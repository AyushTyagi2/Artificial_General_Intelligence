"""Scientific law discovery for digital_baby.

Watches pairs of state variables that share a causal edge and attempts to fit
parametric equations to the (cause_value, effect_delta) observations collected
during world-simulator interventions.

When a good fit is found (R² ≥ MIN_R_SQUARED) the discovered law is stored as
a human-readable equation string, annotated onto the causal edge in the
knowledge graph, and logged at INFO level.

Candidate equation families
-----------------------------
Name          Formula                       Typical use-case
──────────    ────────────────────────────  ──────────────────────────────
linear        y = k × x                     direct proportionality
power2        y = k × x²                    quadratic growth
sqrt          y = k × √x                    sub-linear growth
exponential   y = k × exp(b × x)            Malthusian / compound
logarithmic   y = k × log(x)               diminishing returns
inverse       y = k / x                     inverse proportionality
arrhenius     y = k × exp(−Eₐ / x)         rate vs temperature (chemistry)

Fitting method
--------------
A pure-Python coordinate-descent optimiser is used so the module has no
external dependencies.  For production use, swap ``_coordinate_descent()``
for ``scipy.optimize.minimize`` (see the comment in that method) to get
10–100× speedup and better convergence on noisy data.

Observation recording
---------------------
Call ``record_observation(cause, effect, x_val, y_val)`` from the event loop
whenever a world-step produces non-trivial state changes.  Cap at 200 samples
per (cause, effect) pair to avoid memory bloat.

Fitting trigger
---------------
Call ``attempt_fit(cause, effect)`` every ~25 ticks for active causal pairs.
Returns a ``DiscoveredLaw`` if a fit meeting the threshold was found, else None.

Integration into event loop
---------------------------

    # In _run_stateful_world_step(), after applying deltas:
    for cause_var, delta_cause in deltas.items():
        for effect_var, delta_effect in deltas.items():
            if cause_var != effect_var and abs(delta_cause) > 1e-6:
                x_val = state.get(cause_var, 0.0)
                self.law_discovery.record_observation(
                    cause_var, effect_var, x_val, delta_effect
                )

    # In run(), every 25 ticks:
    if tick % 25 == 0:
        for rule in self.memory.get_causal_rules():
            law = self.law_discovery.attempt_fit(rule.cause, rule.effect)
            if law:
                self.knowledge_graph.ingest_law(law)
                self.logger.info(
                    "[law_discovered] %s r2=%.3f n=%d",
                    law.equation_str, law.r_squared, law.n_datapoints,
                )
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import json
from pathlib import Path
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_DATAPOINTS: int   = 8      # minimum (x, y) pairs needed for fitting
MIN_R_SQUARED:  float = 0.60   # goodness-of-fit threshold (lowered from 0.85 — real
                                # world-sim delta data has σ≈0.3; linear R²≈0.62 at
                                # 25 points, so 0.60 is the practical minimum threshold)
MAX_SAMPLES:    int   = 200    # maximum observations kept per (cause, effect) pair

_COORD_DESCENT_STEPS: int   = 600
_COORD_DESCENT_STEP:  float = 0.05


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DiscoveredLaw:
    """A mathematical relationship discovered between two causal variables."""
    cause:          str
    effect:         str
    equation_name:  str
    params:         List[float]
    r_squared:      float
    n_datapoints:   int
    equation_str:   str     # e.g. "reaction_rate = 2.3 × exp(0.08 × temperature)"
    confidence:     float   # derived from R²: 0.9 × R² + 0.1

    def __str__(self) -> str:
        return f"{self.equation_str}  [R²={self.r_squared:.3f}, n={self.n_datapoints}]"


@dataclass
class _EquationSpec:
    """One candidate equation family."""
    name:    str
    fn:      Callable                 # fn(x, *params) → y
    n_params: int
    p0:      List[float]              # initial parameter guess


# ---------------------------------------------------------------------------
# Candidate equations
# ---------------------------------------------------------------------------

def _safe_exp(x: float, b: float = 1.0) -> float:
    try:
        return math.exp(min(700.0, b * x))
    except (OverflowError, ValueError):
        return float("inf")

def _safe_log(x: float) -> float:
    return math.log(max(x, 1e-9))

def _safe_sqrt(x: float) -> float:
    return math.sqrt(max(x, 0.0))

def _safe_inv(x: float) -> float:
    return 1.0 / max(abs(x), 1e-9) * (1 if x >= 0 else -1)


CANDIDATE_EQUATIONS: List[_EquationSpec] = [
    _EquationSpec(
        "linear",
        lambda x, k: k * x,
        1, [1.0],
    ),
    _EquationSpec(
        "power2",
        lambda x, k: k * x ** 2,
        1, [0.1],
    ),
    _EquationSpec(
        "sqrt",
        lambda x, k: k * _safe_sqrt(x),
        1, [1.0],
    ),
    _EquationSpec(
        "exponential",
        lambda x, k, b: k * _safe_exp(x, b),
        2, [1.0, 0.05],
    ),
    _EquationSpec(
        "logarithmic",
        lambda x, k: k * _safe_log(x),
        1, [1.0],
    ),
    _EquationSpec(
        "inverse",
        lambda x, k: k * _safe_inv(x),
        1, [1.0],
    ),
    _EquationSpec(
        "arrhenius",
        lambda x, k, Ea: k * _safe_exp(1.0, -Ea / max(abs(x), 1e-9)),
        2, [1.0, 1.0],
    ),
]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LawDiscovery:
    """Fits parametric equations to causal (x, y) observations."""

    def __init__(
        self,
        min_datapoints: int   = MIN_DATAPOINTS,
        min_r_squared:  float = MIN_R_SQUARED,
        max_samples:    int   = MAX_SAMPLES,
    ) -> None:
        self.min_datapoints = min_datapoints
        self.min_r_squared  = min_r_squared
        self.max_samples    = max_samples

        # (cause, effect) → list of (x, y) observations
        self._data: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        # (cause, effect) → best discovered law (None if no fit yet)
        self._laws: Dict[Tuple[str, str], Optional[DiscoveredLaw]]  = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def record_observation(
        self,
        cause:  str,
        effect: str,
        x_val:  float,
        y_val:  float,
    ) -> None:
        """Record one (cause_value, effect_delta) observation.

        Filters out near-zero observations that add no signal, and enforces
        the maximum sample cap with a sliding window (keep most recent).
        """
        if not math.isfinite(x_val) or not math.isfinite(y_val):
            return
        if abs(y_val) < 1e-9:
            return   # no observable effect — skip

        key = (cause, effect)
        bucket = self._data.setdefault(key, [])
        bucket.append((x_val, y_val))
        if len(bucket) > self.max_samples:
            self._data[key] = bucket[-self.max_samples:]

    def attempt_fit(
        self,
        cause:  str,
        effect: str,
    ) -> Optional[DiscoveredLaw]:
        """Attempt to fit all candidate equations and return the best.

        Returns a DiscoveredLaw if R² ≥ min_r_squared, otherwise None.
        Already-discovered laws are re-evaluated when new data arrives.
        """
        key    = (cause, effect)
        points = self._data.get(key, [])

        if len(points) < self.min_datapoints:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        best_law: Optional[DiscoveredLaw] = None
        best_r2  = -1.0

        for spec in CANDIDATE_EQUATIONS:
            try:
                params, r2 = self._fit(spec, xs, ys)
            except Exception as exc:
                logger.debug("[law_discovery] fit_failed eq=%s error=%s", spec.name, exc)
                continue

            if r2 > best_r2:
                best_r2 = r2
                if r2 >= self.min_r_squared:
                    eq_str = self._format_equation(spec.name, cause, effect, params)
                    best_law = DiscoveredLaw(
                        cause=cause,
                        effect=effect,
                        equation_name=spec.name,
                        params=params,
                        r_squared=r2,
                        n_datapoints=len(points),
                        equation_str=eq_str,
                        confidence=min(0.99, r2 * 0.9 + 0.1),
                    )

        if best_law is not None:
            prev = self._laws.get(key)
            is_new = prev is None
            is_better = prev is not None and best_law.r_squared > prev.r_squared + 0.02
            eq_changed = prev is not None and best_law.equation_str != prev.equation_str
            if is_new or is_better or eq_changed:
                self._laws[key] = best_law
                logger.info(
                    "[law_discovery] discovered %s  r2=%.3f  n=%d",
                    best_law.equation_str, best_law.r_squared, best_law.n_datapoints,
                )
            # Return None if law unchanged — prevents redundant graph annotations
            return best_law if (is_new or is_better or eq_changed) else None

        return None

    def get_all_laws(self) -> List[DiscoveredLaw]:
        """Return all laws discovered so far."""
        return [law for law in self._laws.values() if law is not None]

    def observation_count(self, cause: str, effect: str) -> int:
        """Return how many (x, y) pairs have been recorded for this pair."""
        return len(self._data.get((cause, effect), []))

    # ── Fitting ────────────────────────────────────────────────────────────────

    def _fit(
        self,
        spec: _EquationSpec,
        xs: List[float],
        ys: List[float],
    ) -> Tuple[List[float], float]:
        """Fit one equation family and return (best_params, r_squared)."""
        params = list(spec.p0)
        best_sse = self._sse(spec.fn, params, xs, ys)

        # ── Coordinate-descent optimiser ──────────────────────────────────────
        # Each iteration walks each parameter dimension in both directions,
        # keeping the move if it reduces SSE.
        #
        # To use scipy instead (recommended for production):
        #
        #   from scipy.optimize import minimize
        #   result = minimize(
        #       lambda p: self._sse(spec.fn, list(p), xs, ys),
        #       x0=spec.p0,
        #       method="Nelder-Mead",
        #       options={"maxiter": 2000, "xatol": 1e-6},
        #   )
        #   params = list(result.x)
        #   best_sse = result.fun

        for _ in range(_COORD_DESCENT_STEPS):
            improved = False
            for i in range(len(params)):
                for delta in (_COORD_DESCENT_STEP, -_COORD_DESCENT_STEP,
                              _COORD_DESCENT_STEP * 0.1, -_COORD_DESCENT_STEP * 0.1):
                    trial = params[:]
                    trial[i] += delta
                    score = self._sse(spec.fn, trial, xs, ys)
                    if score < best_sse:
                        best_sse = score
                        params   = trial
                        improved = True
            if not improved:
                break   # early exit when converged

        # Compute R²
        r2 = self._r_squared(spec.fn, params, xs, ys)
        return params, r2

    @staticmethod
    def _sse(
        fn: Callable,
        params: List[float],
        xs: List[float],
        ys: List[float],
    ) -> float:
        total = 0.0
        for x, y in zip(xs, ys):
            try:
                pred = fn(x, *params)
                if not math.isfinite(pred):
                    return float("inf")
                total += (pred - y) ** 2
            except (ZeroDivisionError, ValueError, OverflowError):
                return float("inf")
        return total

    @staticmethod
    def _r_squared(
        fn: Callable,
        params: List[float],
        xs: List[float],
        ys: List[float],
    ) -> float:
        try:
            y_mean  = statistics.mean(ys)
            ss_tot  = sum((y - y_mean) ** 2 for y in ys)
            if ss_tot < 1e-12:
                return 0.0
            preds   = [fn(x, *params) for x in xs]
            ss_res  = sum((y - p) ** 2 for y, p in zip(ys, preds))
            return max(-1.0, 1.0 - ss_res / ss_tot)
        except Exception:
            return -1.0

    # ── Equation formatting ───────────────────────────────────────────────────

    @staticmethod
    def _format_equation(
        name:   str,
        cause:  str,
        effect: str,
        params: List[float],
    ) -> str:
        k = round(params[0], 4)
        if name == "linear":
            return f"{effect} = {k} × {cause}"
        if name == "power2":
            return f"{effect} = {k} × {cause}²"
        if name == "sqrt":
            return f"{effect} = {k} × √{cause}"
        if name == "exponential":
            b = round(params[1], 4)
            return f"{effect} = {k} × exp({b} × {cause})"
        if name == "logarithmic":
            return f"{effect} = {k} × log({cause})"
        if name == "inverse":
            return f"{effect} = {k} / {cause}"
        if name == "arrhenius":
            ea = round(params[1], 4)
            return f"{effect} = {k} × exp(−{ea} / {cause})"
        return f"{effect} = f({cause})"
    def save(self, path) -> None:
        """Persist observation data and discovered laws to disk."""

        payload = {
            "data": {
                f"{c}||{e}": pts
                for (c, e), pts in self._data.items()
            },
            "laws": {
                f"{c}||{e}": {
                    "cause": law.cause, "effect": law.effect,
                    "equation_name": law.equation_name,
                    "params": law.params, "r_squared": law.r_squared,
                    "n_datapoints": law.n_datapoints,
                    "equation_str": law.equation_str,
                    "confidence": law.confidence,
                }
                for (c, e), law in self._laws.items() if law is not None
            }
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path) -> None:
        """Restore observation data and laws from disk."""
        import json
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return
        for key, pts in payload.get("data", {}).items():
            c, e = key.split("||", 1)
            self._data[(c, e)] = [tuple(p) for p in pts]
        for key, d in payload.get("laws", {}).items():
            c, e = key.split("||", 1)
            self._laws[(c, e)] = DiscoveredLaw(**d)
        logger.info("[law_discovery] loaded %d pairs, %d laws from %s",
                    len(self._data), len(self._laws), path)