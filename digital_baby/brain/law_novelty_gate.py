"""Law Novelty Gate — Architecture v2.

Sits between LawDiscovery and the knowledge graph / logger.
Prevents the system from announcing a "new discovery" when it's merely
re-fitting an already-known equation with slightly different coefficients.

Two equations are considered the SAME discovery if:
  1. Same (cause, effect) pair, AND
  2. Same equation family (linear, exponential, ...), AND
  3. Their predicted outputs agree within PARAM_SIMILARITY_THRESHOLD
     over a test grid of x-values.

A law IS considered genuinely new if:
  - The (cause, effect) pair has never produced a law before, OR
  - The equation family has changed (linear → exponential is novel), OR
  - The R² improved by ≥ R2_IMPROVEMENT_THRESHOLD (meaningfully better fit).

Usage
-----
    gate = LawNoveltyGate()

    # Inside the tick loop (replaces direct law_discovery.attempt_fit call):
    law = law_discovery.attempt_fit(cause, effect)
    if law:
        result = gate.evaluate(law)
        if result.is_novel:
            knowledge_graph.ingest_law(law)
            logger.info("[law_discovered] %s r2=%.3f n=%d", ...)
        else:
            logger.debug("[law_gate] suppressed: %s", result.reason)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

PARAM_SIMILARITY_THRESHOLD: float = 0.20   # relative prediction difference
R2_IMPROVEMENT_THRESHOLD:   float = 0.03   # minimum R² gain to count as new
TEST_X_VALUES: List[float] = [0.5, 1.0, 2.0, 5.0, 10.0]   # realistic state variable range

# Suppress a rediscovery announcement for this many ticks
REDISCOVERY_SUPPRESSION_TICKS: int = 25


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LawEvaluation:
    """Result from the novelty gate."""
    is_novel:         bool
    reason:           str
    previous_law_str: Optional[str] = None
    r2_delta:         float = 0.0


@dataclass
class _StoredLaw:
    cause:         str
    effect:        str
    equation_name: str
    params:        List[float]
    r_squared:     float
    equation_str:  str
    discovered_at: int    # tick


# ---------------------------------------------------------------------------
# Equation evaluator (mirrors LawDiscovery._format_equation logic)
# ---------------------------------------------------------------------------

def _safe_exp(x: float, b: float) -> float:
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


def _predict(name: str, params: List[float], x: float) -> float:
    """Evaluate an equation family at x."""
    k = params[0]
    try:
        if name == "linear":
            return k * x
        if name == "power2":
            return k * x ** 2
        if name == "sqrt":
            return k * _safe_sqrt(x)
        if name == "exponential":
            return k * _safe_exp(x, params[1])
        if name == "logarithmic":
            return k * _safe_log(x)
        if name == "inverse":
            return k * _safe_inv(x)
        if name == "arrhenius":
            return k * _safe_exp(1.0, -params[1] / max(abs(x), 1e-9))
    except (IndexError, ZeroDivisionError, OverflowError, ValueError):
        pass
    return 0.0


def _predictions_similar(
    name_a: str, params_a: List[float],
    name_b: str, params_b: List[float],
    xs: List[float],
    threshold: float,
) -> bool:
    """Return True if two equations produce similar outputs on xs."""
    if name_a != name_b:
        return False
    diffs: List[float] = []
    for x in xs:
        ya = _predict(name_a, params_a, x)
        yb = _predict(name_b, params_b, x)
        denom = max(abs(ya), abs(yb), 1e-9)
        diffs.append(abs(ya - yb) / denom)
    mean_diff = sum(diffs) / len(diffs) if diffs else 1.0
    return mean_diff < threshold


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LawNoveltyGate:
    """Filters out redundant law rediscoveries.

    Integrates with existing LawDiscovery without modifying it.
    """

    def __init__(
        self,
        param_similarity_threshold: float = PARAM_SIMILARITY_THRESHOLD,
        r2_improvement_threshold:   float = R2_IMPROVEMENT_THRESHOLD,
    ) -> None:
        self._threshold   = param_similarity_threshold
        self._r2_min_gain = r2_improvement_threshold
        self._stored: Dict[Tuple[str, str], _StoredLaw] = {}
        self._suppressed_count: Dict[Tuple[str, str], int] = {}
        self._total_suppressed = 0
        self._total_novel      = 0

    def evaluate(self, law, current_tick: int = 0) -> LawEvaluation:
        """Check if `law` (a DiscoveredLaw) is genuinely novel.

        Parameters
        ----------
        law          : DiscoveredLaw from LawDiscovery.attempt_fit()
        current_tick : current agent tick (for suppression timeout)

        Returns
        -------
        LawEvaluation with is_novel=True if this should be announced.
        """
        key = (law.cause, law.effect)
        prev = self._stored.get(key)

        # First discovery of this (cause, effect) pair
        if prev is None:
            self._store(law, current_tick)
            self._total_novel += 1
            logger.debug("[law_gate] novel (first) %s", law.equation_str[:60])
            return LawEvaluation(is_novel=True, reason="first_discovery")

        # Family changed → always novel
        if law.equation_name != prev.equation_name:
            self._store(law, current_tick)
            self._total_novel += 1
            reason = f"family_changed:{prev.equation_name}→{law.equation_name}"
            logger.debug("[law_gate] novel (family_change) %s", reason)
            return LawEvaluation(
                is_novel=True,
                reason=reason,
                previous_law_str=prev.equation_str,
                r2_delta=law.r_squared - prev.r_squared,
            )

        # Substantial R² improvement
        r2_delta = law.r_squared - prev.r_squared
        if r2_delta >= self._r2_min_gain:
            self._store(law, current_tick)
            self._total_novel += 1
            logger.debug("[law_gate] novel (r2_improved +%.3f) %s", r2_delta, law.equation_str[:60])
            return LawEvaluation(
                is_novel=True,
                reason=f"r2_improved:{r2_delta:+.3f}",
                previous_law_str=prev.equation_str,
                r2_delta=r2_delta,
            )

        # Predictions similar OR parameters very close → suppress
        similar = _predictions_similar(
            law.equation_name, law.params,
            prev.equation_name, prev.params,
            TEST_X_VALUES,
            self._threshold,
        )
        # Also consider params directly: if all params within 25% → same law
        params_close = all(
            abs(p1 - p2) / (abs(p1) + 1e-9) < 0.25
            for p1, p2 in zip(law.params, prev.params)
        ) if len(law.params) == len(prev.params) else False

        if similar or params_close:
            self._suppressed_count[key] = self._suppressed_count.get(key, 0) + 1
            self._total_suppressed += 1
            reason = (
                f"reparametrisation of '{prev.equation_str[:50]}' "
                f"(suppressed #{self._suppressed_count[key]})"
            )
            return LawEvaluation(
                is_novel=False,
                reason=reason,
                previous_law_str=prev.equation_str,
                r2_delta=r2_delta,
            )

        # Predictions differ despite same family — count as novel
        self._store(law, current_tick)
        self._total_novel += 1
        return LawEvaluation(
            is_novel=True,
            reason="predictions_diverged",
            previous_law_str=prev.equation_str,
            r2_delta=r2_delta,
        )

    def stats(self) -> Dict:
        return {
            "novel_laws":          self._total_novel,
            "suppressed_laws":     self._total_suppressed,
            "tracked_pairs":       len(self._stored),
            "suppression_ratio":   (
                self._total_suppressed /
                max(1, self._total_suppressed + self._total_novel)
            ),
        }

    def get_known_laws(self) -> List[_StoredLaw]:
        return list(self._stored.values())

    # ── Internal ───────────────────────────────────────────────────────────────

    def _store(self, law, tick: int) -> None:
        key = (law.cause, law.effect)
        self._stored[key] = _StoredLaw(
            cause=law.cause,
            effect=law.effect,
            equation_name=law.equation_name,
            params=list(law.params),
            r_squared=law.r_squared,
            equation_str=law.equation_str,
            discovered_at=tick,
        )
        # Reset suppression counter on genuine update
        self._suppressed_count.pop(key, None)