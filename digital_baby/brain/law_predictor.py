"""Law Predictor — v1.

Takes discovered laws from LawDiscovery and evaluates them numerically to
generate concrete predictions before a world step runs.

The prediction pipeline:
    state_before + discovered_law → predicted_effect_delta
    world_step runs
    state_after → observed_effect_delta
    PredictionEvaluator scores the prediction

Design
------
- Keeps an index of all active laws keyed by (cause, effect)
- When the event loop is about to run an intervention on a domain,
  call predict_all(state, domain_vars) to get a dict of predictions
- After the world step, pass predictions + actual deltas to
  PredictionEvaluator.evaluate_all()

Integration
-----------
In EventLoop.__init__:

    from digital_baby.brain.law_predictor import LawPredictor
    self.law_predictor = LawPredictor()

In the law-discovery block (after absorb_law):

    if law_eval.is_novel:
        self.knowledge_graph.ingest_law(law)
        self.theory_abstraction.absorb_law(law)
        self.law_predictor.register_law(law)   # ← ADD

Before the world step (inside the effective_domain block):

    law_predictions = self.law_predictor.predict_all(
        state=self.generator.state_for_domain(effective_domain),
        tick=tick,
    )

After the world step, pass law_predictions to PredictionEvaluator.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LawPrediction:
    """A single numeric prediction derived from a discovered law."""
    cause:          str
    effect:         str
    cause_value:    float        # x input to the equation
    predicted_delta: float       # predicted change in effect variable
    equation_name:  str
    equation_str:   str
    law_r_squared:  float
    tick:           int


# ---------------------------------------------------------------------------
# Equation evaluators (mirror the families in law_discovery.py)
# ---------------------------------------------------------------------------

def _safe_exp(b_x: float) -> float:
    try:
        return math.exp(min(700.0, b_x))
    except (OverflowError, ValueError):
        return 1e9


def _safe_log(x: float) -> float:
    return math.log(max(abs(x), 1e-9))


def _safe_sqrt(x: float) -> float:
    return math.sqrt(max(x, 0.0))


def _safe_inv(x: float) -> float:
    return 1.0 / max(abs(x), 1e-9) * (1.0 if x >= 0 else -1.0)


def _evaluate(equation_name: str, params: List[float], x: float) -> Optional[float]:
    """Evaluate a named equation at x with given params. Returns None on error."""
    try:
        if equation_name == "linear":
            k = params[0]
            return k * x
        if equation_name == "power2":
            k = params[0]
            return k * x ** 2
        if equation_name == "sqrt":
            k = params[0]
            return k * _safe_sqrt(x)
        if equation_name == "exponential":
            k, b = params[0], params[1]
            return k * _safe_exp(b * x)
        if equation_name == "logarithmic":
            k = params[0]
            return k * _safe_log(x)
        if equation_name == "inverse":
            k = params[0]
            return k * _safe_inv(x)
        if equation_name == "arrhenius":
            k, ea = params[0], params[1]
            return k * _safe_exp(-ea / max(abs(x), 1e-9))
    except Exception as exc:
        logger.debug("[law_predictor] eval_error eq=%s x=%.4f err=%s", equation_name, x, exc)
    return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LawPredictor:
    """Generates numeric predictions from discovered laws."""

    def __init__(self) -> None:
        # (cause, effect) → latest DiscoveredLaw
        self._laws: Dict[Tuple[str, str], object] = {}
        # Recent predictions waiting for evaluation
        self._pending: List[LawPrediction] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def register_law(self, law) -> None:
        """Register or update a discovered law.

        Parameters
        ----------
        law : DiscoveredLaw  (from brain.law_discovery)
        """
        key = (law.cause, law.effect)
        prev = self._laws.get(key)
        self._laws[key] = law
        if prev is None:
            logger.info(
                "[law_predictor] registered cause=%r effect=%r eq=%s r2=%.3f",
                law.cause, law.effect, law.equation_name, law.r_squared,
            )
        elif law.r_squared > getattr(prev, "r_squared", 0) + 0.01:
            logger.info(
                "[law_predictor] updated cause=%r effect=%r eq=%s r2=%.3f→%.3f",
                law.cause, law.effect, law.equation_name,
                getattr(prev, "r_squared", 0), law.r_squared,
            )

    def predict_all(
        self,
        state: Dict[str, float],
        tick: int = 0,
    ) -> List[LawPrediction]:
        """Generate predictions for all registered laws whose cause is in state.

        Returns a list of LawPrediction objects.  Also stores them as pending
        for later evaluation.
        """
        predictions: List[LawPrediction] = []

        for (cause, effect), law in self._laws.items():
            if cause not in state:
                continue
            x = state[cause]
            if not math.isfinite(x):
                continue

            y_pred = _evaluate(law.equation_name, law.params, x)
            if y_pred is None or not math.isfinite(y_pred):
                continue

            pred = LawPrediction(
                cause=cause,
                effect=effect,
                cause_value=x,
                predicted_delta=y_pred,
                equation_name=law.equation_name,
                equation_str=law.equation_str,
                law_r_squared=law.r_squared,
                tick=tick,
            )
            predictions.append(pred)

        self._pending = predictions
        logger.debug("[law_predictor] tick=%d predictions=%d", tick, len(predictions))
        return predictions

    def get_prediction(self, cause: str, effect: str) -> Optional[LawPrediction]:
        """Return the most recent pending prediction for a specific pair."""
        for p in self._pending:
            if p.cause == cause and p.effect == effect:
                return p
        return None

    def law_count(self) -> int:
        return len(self._laws)

    def get_all_laws(self) -> list:
        return list(self._laws.values())