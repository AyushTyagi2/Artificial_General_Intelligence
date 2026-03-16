"""Prediction Evaluator — v1.

Scores LawPredictor predictions against actual world-step outcomes and feeds
the error signal back into:
  1. The knowledge graph (law-backed edges get faster confidence updates)
  2. The curiosity model (high-error pairs become exploration targets)
  3. An internal accuracy ledger (for dashboard display)

Scoring formula
---------------
For each (cause, effect) pair where a prediction existed:

    relative_error = |predicted_delta - observed_delta| / (|observed_delta| + ε)
    accuracy       = max(0, 1 - relative_error)

    If accuracy >= HIT_THRESHOLD  → confident prediction
        → apply CONFIDENCE_BOOST to the graph edge
    If accuracy < MISS_THRESHOLD  → poor prediction
        → apply CONFIDENCE_PENALTY to the graph edge
        → register as high-curiosity target

Integration
-----------
In EventLoop.__init__:

    from digital_baby.brain.prediction_evaluator import PredictionEvaluator
    self.prediction_evaluator = PredictionEvaluator()

After the world step, inside the effective_domain block:

    if law_predictions:
        eval_results = self.prediction_evaluator.evaluate_all(
            predictions=law_predictions,
            state_before=transition.get("state_before") or {},
            state_after=transition.get("state_after") or {},
            knowledge_graph=self.knowledge_graph,
            tick=tick,
        )
        if eval_results.any_hit:
            self.logger.info(
                "[law_eval] tick=%d hits=%d misses=%d mean_acc=%.3f",
                tick, eval_results.hits, eval_results.misses, eval_results.mean_accuracy,
            )
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIT_THRESHOLD:     float = 0.65   # accuracy >= this → confident prediction
MISS_THRESHOLD:    float = 0.30   # accuracy < this  → poor prediction
EPSILON:           float = 1e-6   # prevents division by zero

CONFIDENCE_BOOST:   float = 0.06  # applied to graph edge on correct prediction
CONFIDENCE_PENALTY: float = 0.03  # applied to graph edge on wrong prediction

# Relations considered "law-grade" causal edges
CAUSAL_RELATIONS = frozenset({
    "positive_affects", "negative_affects", "affects",
    "mixed_affects", "conditional_affects",
})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PredictionScore:
    """Score for a single (cause, effect) prediction."""
    cause:            str
    effect:           str
    predicted_delta:  float
    observed_delta:   float
    accuracy:         float       # 0.0 → 1.0
    is_hit:           bool
    is_miss:          bool
    equation_str:     str
    tick:             int


@dataclass
class EvaluationResult:
    """Aggregate result of evaluating all predictions for one tick."""
    scores:         List[PredictionScore] = field(default_factory=list)
    hits:           int = 0
    misses:         int = 0
    skipped:        int = 0        # predictions where observed delta was near zero

    @property
    def total(self) -> int:
        return len(self.scores)

    @property
    def any_hit(self) -> bool:
        return self.hits > 0

    @property
    def mean_accuracy(self) -> float:
        if not self.scores:
            return 0.0
        return sum(s.accuracy for s in self.scores) / len(self.scores)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PredictionEvaluator:
    """Scores law-derived predictions against world-step outcomes."""

    def __init__(self) -> None:
        self._history: List[PredictionScore] = []
        # (cause, effect) → rolling accuracy (exponential moving average)
        self._ema_accuracy: Dict[Tuple[str, str], float] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def evaluate_all(
        self,
        predictions: list,          # List[LawPrediction]
        state_before: Dict[str, float],
        state_after:  Dict[str, float],
        knowledge_graph,            # KnowledgeGraph
        tick: int = 0,
    ) -> EvaluationResult:
        """Evaluate all predictions against actual state transition.

        Parameters
        ----------
        predictions  : output of LawPredictor.predict_all()
        state_before : domain state dict before the world step
        state_after  : domain state dict after the world step
        knowledge_graph : KnowledgeGraph instance (for confidence updates)
        tick         : current tick number
        """
        result = EvaluationResult()

        for pred in predictions:
            effect = pred.effect
            before = state_before.get(effect)
            after  = state_after.get(effect)

            # Skip if we can't observe the effect variable
            if before is None or after is None:
                result.skipped += 1
                continue

            observed_delta = after - before

            # Skip near-zero observations — no signal
            if abs(observed_delta) < EPSILON:
                result.skipped += 1
                continue

            accuracy = self._score_accuracy(pred.predicted_delta, observed_delta)
            is_hit   = accuracy >= HIT_THRESHOLD
            is_miss  = accuracy < MISS_THRESHOLD

            score = PredictionScore(
                cause=pred.cause,
                effect=pred.effect,
                predicted_delta=pred.predicted_delta,
                observed_delta=observed_delta,
                accuracy=accuracy,
                is_hit=is_hit,
                is_miss=is_miss,
                equation_str=pred.equation_str,
                tick=tick,
            )
            result.scores.append(score)
            self._history.append(score)
            if len(self._history) > 1000:
                self._history = self._history[-1000:]

            if is_hit:
                result.hits += 1
                self._update_graph_confidence(
                    knowledge_graph, pred.cause, pred.effect,
                    CONFIDENCE_BOOST, tick,
                )
                logger.info(
                    "[law_eval] HIT  %s → %s  pred=%.4f obs=%.4f acc=%.3f  [%s]",
                    pred.cause, pred.effect,
                    pred.predicted_delta, observed_delta, accuracy,
                    pred.equation_name,
                )
            elif is_miss:
                result.misses += 1
                self._update_graph_confidence(
                    knowledge_graph, pred.cause, pred.effect,
                    -CONFIDENCE_PENALTY, tick,
                )
                logger.info(
                    "[law_eval] MISS %s → %s  pred=%.4f obs=%.4f acc=%.3f  [%s]",
                    pred.cause, pred.effect,
                    pred.predicted_delta, observed_delta, accuracy,
                    pred.equation_name,
                )
            else:
                logger.debug(
                    "[law_eval] NEAR %s → %s  pred=%.4f obs=%.4f acc=%.3f",
                    pred.cause, pred.effect,
                    pred.predicted_delta, observed_delta, accuracy,
                )

            # Update EMA accuracy
            key = (pred.cause, pred.effect)
            prev_ema = self._ema_accuracy.get(key, accuracy)
            self._ema_accuracy[key] = 0.8 * prev_ema + 0.2 * accuracy

        return result

    def get_ema_accuracy(self, cause: str, effect: str) -> Optional[float]:
        """Return rolling accuracy for a pair, or None if never evaluated."""
        return self._ema_accuracy.get((cause, effect))

    def worst_pairs(self, n: int = 5) -> List[Tuple[str, str, float]]:
        """Return the n (cause, effect) pairs with lowest EMA accuracy."""
        items = [
            (c, e, acc)
            for (c, e), acc in self._ema_accuracy.items()
        ]
        items.sort(key=lambda x: x[2])
        return items[:n]

    def best_pairs(self, n: int = 5) -> List[Tuple[str, str, float]]:
        """Return the n (cause, effect) pairs with highest EMA accuracy."""
        items = [
            (c, e, acc)
            for (c, e), acc in self._ema_accuracy.items()
        ]
        items.sort(key=lambda x: -x[2])
        return items[:n]

    def summary(self) -> str:
        total = len(self._history)
        if not total:
            return "PredictionEvaluator: no evaluations yet"
        hits   = sum(1 for s in self._history if s.is_hit)
        misses = sum(1 for s in self._history if s.is_miss)
        mean   = sum(s.accuracy for s in self._history) / total
        return (
            f"PredictionEvaluator: total={total} "
            f"hits={hits} misses={misses} mean_acc={mean:.3f}"
        )

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _score_accuracy(predicted: float, observed: float) -> float:
        """Compute accuracy in [0, 1]. 1.0 = perfect, 0.0 = wildly wrong."""
        if abs(observed) < EPSILON:
            return 0.0
        relative_error = abs(predicted - observed) / (abs(observed) + EPSILON)
        # Cap relative error at 2.0 so accuracy floor is 0
        return max(0.0, 1.0 - min(relative_error, 2.0) / 2.0)

    @staticmethod
    def _update_graph_confidence(
        graph,
        cause: str,
        effect: str,
        delta: float,
        tick: int,
    ) -> None:
        """Apply a confidence delta to all causal edges between cause → effect."""
        c = graph._normalize(cause)
        e = graph._normalize(effect)
        for edge in graph.edges:
            if edge.source == c and edge.target == e and edge.relation in CAUSAL_RELATIONS:
                old_conf = edge.confidence
                edge.confidence = max(0.05, min(0.99, edge.confidence + delta))
                # Update status tier
                from digital_baby.brain.knowledge_graph import _edge_status
                edge.status = _edge_status(edge.confidence)
                edge.last_tick = tick
                logger.debug(
                    "[law_eval] conf_update %s→%s %.3f%+.3f=%.3f",
                    cause, effect, old_conf, delta, edge.confidence,
                )
                break