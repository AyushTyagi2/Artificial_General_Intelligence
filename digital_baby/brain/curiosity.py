"""Curiosity model for topic selection and novelty rewards."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Set


@dataclass
class CuriositySignal:
    topic: str
    score: float
    reason: str


class CuriosityModel:
    """Tracks unknowns, prediction error, novelty, and discovered patterns."""

    UNKNOWN_CONCEPT_BONUS = 5.0
    PREDICTION_ERROR_BONUS = 10.0
    NEW_RELATION_BONUS = 3.0
    NEW_PATTERN_BONUS = 7.0
    ERROR_THRESHOLD = 0.3

    def __init__(self) -> None:
        self.topic_visits: Dict[str, int] = defaultdict(int)
        self.unknown_concepts_by_topic: Dict[str, Set[str]] = defaultdict(set)
        self.pending_concepts: Deque[str] = deque()
        self.prediction_error_by_topic: Dict[str, float] = defaultdict(float)
        self.structural_novelty_by_topic: Dict[str, float] = defaultdict(float)
        self.hypothesis_uncertainty_by_topic: Dict[str, float] = defaultdict(float)
        self.experiment_signal_by_topic: Dict[str, float] = defaultdict(float)
        self.new_patterns_by_topic: Dict[str, int] = defaultdict(int)

    def curiosity_score(
        self,
        unknown_concepts: Iterable[str],
        prediction_error: float,
        unexplored_concepts: Iterable[str],
        new_relation: bool,
        new_pattern: bool = False,
    ) -> float:
        unknown_count = len([c for c in unknown_concepts if c])
        unexplored_count = len([c for c in unexplored_concepts if c])
        score = 0.0
        if unknown_count > 0:
            score += self.UNKNOWN_CONCEPT_BONUS
        if prediction_error > self.ERROR_THRESHOLD:
            score += prediction_error * self.PREDICTION_ERROR_BONUS
        if unexplored_count > 0:
            score += self.UNKNOWN_CONCEPT_BONUS + (unexplored_count - 1)
        if new_relation:
            score += self.NEW_RELATION_BONUS
        if new_pattern:
            score += self.NEW_PATTERN_BONUS
        return score

    def register_unknowns(self, topic: str, unknown_concepts: Iterable[str]) -> None:
        clean = [c.strip().lower() for c in unknown_concepts if c.strip()]
        self.unknown_concepts_by_topic[topic].update(clean)
        for concept in clean:
            if concept not in self.pending_concepts:
                self.pending_concepts.append(concept)

    def register_prediction_error(self, topic: str, error: float) -> None:
        self.prediction_error_by_topic[topic] += max(0.0, error)

    def register_new_pattern(self, topic: str) -> None:
        self.new_patterns_by_topic[topic] += 1

    def register_structural_novelty(self, topic: str, novelty: float) -> None:
        self.structural_novelty_by_topic[topic] += max(0.0, novelty)

    def register_hypothesis_uncertainty(self, topic: str, uncertainty: float) -> None:
        self.hypothesis_uncertainty_by_topic[topic] += max(0.0, uncertainty)

    def register_experiment_signal(self, topic: str, signal: float) -> None:
        self.experiment_signal_by_topic[topic] += max(0.0, signal)

    def reward(self, novelty: float, conflict_bonus: float = 0.0) -> float:
        return max(0.0, novelty + conflict_bonus)

    def pop_goal_concept(self) -> Optional[str]:
        if not self.pending_concepts:
            return None
        return self.pending_concepts.popleft()

    def score_topics(self, topics: Iterable[str], weak_fact_ratio_by_topic: Dict[str, float]) -> List[CuriositySignal]:
        signals: List[CuriositySignal] = []
        for topic in topics:
            visits = self.topic_visits[topic]
            novelty = 1.0 / (1 + visits)
            visit_penalty = visits * 0.25
            unknown_weight = len(self.unknown_concepts_by_topic.get(topic, set())) * 0.3
            weak_bonus = weak_fact_ratio_by_topic.get(topic, 0.0)
            prediction_error = self.prediction_error_by_topic.get(topic, 0.0)
            structural = self.structural_novelty_by_topic.get(topic, 0.0)
            hypothesis_uncertainty = self.hypothesis_uncertainty_by_topic.get(topic, 0.0)
            experiment_signal = self.experiment_signal_by_topic.get(topic, 0.0)
            pattern_bonus = self.new_patterns_by_topic.get(topic, 0) * 0.5

            score = max(0.0, novelty + unknown_weight + weak_bonus + prediction_error + structural + hypothesis_uncertainty + experiment_signal + pattern_bonus - visit_penalty)
            reason = (
                f"novelty={novelty:.2f}, unknown={unknown_weight:.2f}, weak={weak_bonus:.2f}, "
                f"pred_err={prediction_error:.2f}, structural={structural:.2f}, hyp_unc={hypothesis_uncertainty:.2f}, "
                f"exp={experiment_signal:.2f}, pattern={pattern_bonus:.2f}, visits={visits}"
            )
            signals.append(CuriositySignal(topic=topic, score=score, reason=reason))

        return sorted(signals, key=lambda s: s.score, reverse=True)

    def mark_visited(self, topic: str) -> None:
        self.topic_visits[topic] += 1
        if topic in self.unknown_concepts_by_topic:
            self.unknown_concepts_by_topic[topic].clear()
        self.prediction_error_by_topic[topic] *= 0.8
        self.structural_novelty_by_topic[topic] *= 0.6
        self.hypothesis_uncertainty_by_topic[topic] *= 0.7
        self.experiment_signal_by_topic[topic] *= 0.7
        self.new_patterns_by_topic[topic] = max(0, self.new_patterns_by_topic[topic] - 1)
