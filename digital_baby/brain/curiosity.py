"""Curiosity model for topic selection and novelty rewards."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Set


@dataclass
class CuriositySignal:
    """Represents curiosity pressure for a topic."""

    topic: str
    score: float
    reason: str


class CuriosityModel:
    """Tracks unknown concepts, prediction error, and topic novelty over time."""

    def __init__(self) -> None:
        self.topic_visits: Dict[str, int] = defaultdict(int)
        self.unknown_concepts_by_topic: Dict[str, Set[str]] = defaultdict(set)
        self.pending_concepts: Deque[str] = deque()
        self.prediction_error_by_topic: Dict[str, float] = defaultdict(float)

    def register_unknowns(self, topic: str, unknown_concepts: Iterable[str]) -> None:
        """Record concepts encountered but not yet grounded in memory."""
        clean = [c.strip().lower() for c in unknown_concepts if c.strip()]
        self.unknown_concepts_by_topic[topic].update(clean)
        for concept in clean:
            if concept not in self.pending_concepts:
                self.pending_concepts.append(concept)

    def register_prediction_error(self, topic: str, error: float) -> None:
        """Increase prediction error signal for topic; decays over visits."""
        self.prediction_error_by_topic[topic] += max(0.0, error)

    def reward(self, novelty: float, conflict_bonus: float = 0.0) -> float:
        """Compute curiosity reward from novelty and conflict pressure."""
        return max(0.0, novelty + conflict_bonus)

    def pop_goal_concept(self) -> Optional[str]:
        """Pop next concept goal if available."""
        if not self.pending_concepts:
            return None
        return self.pending_concepts.popleft()

    def score_topics(self, topics: Iterable[str], weak_fact_ratio_by_topic: Dict[str, float]) -> List[CuriositySignal]:
        """Prioritize topics with novelty, unknowns, weak confidence, and prediction error."""
        signals: List[CuriositySignal] = []
        for topic in topics:
            visits = self.topic_visits[topic]
            novelty = 1.0 / (1 + visits)
            visit_penalty = visits * 0.25
            unknown_weight = len(self.unknown_concepts_by_topic.get(topic, set())) * 0.3
            weak_bonus = weak_fact_ratio_by_topic.get(topic, 0.0) * 1.0
            prediction_error = self.prediction_error_by_topic.get(topic, 0.0)
            score = max(0.0, unknown_weight + weak_bonus + novelty + prediction_error - visit_penalty)
            reason = (
                f"unknown={unknown_weight:.2f}, weak={weak_bonus:.2f}, novelty={novelty:.2f}, "
                f"pred_err={prediction_error:.2f}, visits={visits}"
            )
            signals.append(CuriositySignal(topic=topic, score=score, reason=reason))

        return sorted(signals, key=lambda s: s.score, reverse=True)

    def mark_visited(self, topic: str) -> None:
        """Update exploration history and decay pressure for visited topic."""
        self.topic_visits[topic] += 1
        if topic in self.unknown_concepts_by_topic:
            self.unknown_concepts_by_topic[topic].clear()
        if topic in self.prediction_error_by_topic:
            self.prediction_error_by_topic[topic] *= 0.8
