"""Curiosity model for topic selection and novelty rewards."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set


@dataclass
class CuriositySignal:
    """Represents curiosity pressure for a topic."""

    topic: str
    score: float
    reason: str


class CuriosityModel:
    """Tracks unknown concepts and topic novelty over time."""

    def __init__(self) -> None:
        self.topic_visits: Dict[str, int] = defaultdict(int)
        self.unknown_concepts_by_topic: Dict[str, Set[str]] = defaultdict(set)

    def register_unknowns(self, topic: str, unknown_concepts: Iterable[str]) -> None:
        """Record concepts encountered but not yet grounded in memory."""
        self.unknown_concepts_by_topic[topic].update(unknown_concepts)

    def reward(self, novelty: float, conflict_bonus: float = 0.0) -> float:
        """Compute curiosity reward from novelty and conflict pressure."""
        return max(0.0, novelty + conflict_bonus)

    def score_topics(self, topics: Iterable[str], weak_fact_ratio_by_topic: Dict[str, float]) -> List[CuriositySignal]:
        """Prioritize topics with novelty and weak-confidence knowledge."""
        signals: List[CuriositySignal] = []
        for topic in topics:
            visits = self.topic_visits[topic]
            exploration_bonus = 0.8 / (1 + visits)
            visit_penalty = visits * 0.1
            unknown_bonus = len(self.unknown_concepts_by_topic.get(topic, set())) * 0.1
            weak_bonus = weak_fact_ratio_by_topic.get(topic, 0.0) * 0.6
            score = max(0.0, 1.0 + exploration_bonus + unknown_bonus + weak_bonus - visit_penalty)
            reason = (
                f"unknown={len(self.unknown_concepts_by_topic.get(topic, set()))}, "
                f"weak_bonus={weak_bonus:.2f}, visits={visits}"
            )
            signals.append(CuriositySignal(topic=topic, score=score, reason=reason))

        return sorted(signals, key=lambda s: s.score, reverse=True)

    def mark_visited(self, topic: str) -> None:
        """Update exploration history."""
        self.topic_visits[topic] += 1
