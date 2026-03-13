"""Hypothesis generation and scoring from discovered patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .patterns import PatternRule


@dataclass
class Hypothesis:
    """Candidate world-model rule with evidence and confidence."""

    rule: str
    concepts: List[str]
    confidence: float
    supporting_evidence: int
    contradicting_evidence: int


class HypothesisEngine:
    """Converts pattern observations into structured hypotheses."""

    def from_patterns(self, patterns: Iterable[PatternRule]) -> List[Hypothesis]:
        hypotheses: List[Hypothesis] = []
        for pattern in patterns:
            concepts = self._concepts_for_relation(pattern.relation)
            support = max(1, pattern.count)
            hypotheses.append(
                Hypothesis(
                    rule=f"X {pattern.relation} Y",
                    concepts=concepts,
                    confidence=support / (support + 2),
                    supporting_evidence=support,
                    contradicting_evidence=2,
                )
            )
        return hypotheses

    @staticmethod
    def _concepts_for_relation(relation: str) -> List[str]:
        mapping = {
            "hunts": ["predator", "prey"],
            "orbits": ["orbiter", "anchor"],
            "is": ["child", "parent"],
            "reacts_with": ["reactant_a", "reactant_b"],
            "food_chain": ["predator", "prey", "producer"],
        }
        return mapping.get(relation, ["entity_a", "entity_b"])
