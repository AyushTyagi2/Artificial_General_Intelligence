"""Hypothesis generation and scoring from discovered patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .concepts import ConceptTypeSystem
from .patterns import PatternRule


@dataclass
class Hypothesis:
    rule: str
    concepts: List[str]
    confidence: float
    supporting_evidence: int
    contradicting_evidence: int


class HypothesisEngine:
    """Converts pattern observations into structured typed hypotheses."""

    def from_patterns(
        self,
        patterns: Iterable[PatternRule],
        concept_types: ConceptTypeSystem,
        triplets: Sequence[Tuple[str, str, str]],
    ) -> List[Hypothesis]:
        hypotheses: List[Hypothesis] = []
        by_relation: dict[str, list[Tuple[str, str]]] = {}
        for s, r, o in triplets:
            by_relation.setdefault(r, []).append((s, o))

        for pattern in patterns:
            pairs = by_relation.get(pattern.relation, [])
            typed_rules = [concept_types.typed_relation(s, pattern.relation, o) for s, o in pairs]
            rule = self._majority_rule(typed_rules) if typed_rules else f"unknown {pattern.relation} unknown"
            support = max(1, pattern.count)
            contradict = max(1, len([r for r in typed_rules if r != rule]))
            confidence = support / (support + contradict)
            concepts = self._concepts_for_relation(pattern.relation)
            hypotheses.append(
                Hypothesis(
                    rule=rule,
                    concepts=concepts,
                    confidence=confidence,
                    supporting_evidence=support,
                    contradicting_evidence=contradict,
                )
            )
        return hypotheses

    @staticmethod
    def _majority_rule(rules: Sequence[str]) -> str:
        counts: dict[str, int] = {}
        for rule in rules:
            counts[rule] = counts.get(rule, 0) + 1
        return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]

    @staticmethod
    def _concepts_for_relation(relation: str) -> List[str]:
        mapping = {
            "hunts": ["predator", "prey"],
            "orbits": ["planet_or_moon", "star_or_planet"],
            "is": ["child", "parent"],
            "reacts_with": ["reactant_a", "reactant_b"],
            "food_chain": ["predator", "prey", "producer"],
        }
        return mapping.get(relation, ["entity_a", "entity_b"])
