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

    semantic_templates: List[Tuple[str, str, str]] = [
        ("animal", "eats", "plant"),
        ("predator", "hunts", "prey"),
        ("animal", "lives_in", "ecosystem"),
        ("plant", "needs", "sunlight"),
    ]

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

        hypotheses.extend(self.generate_semantic_hypotheses(concept_types, triplets))
        return self._deduplicate(hypotheses)

    def generate_semantic_hypotheses(
        self,
        concept_types: ConceptTypeSystem,
        triplets: Sequence[Tuple[str, str, str]],
        max_hypotheses: int = 8,
    ) -> List[Hypothesis]:
        """Generate meaningful developmental hypotheses from semantic templates."""
        relation_set = {r for _, r, _ in triplets}
        hypotheses: List[Hypothesis] = []

        for subj_type, relation, obj_type in self.semantic_templates:
            support = sum(1 for s, r, o in triplets if r == relation and concept_types.get_type(s) == subj_type and concept_types.get_type(o) == obj_type)
            contradict = sum(1 for _s, r, _o in triplets if r == relation) - support
            confidence = (support + 1) / max(1, support + contradict + 2)
            hypotheses.append(
                Hypothesis(
                    rule=f"{subj_type} {relation} {obj_type}",
                    concepts=[subj_type, obj_type],
                    confidence=confidence,
                    supporting_evidence=max(1, support),
                    contradicting_evidence=max(1, contradict),
                )
            )
            if relation not in relation_set:
                hypotheses[-1].confidence *= 0.6

        return hypotheses[:max_hypotheses]

    @staticmethod
    def _deduplicate(hypotheses: Sequence[Hypothesis]) -> List[Hypothesis]:
        best: dict[str, Hypothesis] = {}
        for hypothesis in hypotheses:
            existing = best.get(hypothesis.rule)
            if existing is None or hypothesis.confidence > existing.confidence:
                best[hypothesis.rule] = hypothesis
        return list(best.values())

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
            "lives_in": ["animal", "ecosystem"],
            "needs": ["living_thing", "resource"],
            "eats": ["consumer", "food"],
        }
        return mapping.get(relation, ["entity_a", "entity_b"])
