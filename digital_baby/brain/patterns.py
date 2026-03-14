"""Pattern discovery over the memory relation graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple


@dataclass
class PatternRule:
    """Generalized relational rule discovered from repeated structures."""

    relation: str
    count: int
    label: str
    template: str


class PatternDiscoverer:
    """Detects recurring relation structures from graph triplets."""

    def discover(self, triplets: Iterable[Tuple[str, str, str]], min_support: int = 2) -> List[PatternRule]:
        triplet_list = list(triplets)
        relation_counts: Dict[str, int] = {}
        for _subject, relation, _obj in triplet_list:
            relation_counts[relation] = relation_counts.get(relation, 0) + 1

        rules: List[PatternRule] = []
        for relation, count in relation_counts.items():
            if count >= min_support:
                label = self._label_for_relation(relation)
                rules.append(
                    PatternRule(
                        relation=relation,
                        count=count,
                        label=label,
                        template=f"X {relation} Y -> {label}",
                    )
                )

        predator_rule_support = self._count_predator_eats_prey_patterns(triplet_list)
        if predator_rule_support >= min_support:
            rules.append(
                PatternRule(
                    relation="eats",
                    count=predator_rule_support,
                    label="predator-eats-prey rule",
                    template="predator eats prey",
                )
            )

        chain_count = self._count_food_chain_motifs(triplet_list)
        if chain_count >= min_support:
            rules.append(
                PatternRule(
                    relation="food_chain",
                    count=chain_count,
                    label="three-step food chain",
                    template="X hunts Y; Y eats Z; Z grows_in W -> food-chain relationship",
                )
            )

        return sorted(rules, key=lambda r: r.count, reverse=True)

    @staticmethod
    def _count_predator_eats_prey_patterns(triplets: List[Tuple[str, str, str]]) -> int:
        predators: Set[str] = set()
        prey: Set[str] = set()
        for s, r, o in triplets:
            if r == "hunts":
                predators.add(s)
                prey.add(o)

        matched = 0
        for s, r, o in triplets:
            if r == "eats" and s in predators and o in prey:
                matched += 1
        return matched

    def _count_food_chain_motifs(self, triplets: List[Tuple[str, str, str]]) -> int:
        hunts: Dict[str, Set[str]] = {}
        eats: Dict[str, Set[str]] = {}
        grows_in: Dict[str, Set[str]] = {}

        for s, r, o in triplets:
            if r == "hunts":
                hunts.setdefault(s, set()).add(o)
            elif r == "eats":
                eats.setdefault(s, set()).add(o)
            elif r == "grows_in":
                grows_in.setdefault(s, set()).add(o)

        motifs = 0
        for _predator, prey_set in hunts.items():
            for prey_item in prey_set:
                plants = eats.get(prey_item, set())
                for plant in plants:
                    if plant in grows_in:
                        motifs += 1
        return motifs

    @staticmethod
    def _label_for_relation(relation: str) -> str:
        mapping = {
            "hunts": "predator-prey relationship",
            "orbits": "orbital-system relationship",
            "reacts_with": "chemical interaction",
            "is": "taxonomy relationship",
            "uses": "component-usage relationship",
            "eats": "feeding relationship",
            "lives_in": "habitat relationship",
            "needs": "resource dependency",
        }
        return mapping.get(relation, "repeated relational pattern")
