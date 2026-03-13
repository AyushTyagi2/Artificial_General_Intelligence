"""Pattern discovery over the memory relation graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


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
        relation_counts: Dict[str, int] = {}
        for _subject, relation, _obj in triplets:
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

        return sorted(rules, key=lambda r: r.count, reverse=True)

    @staticmethod
    def _label_for_relation(relation: str) -> str:
        mapping = {
            "hunts": "predator-prey relationship",
            "orbits": "orbital-system relationship",
            "reacts_with": "chemical interaction",
            "is": "taxonomy relationship",
            "uses": "component-usage relationship",
        }
        return mapping.get(relation, "repeated relational pattern")
