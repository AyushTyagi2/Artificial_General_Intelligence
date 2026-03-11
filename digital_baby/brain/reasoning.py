"""Reasoning layer for relation extraction and contradiction detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class Conflict:
    """Represents a contradiction between two facts."""

    entity: str
    relation: str
    values: Tuple[str, str]


class Reasoner:
    """Simple symbolic reasoner with lightweight conflict checks."""

    def extract_relation(self, fact: str) -> Tuple[str, str, str] | None:
        """Extract relation triplets from fact text.

        Expected format: "<subject> <relation> <object>".
        """
        tokens = fact.strip().split()
        if len(tokens) < 3:
            return None
        subject = tokens[0].lower()
        relation = tokens[1].lower()
        obj = " ".join(tokens[2:]).lower()
        return subject, relation, obj

    def detect_contradictions(self, triplets: Iterable[Tuple[str, str, str]]) -> List[Conflict]:
        """Detect contradictions where one entity has competing values for a relation."""
        seen: Dict[Tuple[str, str], str] = {}
        conflicts: List[Conflict] = []

        for subject, relation, obj in triplets:
            key = (subject, relation)
            if key in seen and seen[key] != obj:
                conflicts.append(Conflict(entity=subject, relation=relation, values=(seen[key], obj)))
            else:
                seen[key] = obj

        return conflicts
