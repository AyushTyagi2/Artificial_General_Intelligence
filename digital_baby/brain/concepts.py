"""Concept hierarchy and entity type system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class HierarchyEdge:
    child: str
    parent: str


class ConceptHierarchy:
    """Builds and queries concept hierarchies inferred from 'is' relations."""

    def __init__(self) -> None:
        self.parent_by_child: Dict[str, Set[str]] = {}
        self.children_by_parent: Dict[str, Set[str]] = {}

    def ingest_triplets(self, triplets: Iterable[Tuple[str, str, str]]) -> None:
        self.parent_by_child.clear()
        self.children_by_parent.clear()
        for subject, relation, obj in triplets:
            if relation != "is":
                continue
            self.parent_by_child.setdefault(subject, set()).add(obj)
            self.children_by_parent.setdefault(obj, set()).add(subject)

    def ancestors(self, concept: str) -> List[str]:
        visited: Set[str] = set()
        stack = [concept]
        result: List[str] = []
        while stack:
            node = stack.pop()
            for parent in self.parent_by_child.get(node, set()):
                if parent not in visited:
                    visited.add(parent)
                    result.append(parent)
                    stack.append(parent)
        return sorted(result)

    def top_concepts_by_connectivity(self, top_n: int = 5) -> List[Tuple[str, int]]:
        nodes = set(self.parent_by_child) | set(self.children_by_parent)
        scored = []
        for node in nodes:
            score = len(self.children_by_parent.get(node, set())) + len(self.parent_by_child.get(node, set()))
            scored.append((node, score))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]

    def edges(self) -> List[HierarchyEdge]:
        out: List[HierarchyEdge] = []
        for child, parents in self.parent_by_child.items():
            for parent in parents:
                out.append(HierarchyEdge(child=child, parent=parent))
        return sorted(out, key=lambda e: (e.parent, e.child))


class ConceptTypeSystem:
    """Tracks inferred entity types and relation type constraints."""

    def __init__(self) -> None:
        self.entity_types: Dict[str, str] = {}
        self.type_confidence: Dict[str, float] = {}
        self.relation_rules: Dict[str, Dict[str, List[str]]] = {
            "orbits": {"subject": ["planet", "moon"], "object": ["star", "planet"]},
            "hunts": {"subject": ["predator"], "object": ["animal", "prey", "herbivore"]},
            "eats": {"subject": ["animal", "prey", "predator", "herbivore"], "object": ["plant", "animal"]},
            "reacts_with": {"subject": ["acid", "base", "reactant"], "object": ["acid", "base", "reactant"]},
            "contains": {"subject": ["molecule"], "object": ["atom"]},
            "is": {"subject": ["animal", "planet", "moon", "predator", "herbivore", "unknown"], "object": ["animal", "planet", "moon", "predator", "herbivore", "unknown"]},
        }

    def infer_from_triplets(self, triplets: Iterable[Tuple[str, str, str]]) -> None:
        """Infer coarse entity types from repeated relation participation."""
        role_counts: Dict[str, Dict[str, int]] = {}

        def bump(entity: str, role: str) -> None:
            role_counts.setdefault(entity, {})
            role_counts[entity][role] = role_counts[entity].get(role, 0) + 1

        for s, r, o in triplets:
            if r == "hunts":
                bump(s, "predator")
                bump(o, "prey")
                bump(o, "animal")
            elif r == "eats":
                bump(s, "animal")
                if o in {"grass", "bush", "shrub", "fern", "tree", "kelp", "berries", "lichen", "plant"}:
                    bump(o, "plant")
                else:
                    bump(o, "animal")
            elif r == "orbits":
                bump(s, "planet")
                bump(o, "star")
            elif r == "emits":
                bump(s, "star")
            elif r == "contains":
                bump(s, "molecule")
                bump(o, "atom")
            elif r == "reacts_with":
                bump(s, "reactant")
                bump(o, "reactant")
            elif r == "is":
                bump(s, o)

        for entity, counts in role_counts.items():
            best_type, best_count = sorted(counts.items(), key=lambda it: it[1], reverse=True)[0]
            total = sum(counts.values())
            self.entity_types[entity] = best_type
            self.type_confidence[entity] = best_count / max(1, total)

    def get_type(self, entity: str) -> str:
        return self.entity_types.get(entity, "unknown")

    def get_type_confidence(self, entity: str) -> float:
        return self.type_confidence.get(entity, 0.0)

    def is_relation_valid(self, subject: str, relation: str, obj: str) -> bool:
        if relation not in self.relation_rules:
            return True
        subject_type = self.get_type(subject)
        object_type = self.get_type(obj)
        allowed = self.relation_rules[relation]
        subject_ok = subject_type in allowed.get("subject", []) or "unknown" in allowed.get("subject", [])
        object_ok = object_type in allowed.get("object", []) or "unknown" in allowed.get("object", [])
        return subject_ok and object_ok

    def typed_relation(self, subject: str, relation: str, obj: str) -> str:
        return f"{self.get_type(subject)} {relation} {self.get_type(obj)}"
