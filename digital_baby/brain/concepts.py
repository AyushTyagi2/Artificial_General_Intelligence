"""Concept hierarchy builder from learned is-a relations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class HierarchyEdge:
    """Represents one hierarchy edge child -> parent."""

    child: str
    parent: str


class ConceptHierarchy:
    """Builds and queries concept hierarchies inferred from 'is' relations."""

    def __init__(self) -> None:
        self.parent_by_child: Dict[str, Set[str]] = {}
        self.children_by_parent: Dict[str, Set[str]] = {}

    def ingest_triplets(self, triplets: Iterable[Tuple[str, str, str]]) -> None:
        """Update hierarchy from graph triplets using relation 'is'."""
        self.parent_by_child.clear()
        self.children_by_parent.clear()
        for subject, relation, obj in triplets:
            if relation != "is":
                continue
            self.parent_by_child.setdefault(subject, set()).add(obj)
            self.children_by_parent.setdefault(obj, set()).add(subject)

    def ancestors(self, concept: str) -> List[str]:
        """Return transitive parent chain for a concept."""
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
        """Return most connected concept nodes by number of children + parents."""
        nodes = set(self.parent_by_child) | set(self.children_by_parent)
        scored = []
        for node in nodes:
            score = len(self.children_by_parent.get(node, set())) + len(self.parent_by_child.get(node, set()))
            scored.append((node, score))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]

    def common_parent(self, concepts: Iterable[str]) -> Optional[str]:
        """Return a common parent concept if one exists."""
        concept_list = [c for c in concepts if c]
        if not concept_list:
            return None
        ancestor_sets = [set(self.ancestors(c)) | {c} for c in concept_list]
        common = set.intersection(*ancestor_sets) if ancestor_sets else set()
        if not common:
            return None
        return sorted(common)[0]

    def edges(self) -> List[HierarchyEdge]:
        """Return explicit hierarchy edges."""
        out: List[HierarchyEdge] = []
        for child, parents in self.parent_by_child.items():
            for parent in parents:
                out.append(HierarchyEdge(child=child, parent=parent))
        return sorted(out, key=lambda e: (e.parent, e.child))
