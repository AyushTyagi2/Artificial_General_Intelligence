"""Persistent knowledge graph for digital_baby.

Nodes represent concepts and edges represent typed relations with confidence/evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json
import logging

LOGGER = logging.getLogger(__name__)


@dataclass
class GraphNode:
    name: str
    type: str = "concept"


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    confidence: float = 0.5
    evidence: int = 1


class KnowledgeGraph:
    """Self-growing knowledge graph with persistence and query utilities."""

    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path)
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self._load()

    @staticmethod
    def _normalize(text: str) -> str:
        return text.strip().lower()

    def add_node(self, name: str, node_type: str = "concept") -> bool:
        key = self._normalize(name)
        if key in self.nodes:
            return False
        self.nodes[key] = GraphNode(name=key, type=node_type)
        LOGGER.info("[graph] node_added %s", key)
        return True

    def add_or_update_edge(
        self,
        source: str,
        target: str,
        relation: str,
        confidence: float = 0.6,
        evidence_increment: int = 1,
        source_type: str = "concept",
        target_type: str = "concept",
    ) -> Tuple[GraphEdge, bool]:
        s = self._normalize(source)
        t = self._normalize(target)
        r = self._normalize(relation)
        self.add_node(s, source_type)
        self.add_node(t, target_type)

        for edge in self.edges:
            if edge.source == s and edge.target == t and edge.relation == r:
                edge.evidence += max(1, evidence_increment)
                edge.confidence = min(1.0, max(0.0, (edge.confidence * 0.8) + (confidence * 0.2)))
                LOGGER.info("[graph] edge_updated confidence=%.2f evidence=%s", edge.confidence, edge.evidence)
                return edge, False

        edge = GraphEdge(
            source=s,
            target=t,
            relation=r,
            confidence=min(1.0, max(0.0, confidence)),
            evidence=max(1, evidence_increment),
        )
        self.edges.append(edge)
        LOGGER.info("[graph] edge_added %s %s %s", s, r, t)
        return edge, True

    def ingest_triplets(self, triplets: List[Tuple[str, str, str]], default_confidence: float = 0.58) -> None:
        for subject, relation, obj in triplets:
            self.add_or_update_edge(subject, obj, relation, confidence=default_confidence, evidence_increment=1)

    def ingest_causal_rule(self, cause: str, effect: str, direction: str, confidence: float, evidence: int) -> None:
        relation = f"{direction}_affects" if direction in {"positive", "negative", "mixed"} else "affects"
        self.add_or_update_edge(
            cause,
            effect,
            relation,
            confidence=confidence,
            evidence_increment=max(1, evidence),
            source_type="state_variable",
            target_type="state_variable",
        )

    def get_neighbors(self, node: str) -> List[str]:
        n = self._normalize(node)
        neighbors: Set[str] = set()
        for edge in self.edges:
            if edge.source == n:
                neighbors.add(edge.target)
            if edge.target == n:
                neighbors.add(edge.source)
        return sorted(neighbors)

    def find_paths(self, source: str, target: str, max_depth: int = 3) -> List[List[str]]:
        src = self._normalize(source)
        dst = self._normalize(target)
        if src not in self.nodes or dst not in self.nodes:
            return []

        adjacency: Dict[str, List[str]] = {}
        for edge in self.edges:
            adjacency.setdefault(edge.source, []).append(edge.target)

        paths: List[List[str]] = []

        def dfs(node: str, path: List[str], depth: int) -> None:
            if depth > max_depth:
                return
            if node == dst and len(path) > 1:
                paths.append(path[:])
                return
            for nxt in adjacency.get(node, []):
                if nxt in path:
                    continue
                path.append(nxt)
                dfs(nxt, path, depth + 1)
                path.pop()

        dfs(src, [src], 0)
        return paths

    def save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": {key: asdict(node) for key, node in self.nodes.items()},
            "edges": [asdict(edge) for edge in self.edges],
        }
        self.storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        for key, node in payload.get("nodes", {}).items():
            self.nodes[key] = GraphNode(**node)
        for edge in payload.get("edges", []):
            self.edges.append(GraphEdge(**edge))
