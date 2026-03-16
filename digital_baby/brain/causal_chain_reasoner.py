"""Multi-step causal chain reasoner for digital_baby.

Enables the agent to infer transitive causal relationships from its
knowledge graph rather than relying only on direct observations.

Example
-------
Given direct causal edges:
    wolves  -[negative_affects]->  deer       conf=0.82
    deer    -[positive_affects]->  grass      conf=0.74

Infers:
    wolves  -[inferred_affects]->  grass      conf=0.61  (0.82 × 0.74)

And generates the hypothesis:
    "if wolves decreases then grass may increase"

Algorithm
---------
1. Collect all causal edges from the graph (positive_affects, negative_affects,
   mixed_affects, affects, inferred_affects).
2. BFS/DFS up to MAX_DEPTH hops to enumerate all causal paths A→B→…→Z.
3. For each path compute chain confidence = product of edge confidences.
4. Skip paths below MIN_CHAIN_CONFIDENCE (too weak to be useful).
5. Write inferred edges back to the graph with provenance="causal_chain".
6. Emit Hypothesis objects for the hypothesis engine.

Integration
-----------
Call ``CausalChainReasoner.run(knowledge_graph, hypothesis_engine)`` from the
event loop once per tick after ``_sync_knowledge_graph()``.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_DEPTH:            int   = 3      # maximum hops in a chain (2 = A→B→C)
MIN_CHAIN_CONFIDENCE: float = 0.15   # discard chains weaker than this
MIN_EDGE_CONFIDENCE:  float = 0.30   # only traverse edges above this floor

# Relation types we treat as causal
CAUSAL_RELATIONS: frozenset = frozenset({
    "positive_affects",
    "negative_affects",
    "mixed_affects",
    "affects",
})

# Relation type used for newly inferred edges
INFERRED_RELATION = "inferred_affects"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CausalEdge:
    source:     str
    target:     str
    relation:   str
    confidence: float
    evidence:   int


@dataclass
class CausalChain:
    """A discovered multi-step causal path."""
    nodes:      List[str]          # [A, B, C]  — includes start and end
    edges:      List[CausalEdge]   # len == len(nodes) - 1
    chain_conf: float              # product of all edge confidences
    direction:  str                # "positive" | "negative" | "mixed"

    @property
    def source(self) -> str:
        return self.nodes[0]

    @property
    def target(self) -> str:
        return self.nodes[-1]

    @property
    def length(self) -> int:
        return len(self.nodes) - 1

    def path_str(self) -> str:
        return " → ".join(self.nodes)


@dataclass
class InferredRule:
    """An edge inferred from a causal chain, ready to inject into the graph."""
    source:       str
    target:       str
    confidence:   float
    evidence:     int
    chain:        CausalChain
    timestamp:    float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Direction inference helpers
# ---------------------------------------------------------------------------

def _chain_direction(edges: List[CausalEdge]) -> str:
    """Infer net direction of a chain from its individual edge directions.

    Rules (like multiplying signs):
      positive × positive = positive
      negative × negative = positive
      positive × negative = negative
      anything with mixed  = mixed
    """
    sign = 1
    for edge in edges:
        rel = edge.relation
        if "mixed" in rel:
            return "mixed"
        if "negative" in rel:
            sign *= -1
    return "positive" if sign > 0 else "negative"


def _direction_to_hypothesis(source: str, target: str, direction: str) -> str:
    """Produce a readable hypothesis statement from an inferred chain."""
    if direction == "positive":
        return f"if {source} increases then {target} may increase"
    if direction == "negative":
        return f"if {source} increases then {target} may decrease"
    return f"if {source} changes then {target} may change"


# ---------------------------------------------------------------------------
# Main reasoner
# ---------------------------------------------------------------------------

class CausalChainReasoner:
    """Discovers multi-hop causal chains and injects inferred edges + hypotheses."""

    def __init__(
        self,
        max_depth:            int   = MAX_DEPTH,
        min_chain_confidence: float = MIN_CHAIN_CONFIDENCE,
        min_edge_confidence:  float = MIN_EDGE_CONFIDENCE,
    ) -> None:
        self.max_depth            = max(1, min(5, max_depth))
        self.min_chain_confidence = min_chain_confidence
        self.min_edge_confidence  = min_edge_confidence

        # Cache of already-inferred (source, target) pairs to avoid log spam
        self._known_inferred: Set[Tuple[str, str]] = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        knowledge_graph,
        hypothesis_engine,
    ) -> Tuple[List[InferredRule], List]:
        """Discover chains, inject edges, and generate hypotheses.

        Returns
        -------
        (inferred_rules, new_hypotheses)
            inferred_rules   : list of newly written InferredRule objects
            new_hypotheses   : list of Hypothesis objects for the engine
        """
        causal_edges = self._extract_causal_edges(knowledge_graph)
        if len(causal_edges) < 2:
            return [], []

        chains = self._find_chains(causal_edges)
        if not chains:
            return [], []

        new_rules:       List[InferredRule] = []
        new_hypotheses:  List               = []

        for chain in chains:
            # Skip if a direct edge already exists between source and target
            if self._direct_edge_exists(chain.source, chain.target, knowledge_graph):
                continue

            rule = InferredRule(
                source=chain.source,
                target=chain.target,
                confidence=chain.chain_conf,
                evidence=min(e.evidence for e in chain.edges),
                chain=chain,
            )

            is_new = self._inject_edge(rule, knowledge_graph)
            if is_new:
                new_rules.append(rule)
                logger.info(
                    "[reasoning] causal_chain %s",
                    chain.path_str(),
                )
                logger.info(
                    "[reasoning] inferred %s → %s confidence=%.2f direction=%s",
                    chain.source, chain.target,
                    chain.chain_conf, chain.direction,
                )

            hyp = self._make_hypothesis(chain)
            if hyp is not None:
                new_hypotheses.append(hyp)

        return new_rules, new_hypotheses

    # ── Edge extraction ───────────────────────────────────────────────────────

    def _extract_causal_edges(self, knowledge_graph) -> List[CausalEdge]:
        """Pull all confirmed causal edges above the confidence floor from the graph.

        Edges with provenance="causal_chain" are excluded to prevent
        compounding-confidence feedback loops.

        Tentative edges (status="tentative") are excluded — they are weak
        hypotheses not yet reliable enough for multi-hop inference.
        """
        from digital_baby.brain.knowledge_graph import EDGE_STATUS_TENTATIVE

        edges: List[CausalEdge] = []
        for e in knowledge_graph.edges:
            if e.relation not in CAUSAL_RELATIONS:
                continue
            if e.confidence < self.min_edge_confidence:
                continue
            if getattr(e, "provenance", "") == "causal_chain":
                continue
            # Skip tentative edges — not reliable enough for chain reasoning
            if getattr(e, "status", "") == EDGE_STATUS_TENTATIVE:
                continue
            edges.append(CausalEdge(
                source=e.source,
                target=e.target,
                relation=e.relation,
                confidence=e.confidence,
                evidence=e.evidence,
            ))
        return edges

    # ── Chain discovery ───────────────────────────────────────────────────────

    def _find_chains(self, edges: List[CausalEdge]) -> List[CausalChain]:
        """BFS up to max_depth to enumerate all valid causal paths."""
        # Build adjacency: source -> list of CausalEdge
        adj: Dict[str, List[CausalEdge]] = defaultdict(list)
        for e in edges:
            adj[e.source].append(e)

        chains: List[CausalChain] = []
        # Deduplicate (source, target) pairs — keep only the highest-conf chain
        best_conf: Dict[Tuple[str, str], float] = {}

        def dfs(
            path_nodes: List[str],
            path_edges: List[CausalEdge],
            running_conf: float,
        ) -> None:
            depth = len(path_edges)
            current = path_nodes[-1]

            # Record any chain of length >= 2
            if depth >= 2:
                key = (path_nodes[0], current)
                if running_conf >= self.min_chain_confidence:
                    if running_conf > best_conf.get(key, 0.0):
                        best_conf[key] = running_conf
                        direction = _chain_direction(path_edges)
                        chains.append(CausalChain(
                            nodes=list(path_nodes),
                            edges=list(path_edges),
                            chain_conf=running_conf,
                            direction=direction,
                        ))

            if depth >= self.max_depth:
                return

            for next_edge in adj.get(current, []):
                nxt = next_edge.target
                if nxt in path_nodes:        # prevent cycles
                    continue
                new_conf = running_conf * next_edge.confidence
                if new_conf < self.min_chain_confidence:
                    continue                 # prune weak branches early
                dfs(
                    path_nodes + [nxt],
                    path_edges + [next_edge],
                    new_conf,
                )

        for start_node in list(adj.keys()):
            dfs([start_node], [], 1.0)

        # Deduplicate: for each (source, target) pair, keep the best chain
        best_chains: Dict[Tuple[str, str], CausalChain] = {}
        for c in chains:
            key = (c.source, c.target)
            if c.chain_conf > best_chains.get(key, CausalChain([], [], 0.0, "")).chain_conf:
                best_chains[key] = c

        return sorted(best_chains.values(), key=lambda c: c.chain_conf, reverse=True)

    # ── Graph injection ───────────────────────────────────────────────────────

    @staticmethod
    def _direct_edge_exists(source: str, target: str, knowledge_graph) -> bool:
        """Return True if a direct causal edge (any type) already exists."""
        for e in knowledge_graph.edges:
            if e.source == source and e.target == target and e.relation in CAUSAL_RELATIONS:
                return True
        return False

    def _inject_edge(self, rule: InferredRule, knowledge_graph) -> bool:
        """Write inferred edge to graph. Returns True if genuinely new."""
        from digital_baby.brain.knowledge_graph import GraphEdge

        # Check if this inferred edge already exists
        for e in knowledge_graph.edges:
            if (e.source == rule.source
                    and e.target == rule.target
                    and e.relation == INFERRED_RELATION):
                # Update confidence (weighted average toward new value)
                e.confidence = min(0.99, (e.confidence * 0.85) + (rule.confidence * 0.15))
                e.evidence  += 1
                knowledge_graph.tick_stats.updated_edges += 1
                return False

        # New edge
        knowledge_graph.add_node(rule.source)
        knowledge_graph.add_node(rule.target)
        knowledge_graph.edges.append(GraphEdge(
            source=rule.source,
            target=rule.target,
            relation=INFERRED_RELATION,
            confidence=rule.confidence,
            evidence=rule.evidence,
            provenance="causal_chain",
        ))
        knowledge_graph.tick_stats.new_edges += 1

        key = (rule.source, rule.target)
        if key not in self._known_inferred:
            self._known_inferred.add(key)
            return True
        return False

    # ── Hypothesis generation ─────────────────────────────────────────────────

    @staticmethod
    def _make_hypothesis(chain: CausalChain) -> Optional[object]:
        """Create a Hypothesis from a causal chain for the hypothesis engine."""
        try:
            from digital_baby.brain.hypothesis import Hypothesis  # avoid circular import

            rule = _direction_to_hypothesis(chain.source, chain.target, chain.direction)
            supporting = max(1, min(e.evidence for e in chain.edges))
            contradicting = max(1, int((1.0 - chain.chain_conf) * supporting))

            return Hypothesis(
                rule=rule,
                concepts=[chain.source, chain.target],
                confidence=chain.chain_conf,
                supporting_evidence=supporting,
                contradicting_evidence=contradicting,
            )
        except Exception as exc:
            logger.debug("[reasoning] hypothesis_build_failed error=%s", exc)
            return None

    # ── Query helpers (used by predictor) ────────────────────────────────────

    def get_inferred_edges(self, knowledge_graph) -> List[InferredRule]:
        """Return all currently inferred edges as InferredRule objects."""
        rules: List[InferredRule] = []
        for e in knowledge_graph.edges:
            if e.relation == INFERRED_RELATION:
                # Reconstruct a minimal InferredRule for the predictor
                chain = CausalChain(
                    nodes=[e.source, e.target],
                    edges=[],
                    chain_conf=e.confidence,
                    direction="mixed",
                )
                rules.append(InferredRule(
                    source=e.source,
                    target=e.target,
                    confidence=e.confidence,
                    evidence=e.evidence,
                    chain=chain,
                ))
        return rules