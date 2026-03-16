"""Topology-driven variable role assignment.

Derives abstract roles (predator, resource, driver, relay, output, exogenous)
from a node's structural position in the knowledge graph, replacing the static
_ROLE dictionary in cross_domain_theory.py for any variable not already in
that dictionary.

Roles
-----
suppressor  — high ratio of outgoing negative edges  (wolf, pathogen, inhibitor)
driver      — high ratio of outgoing positive edges  (temperature, force, catalyst)
resource    — high ratio of incoming positive edges  (grass, energy, reactants)
relay       — high betweenness centrality            (deer, enzyme, messenger)
output      — no outgoing edges, has incoming        (reaction_rate, kinetic_energy)
exogenous   — no incoming edges, has outgoing        (co2_level, initial_force)

Usage
-----
    assigner = TopologicalRoleAssigner(knowledge_graph)
    role = assigner.get_role("deer")     # "relay" (or "prey" from static dict)

    # Refresh after many new nodes are added:
    assigner.invalidate_cache()
"""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static fallback dict (from cross_domain_theory.py — kept as fast path)
# ---------------------------------------------------------------------------

_STATIC_ROLE: Dict[str, str] = {
    "wolf": "predator", "wolves": "predator",
    "deer": "relay",
    "grass": "resource",
    "pathogens": "suppressor",
    "immune_response": "output",
    "energy": "resource",
    "proteins": "resource",
    "cells": "resource",
    "robot": "agent", "robots": "agent",
    "sensor_coverage": "output",
    "data_quality": "output",
    "robot_activity": "relay",
    "battery_charge": "resource",
    "temperature": "driver",
    "reactants": "resource",
    "reaction_rate": "output",
    "reaction_energy": "output",
    "force": "driver",
    "acceleration": "output",
    "heat": "driver",
    "kinetic_energy": "output",
    "mass": "resource",
    "asteroid": "agent", "asteroids": "agent",
    "collision_risk": "output",
    "radiation_pressure": "driver",
    "co2_level": "exogenous",
    "temperature_anomaly": "relay",
    "glacier_melt": "relay",
    "sea_level_rise": "output",
    "interest_rate": "driver",
    "investment": "relay",
    "gdp_growth": "output",
    "dopamine_level": "driver",
    "cortisol": "relay",
    "learning_rate": "output",
    "stress_level": "exogenous",
    "conductivity": "output",
    "hardness": "output",
    "catalyst": "driver",
    "activation_energy": "barrier",
}

# Causal relation types that contribute to role scoring
_POSITIVE_RELS: frozenset = frozenset({"positive_affects", "positively_affects"})
_NEGATIVE_RELS: frozenset = frozenset({"negative_affects", "negatively_affects"})
_ALL_CAUSAL:    frozenset = frozenset({
    "positive_affects", "negative_affects", "positively_affects",
    "negatively_affects", "mixed_affects", "affects", "inferred_affects",
    "conditional_affects",
})

# Minimum confidence for an edge to count toward role metrics
MIN_ROLE_CONFIDENCE: float = 0.20

# Betweenness approximation: number of random pivot pairs
BETWEENNESS_PIVOTS: int = 100

# Minimum score to assign a non-unknown role
MIN_ROLE_SCORE: float = 0.35


# ---------------------------------------------------------------------------
# Node metrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class NodeMetrics:
    neg_out:   int    # outgoing negative edges
    pos_out:   int    # outgoing positive edges
    pos_in:    int    # incoming positive edges
    neg_in:    int    # incoming negative edges
    total_out: int
    total_in:  int
    betweenness: float   # normalised [0, 1]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TopologicalRoleAssigner:
    """Assigns abstract roles to graph nodes from their causal topology.

    Parameters
    ----------
    knowledge_graph : KnowledgeGraph
        The live graph.  The assigner holds a reference and reads it on demand.
    betweenness_pivots : int
        Number of random pairs used for approximate betweenness.
        Higher = more accurate but slower.  Default 100.
    """

    def __init__(
        self,
        knowledge_graph,
        betweenness_pivots: int = BETWEENNESS_PIVOTS,
    ) -> None:
        self._graph             = knowledge_graph
        self._pivots            = betweenness_pivots
        self._role_cache:       Dict[str, str]         = {}
        self._metrics_cache:    Dict[str, NodeMetrics] = {}
        self._betweenness:      Dict[str, float]       = {}
        self._betweenness_done: bool                   = False

    # ── Public API ─────────────────────────────────────────────────────────

    def get_role(self, node: str) -> str:
        """Return the abstract role for *node*.

        Fast path: static dict → role cache → topology computation.
        Falls back to "unknown" if topology is ambiguous.
        """
        if node in _STATIC_ROLE:
            return _STATIC_ROLE[node]
        if node in self._role_cache:
            return self._role_cache[node]
        role = self._compute_role(node)
        self._role_cache[node] = role
        return role

    def get_domain(self, node: str) -> str:
        """Infer domain from community membership.

        Returns the most common known-domain label among the node's neighbours,
        or "unknown".  Does not require Louvain — uses 1-hop neighbourhood only,
        which is fast and good enough for the cross-domain engine.
        """
        from digital_baby.brain.cross_domain_theory import _DOMAIN_FOR_VAR
        if node in _DOMAIN_FOR_VAR:
            return _DOMAIN_FOR_VAR[node]

        neighbour_domains = []
        for e in self._graph.edges:
            if e.confidence < MIN_ROLE_CONFIDENCE:
                continue
            if e.source == node and e.target in _DOMAIN_FOR_VAR:
                neighbour_domains.append(_DOMAIN_FOR_VAR[e.target])
            elif e.target == node and e.source in _DOMAIN_FOR_VAR:
                neighbour_domains.append(_DOMAIN_FOR_VAR[e.source])

        if not neighbour_domains:
            return "unknown"
        return Counter(neighbour_domains).most_common(1)[0][0]

    def invalidate_cache(self) -> None:
        """Clear cached roles and metrics.  Call after bulk graph updates."""
        self._role_cache.clear()
        self._metrics_cache.clear()
        self._betweenness.clear()
        self._betweenness_done = False
        logger.debug("[role_assigner] cache_invalidated")

    def precompute_betweenness(self) -> None:
        """Run approximate betweenness over the whole graph.

        Call this every ~500 ticks from the event loop.  Between calls the
        assigner uses cached values.
        """
        self._betweenness = self._approx_betweenness_all()
        self._betweenness_done = True
        logger.info("[role_assigner] betweenness_precomputed nodes=%d", len(self._betweenness))

    # ── Internal: role computation ─────────────────────────────────────────

    def _compute_role(self, node: str) -> str:
        m = self._get_metrics(node)

        scores: Dict[str, float] = {
            "suppressor": m.neg_out  / max(m.total_out, 1),
            "driver":     m.pos_out  / max(m.total_out, 1),
            "resource":   m.pos_in   / max(m.total_in,  1),
            "relay":      min(1.0, self._get_betweenness(node) * 4.0),
            "output":     1.0 if m.total_out == 0 and m.total_in > 0  else 0.0,
            "exogenous":  1.0 if m.total_in  == 0 and m.total_out > 0 else 0.0,
        }

        best_role  = max(scores, key=scores.get)
        best_score = scores[best_role]

        if best_score < MIN_ROLE_SCORE:
            return "unknown"

        logger.debug(
            "[role_assigner] node=%s role=%s score=%.2f scores=%s",
            node, best_role, best_score,
            {k: round(v, 2) for k, v in scores.items()},
        )
        return best_role

    def _get_metrics(self, node: str) -> NodeMetrics:
        if node in self._metrics_cache:
            return self._metrics_cache[node]
        m = self._compute_metrics(node)
        self._metrics_cache[node] = m
        return m

    def _compute_metrics(self, node: str) -> NodeMetrics:
        neg_out = pos_out = pos_in = neg_in = total_out = total_in = 0
        for e in self._graph.edges:
            if e.confidence < MIN_ROLE_CONFIDENCE:
                continue
            if e.relation not in _ALL_CAUSAL:
                continue
            if e.source == node:
                total_out += 1
                if e.relation in _NEGATIVE_RELS:
                    neg_out += 1
                elif e.relation in _POSITIVE_RELS:
                    pos_out += 1
            if e.target == node:
                total_in += 1
                if e.relation in _POSITIVE_RELS:
                    pos_in += 1
                elif e.relation in _NEGATIVE_RELS:
                    neg_in += 1
        return NodeMetrics(neg_out, pos_out, pos_in, neg_in, total_out, total_in, betweenness=0.0)

    # ── Betweenness (approximate) ──────────────────────────────────────────

    def _get_betweenness(self, node: str) -> float:
        if not self._betweenness_done:
            # Lazy: compute once on first request
            self.precompute_betweenness()
        return self._betweenness.get(node, 0.0)

    def _approx_betweenness_all(self) -> Dict[str, float]:
        """Brandes-style approximation using random pivot sampling."""
        nodes = list(self._graph.nodes.keys())
        if len(nodes) < 3:
            return {}

        # Build adjacency (unweighted, undirected for betweenness)
        adj: Dict[str, List[str]] = defaultdict(list)
        for e in self._graph.edges:
            if e.confidence >= MIN_ROLE_CONFIDENCE and e.relation in _ALL_CAUSAL:
                adj[e.source].append(e.target)
                adj[e.target].append(e.source)

        counts: Dict[str, float] = defaultdict(float)
        pivots = random.sample(nodes, min(self._pivots, len(nodes)))

        for source in pivots:
            # BFS from source, recording predecessors and path counts
            dist:  Dict[str, int]         = {source: 0}
            sigma: Dict[str, float]       = {source: 1.0}
            pred:  Dict[str, List[str]]   = defaultdict(list)
            queue: List[str]              = [source]
            order: List[str]              = []

            head = 0
            while head < len(queue):
                v = queue[head]; head += 1
                order.append(v)
                for w in adj.get(v, []):
                    if w not in dist:
                        dist[w]  = dist[v] + 1
                        queue.append(w)
                    if dist[w] == dist[v] + 1:
                        sigma[w] = sigma.get(w, 0.0) + sigma[v]
                        pred[w].append(v)

            # Back-propagate dependency
            delta: Dict[str, float] = defaultdict(float)
            for w in reversed(order):
                for v in pred[w]:
                    delta[v] += (sigma.get(v, 1.0) / max(sigma.get(w, 1.0), 1e-9)) * (1 + delta[w])
                if w != source:
                    counts[w] += delta[w]

        # Normalise to [0, 1]
        max_count = max(counts.values(), default=1.0) or 1.0
        return {n: v / max_count for n, v in counts.items()}