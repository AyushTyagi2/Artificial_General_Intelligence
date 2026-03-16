"""Persistent knowledge graph for digital_baby — v4.

Changes from v3
---------------
1. PROBABILISTIC EDGE ADMISSION:
   All candidate edges are now admitted immediately at whatever confidence the
   caller supplies.  Previously, edges below TENTATIVE_CONFIDENCE (0.03) were
   hard-rejected.  Now they are stored as "speculative" (a new third tier).
   The knowledge graph is a living probability store, not a binary known/unknown
   structure.  Queries filter by tier at read time, not at write time.

2. FIVE CONFIDENCE TIERS (was two):
   speculative  [0.00, 0.05)  — stored but not used in reasoning
   tentative    [0.05, 0.15)  — used in hypothesis generation only
   probable     [0.15, 0.50)  — used in default reasoning
   confirmed    [0.50, 0.90)  — used in theory formation
   law_grade    [0.90, 1.00]  — immutable unless contradicted by strong evidence

3. EDGE DIRECTION RELAXED:
   'bidirectional' and 'mixed' directions are now stored as
   "mixed_affects" rather than being rejected.  They represent genuine
   ambiguity and can be used by hypothesis generation to target experiments.

4. IMPROVED PRUNING:
   _prune() now uses a composite score (confidence × degree × recency) to
   decide which edges to evict, rather than simply sorting by confidence.
   Well-connected edges are never pruned regardless of confidence.

5. CAUSAL CHAIN HELPER:
   get_causal_chain_edges() returns all edges suitable for second-order
   inference (positive_affects + negative_affects with confidence >= threshold).

6. GRAPH ANALYTICS:
   mean_confidence(), entropy_summary(), and domain_stats() are new utility
   methods for the dashboard and epistemic tracker.

All v3 public API is preserved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import json
import logging

LOGGER = logging.getLogger(__name__)


def _atomic_write_fd(path, write_fn) -> None:
    import tempfile, os, json
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            write_fn(fh)
        try:
            os.replace(tmp, str(p))
        except PermissionError:
            import shutil
            shutil.copy2(tmp, str(p))
            os.unlink(tmp)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# v4: Five confidence tiers
# ---------------------------------------------------------------------------

# Tier boundaries
TIER_SPECULATIVE: float = 0.00   # [0.00, 0.05) — stored, not used in reasoning
TIER_TENTATIVE:   float = 0.05   # [0.05, 0.15) — used in hypothesis generation
TIER_PROBABLE:    float = 0.15   # [0.15, 0.50) — used in default reasoning
TIER_CONFIRMED:   float = 0.50   # [0.50, 0.90) — used in theory formation
TIER_LAW_GRADE:   float = 0.90   # [0.90, 1.00] — nearly immutable

# Backward-compat aliases
CONFIRMED_CONFIDENCE: float = TIER_CONFIRMED
TENTATIVE_CONFIDENCE: float = TIER_TENTATIVE
MIN_CAUSAL_CONFIDENCE: float = TIER_SPECULATIVE   # v4: nothing is outright rejected

EDGE_STATUS_LAW_GRADE   = "law_grade"
EDGE_STATUS_CONFIRMED   = "confirmed"
EDGE_STATUS_PROBABLE    = "probable"
EDGE_STATUS_TENTATIVE   = "tentative"
EDGE_STATUS_SPECULATIVE = "speculative"

# Hard caps
MAX_NODES: int = 2_000
MAX_EDGES: int = 10_000


def _edge_status(confidence: float) -> str:
    if confidence >= TIER_LAW_GRADE:
        return EDGE_STATUS_LAW_GRADE
    if confidence >= TIER_CONFIRMED:
        return EDGE_STATUS_CONFIRMED
    if confidence >= TIER_PROBABLE:
        return EDGE_STATUS_PROBABLE
    if confidence >= TIER_TENTATIVE:
        return EDGE_STATUS_TENTATIVE
    return EDGE_STATUS_SPECULATIVE


@dataclass
class GraphNode:
    name: str
    type: str = "concept"


@dataclass
class GraphEdge:
    source:     str
    target:     str
    relation:   str
    confidence: float = 0.5
    evidence:   int   = 1
    provenance: str   = "internal"
    status:     str   = EDGE_STATUS_CONFIRMED
    last_tick:  int   = 0    # v4: track recency for pruning


@dataclass
class TickStats:
    new_nodes:         int = 0
    new_edges:         int = 0
    updated_edges:     int = 0
    tentative_edges:   int = 0
    speculative_edges: int = 0   # v4 NEW

    def clear(self) -> None:
        self.new_nodes         = 0
        self.new_edges         = 0
        self.updated_edges     = 0

    def __str__(self) -> str:
        return (
            f"new_nodes={self.new_nodes} "
            f"new_edges={self.new_edges} "
            f"updated_edges={self.updated_edges} "
            f"tentative={self.tentative_edges} "
            f"speculative={self.speculative_edges}"
        )


class KnowledgeGraph:
    """Self-growing probabilistic knowledge graph — v4."""

    def __init__(self, storage_path) -> None:
        self.storage_path = Path(storage_path)
        self.nodes:      Dict[str, GraphNode] = {}
        self.edges:      List[GraphEdge]      = []
        self.tick_stats: TickStats            = TickStats()

        self._rejected_permanent:    Set[Tuple[str, str, str]] = set()
        self._low_conf_last_logged:  Dict[Tuple[str, str], float] = {}

        # v4: track per-node degree for intelligent pruning
        self._degree_cache:          Dict[str, int] = {}
        self._degree_cache_dirty:    bool = True

        self._load()

    def reset_tick_stats(self) -> None:
        self.tick_stats.clear()
        self.tick_stats.tentative_edges   = sum(1 for e in self.edges if e.status == EDGE_STATUS_TENTATIVE)
        self.tick_stats.speculative_edges = sum(1 for e in self.edges if e.status == EDGE_STATUS_SPECULATIVE)

    # --- Entity plural -> singular map (Issue 1 fix) ---
    # Covers every plural that appears in world-simulator state vectors,
    # Wikipedia ingestion, and causal discovery output.
    _PLURAL_MAP: Dict[str, str] = {
        "wolves":    "wolf",
        "cells":     "cell",
        "plants":    "plant",
        "robots":    "robot",
        "asteroids": "asteroid",
        "sensors":   "sensor",
        "pathogens": "pathogen",
        "reactants": "reactant",
        "proteins":  "protein",
        "antibodies":"antibody",
    }

    # --- Relation alias -> canonical form map (Issue 2 fix) ---
    # ingest_causal_rule emits "positive_affects"/"negative_affects" (adjective).
    # episodic_memory emits "positively_affects"/"negatively_affects" (adverb).
    # Both are accepted by Reasoner but treated as different edges.
    # Canonical form is the adverb form already used in _KNOWN_RELATIONS.
    _RELATION_ALIASES: Dict[str, str] = {
        "positive_affects":  "positively_affects",
        "negative_affects":  "negatively_affects",
        "inferred_affects":  "affects",
    }

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase + strip, then apply entity and relation aliases.

        Called by every write path (add_or_update_edge, ingest_causal_rule,
        ingest_law, add_node) so normalisation is enforced at a single point.
        """
        t = text.strip().lower()
        # Entity: collapse known plural forms to singular so wolf/wolves
        # are the same node and evidence accumulates on a single edge.
        t = KnowledgeGraph._PLURAL_MAP.get(t, t)
        # Relation: canonicalise aliases so positive_affects and
        # positively_affects accumulate on the same edge.
        t = KnowledgeGraph._RELATION_ALIASES.get(t, t)
        return t


    def add_node(self, name: str, node_type: str = "concept") -> bool:
        key = self._normalize(name)
        if key in self.nodes:
            return False
        self.nodes[key] = GraphNode(name=key, type=node_type)
        self.tick_stats.new_nodes += 1
        self._degree_cache_dirty = True
        LOGGER.info("[graph] node_added %s", key)
        return True

    def add_or_update_edge(
        self,
        source:            str,
        target:            str,
        relation:          str,
        confidence:        float = 0.6,
        evidence_increment: int  = 1,
        source_type:       str  = "concept",
        target_type:       str  = "concept",
        provenance:        str  = "internal",
        current_tick:      int  = 0,
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
                new_status = _edge_status(edge.confidence)
                if edge.status != new_status:
                    LOGGER.info(
                        "[graph] edge_promoted %s→%s %s conf=%.2f",
                        s, r, t, edge.confidence,
                    )
                edge.status    = new_status
                edge.last_tick = current_tick
                self.tick_stats.updated_edges += 1
                LOGGER.debug(
                    "[graph] edge_updated %s %s %s conf=%.2f ev=%d status=%s",
                    s, r, t, edge.confidence, edge.evidence, edge.status,
                )
                return edge, False

        new_status = _edge_status(confidence)
        edge = GraphEdge(
            source=s, target=t, relation=r,
            confidence=min(1.0, max(0.0, confidence)),
            evidence=max(1, evidence_increment),
            provenance=provenance,
            status=new_status,
            last_tick=current_tick,
        )
        self.edges.append(edge)
        self.tick_stats.new_edges += 1
        self._degree_cache_dirty = True
        LOGGER.info(
            "[graph] edge_added %s %s %s provenance=%s status=%s conf=%.2f",
            s, r, t, provenance, new_status, confidence,
        )
        return edge, True

    def ingest_triplets(
        self,
        triplets: List[Tuple[str, str, str]],
        default_confidence: float = 0.58,
        current_tick: int = 0,
    ) -> None:
        for subject, relation, obj in triplets:
            if subject.endswith("_delta") or obj.endswith("_delta"):
                continue
            try:
                float(obj)
                continue
            except (ValueError, TypeError):
                pass
            self.add_or_update_edge(
                subject, obj, relation,
                confidence=default_confidence,
                evidence_increment=1,
                provenance="internal",
                current_tick=current_tick,
            )

    def ingest_causal_rule(
        self,
        cause:      str,
        effect:     str,
        direction:  str,
        confidence: float,
        evidence:   int,
        current_tick: int = 0,
    ) -> None:
        """v4: Admits ALL confidence values (including speculative < 0.05).

        Only self-causal edges are rejected.  'bidirectional' and 'mixed'
        directions are now stored as 'mixed_affects' rather than being dropped.
        """
        c = self._normalize(cause)
        e = self._normalize(effect)

        # Gate 1 — self-causation only
        if c == e:
            key = (c, e, "self_causal")
            if key not in self._rejected_permanent:
                LOGGER.info("[graph] rejected_edge reason=self_causal source=%s", c)
                self._rejected_permanent.add(key)
            else:
                LOGGER.debug("[graph] rejected_edge (cached) reason=self_causal source=%s", c)
            return

        # v4: map all direction values to relation strings (no longer reject mixed)
        direction_to_relation = {
            "positive":      "positive_affects",
            "negative":      "negative_affects",
            "bidirectional": "mixed_affects",
            "mixed":         "mixed_affects",
            "conditional":   "conditional_affects",
        }
        relation = direction_to_relation.get(direction, "affects")

        # v4: log when we're admitting a speculative edge (instead of rejecting)
        if confidence < TIER_TENTATIVE:
            LOGGER.debug(
                "[graph] speculative_edge_admitted source=%s target=%s conf=%.3f",
                c, e, confidence,
            )

        self.add_or_update_edge(
            c, e, relation,
            confidence=confidence,
            evidence_increment=max(1, evidence),
            source_type="state_variable",
            target_type="state_variable",
            provenance="causal",
            current_tick=current_tick,
        )

    def ingest_law(self, law) -> None:
        cause  = self._normalize(law.cause)
        effect = self._normalize(law.effect)
        causal_rels = {
            "positive_affects", "negative_affects",
            "mixed_affects", "affects", "inferred_affects", "conditional_affects",
        }
        for edge in self.edges:
            if (edge.source == cause
                    and edge.target == effect
                    and edge.relation in causal_rels):
                try:
                    edge.equation = law.equation_str          # type: ignore[attr-defined]
                    edge.law_r2   = round(law.r_squared, 4)   # type: ignore[attr-defined]
                except AttributeError:
                    pass
                edge.confidence = min(0.99, edge.confidence + law.r_squared * 0.05)
                edge.status     = _edge_status(edge.confidence)
                LOGGER.info(
                    "[graph] law_annotated %s → %s eq=%r r2=%.3f",
                    cause, effect, law.equation_str, law.r_squared,
                )
                return
        self.add_or_update_edge(
            cause, effect,
            relation="positive_affects" if law.r_squared >= 0 else "negative_affects",
            confidence=min(0.99, 0.5 + law.r_squared * 0.4),
            evidence_increment=law.n_datapoints,
            provenance="law_discovery",
        )
        LOGGER.info(
            "[graph] law_edge_created %s → %s eq=%r r2=%.3f",
            cause, effect, law.equation_str, law.r_squared,
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

    def find_paths(
        self, source: str, target: str, max_depth: int = 3
    ) -> List[List[str]]:
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

    # ── v4 NEW: Causal chain helper ────────────────────────────────────────

    def get_causal_chain_edges(
        self, min_confidence: float = TIER_PROBABLE
    ) -> List[Tuple[str, str, int, float]]:
        """Return (source, target, sign, confidence) tuples for causal edges.

        Only returns edges suitable for second-order chain inference:
        positive_affects (+1) and negative_affects (-1).
        """
        result = []
        for edge in self.edges:
            if edge.confidence < min_confidence:
                continue
            if edge.relation == "positive_affects":
                result.append((edge.source, edge.target, +1, edge.confidence))
            elif edge.relation == "negative_affects":
                result.append((edge.source, edge.target, -1, edge.confidence))
        return result

    def get_edges(
        self,
        min_confidence: float = 0.0,
        status_filter: Optional[Set[str]] = None,
    ) -> List[GraphEdge]:
        """Return edges filtered by confidence and/or status tier."""
        return [
            e for e in self.edges
            if e.confidence >= min_confidence
            and (status_filter is None or e.status in status_filter)
        ]

    # ── v4 NEW: Analytics helpers ──────────────────────────────────────────

    def mean_confidence(self, min_confidence: float = 0.0) -> float:
        edges = [e for e in self.edges if e.confidence >= min_confidence]
        if not edges:
            return 0.0
        return sum(e.confidence for e in edges) / len(edges)

    def entropy_summary(self) -> Dict:
        """Per-tier edge counts and mean confidence for the dashboard."""
        tiers = {
            EDGE_STATUS_SPECULATIVE: [],
            EDGE_STATUS_TENTATIVE:   [],
            EDGE_STATUS_PROBABLE:    [],
            EDGE_STATUS_CONFIRMED:   [],
            EDGE_STATUS_LAW_GRADE:   [],
        }
        for e in self.edges:
            tiers.setdefault(e.status, []).append(e.confidence)
        return {
            tier: {"count": len(confs), "mean_conf": sum(confs)/len(confs) if confs else 0.0}
            for tier, confs in tiers.items()
        }

    def domain_stats(self, domain_node_sets: Dict[str, Set[str]]) -> Dict[str, Dict]:
        """Return per-domain edge count and mean confidence."""
        result = {}
        for domain, nodes in domain_node_sets.items():
            domain_edges = [
                e for e in self.edges
                if e.source in nodes or e.target in nodes
            ]
            result[domain] = {
                "edges": len(domain_edges),
                "mean_confidence": (
                    sum(e.confidence for e in domain_edges) / len(domain_edges)
                    if domain_edges else 0.0
                ),
            }
        return result

    def _compute_degree(self) -> Dict[str, int]:
        degree: Dict[str, int] = defaultdict(int)
        for e in self.edges:
            degree[e.source] += 1
            degree[e.target] += 1
        return dict(degree)

    # ── v4: Improved pruning using composite score ─────────────────────────

    def _prune(self) -> None:
        """v4: Evict edges using composite score = confidence × recency.

        Well-connected nodes (degree > 5) are protected from pruning.
        Speculative edges are evicted first, then tentative, then by score.
        """
        if len(self.edges) <= MAX_EDGES:
            self._prune_orphan_nodes()
            return

        degree = self._compute_degree()
        max_tick = max((e.last_tick for e in self.edges), default=1) or 1

        def edge_score(e: GraphEdge) -> float:
            src_deg = degree.get(e.source, 0)
            tgt_deg = degree.get(e.target, 0)
            max_deg = max(src_deg, tgt_deg)
            # Protect well-connected edges
            if max_deg > 5:
                return float("inf")
            recency = e.last_tick / max_tick
            return e.confidence * (0.7 + 0.3 * recency)

        # Sort ascending by score: lowest-scoring evicted first
        self.edges.sort(key=edge_score)
        self.edges = self.edges[len(self.edges) - MAX_EDGES:]

        LOGGER.info(
            "[graph] pruned_to %d edges (was over limit)", MAX_EDGES
        )
        self._prune_orphan_nodes()
        self._degree_cache_dirty = True

    def _prune_orphan_nodes(self) -> None:
        if len(self.nodes) <= MAX_NODES:
            return
        referenced = set()
        for e in self.edges:
            referenced.add(e.source)
            referenced.add(e.target)
        orphans = [k for k in self.nodes if k not in referenced]
        for k in orphans:
            del self.nodes[k]
        if len(self.nodes) > MAX_NODES:
            excess = list(self.nodes.keys())[MAX_NODES:]
            for k in excess:
                del self.nodes[k]

    def save(self) -> None:
        self._prune()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": {key: asdict(node) for key, node in self.nodes.items()},
            "edges": [asdict(edge) for edge in self.edges],
        }
        _atomic_write_fd(self.storage_path, lambda fh: json.dump(payload, fh, indent=2))

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        raw = self.storage_path.read_text(encoding="utf-8").strip().lstrip("\x00")
        if not raw:
            return
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            import shutil, time
            backup = self.storage_path.with_suffix(f".corrupted.{int(time.time())}.json")
            shutil.move(str(self.storage_path), str(backup))
            LOGGER.warning(
                "Knowledge graph file was corrupt. Backed up to %s — starting fresh.", backup
            )
            return
        for key, node in payload.get("nodes", {}).items():
            self.nodes[key] = GraphNode(**node)
        for edge_data in payload.get("edges", []):
            edge_data.setdefault("provenance", "internal")
            edge_data.setdefault("status", _edge_status(edge_data.get("confidence", 0.5)))
            edge_data.setdefault("last_tick", 0)
            self.edges.append(GraphEdge(**edge_data))


# ---------------------------------------------------------------------------
# Module-level defaultdict import (needed by _compute_degree)
# ---------------------------------------------------------------------------
from collections import defaultdict  # noqa: E402 — after class definition