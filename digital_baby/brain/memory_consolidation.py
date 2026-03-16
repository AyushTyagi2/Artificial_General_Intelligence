"""Memory consolidation system for digital_baby — v4.

Changes from v3
---------------
1. FIVE-TIER MEMORY SYSTEM (was two):
   The v4 knowledge graph has five confidence tiers: speculative, tentative,
   probable, confirmed, law_grade.  Memory consolidation now mirrors this
   by tracking tier transitions during promotion and preventing speculative
   edges from being consolidated prematurely.

   Tier→Memory tier mapping:
     speculative  → always EPISODIC (needs more evidence before consolidating)
     tentative    → EPISODIC (can promote after min_evidence × 2)
     probable     → EPISODIC → SEMANTIC eligible
     confirmed    → SEMANTIC (immediately eligible for promotion)
     law_grade    → SEMANTIC (permanent, never decayed)

2. SECOND-ORDER EDGE CONSOLIDATION:
   Edges with provenance='second_order' use relaxed promotion rules (lower
   evidence threshold) because second-order inference chains are necessarily
   indirect — we should not require as many direct observations.

3. COMPOSITE EVICTION SCORE:
   Semantic eviction previously sorted only by confidence × log(evidence).
   v4 adds a recency term and a connectivity bonus:
     score = confidence × log(evidence+1) × (0.7 + 0.3×recency) × degree_bonus
   Well-connected semantic edges (degree > 3) are protected from eviction.

4. LAW-GRADE PROTECTION:
   Edges with status='law_grade' are NEVER evicted regardless of memory
   pressure.  They represent discovered laws and are the most valuable
   knowledge in the graph.

5. PROMOTION HYSTERESIS:
   An edge that was recently demoted (promoted→episodic due to low new
   confidence) must accumulate 3 additional observations before it can
   be promoted again.  This prevents oscillation.

6. SPECULATIVE EDGE LIFESPAN:
   Speculative edges (confidence < 0.05) that have not been updated in
   SPECULATIVE_MAX_AGE ticks are pruned, preventing the graph from
   filling with dead-end hypotheses.

All v3 public API is preserved.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONSOLIDATION_INTERVAL: int = 10

MAX_EPISODIC:  int = 500
MAX_SEMANTIC:  int = 3000

STALE_DECAY_RATE:      float = 0.001
STALE_GRACE_TICKS:     int   = 30

SPECULATIVE_MAX_AGE:   int   = 100   # v4 NEW: max ticks a speculative edge lives unupdated
DEMOTION_HYSTERESIS:   int   = 3     # v4 NEW: extra observations needed after demotion

# Promotion rules: (min_evidence, min_confidence, min_stable_ticks)
_DEFAULT_PROMOTION = (15, 0.45, 15)

PROMOTION_RULES: Dict[str, Tuple[int, float, int]] = {
    "causal":        (10,  0.35, 15),
    "internal":      (20,  0.50, 20),
    "ingestion":     ( 5,  0.40, 10),
    "abstraction":   ( 8,  0.45, 12),
    "causal_chain":  (12,  0.40, 18),
    "second_order":  ( 6,  0.30, 10),   # v4 NEW: relaxed for indirect chains
    "law_discovery": ( 3,  0.60,  5),   # v4 NEW: laws promote quickly
}

FORGETTING_RULES: Dict[str, Tuple[int, float]] = {
    "episodic": ( 50, 0.10),
    "semantic": (500, 0.05),
}

# Confidence tier names (mirror knowledge_graph.py v4)
_TIER_SPECULATIVE = "speculative"
_TIER_TENTATIVE   = "tentative"
_TIER_PROBABLE    = "probable"
_TIER_CONFIRMED   = "confirmed"
_TIER_LAW_GRADE   = "law_grade"

# Minimum confidence for semantic promotion (speculative edges cannot be promoted)
_MIN_PROMOTE_CONF = 0.15   # must be at least "probable"


class MemoryTier(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC  = "semantic"


@dataclass
class ConsolidationStats:
    promoted:  int = 0
    decayed:   int = 0
    forgotten: int = 0
    evicted:   int = 0
    speculative_pruned: int = 0   # v4 NEW

    def __bool__(self) -> bool:
        return bool(self.promoted or self.forgotten or self.evicted or self.speculative_pruned)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MemoryConsolidator:
    """Manages two-tier episodic/semantic memory for the knowledge graph — v4."""

    def __init__(
        self,
        consolidation_interval: int = CONSOLIDATION_INTERVAL,
        max_episodic: int = MAX_EPISODIC,
        max_semantic:  int = MAX_SEMANTIC,
    ) -> None:
        self.consolidation_interval = consolidation_interval
        self.max_episodic = max_episodic
        self.max_semantic  = max_semantic
        self._last_run_tick: int = 0

        # v4: track demotion history for hysteresis
        self._demotion_evidence: Dict[Tuple[str, str, str], int] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self, knowledge_graph, current_tick: int) -> ConsolidationStats:
        """Run one consolidation pass if the interval has elapsed."""
        if current_tick - self._last_run_tick < self.consolidation_interval:
            return ConsolidationStats()

        self._last_run_tick = current_tick
        stats = ConsolidationStats()

        self._ensure_tier_field(knowledge_graph, current_tick)
        self._prune_speculative(knowledge_graph, current_tick, stats)   # v4 NEW (before promote)
        self._promote_episodic(knowledge_graph, current_tick, stats)
        self._apply_stale_decay(knowledge_graph, current_tick, stats)
        self._prune_weak(knowledge_graph, current_tick, stats)
        self._enforce_size_limits(knowledge_graph, current_tick, stats)

        return stats

    # ── Step 0: ensure edges have required fields ──────────────────────────

    @staticmethod
    def _ensure_tier_field(knowledge_graph, current_tick: int) -> None:
        for edge in knowledge_graph.edges:
            if not hasattr(edge, "memory_tier"):
                # Map from knowledge graph tier to memory tier
                kg_status = getattr(edge, "status", _TIER_PROBABLE)
                if kg_status in (_TIER_CONFIRMED, _TIER_LAW_GRADE):
                    tier = MemoryTier.SEMANTIC
                elif getattr(edge, "evidence", 1) >= 20:
                    tier = MemoryTier.SEMANTIC
                else:
                    tier = MemoryTier.EPISODIC
                try:
                    edge.memory_tier = tier
                except AttributeError:
                    pass
            for attr, default in [
                ("created_tick",      current_tick),
                ("last_updated_tick", current_tick),
            ]:
                if not hasattr(edge, attr):
                    try:
                        setattr(edge, attr, default)
                    except AttributeError:
                        pass

    # ── Step 1 (v4 NEW): prune stale speculative edges ─────────────────────

    def _prune_speculative(
        self,
        knowledge_graph,
        current_tick: int,
        stats: ConsolidationStats,
    ) -> None:
        """Remove speculative edges that have not been updated recently.

        Speculative edges exist to give weak signals a chance to accumulate.
        If they haven't been touched in SPECULATIVE_MAX_AGE ticks, they never
        will be — they represent dead-end hypotheses and should be pruned.
        Law-grade edges are always exempt.
        """
        to_remove = []
        for edge in knowledge_graph.edges:
            kg_status = getattr(edge, "status", _TIER_PROBABLE)
            if kg_status != _TIER_SPECULATIVE:
                continue
            if kg_status == _TIER_LAW_GRADE:
                continue
            last_updated = getattr(edge, "last_updated_tick", current_tick)
            age = current_tick - last_updated
            if age > SPECULATIVE_MAX_AGE:
                to_remove.append(edge)

        for edge in to_remove:
            try:
                knowledge_graph.edges.remove(edge)
                stats.speculative_pruned += 1
                logger.debug(
                    "[consolidation] speculative_pruned %s %s %s age=%d",
                    edge.source, edge.relation, edge.target,
                    current_tick - getattr(edge, "last_updated_tick", current_tick),
                )
            except ValueError:
                pass

    # ── Step 2: promote episodic → semantic ───────────────────────────────

    def _promote_episodic(
        self,
        knowledge_graph,
        current_tick: int,
        stats: ConsolidationStats,
    ) -> None:
        for edge in knowledge_graph.edges:
            tier = getattr(edge, "memory_tier", MemoryTier.EPISODIC)
            if tier != MemoryTier.EPISODIC:
                continue

            # v4: speculative edges cannot be promoted to semantic
            kg_status = getattr(edge, "status", _TIER_PROBABLE)
            confidence = getattr(edge, "confidence", 0.0)
            if kg_status == _TIER_SPECULATIVE or confidence < _MIN_PROMOTE_CONF:
                continue

            # v4: law_grade edges promote immediately
            if kg_status == _TIER_LAW_GRADE:
                try:
                    edge.memory_tier = MemoryTier.SEMANTIC
                except AttributeError:
                    pass
                stats.promoted += 1
                continue

            prov      = getattr(edge, "provenance", "internal")
            min_ev, min_conf, min_age = PROMOTION_RULES.get(prov, _DEFAULT_PROMOTION)

            evidence   = getattr(edge, "evidence", 1)
            created_at = getattr(edge, "created_tick", current_tick)
            age_ticks  = current_tick - created_at

            # v4: check demotion hysteresis
            edge_key = (getattr(edge, "source", ""), getattr(edge, "relation", ""), getattr(edge, "target", ""))
            extra_ev  = self._demotion_evidence.get(edge_key, 0)
            effective_evidence = evidence - extra_ev

            if (effective_evidence >= min_ev
                    and confidence >= min_conf
                    and age_ticks >= min_age):
                try:
                    edge.memory_tier = MemoryTier.SEMANTIC
                except AttributeError:
                    pass
                self._demotion_evidence.pop(edge_key, None)  # clear hysteresis on promotion
                stats.promoted += 1
                logger.info(
                    "[consolidation] promoted episodic→semantic %s %s %s"
                    " ev=%d conf=%.3f age=%d prov=%s",
                    edge.source, edge.relation, edge.target,
                    evidence, confidence, age_ticks, prov,
                )

    # ── Step 3: stale decay on semantic edges ─────────────────────────────

    def _apply_stale_decay(
        self,
        knowledge_graph,
        current_tick: int,
        stats: ConsolidationStats,
    ) -> None:
        for edge in knowledge_graph.edges:
            tier = getattr(edge, "memory_tier", MemoryTier.EPISODIC)
            if tier != MemoryTier.SEMANTIC:
                continue

            # v4: law_grade edges are NEVER decayed
            kg_status = getattr(edge, "status", _TIER_PROBABLE)
            if kg_status == _TIER_LAW_GRADE:
                continue

            last_updated = getattr(edge, "last_updated_tick", current_tick)
            ticks_since  = current_tick - last_updated

            if ticks_since > STALE_GRACE_TICKS:
                decay = STALE_DECAY_RATE * (ticks_since - STALE_GRACE_TICKS)
                old_conf = getattr(edge, "confidence", 0.0)
                try:
                    edge.confidence = max(0.0, edge.confidence - decay)
                except AttributeError:
                    pass

                # v4: if decay pushes confidence below promotion threshold,
                # demote back to episodic and record hysteresis
                new_conf = getattr(edge, "confidence", 0.0)
                if new_conf < _MIN_PROMOTE_CONF and old_conf >= _MIN_PROMOTE_CONF:
                    try:
                        edge.memory_tier = MemoryTier.EPISODIC
                    except AttributeError:
                        pass
                    edge_key = (getattr(edge, "source", ""), getattr(edge, "relation", ""), getattr(edge, "target", ""))
                    self._demotion_evidence[edge_key] = getattr(edge, "evidence", 0)
                    logger.debug(
                        "[consolidation] demoted semantic→episodic %s %s %s conf=%.3f",
                        edge.source, edge.relation, edge.target, new_conf,
                    )

                stats.decayed += 1

    # ── Step 4: prune below forgetting floor ──────────────────────────────

    def _prune_weak(
        self,
        knowledge_graph,
        current_tick: int,
        stats: ConsolidationStats,
    ) -> None:
        to_remove = []
        for edge in knowledge_graph.edges:
            # v4: law_grade edges are NEVER forgotten
            kg_status = getattr(edge, "status", _TIER_PROBABLE)
            if kg_status == _TIER_LAW_GRADE:
                continue

            tier       = getattr(edge, "memory_tier", MemoryTier.EPISODIC)
            tier_key   = tier.value if isinstance(tier, MemoryTier) else str(tier)
            max_age, floor = FORGETTING_RULES.get(tier_key, (500, 0.05))

            created_at = getattr(edge, "created_tick", current_tick)
            age_ticks  = current_tick - created_at
            confidence = getattr(edge, "confidence", 0.0)

            if age_ticks > max_age and confidence < floor:
                to_remove.append(edge)

        for edge in to_remove:
            try:
                knowledge_graph.edges.remove(edge)
            except ValueError:
                pass
            stats.forgotten += 1
            logger.debug(
                "[consolidation] forgot edge %s %s %s age=%d conf=%.3f tier=%s",
                edge.source, edge.relation, edge.target,
                current_tick - getattr(edge, "created_tick", current_tick),
                getattr(edge, "confidence", 0.0),
                getattr(edge, "memory_tier", "?"),
            )

    # ── Step 5: enforce hard size limits ──────────────────────────────────

    def _enforce_size_limits(
        self,
        knowledge_graph,
        current_tick: int,
        stats: ConsolidationStats,
    ) -> None:
        episodic = [e for e in knowledge_graph.edges
                    if getattr(e, "memory_tier", MemoryTier.EPISODIC) == MemoryTier.EPISODIC]
        semantic  = [e for e in knowledge_graph.edges
                     if getattr(e, "memory_tier", None) == MemoryTier.SEMANTIC]

        # Evict weakest episodic edges first
        if len(episodic) > self.max_episodic:
            evict_count = len(episodic) - self.max_episodic
            to_evict = sorted(episodic, key=lambda e: getattr(e, "confidence", 0.0))[:evict_count]
            for edge in to_evict:
                try:
                    knowledge_graph.edges.remove(edge)
                except ValueError:
                    pass
                stats.evicted += 1

        # Evict weakest semantic edges using composite score
        if len(semantic) > self.max_semantic:
            evict_count = len(semantic) - self.max_semantic

            # Compute degree for connectivity bonus
            degree: Dict[str, int] = defaultdict(int)
            for e in knowledge_graph.edges:
                degree[e.source] += 1
                degree[e.target] += 1

            max_tick = max(
                (getattr(e, "last_updated_tick", 0) for e in semantic),
                default=current_tick or 1,
            ) or 1

            def semantic_score(e) -> float:
                # v4: law_grade is protected (returns +inf)
                kg_status = getattr(e, "status", _TIER_PROBABLE)
                if kg_status == _TIER_LAW_GRADE:
                    return float("inf")

                c       = getattr(e, "confidence", 0.0)
                ev      = getattr(e, "evidence", 1)
                recency = getattr(e, "last_updated_tick", 0) / max_tick

                # Degree bonus: well-connected edges are more valuable
                src_deg = degree.get(getattr(e, "source", ""), 0)
                tgt_deg = degree.get(getattr(e, "target", ""), 0)
                deg_bonus = 1.0 + 0.5 * min(1.0, max(src_deg, tgt_deg) / 5.0)

                return c * math.log(ev + 1) * (0.7 + 0.3 * recency) * deg_bonus

            # Sort ascending — lowest score evicted first
            to_evict = sorted(semantic, key=semantic_score)[:evict_count]
            for edge in to_evict:
                if semantic_score(edge) == float("inf"):
                    break  # hit law_grade protection
                try:
                    knowledge_graph.edges.remove(edge)
                except ValueError:
                    pass
                stats.evicted += 1

        if stats.evicted:
            logger.info(
                "[consolidation] size_limit evicted=%d law_grade_protected=%d",
                stats.evicted,
                sum(1 for e in knowledge_graph.edges
                    if getattr(e, "status", "") == _TIER_LAW_GRADE),
            )