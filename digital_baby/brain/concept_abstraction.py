"""Improved concept abstraction engine for digital_baby — v2.

Changes from v1
---------------
Five quality gates now guard abstraction promotion, preventing "container"
and other cross-domain noise abstractions that triggered too early:

Gate 1  Minimum member count    ≥ 3 entities must share the role
Gate 2  Minimum evidence        ≥ 15 total triplet observations across members
Gate 3  Domain purity           ≥ 60 % of members from the same source domain
Gate 4  Stability               cluster membership must not have changed for ≥ 5 ticks
Gate 5  Confidence threshold    composite score (cohesion × coverage × purity) ≥ 0.50

A pending dict tracks candidate abstractions between ticks so stability can
be measured. Membership changes reset the stability timer.

The three discovery mechanisms (role clustering, chain abstraction,
co-occurrence grouping) are unchanged from v1; only the promotion gate is new.

Usage
-----
Pass ``current_tick`` to ``run()`` so stability timers work correctly::

    new_abstractions = self.abstraction_engine.run(
        all_triplets, self.knowledge_graph,
        self.type_system, self.curiosity,
        current_tick=tick,
    )
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Relation → role mappings  (unchanged from v1)
# ---------------------------------------------------------------------------

SUBJECT_ROLE: Dict[str, str] = {
    "hunts":       "predator",
    "eats":        "consumer",
    "subclass_of": "subtype",
    "instance_of": "instance",
    "part_of":     "component",
    "located_in":  "inhabitant",
    "has_part":    "composite",
    "produces":    "producer",
    "contains":    "container",
    "orbits":      "orbiting_body",
    "reacts_with": "reactant",
}

OBJECT_ROLE: Dict[str, str] = {
    "hunts":       "prey",
    "eats":        "food_source",
    "subclass_of": "supertype",
    "instance_of": "category",
    "part_of":     "whole",
    "located_in":  "habitat",
    "has_part":    "component",
    "produces":    "product",
    "contains":    "contents",
    "orbits":      "orbital_center",
    "reacts_with": "reactant",
}

CHAIN_LABELS: Dict[str, str] = {
    "subclass_of": "taxonomy_chain",
    "part_of":     "composition_hierarchy",
    "located_in":  "spatial_hierarchy",
    "eats":        "food_chain",
    "hunts":       "predation_chain",
}

# Relations that carry no meaningful abstraction signal (noise)
STRUCTURAL_NOISE_RELATIONS: FrozenSet[str] = frozenset({"is", "measures", "has_value"})

# ---------------------------------------------------------------------------
# Quality gate constants
# ---------------------------------------------------------------------------

MIN_MEMBER_COUNT:            int   = 3
MIN_CLUSTER_EVIDENCE:        int   = 15
MIN_DOMAIN_PURITY:           float = 0.60
MIN_CLUSTER_STABILITY_TICKS: int   = 5
MIN_ABSTRACTION_CONFIDENCE:  float = 0.50
MIN_CHAIN_LENGTH:            int   = 3
MIN_COOCCURRENCE:            int   = 3

# Maximum number of pending candidates (LRU eviction above this)
MAX_PENDING_CANDIDATES: int = 1_000

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AbstractConcept:
    """A latent concept inferred from structural graph patterns."""
    name:       str
    members:    Set[str]
    source_rel: str
    evidence:   int
    confidence: float
    origin:     str
    domain_purity: float = 1.0
    timestamp:  float = field(default_factory=time.time)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AbstractConcept) and self.name == other.name


@dataclass
class AbstractionCandidate:
    """Tracks a candidate abstraction between ticks for stability measurement."""
    name:              str
    members:           Set[str]
    relation:          str
    first_seen_tick:   int
    last_changed_tick: int
    evidence_total:    int
    domain_counts:     Dict[str, int]   # domain_key → member count

    @property
    def domain_purity(self) -> float:
        if not self.domain_counts:
            return 0.0
        top = max(self.domain_counts.values())
        return top / max(1, sum(self.domain_counts.values()))

    @property
    def confidence(self) -> float:
        """Composite score: cohesion × coverage × purity."""
        cohesion = min(1.0, self.evidence_total / max(1, len(self.members) * 20))
        coverage = min(1.0, len(self.members) / 10.0)
        purity   = self.domain_purity
        return round(cohesion * 0.4 + coverage * 0.3 + purity * 0.3, 3)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class ConceptAbstractionEngine:
    """Discovers latent abstract concepts with evidence/stability/purity gates."""

    def __init__(self, min_cluster_size: int = MIN_MEMBER_COUNT) -> None:
        self.min_cluster_size = max(2, min_cluster_size)
        self._known_abstractions: Set[str] = set()
        self._pending: Dict[str, AbstractionCandidate] = {}
        self._current_tick: int = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        triplets: List[Tuple[str, str, str]],
        knowledge_graph,
        type_system,
        curiosity_model,
        current_tick: int = 0,
    ) -> List[AbstractConcept]:
        """Run abstraction discovery with all quality gates applied.

        Parameters
        ----------
        triplets:       All current (subject, relation, object) triplets.
        knowledge_graph: Live KnowledgeGraph instance.
        type_system:    ConceptTypeSystem for registering new types.
        curiosity_model: CuriosityModel to notify of new concepts.
        current_tick:   Current agent tick (required for stability gate).

        Returns list of newly promoted AbstractConcept objects this tick.
        """
        self._current_tick = current_tick

        # Collect raw candidates from all three mechanisms
        raw_candidates = self._collect_candidates(triplets, knowledge_graph)

        # Update pending dict (track membership changes, reset stability timer)
        self._update_pending(raw_candidates, current_tick)

        # Apply all gates and promote qualifying candidates
        new_this_tick: List[AbstractConcept] = []

        for name, candidate in list(self._pending.items()):
            if name in self._known_abstractions:
                self._update_evidence(name, knowledge_graph)
                continue

            rejection = self._check_gates(candidate, current_tick)
            if rejection:
                logger.debug("[abstraction] rejected %s reason=%s", name, rejection)
                continue

            concept = AbstractConcept(
                name=name,
                members=set(candidate.members),
                source_rel=candidate.relation,
                evidence=candidate.evidence_total,
                confidence=candidate.confidence,
                origin="role_clustering",
                domain_purity=candidate.domain_purity,
            )
            self._inject(concept, knowledge_graph, type_system)
            self._known_abstractions.add(name)
            new_this_tick.append(concept)
            logger.info(
                "[abstraction] promoted name=%s members=%d conf=%.3f"
                " purity=%.2f evidence=%d stable_ticks=%d",
                name, len(concept.members), concept.confidence,
                concept.domain_purity, concept.evidence,
                current_tick - candidate.last_changed_tick,
            )

        # Notify curiosity
        if new_this_tick and curiosity_model is not None:
            try:
                curiosity_model.register_unknowns("abstraction", [c.name for c in new_this_tick])
                curiosity_model.register_new_pattern("abstraction")
            except Exception as exc:
                logger.debug("[abstraction] curiosity_update_failed error=%s", exc)

        # LRU eviction of pending candidates if dict is too large
        if len(self._pending) > MAX_PENDING_CANDIDATES:
            oldest = sorted(
                self._pending.items(),
                key=lambda kv: kv[1].first_seen_tick
            )[:len(self._pending) - MAX_PENDING_CANDIDATES]
            for k, _ in oldest:
                del self._pending[k]

        return new_this_tick

    # ── Gate checks ────────────────────────────────────────────────────────────

    def _check_gates(
        self,
        candidate: AbstractionCandidate,
        current_tick: int,
    ) -> Optional[str]:
        """Return the name of the first failed gate, or None if all pass."""
        if len(candidate.members) < self.min_cluster_size:
            return f"min_members({len(candidate.members)}<{self.min_cluster_size})"
        if candidate.evidence_total < MIN_CLUSTER_EVIDENCE:
            return f"min_evidence({candidate.evidence_total}<{MIN_CLUSTER_EVIDENCE})"
        if candidate.domain_purity < MIN_DOMAIN_PURITY:
            return f"domain_purity({candidate.domain_purity:.2f}<{MIN_DOMAIN_PURITY})"
        stability_age = current_tick - candidate.last_changed_tick
        if stability_age < MIN_CLUSTER_STABILITY_TICKS:
            return f"stability({stability_age}<{MIN_CLUSTER_STABILITY_TICKS})"
        if candidate.confidence < MIN_ABSTRACTION_CONFIDENCE:
            return f"confidence({candidate.confidence:.3f}<{MIN_ABSTRACTION_CONFIDENCE})"
        return None

    # ── Candidate collection ──────────────────────────────────────────────────

    def _collect_candidates(
        self,
        triplets: List[Tuple[str, str, str]],
        knowledge_graph,
    ) -> List[AbstractionCandidate]:
        """Build raw candidates from role clustering, chains, and co-occurrence."""
        candidates: Dict[str, AbstractionCandidate] = {}

        # 1. Role clustering
        subject_by_role: Dict[str, List[str]] = defaultdict(list)
        object_by_role:  Dict[str, List[str]] = defaultdict(list)

        for subj, rel, obj in triplets:
            if rel in STRUCTURAL_NOISE_RELATIONS:
                continue
            if rel in SUBJECT_ROLE:
                subject_by_role[SUBJECT_ROLE[rel]].append(subj)
            if rel in OBJECT_ROLE:
                object_by_role[OBJECT_ROLE[rel]].append(obj)

        for role, members in {**subject_by_role, **object_by_role}.items():
            if len(set(members)) < self.min_cluster_size:
                continue
            source_rel = next(
                (r for r, rn in {**SUBJECT_ROLE, **OBJECT_ROLE}.items() if rn == role),
                "unknown",
            )
            evidence = len(members)
            domain_counts = self._get_domain_counts(list(set(members)), knowledge_graph)
            candidates[role] = AbstractionCandidate(
                name=role, members=set(members), relation=source_rel,
                first_seen_tick=self._current_tick,
                last_changed_tick=self._current_tick,
                evidence_total=evidence,
                domain_counts=domain_counts,
            )

        # 2. Transitive chain abstraction
        by_relation: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for subj, rel, obj in triplets:
            by_relation[rel].append((subj, obj))

        for rel, pairs in by_relation.items():
            if rel not in CHAIN_LABELS or len(pairs) < MIN_CHAIN_LENGTH:
                continue
            adj: Dict[str, Set[str]] = defaultdict(set)
            for a, b in pairs:
                adj[a].add(b)
            chains = self._find_chains(adj, min_length=3)
            if not chains:
                continue
            all_members: Set[str] = set()
            for chain in chains:
                all_members.update(chain)
            label = CHAIN_LABELS[rel]
            domain_counts = self._get_domain_counts(list(all_members), knowledge_graph)
            candidates[label] = AbstractionCandidate(
                name=label, members=all_members, relation=rel,
                first_seen_tick=self._current_tick,
                last_changed_tick=self._current_tick,
                evidence_total=len(chains),
                domain_counts=domain_counts,
            )

        # 3. Co-occurrence grouping
        cooc: Dict[Tuple[str, str], int] = defaultdict(int)
        for subj, _rel, obj in triplets:
            if _rel in STRUCTURAL_NOISE_RELATIONS:
                continue
            pair = (min(subj, obj), max(subj, obj))
            cooc[pair] += 1

        entity_partners: Dict[str, Set[str]] = defaultdict(set)
        for (a, b), count in cooc.items():
            if count >= MIN_COOCCURRENCE:
                entity_partners[a].add(b)
                entity_partners[b].add(a)

        assigned: Set[str] = set()
        for entity, partners in sorted(entity_partners.items(), key=lambda x: -len(x[1])):
            if entity in assigned:
                continue
            group = ({entity} | partners) - assigned
            if len(group) < self.min_cluster_size:
                continue
            rel_counts: Dict[str, int] = defaultdict(int)
            for subj, rel, obj in triplets:
                if (subj in group or obj in group) and rel not in STRUCTURAL_NOISE_RELATIONS:
                    rel_counts[rel] += 1
            dominant_rel = max(rel_counts, key=rel_counts.__getitem__) if rel_counts else "related"
            role_name = SUBJECT_ROLE.get(dominant_rel, f"co_{dominant_rel}_group")
            label = f"{role_name}_cluster"
            domain_counts = self._get_domain_counts(list(group), knowledge_graph)
            candidates[label] = AbstractionCandidate(
                name=label, members=group, relation=dominant_rel,
                first_seen_tick=self._current_tick,
                last_changed_tick=self._current_tick,
                evidence_total=sum(rel_counts.values()),
                domain_counts=domain_counts,
            )
            assigned.update(group)

        return list(candidates.values())

    # ── Pending dict management ───────────────────────────────────────────────

    def _update_pending(
        self,
        candidates: List[AbstractionCandidate],
        current_tick: int,
    ) -> None:
        for cand in candidates:
            existing = self._pending.get(cand.name)
            if existing is None:
                cand.first_seen_tick = current_tick
                cand.last_changed_tick = current_tick
                self._pending[cand.name] = cand
            else:
                # If membership changed, reset stability timer
                if frozenset(cand.members) != frozenset(existing.members):
                    existing.last_changed_tick = current_tick
                    existing.members = cand.members
                    existing.domain_counts = cand.domain_counts
                # Always update evidence total
                existing.evidence_total = cand.evidence_total

    # ── Domain counting ───────────────────────────────────────────────────────

    @staticmethod
    def _get_domain_counts(
        members: List[str],
        knowledge_graph,
    ) -> Dict[str, int]:
        """Count how many members belong to each source domain via graph edges."""
        counts: Dict[str, int] = defaultdict(int)
        for member in members:
            # Try to find a domain edge for this member in the graph
            found = False
            for edge in getattr(knowledge_graph, "edges", []):
                if edge.source == member and edge.relation in ("instance_of", "subclass_of", "located_in"):
                    counts[edge.target] += 1
                    found = True
                    break
            if not found:
                # Infer domain from node type
                node = getattr(knowledge_graph, "nodes", {}).get(member)
                if node:
                    counts[getattr(node, "type", "unknown")] += 1
                else:
                    counts["unknown"] += 1
        return dict(counts)

    # ── Chain finding (unchanged from v1) ────────────────────────────────────

    @staticmethod
    def _find_chains(
        adj: Dict[str, Set[str]],
        min_length: int,
    ) -> List[List[str]]:
        chains: List[List[str]] = []
        visited: Set[str] = set()

        def dfs(node: str, path: List[str]) -> None:
            if len(path) >= min_length:
                chains.append(list(path))
            if node in visited or len(path) > 6:
                return
            visited.add(node)
            for neighbour in adj.get(node, set()):
                dfs(neighbour, path + [neighbour])
            visited.discard(node)

        for start in list(adj.keys()):
            dfs(start, [start])
        return chains

    # ── Graph injection (unchanged from v1) ──────────────────────────────────

    def _inject(
        self,
        concept: AbstractConcept,
        knowledge_graph,
        type_system,
    ) -> None:
        from digital_baby.brain.knowledge_graph import GraphEdge

        knowledge_graph.add_node(concept.name, node_type="abstract_concept")
        existing_keys = {(e.source, e.relation, e.target) for e in knowledge_graph.edges}

        for member in concept.members:
            knowledge_graph.add_node(member)
            key = (member, "is_instance_of", concept.name)
            if key not in existing_keys:
                knowledge_graph.edges.append(GraphEdge(
                    source=member,
                    target=concept.name,
                    relation="is_instance_of",
                    confidence=concept.confidence,
                    evidence=concept.evidence,
                    provenance="abstraction",
                ))
                existing_keys.add(key)
                knowledge_graph.tick_stats.new_edges += 1

        try:
            type_system.infer_from_triplets(
                [(m, "is_instance_of", concept.name) for m in concept.members]
            )
        except Exception as exc:
            logger.debug("[abstraction] type_system_update_failed error=%s", exc)

    def _update_evidence(self, name: str, knowledge_graph) -> None:
        for edge in knowledge_graph.edges:
            if edge.target == name and edge.relation == "is_instance_of":
                edge.evidence = getattr(edge, "evidence", 1) + 1
                edge.confidence = min(0.99, getattr(edge, "confidence", 0.5) + 0.005)