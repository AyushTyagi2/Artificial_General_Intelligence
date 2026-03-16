"""Memory subsystem for the digital baby agent.

Stores unique facts with evidence, a knowledge graph, world-model rules/predictions,
and persistence for long-term runs.

Confidence model (Upgrade 1)
-----------------------------
Fact confidence is derived purely from accumulated evidence using the formula:

    confidence = evidence / (evidence + CONFIDENCE_PRIOR_K)

where CONFIDENCE_PRIOR_K = 10 acts as a Bayesian prior strength (equivalent to
starting with 10 pseudo-observations of uncertainty).  This gives:

    evidence =   1  ->  conf ≈ 0.09   (very uncertain – only seen once)
    evidence =  10  ->  conf = 0.50   (prior balanced by observation)
    evidence = 100  ->  conf ≈ 0.91   (well-supported belief)
    evidence = 990  ->  conf ≈ 0.99   (near-certain)

The computed value is always monotonically increasing with evidence and is
recomputed on every update, so no numeric drift accumulates over time.

Stale decay (optional, separate concern)
-----------------------------------------
Facts that have NOT been re-observed for more than STALE_TICKS ticks receive a
small linear penalty applied to the *stored* confidence float.  The underlying
evidence count is NEVER modified by decay.  On any re-observation the confidence
is immediately recomputed from evidence, erasing the stale penalty.  This lets
the agent treat genuinely forgotten knowledge as less reliable without discarding
it entirely.

Backward compatibility
-----------------------
The JSON schema is unchanged.  On load, stored confidence values are replaced
with the evidence-based formula so that old stores are automatically migrated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import json
import time


def _atomic_write_fd(path, write_fn) -> None:
    """Write via callback to path atomically using a sibling temp file."""
    import tempfile, os, shutil
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
            shutil.copy2(tmp, str(p))
            os.unlink(tmp)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── Confidence model constants ──────────────────────────────────────────────
# Prior strength: equivalent to k pseudo-observations of uncertainty.
# Increase k to make the agent more sceptical of sparse evidence.
CONFIDENCE_PRIOR_K: int = 10

# Facts not re-observed for this many ticks begin to receive stale decay.
STALE_TICKS: int = 10

# Linear confidence penalty per tick while a fact is stale.
# At this rate a fact with evidence=100 (base conf≈0.909) remains above 0.5
# for ~409 ticks of total non-observation before degrading further.
STALE_DECAY_RATE: float = 0.001

# Hard caps to keep memory_store.json from growing without bound.
MAX_FACTS: int = 10_000          # raised: delta facts no longer flood this
MAX_RELATION_TRIPLES: int = 10_000
MIN_CONFIDENCE_TO_KEEP: float = 0.05

# Ring-buffer caps for world_model lists (raised from 300 / 200)
MAX_PREDICTIONS: int = 2_000
MAX_EXPERIMENTS: int = 2_000
MAX_WM_RULES: int = 500


def evidence_to_confidence(evidence: int, k: int = CONFIDENCE_PRIOR_K) -> float:
    """Compute confidence from evidence count using a Beta-distribution mean.

    confidence = evidence / (evidence + k)

    This is monotonically increasing, bounded in (0, 1), and converges to 1
    asymptotically.  k sets how many observations are needed to reach 0.5.
    """
    return evidence / (evidence + k)


@dataclass
class FactRecord:
    """Represents one learned unique fact and metadata.

    Fields
    ------
    evidence       : cumulative observation count (never decremented)
    confidence     : derived from evidence via evidence_to_confidence(); also
                     receives a small linear stale penalty when the fact has
                     not been re-observed for STALE_TICKS ticks
    last_tick      : agent tick at which this fact was most recently observed
                     (0 = loaded from a pre-Upgrade-1 store, treated as fresh)
    compressed     : True when this record is a memory-compression summary
    """

    statement: str
    confidence: float
    source_topic: str
    timestamp: float
    evidence: int = 1
    compressed: bool = False
    last_tick: int = 0


@dataclass
class PatternRecord:
    """Stores one discovered generalized rule."""

    template: str
    relation: str
    support: int
    label: str
    timestamp: float


@dataclass
class BeliefState:
    """Resolved belief for one (entity, relation) pair."""

    best: str
    alternatives: List[str]
    confidence: float


@dataclass
class HypothesisRecord:
    """Persistent hypothesis in world model."""

    rule: str
    concepts: List[str]
    confidence: float
    supporting_evidence: int
    contradicting_evidence: int


@dataclass
class PredictionRecord:
    """Prediction generated from a hypothesis."""

    rule: str
    statement: str
    success: Optional[bool]
    timestamp: float


@dataclass
class ExperimentRecord:
    """Experiment result log entry."""

    rule: str
    name: str
    supported: int
    contradicted: int
    timestamp: float


@dataclass
class CausalRuleRecord:
    """Persistent directional causal rule discovered from transitions."""

    cause: str
    effect: str
    direction: str
    confidence: float
    observations: int


class Memory:
    """Persistent memory containing factual beliefs, graph, and world model."""

    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path)
        self.facts: Dict[str, FactRecord] = {}
        self.entity_relations: Dict[str, Dict[str, Set[str]]] = {}
        self.relation_evidence: Dict[Tuple[str, str, str], int] = {}
        self.fact_index: Dict[Tuple[str, str, str], str] = {}
        self.patterns: List[PatternRecord] = []
        self.causal_rules: List[CausalRuleRecord] = []
        self.world_model: Dict[str, List[dict]] = {"rules": [], "predictions": [], "experiments": []}
        # Cumulative counters -- never reset, grow monotonically.
        # Dashboard reads these to show total activity, not just the ring-buffer size.
        self.total_predictions: int = 0
        self.total_experiments: int = 0
        # Current agent tick, set externally by the event loop each tick.
        # Used to compute stale penalties without coupling Memory to wall-clock time.
        self.current_tick: int = 0
        self._load()

    # ── Fact ingestion ────────────────────────────────────────────────────────

    def upsert_fact(self, statement: str, confidence: float, source_topic: str, evidence_increment: int = 1) -> None:
        # Drop transient numeric delta facts -- they are unique every tick
        # (e.g. "temperature_delta is 5.2" vs "5.3"), never accumulate evidence,
        # and churn through the facts store crowding out structural knowledge.
        # law_discovery reads state deltas directly from the world step, so
        # dropping them here has no impact on causal inference.
        if "_delta" in statement:
            return
        """Insert or update a fact using the evidence-based confidence model.

        The ``confidence`` parameter is intentionally ignored for *existing*
        facts – confidence is always recomputed from the accumulated evidence
        count so that no caller can accidentally corrupt the signal by passing
        a stale or hard-coded value.

        Evidence is the single source of truth.  The stored ``confidence``
        float is a derived, display-ready value.  Any stale penalty is applied
        separately via ``apply_stale_decay``.
        """
        now = time.time()
        if statement in self.facts:
            existing = self.facts[statement]
            existing.evidence += max(1, evidence_increment)
            # Recompute confidence purely from evidence – no weighted blend,
            # no additive drift.
            existing.confidence = evidence_to_confidence(existing.evidence)
            existing.source_topic = source_topic
            existing.timestamp = now
            existing.last_tick = self.current_tick
        else:
            initial_evidence = max(1, evidence_increment)
            self.facts[statement] = FactRecord(
                statement=statement,
                confidence=evidence_to_confidence(initial_evidence),
                source_topic=source_topic,
                timestamp=now,
                evidence=initial_evidence,
                last_tick=self.current_tick,
            )

    def add_relation_fact(self, subject: str, relation: str, obj: str, source_topic: str, base_confidence: float = 0.6) -> bool:
        """Deduplicated fact ingestion using (subject, relation, object) index.

        ``base_confidence`` is kept in the signature for call-site compatibility
        but is no longer used to compute confidence – evidence drives everything.

        Returns True if this was a new unique fact, False if it already existed.
        """
        key = (subject, relation, obj)
        statement = f"{subject} {relation} {obj}"

        self.entity_relations.setdefault(subject, {}).setdefault(relation, set()).add(obj)
        self.relation_evidence[key] = self.relation_evidence.get(key, 0) + 1

        is_new = key not in self.fact_index
        if is_new:
            self.fact_index[key] = statement
            self.upsert_fact(statement, base_confidence, source_topic=source_topic, evidence_increment=1)
        else:
            canonical = self.fact_index[key]
            # base_confidence arg kept for compat; upsert ignores it for
            # existing facts and recomputes from evidence instead.
            self.upsert_fact(canonical, base_confidence, source_topic=source_topic, evidence_increment=1)

        return is_new

    # ── Confidence / decay ────────────────────────────────────────────────────

    def apply_stale_decay(self) -> int:
        """Apply a gentle confidence penalty to facts not seen recently.

        Only facts whose ``last_tick`` is more than STALE_TICKS ticks behind
        the current tick are penalised.  The penalty is linear at
        STALE_DECAY_RATE per tick of staleness beyond the threshold.

        Crucially, the underlying ``evidence`` count is NEVER modified.  The
        stale penalty is applied to the stored ``confidence`` float only, which
        means a single re-observation instantly restores full evidence-based
        confidence.

        Returns the number of facts that received a stale penalty this call.
        """
        penalised = 0
        for fact in self.facts.values():
            ticks_since_seen = self.current_tick - fact.last_tick
            if ticks_since_seen <= STALE_TICKS:
                continue
            stale_ticks = ticks_since_seen - STALE_TICKS
            base = evidence_to_confidence(fact.evidence)
            # Clamp to 0.01 — a fact can become very uncertain but should
            # never reach zero or negative confidence, which would corrupt
            # belief resolution and downstream hypothesis scoring.
            fact.confidence = max(0.01, base - STALE_DECAY_RATE * stale_ticks)
            penalised += 1
        return penalised

    def decay_confidence(self, decay_rate: float = 0.01) -> None:
        """Deprecated – replaced by apply_stale_decay().

        Kept for call-site compatibility.  Calling this is now a no-op so that
        any existing call in the event loop does not silently corrupt the new
        confidence model.  The event loop should be updated to call
        ``apply_stale_decay()`` instead, but even if it still calls this method
        the model remains correct.
        """
        # Intentionally empty – see apply_stale_decay() for the replacement.

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_fact(self, statement: str) -> Optional[FactRecord]:
        return self.facts.get(statement)

    def has_fact(self, statement: str) -> bool:
        return statement in self.facts

    def get_relations(self, subject: str) -> Dict[str, Set[str]]:
        return self.entity_relations.get(subject, {})

    def relation_evidence_count(self, subject: str, relation: str, obj: str) -> int:
        return self.relation_evidence.get((subject, relation, obj), 0)

    def find_topics_for_concept(self, concept: str) -> List[str]:
        lowered = concept.lower().strip()
        topics = {record.source_topic for record in self.facts.values() if lowered in record.statement.lower()}
        return sorted(topics)

    def all_entities(self) -> Set[str]:
        entities: Set[str] = set(self.entity_relations.keys())
        for rel_map in self.entity_relations.values():
            for objects in rel_map.values():
                entities.update(objects)
        return entities

    def relation_triplets(self) -> List[Tuple[str, str, str]]:
        triplets: List[Tuple[str, str, str]] = []
        for subject, rel_map in self.entity_relations.items():
            for relation, objects in rel_map.items():
                for obj in objects:
                    triplets.append((subject, relation, obj))
        return triplets

    def conflicting_relations_with_evidence(self) -> List[Tuple[str, str, List[Tuple[str, int]]]]:
        enriched: List[Tuple[str, str, List[Tuple[str, int]]]] = []
        for subject, rel_map in self.entity_relations.items():
            for relation, objects in rel_map.items():
                if len(objects) <= 1:
                    continue
                ranked = sorted(
                    [(obj, self.relation_evidence_count(subject, relation, obj)) for obj in objects],
                    key=lambda item: item[1],
                    reverse=True,
                )
                enriched.append((subject, relation, ranked))
        return enriched

    def resolve_belief(self, subject: str, relation: str) -> Optional[BeliefState]:
        values = self.entity_relations.get(subject, {}).get(relation, set())
        if not values:
            return None
        ranked = sorted(
            [(value, self.relation_evidence_count(subject, relation, value)) for value in values],
            key=lambda item: item[1],
            reverse=True,
        )
        total = sum(ev for _, ev in ranked)
        best, best_ev = ranked[0]
        alternatives = [value for value, _ in ranked[1:]]
        confidence = (best_ev / total) if total else 0.0
        return BeliefState(best=best, alternatives=alternatives, confidence=confidence)

    # ── Causal rules ──────────────────────────────────────────────────────────

    def upsert_causal_rule(self, cause: str, effect: str, direction: str, observations_increment: int = 1) -> Tuple[CausalRuleRecord, bool]:
        """Insert or update a causal rule with confidence tracking.

        Confidence is normalized by total observations for the same cause.
        Returns (rule, created_new).
        """
        cause = cause.strip().lower()
        effect = effect.strip().lower()
        direction = direction.strip().lower()
        existing = next((r for r in self.causal_rules if r.cause == cause and r.effect == effect), None)

        if existing is None:
            existing = CausalRuleRecord(
                cause=cause,
                effect=effect,
                direction=direction,
                confidence=0.0,
                observations=max(1, observations_increment),
            )
            self.causal_rules.append(existing)
            created_new = True
        else:
            created_new = False
            existing.observations += max(1, observations_increment)
            if existing.direction != direction:
                existing.direction = "mixed"

        total_cause_observations = sum(r.observations for r in self.causal_rules if r.cause == cause)
        existing.confidence = existing.observations / max(1, total_cause_observations)
        return existing, created_new

    def get_causal_rules(self) -> List[CausalRuleRecord]:
        return list(self.causal_rules)

    # ── Patterns / world model ────────────────────────────────────────────────

    def update_patterns(self, rules: List[PatternRecord]) -> None:
        self.patterns = rules

    def set_world_model_rules(self, rules: List[HypothesisRecord]) -> None:
        """Merge new rules into world_model without wiping existing evidence.

        Existing rules keep their accumulated supported/contradicted counts.
        New rules are appended. The list is capped at 200 entries (highest
        confidence first) so it doesn't grow unboundedly.
        """
        existing = {r["rule"]: r for r in self.world_model.get("rules", [])}
        for r in rules:
            d = asdict(r)
            if d["rule"] in existing:
                # Preserve accumulated evidence — only update confidence
                existing[d["rule"]]["confidence"] = d["confidence"]
            else:
                existing[d["rule"]] = d
        # Keep the 200 highest-confidence rules
        merged = sorted(existing.values(), key=lambda x: x.get("confidence", 0), reverse=True)
        self.world_model["rules"] = merged[:MAX_WM_RULES]

    def add_prediction(self, prediction: PredictionRecord) -> None:
        self.world_model.setdefault("predictions", []).append(asdict(prediction))
        self.world_model["predictions"] = self.world_model["predictions"][-MAX_PREDICTIONS:]
        self.total_predictions += 1  # cumulative counter

    def add_experiment(self, experiment: ExperimentRecord) -> None:
        self.world_model.setdefault("experiments", []).append(asdict(experiment))
        self.world_model["experiments"] = self.world_model["experiments"][-MAX_EXPERIMENTS:]
        self.total_experiments += 1  # cumulative counter

    def update_hypothesis_evidence(self, rule: str, supported: int, contradicted: int) -> Optional[dict]:
        """Update stored rule evidence/confidence from experiment outcomes."""
        rules = self.world_model.get("rules", [])
        for record in rules:
            if record.get("rule") != rule:
                continue
            record["supporting_evidence"] = int(record.get("supporting_evidence", 0)) + int(supported)
            record["contradicting_evidence"] = int(record.get("contradicting_evidence", 0)) + int(contradicted)
            sup = record["supporting_evidence"]
            con = record["contradicting_evidence"]
            record["confidence"] = sup / max(1, sup + con)
            return record
        return None

    def hypothesis_uncertainty(self) -> float:
        rules = self.world_model.get("rules", [])
        if not rules:
            return 0.0
        return sum(1.0 - float(rule.get("confidence", 0.0)) for rule in rules) / len(rules)

    def compress_relation_facts(self, relation: str, min_objects: int = 2) -> List[str]:
        """Summarise subjects with many objects for a relation into a single fact.

        The summary fact is given evidence equal to the number of objects it
        covers, so its confidence reflects how many observations back it up.
        """
        summaries: List[str] = []
        herbivores = {"deer", "zebra", "rabbit", "antelope", "buffalo", "goat", "hare", "rodent", "seal"}
        for subject, rel_map in self.entity_relations.items():
            objs = rel_map.get(relation, set())
            if len(objs) >= min_objects:
                category = "herbivores" if all(o in herbivores for o in objs) else "multiple_entities"
                summary = f"{subject} {relation} {category}"
                if summary not in self.facts:
                    # evidence_increment = number of individual facts being
                    # compressed; confidence is then derived automatically.
                    self.upsert_fact(
                        summary,
                        confidence=0.0,  # ignored for new facts; evidence drives conf
                        source_topic="memory_compression",
                        evidence_increment=len(objs),
                    )
                    self.facts[summary].compressed = True
                summaries.append(summary)
        return sorted(set(summaries))

    # ── Persistence ───────────────────────────────────────────────────────────

    # ── Size management ──────────────────────────────────────────────────────

    def _prune(self) -> None:
        """Evict weakest facts/relation triples when hard caps are exceeded."""
        # --- facts cap ---
        if len(self.facts) > MAX_FACTS:
            sorted_stmts = sorted(self.facts, key=lambda s: self.facts[s].confidence)
            for stmt in sorted_stmts[:len(self.facts) - MAX_FACTS]:
                del self.facts[stmt]

        # --- relation triples cap ---
        if len(self.relation_evidence) > MAX_RELATION_TRIPLES:
            sorted_triples = sorted(self.relation_evidence, key=lambda k: self.relation_evidence[k])
            for triple in sorted_triples[:len(self.relation_evidence) - MAX_RELATION_TRIPLES]:
                self.relation_evidence.pop(triple, None)
                self.fact_index.pop(triple, None)
                s, r, o = triple
                rel_map = self.entity_relations.get(s, {})
                objs = rel_map.get(r)
                if objs:
                    objs.discard(o)
                    if not objs:
                        del rel_map[r]
                    if not rel_map:
                        del self.entity_relations[s]

    def save(self) -> None:
        self._prune()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "facts": [asdict(record) for record in self.facts.values()],
            "entity_relations": {
                subject: {relation: sorted(list(objects)) for relation, objects in rel_map.items()}
                for subject, rel_map in self.entity_relations.items()
            },
            "relation_evidence": [
                {"subject": s, "relation": r, "object": o, "evidence": ev}
                for (s, r, o), ev in self.relation_evidence.items()
            ],
            "fact_index": [
                {"subject": s, "relation": r, "object": o, "statement": st}
                for (s, r, o), st in self.fact_index.items()
            ],
            "patterns": [asdict(pattern) for pattern in self.patterns],
            "causal_rules": [asdict(rule) for rule in self.causal_rules],
            "world_model": self.world_model,
            "total_predictions": self.total_predictions,
            "total_experiments": self.total_experiments,
        }
        # Atomic write: write to a temp file, then replace.
        # Prevents a corrupt/null-byte file if the process crashes mid-write.
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
            import logging, shutil, time
            backup = self.storage_path.with_suffix(f".corrupted.{int(time.time())}.json")
            shutil.move(str(self.storage_path), str(backup))
            logging.getLogger(__name__).warning(
                "Memory file was corrupt and could not be parsed. "
                "Backed up to %s — starting with fresh memory.", backup
            )
            return

        for item in payload.get("facts", []):
            item.setdefault("evidence", 1)
            item.setdefault("compressed", False)
            # Upgrade-1 migration: last_tick may be absent in older stores.
            # Default to 0 (treated as "seen at tick 0", i.e. not yet stale).
            item.setdefault("last_tick", 0)
            record = FactRecord(**item)
            # Always recompute confidence from evidence on load so that stores
            # written by the old decay-based model are automatically corrected.
            record.confidence = evidence_to_confidence(record.evidence)
            self.facts[record.statement] = record

        for subject, rel_map in payload.get("entity_relations", {}).items():
            self.entity_relations[subject] = {relation: set(objects) for relation, objects in rel_map.items()}

        for entry in payload.get("relation_evidence", []):
            self.relation_evidence[(entry["subject"], entry["relation"], entry["object"])] = int(entry.get("evidence", 1))

        for entry in payload.get("fact_index", []):
            self.fact_index[(entry["subject"], entry["relation"], entry["object"])] = entry["statement"]

        # backward-compatible reconstruction if fact_index absent
        if not self.fact_index:
            for statement in self.facts:
                tokens = statement.split()
                if len(tokens) >= 3:
                    s = tokens[0].lower()
                    r = tokens[1].lower()
                    o = " ".join(tokens[2:]).lower()
                    self.fact_index[(s, r, o)] = statement

        for entry in payload.get("patterns", []):
            self.patterns.append(PatternRecord(**entry))

        for entry in payload.get("causal_rules", []):
            self.causal_rules.append(CausalRuleRecord(**entry))

        self.world_model = payload.get("world_model", {"rules": [], "predictions": [], "experiments": []})

        # Restore cumulative counters (backwards-compatible: default to buffer size)
        self.total_predictions = payload.get(
            "total_predictions",
            len(self.world_model.get("predictions", [])),
        )
        self.total_experiments = payload.get(
            "total_experiments",
            len(self.world_model.get("experiments", [])),
        )