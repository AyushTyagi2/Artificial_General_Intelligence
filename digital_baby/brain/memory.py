"""Memory subsystem for the digital baby agent.

Stores unique facts with evidence, a knowledge graph, world-model rules/predictions,
and persistence for long-term runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import json
import time


@dataclass
class FactRecord:
    """Represents one learned unique fact and metadata."""

    statement: str
    confidence: float
    source_topic: str
    timestamp: float
    evidence: int = 1
    compressed: bool = False


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


class Memory:
    """Persistent memory containing factual beliefs, graph, and world model."""

    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path)
        self.facts: Dict[str, FactRecord] = {}
        self.entity_relations: Dict[str, Dict[str, Set[str]]] = {}
        self.relation_evidence: Dict[Tuple[str, str, str], int] = {}
        self.fact_index: Dict[Tuple[str, str, str], str] = {}
        self.patterns: List[PatternRecord] = []
        self.world_model: Dict[str, List[dict]] = {"rules": [], "predictions": [], "experiments": []}
        self._load()

    def upsert_fact(self, statement: str, confidence: float, source_topic: str, evidence_increment: int = 1) -> None:
        """Insert or update a fact with confidence clamped to [0, 1]."""
        bounded_confidence = max(0.0, min(1.0, confidence))
        now = time.time()
        if statement in self.facts:
            existing = self.facts[statement]
            existing.evidence += max(1, evidence_increment)
            existing.confidence = min(1.0, (existing.confidence * 0.85) + (bounded_confidence * 0.15) + 0.01)
            existing.source_topic = source_topic
            existing.timestamp = now
        else:
            self.facts[statement] = FactRecord(
                statement=statement,
                confidence=bounded_confidence,
                source_topic=source_topic,
                timestamp=now,
                evidence=max(1, evidence_increment),
            )

    def add_relation_fact(self, subject: str, relation: str, obj: str, source_topic: str, base_confidence: float = 0.6) -> bool:
        """Deduplicated fact ingestion using (subject, relation, object) index.

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
            self.upsert_fact(canonical, self.facts[canonical].confidence, source_topic=source_topic, evidence_increment=1)

        return is_new

    def decay_confidence(self, decay_rate: float = 0.01) -> None:
        for fact in self.facts.values():
            fact.confidence = max(0.0, fact.confidence - decay_rate)

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

    def update_patterns(self, rules: List[PatternRecord]) -> None:
        self.patterns = rules

    def set_world_model_rules(self, rules: List[HypothesisRecord]) -> None:
        self.world_model["rules"] = [asdict(r) for r in rules]

    def add_prediction(self, prediction: PredictionRecord) -> None:
        self.world_model.setdefault("predictions", []).append(asdict(prediction))
        # keep bounded history
        self.world_model["predictions"] = self.world_model["predictions"][-300:]


    def add_experiment(self, experiment: ExperimentRecord) -> None:
        self.world_model.setdefault("experiments", []).append(asdict(experiment))
        self.world_model["experiments"] = self.world_model["experiments"][-300:]

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
        summaries: List[str] = []
        herbivores = {"deer", "zebra", "rabbit", "antelope", "buffalo", "goat", "hare", "rodent", "seal"}
        for subject, rel_map in self.entity_relations.items():
            objs = rel_map.get(relation, set())
            if len(objs) >= min_objects:
                category = "herbivores" if all(o in herbivores for o in objs) else "multiple_entities"
                summary = f"{subject} {relation} {category}"
                if summary not in self.facts:
                    self.upsert_fact(summary, confidence=0.58, source_topic="memory_compression", evidence_increment=len(objs))
                    self.facts[summary].compressed = True
                summaries.append(summary)
        return sorted(set(summaries))

    def save(self) -> None:
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
            "world_model": self.world_model,
        }
        self.storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        payload = json.loads(self.storage_path.read_text(encoding="utf-8"))

        for item in payload.get("facts", []):
            item.setdefault("evidence", 1)
            item.setdefault("compressed", False)
            record = FactRecord(**item)
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

        self.world_model = payload.get("world_model", {"rules": [], "predictions": [], "experiments": []})
