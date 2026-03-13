"""Memory subsystem for the digital baby agent.

Stores facts with confidence/evidence, a scalable knowledge graph, discovered patterns,
and supports persistence so learning survives across runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import json
import time


@dataclass
class FactRecord:
    """Represents one learned fact and metadata."""

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


class Memory:
    """Persistent memory containing factual beliefs and a simple knowledge graph."""

    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path)
        self.facts: Dict[str, FactRecord] = {}
        # Graph shape: entity_relations[subject][relation] -> set(objects)
        self.entity_relations: Dict[str, Dict[str, Set[str]]] = {}
        # Evidence shape: relation_evidence[(subject, relation, object)] -> count
        self.relation_evidence: Dict[Tuple[str, str, str], int] = {}
        self.patterns: List[PatternRecord] = []
        self._load()

    def upsert_fact(self, statement: str, confidence: float, source_topic: str, evidence_increment: int = 1) -> None:
        """Insert or update a fact with confidence clamped to [0, 1]."""
        bounded_confidence = max(0.0, min(1.0, confidence))
        now = time.time()
        if statement in self.facts:
            existing = self.facts[statement]
            existing.evidence += max(1, evidence_increment)
            # Evidence-weighted confidence update.
            existing.confidence = min(1.0, existing.confidence + (0.03 * evidence_increment))
            blended = (existing.confidence * 0.8) + (bounded_confidence * 0.2)
            existing.confidence = max(0.0, min(1.0, blended))
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

    def decay_confidence(self, decay_rate: float = 0.01) -> None:
        """Decay confidence for all facts by a fixed amount each tick."""
        for fact in self.facts.values():
            fact.confidence = max(0.0, fact.confidence - decay_rate)

    def get_fact(self, statement: str) -> Optional[FactRecord]:
        """Retrieve one fact if it exists."""
        return self.facts.get(statement)

    def weakest_facts(self, top_n: int = 5) -> List[FactRecord]:
        """Return the lowest-confidence facts to drive curiosity."""
        return sorted(self.facts.values(), key=lambda f: f.confidence)[:top_n]

    def add_relation(self, subject: str, relation: str, obj: str, evidence_increment: int = 1) -> None:
        """Add a relation edge in the graph and track evidence count."""
        relation_map = self.entity_relations.setdefault(subject, {})
        relation_map.setdefault(relation, set()).add(obj)
        key = (subject, relation, obj)
        self.relation_evidence[key] = self.relation_evidence.get(key, 0) + max(1, evidence_increment)

    def relation_evidence_count(self, subject: str, relation: str, obj: str) -> int:
        """Return evidence count for one relation edge."""
        return self.relation_evidence.get((subject, relation, obj), 0)

    def get_relations(self, subject: str) -> Dict[str, Set[str]]:
        """Get all outgoing relations for a subject."""
        return self.entity_relations.get(subject, {})

    def query_entities(self, keyword: str) -> List[str]:
        """Query known entities by keyword match for fuzzy concept discovery."""
        lowered = keyword.lower().strip()
        return sorted([entity for entity in self.all_entities() if lowered in entity])

    def find_topics_for_concept(self, concept: str) -> List[str]:
        """Find source topics that mention a concept in their fact statement."""
        lowered = concept.lower().strip()
        topics = {
            record.source_topic
            for record in self.facts.values()
            if lowered in record.statement.lower()
        }
        return sorted(topics)

    def all_entities(self) -> Set[str]:
        """Return a set of known entities from the graph."""
        entities: Set[str] = set(self.entity_relations.keys())
        for rel_map in self.entity_relations.values():
            for objects in rel_map.values():
                entities.update(objects)
        return entities

    def relation_triplets(self) -> List[Tuple[str, str, str]]:
        """Return all graph edges as triplets."""
        triplets: List[Tuple[str, str, str]] = []
        for subject, rel_map in self.entity_relations.items():
            for relation, objects in rel_map.items():
                for obj in objects:
                    triplets.append((subject, relation, obj))
        return triplets

    def conflicting_relations(self) -> List[Tuple[str, str, List[str]]]:
        """Return conflicting edges where one (subject, relation) has multiple values."""
        conflicts: List[Tuple[str, str, List[str]]] = []
        for subject, rel_map in self.entity_relations.items():
            for relation, objects in rel_map.items():
                if len(objects) > 1:
                    conflicts.append((subject, relation, sorted(objects)))
        return conflicts

    def conflicting_relations_with_evidence(self) -> List[Tuple[str, str, List[Tuple[str, int]]]]:
        """Return conflicts enriched with evidence counts per competing value."""
        enriched: List[Tuple[str, str, List[Tuple[str, int]]]] = []
        for subject, relation, values in self.conflicting_relations():
            ranked = sorted(
                [(value, self.relation_evidence_count(subject, relation, value)) for value in values],
                key=lambda item: item[1],
                reverse=True,
            )
            enriched.append((subject, relation, ranked))
        return enriched

    def update_patterns(self, rules: List[PatternRecord]) -> None:
        """Replace stored pattern records with latest discovery pass."""
        self.patterns = rules

    def has_fact(self, statement: str) -> bool:
        """Return whether a fact statement already exists in memory."""
        return statement in self.facts

    def compress_relation_facts(self, relation: str, min_objects: int = 3) -> List[str]:
        """Compress redundant relation facts for same subject into summary facts."""
        summaries: List[str] = []
        for subject, rel_map in self.entity_relations.items():
            objs = rel_map.get(relation, set())
            if len(objs) >= min_objects:
                # Try concept-aware category summarization first.
                category = "multiple_entities"
                herbivores = {"deer", "zebra", "rabbit", "antelope", "buffalo", "goat", "hare", "rodent"}
                if objs and all(obj in herbivores for obj in objs):
                    category = "herbivores"

                summary = f"{subject} {relation} {category}"
                if summary not in self.facts:
                    self.upsert_fact(summary, confidence=0.58, source_topic="memory_compression", evidence_increment=len(objs))
                    self.facts[summary].compressed = True
                summaries.append(summary)

        # Global relation summary when many subjects share structure.
        subjects_with_relation = [s for s, m in self.entity_relations.items() if relation in m]
        if len(subjects_with_relation) >= 4:
            global_summary = f"multiple_predators {relation} multiple_prey"
            if global_summary not in self.facts:
                self.upsert_fact(global_summary, confidence=0.56, source_topic="memory_compression", evidence_increment=len(subjects_with_relation))
                self.facts[global_summary].compressed = True
            summaries.append(global_summary)

        return sorted(set(summaries))

    def save(self) -> None:
        """Persist memory state to disk as JSON."""
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
            "patterns": [asdict(pattern) for pattern in self.patterns],
        }
        self.storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> None:
        """Load memory state if it exists."""
        if not self.storage_path.exists():
            return

        payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        for item in payload.get("facts", []):
            if "evidence" not in item:
                item["evidence"] = 1
            if "compressed" not in item:
                item["compressed"] = False
            record = FactRecord(**item)
            self.facts[record.statement] = record

        for subject, rel_map in payload.get("entity_relations", {}).items():
            self.entity_relations[subject] = {relation: set(objects) for relation, objects in rel_map.items()}

        for entry in payload.get("relation_evidence", []):
            key = (entry["subject"], entry["relation"], entry["object"])
            self.relation_evidence[key] = int(entry.get("evidence", 1))

        for entry in payload.get("patterns", []):
            self.patterns.append(PatternRecord(**entry))
