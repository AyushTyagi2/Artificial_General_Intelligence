"""Memory subsystem for the digital baby agent.

Stores facts with confidence scores, a lightweight knowledge graph, and supports
persistence so learning survives across runs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
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


class Memory:
    """Persistent memory containing factual beliefs and a simple knowledge graph."""

    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path)
        self.facts: Dict[str, FactRecord] = {}
        # Graph shape: entity_relations[subject][relation] -> set(objects)
        self.entity_relations: Dict[str, Dict[str, Set[str]]] = {}
        self._load()

    def upsert_fact(self, statement: str, confidence: float, source_topic: str) -> None:
        """Insert or update a fact with confidence clamped to [0, 1]."""
        bounded_confidence = max(0.0, min(1.0, confidence))
        now = time.time()
        if statement in self.facts:
            existing = self.facts[statement]
            # Blend old and new confidence for stability.
            blended = (existing.confidence * 0.7) + (bounded_confidence * 0.3)
            existing.confidence = max(0.0, min(1.0, blended))
            existing.source_topic = source_topic
            existing.timestamp = now
        else:
            self.facts[statement] = FactRecord(
                statement=statement,
                confidence=bounded_confidence,
                source_topic=source_topic,
                timestamp=now,
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

    def add_relation(self, subject: str, relation: str, obj: str) -> None:
        """Add a relation edge in the graph."""
        relation_map = self.entity_relations.setdefault(subject, {})
        relation_map.setdefault(relation, set()).add(obj)

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

    def save(self) -> None:
        """Persist memory state to disk as JSON."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "facts": [asdict(record) for record in self.facts.values()],
            "entity_relations": {
                subject: {relation: sorted(list(objects)) for relation, objects in rel_map.items()}
                for subject, rel_map in self.entity_relations.items()
            },
        }
        self.storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> None:
        """Load memory state if it exists."""
        if not self.storage_path.exists():
            return

        payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        for item in payload.get("facts", []):
            record = FactRecord(**item)
            self.facts[record.statement] = record

        for subject, rel_map in payload.get("entity_relations", {}).items():
            self.entity_relations[subject] = {
                relation: set(objects) for relation, objects in rel_map.items()
            }
