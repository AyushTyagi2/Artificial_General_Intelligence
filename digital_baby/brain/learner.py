"""Learning logic for reading world pages and updating memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from .memory import Memory
from .reasoning import Reasoner


@dataclass
class LearningResult:
    """Outcome of a single topic learning step."""

    topic: str
    learned_facts: int
    unknown_concepts: Set[str]
    weak_fact_ratio: float


class Learner:
    """Parses facts, updates confidence, and populates knowledge graph edges."""

    def __init__(self, memory: Memory, reasoner: Reasoner) -> None:
        self.memory = memory
        self.reasoner = reasoner

    def learn_from_page(self, page: Dict) -> LearningResult:
        """Learn facts from one knowledge page dictionary."""
        topic = page["topic"]
        facts: List[str] = page.get("facts", [])
        unknown_concepts: Set[str] = set()
        weak_facts = 0

        for fact in facts:
            relation = self.reasoner.extract_relation(fact)
            confidence = 0.6
            if relation:
                subj, rel, obj = relation
                known_entities = self.memory.all_entities()
                # New entities indicate uncertainty and future curiosity.
                if subj not in known_entities:
                    unknown_concepts.add(subj)
                if obj not in known_entities:
                    unknown_concepts.add(obj)
                self.memory.add_relation(subj, rel, obj)

            existing = self.memory.get_fact(fact)
            if existing is not None:
                confidence = min(1.0, existing.confidence + 0.1)

            self.memory.upsert_fact(fact, confidence, source_topic=topic)
            if confidence < 0.5:
                weak_facts += 1

        weak_ratio = (weak_facts / len(facts)) if facts else 0.0
        return LearningResult(
            topic=topic,
            learned_facts=len(facts),
            unknown_concepts=unknown_concepts,
            weak_fact_ratio=weak_ratio,
        )
