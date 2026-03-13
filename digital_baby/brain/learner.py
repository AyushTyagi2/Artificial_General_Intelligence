"""Learning logic for reading world pages and updating memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from .memory import Memory
from .reasoning import Reasoner


@dataclass
class LearningResult:
    topic: str
    learned_facts: int
    new_facts: int
    unknown_concepts: Set[str]
    unknown_relations: Set[str]
    new_entities: Set[str]
    new_relation_types: Set[str]
    weak_fact_ratio: float


class Learner:
    def __init__(self, memory: Memory, reasoner: Reasoner) -> None:
        self.memory = memory
        self.reasoner = reasoner

    def learn_from_page(self, page: Dict) -> LearningResult:
        topic = page["topic"]
        facts: List[str] = page.get("facts", [])
        unknown_concepts: Set[str] = set()
        unknown_relations: Set[str] = set()
        new_entities: Set[str] = set()
        new_relation_types: Set[str] = set()
        weak_facts = 0
        new_facts = 0

        for fact in facts:
            relation = self.reasoner.extract_relation(fact)
            confidence = 0.6
            if relation:
                subj, rel, obj = relation
                known_entities = self.memory.all_entities()
                known_subject_relations = self.memory.get_relations(subj)

                if subj not in known_entities:
                    unknown_concepts.add(subj)
                    new_entities.add(subj)
                if obj not in known_entities:
                    unknown_concepts.add(obj)
                    new_entities.add(obj)
                if rel not in known_subject_relations:
                    unknown_relations.add(rel)
                    new_relation_types.add(rel)

                is_new = self.memory.add_relation_fact(subj, rel, obj, source_topic=topic, base_confidence=confidence)
                if is_new:
                    new_facts += 1
            else:
                # fallback for unparsable text facts
                existing = self.memory.get_fact(fact)
                if existing is None:
                    new_facts += 1
                self.memory.upsert_fact(fact, confidence, source_topic=topic, evidence_increment=1)

            existing = self.memory.get_fact(fact)
            if existing and existing.confidence < 0.5:
                weak_facts += 1

        weak_ratio = (weak_facts / len(facts)) if facts else 0.0
        return LearningResult(
            topic=topic,
            learned_facts=len(facts),
            new_facts=new_facts,
            unknown_concepts=unknown_concepts,
            unknown_relations=unknown_relations,
            new_entities=new_entities,
            new_relation_types=new_relation_types,
            weak_fact_ratio=weak_ratio,
        )
