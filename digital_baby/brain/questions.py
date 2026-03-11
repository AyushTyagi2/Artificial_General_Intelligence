"""Question generation for goal-directed exploration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass
class Question:
    """Represents a curiosity question that can drive topic selection."""

    text: str
    target_concept: str
    reason: str


class QuestionGenerator:
    """Creates simple questions from unknown concepts and conflicts."""

    def from_unknown_concepts(self, concepts: Iterable[str]) -> List[Question]:
        questions: List[Question] = []
        for concept in sorted({c.strip().lower() for c in concepts if c.strip()}):
            questions.append(
                Question(
                    text=f"What is {concept}?",
                    target_concept=concept,
                    reason="unknown_entity",
                )
            )
        return questions

    def from_conflicts(self, conflicts: Sequence[Tuple[str, str, Sequence[str]]]) -> List[Question]:
        questions: List[Question] = []
        for entity, relation, values in conflicts:
            values_text = " vs ".join(values)
            questions.append(
                Question(
                    text=f"How does {entity} {relation} relate to {values_text}?",
                    target_concept=entity,
                    reason="conflict_investigation",
                )
            )
        return questions
