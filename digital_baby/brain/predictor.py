"""Prediction engine driven by hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import random

from .hypothesis import Hypothesis


@dataclass
class Prediction:
    """Predicted relation expected from a hypothesis."""

    rule: str
    statement: str
    relation: str


class Predictor:
    """Generates predicted facts and evaluates outcomes."""

    def __init__(self, seed: int | None = None) -> None:
        self.random = random.Random(seed)

    def predict(self, hypotheses: Sequence[Hypothesis], entities: Iterable[str]) -> List[Prediction]:
        entity_list = sorted(set(entities))
        if len(entity_list) < 2:
            return []

        predictions: List[Prediction] = []
        for hypothesis in hypotheses[:5]:
            parts = hypothesis.rule.split()
            if len(parts) < 3:
                continue
            relation = parts[1]
            subj = self.random.choice(entity_list)
            obj = self.random.choice([e for e in entity_list if e != subj] or entity_list)
            predictions.append(
                Prediction(
                    rule=hypothesis.rule,
                    relation=relation,
                    statement=f"{subj} {relation} {obj}",
                )
            )
        return predictions

    def evaluate(self, prediction: Prediction, observed_triplets: Sequence[Tuple[str, str, str]]) -> bool:
        expected = tuple(prediction.statement.split(maxsplit=2))
        return expected in observed_triplets
