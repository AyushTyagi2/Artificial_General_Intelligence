"""Prediction engine driven by hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import random

from .concepts import ConceptTypeSystem
from .hypothesis import Hypothesis


@dataclass
class Prediction:
    rule: str
    statement: str
    relation: str
    valid: bool


class Predictor:
    """Generates predicted facts and evaluates outcomes."""

    def __init__(self, seed: int | None = None) -> None:
        self.random = random.Random(seed)

    def predict(
        self,
        hypotheses: Sequence[Hypothesis],
        entities: Iterable[str],
        concept_types: ConceptTypeSystem,
    ) -> List[Prediction]:
        entity_list = sorted(set(entities))
        if len(entity_list) < 2:
            return []

        predictions: List[Prediction] = []
        for hypothesis in hypotheses[:6]:
            parts = hypothesis.rule.split()
            if len(parts) < 3:
                continue
            # typed rules like "predator hunts prey"
            relation = parts[1]
            subj_candidates = [e for e in entity_list if concept_types.get_type(e) == parts[0] or parts[0] == "unknown"]
            obj_candidates = [e for e in entity_list if concept_types.get_type(e) == parts[2] or parts[2] == "unknown"]
            if not subj_candidates:
                subj_candidates = entity_list
            if not obj_candidates:
                obj_candidates = entity_list
            subj = self.random.choice(subj_candidates)
            obj_pool = [e for e in obj_candidates if e != subj] or obj_candidates
            obj = self.random.choice(obj_pool)

            valid = concept_types.is_relation_valid(subj, relation, obj)
            predictions.append(Prediction(rule=hypothesis.rule, relation=relation, statement=f"{subj} {relation} {obj}", valid=valid))
        return predictions

    def evaluate(self, prediction: Prediction, observed_triplets: Sequence[Tuple[str, str, str]]) -> bool:
        expected = tuple(prediction.statement.split(maxsplit=2))
        return expected in observed_triplets
