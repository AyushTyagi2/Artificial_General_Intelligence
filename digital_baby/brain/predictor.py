"""Prediction engine driven by hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
import random

from .concepts import ConceptTypeSystem
from .hypothesis import Hypothesis


@dataclass
class Prediction:
    rule: str
    statement: str
    relation: str
    valid: bool


@dataclass
class PredictionEvaluation:
    """Outcome of predicted-vs-observed comparison."""

    success: bool
    prediction_error: float
    curiosity_score: float


@dataclass
class StatePrediction:
    """Predicted scalar state deltas for causal evaluation."""

    action: str
    expected_deltas: Dict[str, float]


class Predictor:
    """Generates predictions for facts and state transitions."""

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

    @staticmethod
    def evaluate_outcome(predicted_success: bool, actual_success: bool) -> PredictionEvaluation:
        success = predicted_success == actual_success
        prediction_error = 0.0 if success else 1.0
        curiosity_score = prediction_error * 10.0
        return PredictionEvaluation(success=success, prediction_error=prediction_error, curiosity_score=curiosity_score)

    @staticmethod
    def predict_state_transition(domain: str, action: str, state: Dict[str, float]) -> StatePrediction:
        """Predict high-level state deltas from an action."""
        expected: Dict[str, float] = {}
        if domain == "ecosystem":
            if action == "remove_predator":
                expected = {"wolves": -1.0, "deer": 1.0}
            elif action == "add_predator":
                expected = {"wolves": 1.0, "deer": -1.0}
            else:
                expected = {"deer": 0.5, "grass": -0.5}
        elif domain == "chemistry":
            if action == "increase_temperature":
                expected = {"temperature": 4.0, "reaction_energy": 0.6}
            elif action == "add_chemical":
                expected = {"reactants": 1.0, "reaction_energy": 0.4}
        elif domain == "astronomy":
            expected = {"asteroids": 1.0 if action == "introduce_species" else -1.0, "solar_energy": 0.0}
        return StatePrediction(action=action, expected_deltas=expected)

    @staticmethod
    def evaluate_state_prediction(prediction: StatePrediction, observed_deltas: Dict[str, float]) -> PredictionEvaluation:
        """Compute absolute error between expected and observed state transitions."""
        if not prediction.expected_deltas:
            return PredictionEvaluation(success=True, prediction_error=0.0, curiosity_score=0.0)

        error_sum = 0.0
        for key, expected in prediction.expected_deltas.items():
            observed = float(observed_deltas.get(key, 0.0))
            error_sum += abs(expected - observed)
        error = error_sum / max(1, len(prediction.expected_deltas))
        success = error < 0.75
        return PredictionEvaluation(success=success, prediction_error=error, curiosity_score=error * 10.0)
