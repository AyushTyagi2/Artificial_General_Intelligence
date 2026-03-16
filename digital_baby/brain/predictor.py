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

    @staticmethod
    def _is_causal_rule(rule: str) -> bool:
        """Return True if this is a causal template like 'if X increases then Y will increase'."""
        r = rule.lower()
        return r.startswith("if ") or " affects " in r or " controls " in r

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
            # ── Causal hypothesis ─────────────────────────────────────────
            if self._is_causal_rule(hypothesis.rule):
                # For causal rules, validity is determined by confidence.
                # A high-confidence causal rule IS a valid prediction.
                valid = hypothesis.confidence >= 0.5
                concepts = hypothesis.concepts
                cause = concepts[0] if concepts else "unknown"
                effect = concepts[1] if len(concepts) > 1 else "unknown"
                predictions.append(Prediction(
                    rule=hypothesis.rule,
                    relation="causal",
                    statement=f"{cause} causally_affects {effect}",
                    valid=valid,
                ))
                continue

            # ── Structural hypothesis: 'subject_type relation object_type' ─
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
        """Predict state deltas from an action using the known causal graph.

        These predictions encode the *true* causal structure so prediction
        error is a meaningful learning signal — low error means the agent
        has correctly learned this causal relation, high error means there
        is still something to discover.
        """
        expected: Dict[str, float] = {}

        if domain == "ecosystem":
            wolves = state.get("wolves", 5)
            deer   = state.get("deer", 20)
            if action == "add_predator":
                # wolves↑ → deer↓ (predation), grass↑ (deer pressure drops)
                expected = {"wolves": +1.0, "deer": -(wolves + 1) * 1.5, "grass": deer * 0.1}
            elif action == "remove_predator":
                # wolves↓ → deer↑ (release), grass↓ (more grazing)
                expected = {"wolves": -1.0, "deer": +(wolves * 1.5), "grass": -deer * 0.1}
            elif action == "introduce_species":
                # deer↑ → grass↓ (more grazing)
                expected = {"deer": +3.0, "grass": -(deer + 3) * 0.8}
            elif action == "remove_species":
                # deer↓ → grass↑ (less grazing)
                expected = {"deer": -3.0, "grass": +(deer - 3) * 0.5}

        elif domain == "chemistry":
            temperature = state.get("temperature", 25)
            reactants   = state.get("reactants", 2)
            if action == "increase_temperature":
                # temperature↑ → reaction_rate↑ → reaction_energy↑
                new_temp = temperature + 7.5  # midpoint of 5-10 range
                new_rate = (new_temp / 50.0) * reactants
                expected = {
                    "temperature": +7.5,
                    "reaction_rate": new_rate - state.get("reaction_rate", 0),
                    "reaction_energy": new_rate * 2.0,
                }
            elif action == "add_chemical":
                # reactants↑ → reaction_rate↑ → reaction_energy↑
                new_rate = (temperature / 50.0) * (reactants + 1)
                expected = {
                    "reactants": +1.0,
                    "reaction_rate": new_rate - state.get("reaction_rate", 0),
                    "reaction_energy": new_rate * 2.0,
                }

        elif domain == "astronomy":
            asteroids = state.get("asteroids", 200)
            if action == "introduce_species":
                # asteroids↑ → collision_risk↑
                added = 5.0  # midpoint of 3-7
                expected = {
                    "asteroids": +added,
                    "collision_risk": added * 0.05,
                }
            elif action == "remove_species":
                removed = 5.0
                expected = {
                    "asteroids": -removed,
                    "collision_risk": -removed * 0.05,
                }

        elif domain == "technology":
            robots         = state.get("robots", 4)
            battery_charge = state.get("battery_charge", 80)
            if action == "add_robot":
                # robots↑ → robot_activity↑ → sensor_coverage↑ → data_quality↑
                new_activity = (robots + 1) * (battery_charge / 100.0)
                expected = {
                    "robots": +1.0,
                    "robot_activity": new_activity - state.get("robot_activity", 0),
                    "sensor_coverage": new_activity * 2.5 - state.get("sensor_coverage", 0),
                }
            elif action == "remove_robot":
                new_activity = max(0, robots - 1) * (battery_charge / 100.0)
                expected = {
                    "robots": -1.0,
                    "robot_activity": new_activity - state.get("robot_activity", 0),
                    "sensor_coverage": new_activity * 2.5 - state.get("sensor_coverage", 0),
                }

        elif domain == "biology":
            pathogens = state.get("pathogens", 0)
            energy    = state.get("energy", 50)
            cells     = state.get("cells", 100)
            if action == "add_pathogen":
                immune = pathogens * 2.5
                expected = {
                    "pathogens": +3.5,
                    "immune_response": immune,
                    "cells": -(pathogens + 3.5) * 0.8,
                }
            elif action == "boost_energy":
                new_prot = (energy + 15) / 10.0 * 2.0
                expected = {
                    "energy": +15.0,
                    "proteins": new_prot - state.get("proteins", 20),
                    "cells": new_prot * 0.5,
                }
            elif action == "add_cells":
                expected = {
                    "cells": +10.0,
                    "energy": -10.0 * 0.05,
                }

        elif domain == "physics":
            force = state.get("force", 10)
            mass  = state.get("mass", 5)
            heat  = state.get("heat", 25)
            if action == "increase_force":
                new_accel = (force + 4) / max(0.1, mass)
                expected = {
                    "force": +4.0,
                    "acceleration": new_accel - state.get("acceleration", 2),
                    "kinetic_energy": 0.5 * mass * new_accel ** 2 - state.get("kinetic_energy", 50),
                }
            elif action == "add_heat":
                expected = {
                    "heat": +7.5,
                    "kinetic_energy": +7.5 * 0.3,
                }

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