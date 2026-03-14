"""Active experiment generation and evaluation for hypothesis testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
import random

from .concepts import ConceptTypeSystem
from .hypothesis import Hypothesis


@dataclass
class Experiment:
    """Synthetic experiment generated to test one hypothesis."""

    name: str
    rule: str
    facts: List[Tuple[str, str, str]]


@dataclass
class ExperimentOutcome:
    """Outcome summary for an executed experiment."""

    rule: str
    supported: int
    contradicted: int


class Experimenter:
    """Builds hypothesis-driven tests and evaluates them with noise + dynamics."""

    def __init__(self, seed: int | None = None) -> None:
        self.random = random.Random(seed)
        self.success_probability_by_relation = {
            "hunts": 0.8,
            "eats": 0.9,
            "orbits": 0.97,
            "reacts_with": 0.7,
            "part_of": 0.95,
            "located_in": 0.92,
        }

    def should_schedule(self, hypothesis: Hypothesis, prediction_error: float) -> bool:
        return hypothesis.confidence < 0.6 or hypothesis.contradicting_evidence > hypothesis.supporting_evidence // 2 or prediction_error > 0.4

    def generate(self, hypothesis: Hypothesis, entities: Iterable[str], type_system: ConceptTypeSystem, n: int = 3) -> Experiment:
        parts = hypothesis.rule.split()
        if len(parts) < 3:
            return Experiment(name="invalid_rule_test", rule=hypothesis.rule, facts=[])

        subject_type, relation, object_type = parts[0], parts[1], parts[2]
        entity_list = sorted(set(entities))
        subj_candidates = [e for e in entity_list if type_system.get_type(e) == subject_type] or self._fallback_entities_for_type(subject_type)
        obj_candidates = [e for e in entity_list if type_system.get_type(e) == object_type] or self._fallback_entities_for_type(object_type)

        facts: List[Tuple[str, str, str]] = []
        for _ in range(n):
            s = self.random.choice(subj_candidates)
            o = self.random.choice([x for x in obj_candidates if x != s] or obj_candidates)
            facts.append((s, relation, o))

        return Experiment(name=f"{relation}_test", rule=hypothesis.rule, facts=facts)

    def evaluate(self, experiment: Experiment, observed: Sequence[Tuple[str, str, str]], type_system: ConceptTypeSystem) -> ExperimentOutcome:
        observed_set = set(observed)
        supported = 0
        contradicted = 0
        for s, r, o in experiment.facts:
            if not type_system.is_relation_valid(s, r, o):
                contradicted += 1
                continue

            probability = self.success_probability_by_relation.get(r, 0.75)
            stochastic_success = self.random.random() <= probability
            observed_success = (s, r, o) in observed_set
            if stochastic_success and (observed_success or self.random.random() < 0.5):
                supported += 1
            else:
                contradicted += 1
        return ExperimentOutcome(rule=experiment.rule, supported=supported, contradicted=contradicted)

    def apply_environment_dynamics(self, domain: str, state: Dict[str, float], action: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Apply action + probabilistic domain dynamics to produce next state."""
        new_state = dict(state)
        before = dict(state)

        if domain == "ecosystem":
            if action == "remove_predator":
                new_state["wolves"] = max(0, new_state.get("wolves", 0) - 1)
            elif action == "add_predator":
                new_state["wolves"] = new_state.get("wolves", 0) + 1
            elif action == "introduce_species":
                new_state["deer"] = new_state.get("deer", 0) + 2
            elif action == "remove_species":
                new_state["deer"] = max(0, new_state.get("deer", 0) - 2)

            if self.random.random() < 0.8 and new_state.get("wolves", 0) > 0:
                new_state["deer"] = max(0, new_state.get("deer", 0) - 1)
            if new_state.get("deer", 0) < 10:
                new_state["grass"] = new_state.get("grass", 0) + 5
            else:
                new_state["grass"] = max(0, new_state.get("grass", 0) - self.random.randint(1, 3))

        elif domain == "chemistry":
            if action == "increase_temperature":
                new_state["temperature"] = new_state.get("temperature", 25) + self.random.randint(2, 8)
            elif action == "add_chemical":
                new_state["reactants"] = new_state.get("reactants", 1) + 1
            if self.random.random() < 0.65:
                new_state["reaction_energy"] = new_state.get("reaction_energy", 0) + (new_state.get("temperature", 25) / 20.0)

        elif domain == "astronomy":
            if action == "introduce_species":
                new_state["asteroids"] = new_state.get("asteroids", 200) + self.random.randint(1, 6)
            elif action == "remove_species":
                new_state["asteroids"] = max(0, new_state.get("asteroids", 200) - self.random.randint(1, 4))
            new_state["solar_energy"] = max(0, new_state.get("solar_energy", 1000) + self.random.randint(-25, 25))

        deltas = {k: new_state.get(k, 0) - before.get(k, 0) for k in set(new_state) | set(before)}
        return new_state, deltas

    @staticmethod
    def _fallback_entities_for_type(entity_type: str) -> List[str]:
        defaults = {
            "predator": ["lion", "wolf", "hawk", "lynx"],
            "prey": ["deer", "rabbit", "zebra", "mouse"],
            "animal": ["deer", "rabbit", "cat", "wolf"],
            "plant": ["grass", "fern", "tree", "bush"],
            "planet": ["earth", "mars", "venus"],
            "moon": ["luna", "titan", "europa"],
            "star": ["sun", "sirius", "vega"],
            "reactant": ["acid", "base", "salt"],
            "molecule": ["water", "methane", "glucose"],
            "atom": ["oxygen", "carbon", "hydrogen"],
            "unknown": ["entity_a", "entity_b", "entity_c"],
        }
        return defaults.get(entity_type, ["entity_a", "entity_b", "entity_c"])
