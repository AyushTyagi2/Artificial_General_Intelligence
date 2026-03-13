"""Active experiment generation and evaluation for hypothesis testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
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
    """Builds hypothesis-driven synthetic tests and evaluates them."""

    def __init__(self, seed: int | None = None) -> None:
        self.random = random.Random(seed)

    def should_schedule(self, hypothesis: Hypothesis, prediction_error: float) -> bool:
        """Schedule experiments for uncertain/conflicted/high-error hypotheses."""
        return hypothesis.confidence < 0.6 or hypothesis.contradicting_evidence > hypothesis.supporting_evidence // 2 or prediction_error > 0.4

    def generate(self, hypothesis: Hypothesis, entities: Iterable[str], type_system: ConceptTypeSystem, n: int = 3) -> Experiment:
        """Generate synthetic typed facts to test a hypothesis rule."""
        parts = hypothesis.rule.split()
        subject_type, relation, object_type = parts[0], parts[1], parts[2]

        entity_list = sorted(set(entities))
        subj_candidates = [e for e in entity_list if type_system.get_type(e) == subject_type]
        obj_candidates = [e for e in entity_list if type_system.get_type(e) == object_type]

        if not subj_candidates:
            subj_candidates = self._fallback_entities_for_type(subject_type)
        if not obj_candidates:
            obj_candidates = self._fallback_entities_for_type(object_type)

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
            if (s, r, o) in observed_set:
                supported += 1
            else:
                contradicted += 1
        return ExperimentOutcome(rule=experiment.rule, supported=supported, contradicted=contradicted)

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
