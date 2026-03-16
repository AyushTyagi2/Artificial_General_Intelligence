"""Active experiment generation and evaluation — v4.1 (hotfix).

Hotfix changes vs v4
--------------------
1. FOUR NEW DOMAIN SIMULATORS added:
   - neuroscience: stress→cortisol→hippocampus→memory, dopamine→learning
   - climate:      co2→temperature_anomaly→glacier_melt→sea_level
   - economics:    interest_rate→investment→gdp_growth, inflation→purchasing_power
   - materials:    stress→strain→crack_growth, heat_treatment→hardness

   These domains existed in the generator (domain_states + actions_by_domain)
   but apply_environment_dynamics() had no branch for them, causing every tick
   on those domains to return action=none because _SIMULATION_DOMAINS didn't
   include them and the event loop's world step was never called.

2. _SIMULATION_DOMAINS in event_loop.py is extended to include all 10 domains.

3. DOMAIN_INITIAL_STATES extended with the 4 new domains.

4. _ACTION_INTERVENED_VAR extended for all new actions.

All v4 public API preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import copy
import random

from .concepts import ConceptTypeSystem
from .hypothesis import Hypothesis


@dataclass
class FixVariableAction:
    """Clamp a state variable to a fixed value during apply_environment_dynamics.

    Pass a list of these as ``fix_actions`` to hold mediator variables constant
    during mediator-blocking experiments.  The clamping happens both before and
    after dynamics so no indirect propagation can shift the variable.
    """
    variable: str
    value:    float


@dataclass
class Experiment:
    name:  str
    rule:  str
    facts: List[Tuple[str, str, str]]


@dataclass
class ExperimentOutcome:
    rule:         str
    supported:    int
    contradicted: int


@dataclass
class PairedResult:
    control_state:   Dict[str, float]
    treatment_state: Dict[str, float]
    causal_deltas:   Dict[str, float]
    intervened_vars: List[str]
    action:          str
    domain:          str


# ---------------------------------------------------------------------------
# Domain initial states
# ---------------------------------------------------------------------------

DOMAIN_INITIAL_STATES: Dict[str, Dict[str, float]] = {
    "ecosystem": {
        "wolves": 5.0, "deer": 20.0, "grass": 100.0, "season_factor": 1.0,
    },
    "chemistry": {
        "temperature": 25.0, "reactants": 2.0, "reaction_rate": 0.0,
        "reaction_energy": 0.0, "catalyst": 0.0, "product_concentration": 0.0,
        "pH": 7.0, "activation_energy": 50.0,
    },
    "astronomy": {
        "asteroids": 200.0, "solar_energy": 1000.0, "collision_risk": 0.0,
        "radiation_pressure": 0.0, "asteroid_drift": 0.0,
    },
    "technology": {
        "robots": 4.0, "battery_charge": 80.0, "robot_activity": 0.0,
        "sensor_coverage": 0.0, "data_quality": 0.0, "maintenance_load": 2.0,
        "network_latency": 5.0, "sensor_threshold": 5.0,
    },
    "biology": {
        "cells": 100.0, "energy": 50.0, "proteins": 20.0, "pathogens": 0.0,
        "immune_response": 0.0, "toxin_level": 0.0, "antibodies": 0.0,
        "cell_cycle_rate": 0.5,
    },
    "physics": {
        "force": 10.0, "mass": 5.0, "acceleration": 2.0, "heat": 25.0,
        "kinetic_energy": 50.0, "velocity": 0.0, "friction": 2.0, "momentum": 0.0,
    },
    "neuroscience": {
        "stress_level": 5.0, "cortisol": 3.0, "dopamine_level": 5.0,
        "serotonin_level": 5.0, "neural_activity": 6.0, "synaptic_strength": 4.0,
        "memory_consolidation": 3.0, "learning_rate": 2.0,
    },
    "climate": {
        "co2_level": 415.0, "temperature_anomaly": 1.2, "glacier_melt": 3.0,
        "sea_level_rise": 0.3, "precipitation": 60.0, "vegetation_cover": 40.0,
        "albedo": 0.3, "ocean_heat": 80.0,
    },
    "economics": {
        "interest_rate": 3.0, "inflation": 2.5, "gdp_growth": 2.0,
        "unemployment": 5.0, "investment": 20.0, "consumption": 60.0,
        "productivity": 3.0, "debt_level": 80.0,
    },
    "materials": {
        "temperature": 25.0, "stress_level": 5.0, "strain": 0.1,
        "hardness": 60.0, "yield_strength": 250.0, "conductivity": 6.0,
        "crack_growth": 0.0, "porosity": 2.0,
    },
}

# Variables directly intervened on by each action
_ACTION_INTERVENED_VAR: Dict[str, str] = {
    # ecosystem
    "add_predator": "wolves", "remove_predator": "wolves",
    "introduce_species": "deer", "remove_species": "deer",
    "change_season": "season_factor",
    # chemistry
    "increase_temperature": "temperature", "add_chemical": "reactants",
    "add_catalyst": "catalyst", "adjust_pH": "pH",
    "remove_product": "product_concentration",
    # astronomy
    # technology
    "add_robot": "robots", "remove_robot": "robots",
    "upgrade_sensor": "sensor_threshold",
    "increase_maintenance": "maintenance_load",
    # biology
    "add_pathogen": "pathogens", "boost_energy": "energy",
    "add_cells": "cells", "neutralise_toxin": "toxin_level",
    "add_antibody": "antibodies",
    # physics
    "increase_force": "force", "add_heat": "heat",
    "add_friction": "friction", "apply_impulse": "momentum",
    "reduce_mass": "mass",
    # neuroscience
    "induce_stress": "stress_level", "reduce_stress": "stress_level",
    "boost_dopamine": "dopamine_level", "improve_sleep": "memory_consolidation",
    "stimulate_neurons": "neural_activity",
    # climate
    "emit_co2": "co2_level", "plant_forest": "vegetation_cover",
    "melt_glacier": "glacier_melt", "increase_albedo": "albedo",
    "warm_ocean": "ocean_heat",
    # economics
    "raise_interest_rate": "interest_rate", "lower_interest_rate": "interest_rate",
    "increase_spending": "consumption", "boost_productivity": "productivity",
    "add_debt": "debt_level",
    # materials
    "apply_stress": "stress_level", "heat_treat": "temperature",
    "add_porosity": "porosity", "quench": "temperature", "anneal": "temperature",
}

# All domains with simulation support
ALL_SIMULATION_DOMAINS = frozenset(DOMAIN_INITIAL_STATES.keys())


class Experimenter:
    """Builds hypothesis-driven tests and evaluates them — v4.1."""

    def __init__(self, seed: int | None = None) -> None:
        self.random = random.Random(seed)
        self.success_probability_by_relation = {
            "hunts": 0.8, "eats": 0.9, "orbits": 0.97,
            "reacts_with": 0.7, "part_of": 0.95, "located_in": 0.92,
        }

    # ── Paired experiment ──────────────────────────────────────────────────

    @staticmethod
    def capture_state(state: Dict[str, float]) -> Dict[str, float]:
        return copy.deepcopy(state)

    def run_paired(
        self, domain: str, base_state: Dict[str, float], action: str
    ) -> PairedResult:
        control_in  = copy.deepcopy(base_state)
        treatment_in = copy.deepcopy(base_state)

        control_out,  _ = self.apply_environment_dynamics(domain, control_in,  "none")
        treatment_out, _ = self.apply_environment_dynamics(domain, treatment_in, action)

        all_vars = set(control_out) | set(treatment_out)
        causal_deltas = {
            var: treatment_out.get(var, 0.0) - control_out.get(var, 0.0)
            for var in all_vars
        }
        intervened_var = _ACTION_INTERVENED_VAR.get(action, "")
        return PairedResult(
            control_state=control_out,
            treatment_state=treatment_out,
            causal_deltas=causal_deltas,
            intervened_vars=[intervened_var] if intervened_var else [],
            action=action,
            domain=domain,
        )

    def run_averaged(
        self, domain: str, state: Dict[str, float], action: str, repeat_n: int = 3
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        if repeat_n <= 1:
            return self.apply_environment_dynamics(domain, copy.deepcopy(state), action)
        all_finals, all_deltas = [], []
        for _ in range(repeat_n):
            f, d = self.apply_environment_dynamics(domain, copy.deepcopy(state), action)
            all_finals.append(f)
            all_deltas.append(d)
        all_vars = set()
        for f in all_finals:
            all_vars.update(f.keys())
        mean_final  = {v: sum(f.get(v, 0.0) for f in all_finals) / repeat_n for v in all_vars}
        mean_deltas = {v: sum(d.get(v, 0.0) for d in all_deltas) / repeat_n for v in all_vars}
        return mean_final, mean_deltas

    @staticmethod
    def extract_causal_delta(
        paired_result: PairedResult, min_relative_change: float = 0.01
    ) -> Dict[str, float]:
        clean: Dict[str, float] = {}
        for var, delta in paired_result.causal_deltas.items():
            if var in paired_result.intervened_vars:
                continue
            if abs(delta) < 1e-9:
                continue
            control_val = paired_result.control_state.get(var, 0.0)
            if abs(control_val) > 1e-9 and abs(delta) / abs(control_val) < min_relative_change:
                continue
            clean[var] = delta
        return clean

    @staticmethod
    def get_initial_state(domain: str) -> Dict[str, float]:
        return copy.deepcopy(DOMAIN_INITIAL_STATES.get(domain, {}))

    # ── v3 API ─────────────────────────────────────────────────────────────

    def should_schedule(self, hypothesis: Hypothesis, prediction_error: float) -> bool:
        if hypothesis.confidence < 0.6:
            return True
        if hypothesis.contradicting_evidence > hypothesis.supporting_evidence // 2:
            return True
        if prediction_error > 0.3:
            return True
        return self.random.random() < 0.20

    @staticmethod
    def _parse_causal_rule(rule: str):
        import re as _re
        r = rule.lower().strip()
        m = _re.match(
            r'if (\w+) (?:increases|decreases|changes) then (\w+) (?:will|may) (increase|decrease|change)',
            r,
        )
        if m:
            return m.group(1), m.group(3), m.group(2)
        m2 = _re.match(r'(\w+) affects (\w+)', r)
        if m2:
            return m2.group(1), 'change', m2.group(2)
        return None

    def generate(
        self, hypothesis: Hypothesis, entities: Iterable[str],
        type_system: ConceptTypeSystem, n: int = 3,
    ) -> Experiment:
        parsed = self._parse_causal_rule(hypothesis.rule)
        if parsed is not None:
            cause, direction, effect = parsed
            relation = f"causes_{direction}_in"
            facts = [(cause, relation, effect)] * n
            return Experiment(name=f"causal_{cause}_{effect}_test", rule=hypothesis.rule, facts=facts)
        parts = hypothesis.rule.split()
        if len(parts) < 3:
            return Experiment(name="invalid_rule_test", rule=hypothesis.rule, facts=[])
        subject_type, relation, object_type = parts[0], parts[1], parts[2]
        entity_list     = sorted(set(entities))
        subj_candidates = ([e for e in entity_list if type_system.get_type(e) == subject_type]
                           or self._fallback_entities_for_type(subject_type))
        obj_candidates  = ([e for e in entity_list if type_system.get_type(e) == object_type]
                           or self._fallback_entities_for_type(object_type))
        facts: List[Tuple[str, str, str]] = []
        for _ in range(n):
            s = self.random.choice(subj_candidates)
            o = self.random.choice([x for x in obj_candidates if x != s] or obj_candidates)
            facts.append((s, relation, o))
        return Experiment(name=f"{relation}_test", rule=hypothesis.rule, facts=facts)

    def evaluate(
        self, experiment: Experiment, observed: Sequence[Tuple[str, str, str]],
        type_system: ConceptTypeSystem, state_before: Optional[Dict[str, float]] = None,
        state_after: Optional[Dict[str, float]] = None,
    ) -> ExperimentOutcome:
        observed_set = set(observed)
        supported, contradicted = 0, 0
        for s, r, o in experiment.facts:
            if r.startswith("causes_"):
                if state_before is not None and state_after is not None:
                    delta = state_after.get(o, 0.0) - state_before.get(o, 0.0)
                    expected_increase = "increase" in r
                    if delta == 0:
                        contradicted += 1
                    elif (expected_increase and delta > 0) or (not expected_increase and delta < 0):
                        supported += 1
                    else:
                        contradicted += 1
                else:
                    causal_observed = any(
                        ts == s and ("affects" in tr or "causes" in tr) and to == o
                        for ts, tr, to in observed
                    )
                    if causal_observed or self.random.random() < 0.55:
                        supported += 1
                    else:
                        contradicted += 1
                continue
            if not type_system.is_relation_valid(s, r, o):
                contradicted += 1
                continue
            probability = self.success_probability_by_relation.get(r, 0.75)
            stochastic  = self.random.random() <= probability
            observed_ok = (s, r, o) in observed_set
            if stochastic and (observed_ok or self.random.random() < 0.5):
                supported += 1
            else:
                contradicted += 1
        return ExperimentOutcome(rule=experiment.rule, supported=supported, contradicted=contradicted)

    # ── Domain simulators ──────────────────────────────────────────────────

    def apply_environment_dynamics(
        self,
        domain:      str,
        state:       Dict[str, float],
        action:      str,
        fix_actions: Optional[List["FixVariableAction"]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Apply action + causal dynamics + noise. action='none' for control arm.

        Parameters
        ----------
        fix_actions : list[FixVariableAction], optional
            Variables to hold constant throughout the step (pre- and post-clamp).
            Used by MediatorBlockingPlanner to run mediator-blocking experiments.
        """
        new_state = dict(state)
        before    = dict(state)

        # Pre-clamp: fix mediator variables before dynamics compute deltas
        if fix_actions:
            for fa in fix_actions:
                new_state[fa.variable] = fa.value

        def _noise(scale: float = 1.0) -> float:
            return max(-scale, min(scale, self.random.gauss(0, scale * 0.4)))

        # ── Ecosystem ─────────────────────────────────────────────────────
        if domain == "ecosystem":
            wolves        = new_state.get("wolves",        5.0)
            deer          = new_state.get("deer",         20.0)
            grass         = new_state.get("grass",       100.0)
            season_factor = new_state.get("season_factor", 1.0)
            if action == "add_predator":      wolves = wolves + 1
            elif action == "remove_predator": wolves = max(0.0, wolves - 1)
            elif action == "introduce_species": deer = deer + 3
            elif action == "remove_species":    deer = max(0.0, deer - 3)
            elif action == "change_season":     season_factor = self.random.choice([0.4, 0.6, 0.8, 1.0, 1.2])
            predation      = wolves * 1.5
            deer           = max(0.0, deer - predation + _noise(0.5))
            grazing        = deer * 0.8
            grass          = max(0.0, grass - grazing + _noise(1.0))
            grass_carrying = grass / 20.0
            deer_growth    = deer * 0.15 * min(1.0, grass_carrying)
            deer           = deer + deer_growth + _noise(0.3)
            grass          = grass + 2.0 * season_factor + _noise(0.5)
            new_state.update({
                "wolves": round(max(0.0, wolves), 2),
                "deer":   round(max(0.0, deer),   2),
                "grass":  round(max(0.0, grass),  2),
                "season_factor": round(max(0.1, min(2.0, season_factor)), 2),
            })

        # ── Chemistry ─────────────────────────────────────────────────────
        elif domain == "chemistry":
            temperature           = new_state.get("temperature",           25.0)
            reactants             = new_state.get("reactants",              2.0)
            reaction_rate         = new_state.get("reaction_rate",          0.0)
            reaction_energy       = new_state.get("reaction_energy",        0.0)
            catalyst              = new_state.get("catalyst",               0.0)
            product_concentration = new_state.get("product_concentration",  0.0)
            pH                    = new_state.get("pH",                     7.0)
            activation_energy     = new_state.get("activation_energy",     50.0)
            if action == "increase_temperature": temperature = temperature + self.random.randint(5, 10)
            elif action == "add_chemical":       reactants  = reactants + 1
            elif action == "add_catalyst":       catalyst   = catalyst + self.random.uniform(0.5, 1.5)
            elif action == "adjust_pH":          pH = max(0.0, min(14.0, pH + self.random.choice([-1.0, -0.5, 0.5, 1.0])))
            elif action == "remove_product":     product_concentration = max(0.0, product_concentration - 2.0)
            activation_energy     = max(5.0, 50.0 - temperature * 0.5 + _noise(0.5))
            ph_factor             = max(0.05, 1.0 - abs(pH - 7.0) / 7.0)
            catalyst_boost        = 1.0 + catalyst * 0.4
            reaction_rate         = max(0.0, (temperature / max(1.0, activation_energy)) * reactants * catalyst_boost * ph_factor + _noise(0.1))
            product_concentration = min(50.0, product_concentration + reaction_rate * 0.5)
            reaction_rate         = reaction_rate * max(0.1, 1.0 - product_concentration / 40.0)
            reaction_energy       = min(5000.0, reaction_energy + reaction_rate * 2.0 + _noise(0.2))
            temperature           = max(20.0, temperature - 1.0 + _noise(0.5))
            new_state.update({
                "temperature": round(temperature, 2), "reactants": round(reactants, 2),
                "reaction_rate": round(reaction_rate, 2), "reaction_energy": round(reaction_energy, 2),
                "catalyst": round(catalyst, 2), "product_concentration": round(product_concentration, 2),
                "pH": round(pH, 2), "activation_energy": round(activation_energy, 2),
            })

        # ── Astronomy ─────────────────────────────────────────────────────
        elif domain == "astronomy":
            asteroids          = new_state.get("asteroids",          200.0)
            solar_energy       = new_state.get("solar_energy",       1000.0)
            collision_risk     = new_state.get("collision_risk",        0.0)
            radiation_pressure = new_state.get("radiation_pressure",    0.0)
            asteroid_drift     = new_state.get("asteroid_drift",        0.0)
            if action == "introduce_species": asteroids = asteroids + self.random.randint(3, 7)
            elif action == "remove_species":  asteroids = max(0.0, asteroids - self.random.randint(3, 7))
            collision_risk     = max(0.0, asteroids * 0.05 + _noise(0.5))
            solar_energy       = max(800.0, min(1200.0, solar_energy + _noise(5.0)))
            radiation_pressure = max(0.0, solar_energy * 0.002 + _noise(0.05))
            asteroid_drift     = max(0.0, radiation_pressure * 1.5 + _noise(0.1))
            new_state.update({
                "asteroids": round(max(0.0, asteroids), 2), "solar_energy": round(solar_energy, 2),
                "collision_risk": round(collision_risk, 2), "radiation_pressure": round(radiation_pressure, 2),
                "asteroid_drift": round(asteroid_drift, 2),
            })

        # ── Technology ────────────────────────────────────────────────────
        elif domain == "technology":
            robots           = new_state.get("robots",           4.0)
            battery_charge   = new_state.get("battery_charge",  80.0)
            robot_activity   = new_state.get("robot_activity",   0.0)
            sensor_coverage  = new_state.get("sensor_coverage",  0.0)
            data_quality     = new_state.get("data_quality",     0.0)
            maintenance_load = new_state.get("maintenance_load", 2.0)
            network_latency  = new_state.get("network_latency",  5.0)
            sensor_threshold = new_state.get("sensor_threshold", 5.0)
            if action == "add_robot":             robots           = robots + 1
            elif action == "remove_robot":        robots           = max(0.0, robots - 1)
            elif action == "upgrade_sensor":      sensor_threshold = sensor_threshold + 2.0
            elif action == "increase_maintenance": maintenance_load = maintenance_load + 1.0
            maintenance_load    = min(20.0, robots * 0.5 + _noise(0.2))
            maintenance_penalty = max(0.1, 1.0 - maintenance_load / 20.0)
            robot_activity      = max(0.0, robots * (battery_charge / 100.0) * maintenance_penalty + _noise(0.3))
            sensor_coverage     = max(0.0, robot_activity * 2.5 + _noise(0.5))
            data_quality = max(0.0, (min(100.0, sensor_coverage * 0.8 - network_latency * 0.5 + _noise(0.3))
                               if sensor_coverage >= sensor_threshold
                               else min(30.0, sensor_coverage * 0.3 + _noise(0.5))))
            network_latency = max(1.0, min(20.0, sensor_coverage * 0.3 + _noise(0.5)))
            battery_charge  = max(10.0, battery_charge - robot_activity * 0.5 + _noise(0.2))
            new_state.update({
                "robots": round(max(0.0, robots), 2), "battery_charge": round(battery_charge, 2),
                "robot_activity": round(robot_activity, 2), "sensor_coverage": round(sensor_coverage, 2),
                "data_quality": round(data_quality, 2), "maintenance_load": round(maintenance_load, 2),
                "network_latency": round(network_latency, 2), "sensor_threshold": round(sensor_threshold, 2),
            })

        # ── Biology ───────────────────────────────────────────────────────
        elif domain == "biology":
            cells           = new_state.get("cells",            100.0)
            energy          = new_state.get("energy",            50.0)
            proteins        = new_state.get("proteins",          20.0)
            pathogens       = new_state.get("pathogens",          0.0)
            immune_response = new_state.get("immune_response",    0.0)
            toxin_level     = new_state.get("toxin_level",        0.0)
            antibodies      = new_state.get("antibodies",         0.0)
            cell_cycle_rate = new_state.get("cell_cycle_rate",    0.5)
            if action == "add_pathogen":   pathogens   = pathogens   + self.random.randint(2, 5)
            elif action == "boost_energy": energy      = energy      + self.random.randint(10, 20)
            elif action == "add_cells":    cells       = cells       + self.random.randint(5, 15)
            elif action == "neutralise_toxin": toxin_level = max(0.0, toxin_level - 3.0)
            elif action == "add_antibody": antibodies  = antibodies  + self.random.uniform(2.0, 5.0)
            toxin_level      = max(0.0, min(50.0, toxin_level + max(0.0, pathogens * 0.4 + _noise(0.1))))
            immune_response  = max(0.0, min(100.0, pathogens * 2.5 + _noise(0.5)))
            antibodies       = min(50.0, antibodies + max(0.0, immune_response * 0.3 + _noise(0.1)))
            pathogens        = max(0.0, pathogens - max(0.0, antibodies * 0.4 + _noise(0.1)))
            cells            = max(1.0, cells - max(0.0, toxin_level * 0.5 + pathogens * 0.3 + _noise(0.3)))
            cell_cycle_rate  = max(0.0, min(2.0, (energy / 50.0) * (proteins / 20.0) + _noise(0.05)))
            proteins         = max(0.0, (energy / 10.0) * 2.0 + _noise(0.3))
            cells            = min(1000.0, cells + max(0.0, proteins * 0.5 * cell_cycle_rate + _noise(0.2)))
            energy           = max(5.0,  energy - cells * 0.05 + _noise(0.5))
            pathogens        = max(0.0,  pathogens - immune_response * 0.1 + _noise(0.1))
            toxin_level      *= 0.9
            antibodies       *= 0.95
            new_state.update({
                "cells": round(max(0.0, cells), 2), "energy": round(max(0.0, energy), 2),
                "proteins": round(max(0.0, proteins), 2), "pathogens": round(max(0.0, pathogens), 2),
                "immune_response": round(max(0.0, immune_response), 2),
                "toxin_level": round(max(0.0, toxin_level), 2),
                "antibodies": round(max(0.0, antibodies), 2),
                "cell_cycle_rate": round(max(0.0, cell_cycle_rate), 2),
            })

        # ── Physics ───────────────────────────────────────────────────────
        elif domain == "physics":
            force          = new_state.get("force",          10.0)
            mass           = new_state.get("mass",            5.0)
            acceleration   = new_state.get("acceleration",    2.0)
            heat           = new_state.get("heat",           25.0)
            kinetic_energy = new_state.get("kinetic_energy", 50.0)
            velocity       = new_state.get("velocity",        0.0)
            friction       = new_state.get("friction",        2.0)
            momentum       = new_state.get("momentum",        0.0)
            if action == "increase_force": force    = force    + self.random.randint(2, 6)
            elif action == "add_heat":     heat     = heat     + self.random.randint(5, 10)
            elif action == "add_friction": friction = friction + self.random.uniform(0.5, 2.0)
            elif action == "apply_impulse": momentum = momentum + self.random.uniform(5.0, 15.0)
            elif action == "reduce_mass":  mass     = max(0.5, mass - self.random.uniform(0.5, 1.5))
            net_force      = max(0.0, force - friction)
            acceleration   = max(0.0, net_force / max(0.1, mass) + _noise(0.2))
            velocity       = min(200.0, max(0.0, velocity + acceleration * 0.1 - friction * 0.05 + _noise(0.1)))
            momentum       = max(0.0, mass * velocity + _noise(0.5))
            kinetic_energy = max(0.0, 0.5 * mass * velocity ** 2 + heat * 0.3 + _noise(0.5))
            force    = max(0.0, force    * 0.95 + _noise(0.1))
            heat     = max(15.0, heat   - 1.0  + _noise(0.3))
            velocity = max(0.0, velocity * 0.98)
            new_state.update({
                "force": round(force, 2), "mass": round(mass, 2),
                "acceleration": round(acceleration, 2), "heat": round(heat, 2),
                "kinetic_energy": round(kinetic_energy, 2), "velocity": round(velocity, 2),
                "friction": round(friction, 2), "momentum": round(momentum, 2),
            })

        # ── Neuroscience ──────────────────────────────────────────────────
        elif domain == "neuroscience":
            stress_level         = new_state.get("stress_level",         5.0)
            cortisol             = new_state.get("cortisol",             3.0)
            dopamine_level       = new_state.get("dopamine_level",       5.0)
            serotonin_level      = new_state.get("serotonin_level",      5.0)
            neural_activity      = new_state.get("neural_activity",      6.0)
            synaptic_strength    = new_state.get("synaptic_strength",    4.0)
            memory_consolidation = new_state.get("memory_consolidation", 3.0)
            learning_rate        = new_state.get("learning_rate",        2.0)

            if action == "induce_stress":    stress_level    = min(10.0, stress_level + self.random.uniform(1.0, 2.5))
            elif action == "reduce_stress":  stress_level    = max(0.0,  stress_level - self.random.uniform(1.0, 2.0))
            elif action == "boost_dopamine": dopamine_level  = min(10.0, dopamine_level + self.random.uniform(1.0, 3.0))
            elif action == "improve_sleep":  memory_consolidation = min(10.0, memory_consolidation + self.random.uniform(1.0, 2.0))
            elif action == "stimulate_neurons": neural_activity = min(10.0, neural_activity + self.random.uniform(1.0, 3.0))

            # stress → cortisol ↑ → hippocampus activity ↓ → memory ↓
            cortisol        = max(0.0, min(10.0, stress_level * 0.6 + _noise(0.3)))
            # high cortisol suppresses memory consolidation
            memory_consolidation = max(0.0, min(10.0,
                memory_consolidation - cortisol * 0.2 + serotonin_level * 0.1 + _noise(0.2)))
            # dopamine → learning_rate ↑
            learning_rate   = max(0.0, min(5.0, dopamine_level * 0.3 + _noise(0.1)))
            # neural_activity → synaptic_strength (Hebbian plasticity)
            synaptic_strength = max(0.0, min(10.0,
                synaptic_strength * 0.95 + neural_activity * 0.08 + _noise(0.15)))
            # serotonin regulates mood; partially offsets stress
            serotonin_level = max(0.0, min(10.0,
                serotonin_level - stress_level * 0.05 + dopamine_level * 0.03 + _noise(0.2)))
            # natural decay / homeostasis
            stress_level    = max(0.5, stress_level * 0.97 + _noise(0.1))
            dopamine_level  = max(1.0, min(10.0, dopamine_level * 0.98 + _noise(0.15)))
            neural_activity = max(1.0, min(10.0, neural_activity * 0.97 + _noise(0.2)))

            new_state.update({
                "stress_level": round(stress_level, 2), "cortisol": round(cortisol, 2),
                "dopamine_level": round(dopamine_level, 2), "serotonin_level": round(serotonin_level, 2),
                "neural_activity": round(neural_activity, 2), "synaptic_strength": round(synaptic_strength, 2),
                "memory_consolidation": round(memory_consolidation, 2), "learning_rate": round(learning_rate, 2),
            })

        # ── Climate ───────────────────────────────────────────────────────
        elif domain == "climate":
            co2_level          = new_state.get("co2_level",          415.0)
            temperature_anomaly = new_state.get("temperature_anomaly", 1.2)
            glacier_melt       = new_state.get("glacier_melt",         3.0)
            sea_level_rise     = new_state.get("sea_level_rise",       0.3)
            precipitation      = new_state.get("precipitation",       60.0)
            vegetation_cover   = new_state.get("vegetation_cover",    40.0)
            albedo             = new_state.get("albedo",               0.3)
            ocean_heat         = new_state.get("ocean_heat",          80.0)

            if action == "emit_co2":        co2_level       = min(600.0, co2_level + self.random.uniform(2.0, 8.0))
            elif action == "plant_forest":  vegetation_cover = min(80.0, vegetation_cover + self.random.uniform(2.0, 5.0))
            elif action == "melt_glacier":  glacier_melt    = min(20.0, glacier_melt + self.random.uniform(0.5, 2.0))
            elif action == "increase_albedo": albedo        = min(0.8, albedo + self.random.uniform(0.02, 0.08))
            elif action == "warm_ocean":    ocean_heat      = min(200.0, ocean_heat + self.random.uniform(2.0, 6.0))

            # co2 → temperature_anomaly (greenhouse effect)
            co2_forcing = (co2_level - 280.0) / 280.0 * 3.0   # ~3°C per doubling
            albedo_cooling = (albedo - 0.3) * 5.0
            temperature_anomaly = max(-1.0, min(5.0,
                co2_forcing - albedo_cooling + ocean_heat * 0.005 + _noise(0.05)))
            # temperature → glacier_melt
            glacier_melt = max(0.0, min(20.0,
                glacier_melt + temperature_anomaly * 0.3 - albedo * 0.5 + _noise(0.1)))
            # glacier_melt → sea_level_rise
            sea_level_rise = max(0.0, sea_level_rise + glacier_melt * 0.01 + _noise(0.005))
            # vegetation absorbs co2
            co2_absorption = vegetation_cover * 0.02
            co2_level = max(280.0, co2_level - co2_absorption + _noise(0.5))
            # ocean_heat → precipitation
            precipitation = max(10.0, min(120.0, ocean_heat * 0.6 + _noise(2.0)))
            # precipitation → vegetation
            vegetation_cover = max(0.0, min(80.0,
                vegetation_cover + precipitation * 0.01 - temperature_anomaly * 0.3 + _noise(0.5)))
            ocean_heat = max(40.0, min(200.0, ocean_heat + temperature_anomaly * 0.2 + _noise(1.0)))

            new_state.update({
                "co2_level": round(co2_level, 2), "temperature_anomaly": round(temperature_anomaly, 3),
                "glacier_melt": round(glacier_melt, 2), "sea_level_rise": round(sea_level_rise, 4),
                "precipitation": round(precipitation, 2), "vegetation_cover": round(vegetation_cover, 2),
                "albedo": round(albedo, 3), "ocean_heat": round(ocean_heat, 2),
            })

        # ── Economics ─────────────────────────────────────────────────────
        elif domain == "economics":
            interest_rate = new_state.get("interest_rate",  3.0)
            inflation     = new_state.get("inflation",      2.5)
            gdp_growth    = new_state.get("gdp_growth",     2.0)
            unemployment  = new_state.get("unemployment",   5.0)
            investment    = new_state.get("investment",    20.0)
            consumption   = new_state.get("consumption",   60.0)
            productivity  = new_state.get("productivity",   3.0)
            debt_level    = new_state.get("debt_level",    80.0)

            if action == "raise_interest_rate":  interest_rate = min(15.0, interest_rate + self.random.uniform(0.25, 1.0))
            elif action == "lower_interest_rate": interest_rate = max(0.0,  interest_rate - self.random.uniform(0.25, 0.75))
            elif action == "increase_spending":  consumption   = min(100.0, consumption + self.random.uniform(2.0, 6.0))
            elif action == "boost_productivity": productivity  = min(10.0,  productivity + self.random.uniform(0.2, 0.8))
            elif action == "add_debt":           debt_level    = min(200.0, debt_level + self.random.uniform(3.0, 8.0))

            # interest_rate → investment ↓ (higher rates = lower investment)
            investment_effect = -interest_rate * 1.2 + productivity * 2.0
            investment  = max(0.0, min(100.0, investment + investment_effect * 0.1 + _noise(0.5)))
            # investment + consumption → gdp_growth
            gdp_growth  = max(-5.0, min(10.0, (investment * 0.15 + consumption * 0.05) - inflation * 0.3 + _noise(0.2)))
            # gdp_growth → unemployment ↓ (Okun's law)
            unemployment = max(1.0, min(20.0, unemployment - gdp_growth * 0.4 + _noise(0.2)))
            # inflation: wage-push from low unemployment, demand-pull from consumption
            inflation   = max(-1.0, min(15.0, inflation + (5.0 - unemployment) * 0.1 + consumption * 0.01 + _noise(0.1)))
            # debt pressure on investment
            debt_drag   = max(0.0, (debt_level - 60.0) / 200.0)
            investment  = max(0.0, investment * (1.0 - debt_drag * 0.05))
            # productivity drift
            productivity = max(0.5, min(10.0, productivity * 0.999 + _noise(0.05)))

            new_state.update({
                "interest_rate": round(interest_rate, 2), "inflation": round(inflation, 2),
                "gdp_growth": round(gdp_growth, 2), "unemployment": round(unemployment, 2),
                "investment": round(investment, 2), "consumption": round(consumption, 2),
                "productivity": round(productivity, 2), "debt_level": round(debt_level, 2),
            })

        # ── Materials ─────────────────────────────────────────────────────
        elif domain == "materials":
            temperature   = new_state.get("temperature",   25.0)
            stress_level  = new_state.get("stress_level",   5.0)
            strain        = new_state.get("strain",         0.1)
            hardness      = new_state.get("hardness",      60.0)
            yield_strength = new_state.get("yield_strength", 250.0)
            conductivity  = new_state.get("conductivity",   6.0)
            crack_growth  = new_state.get("crack_growth",   0.0)
            porosity      = new_state.get("porosity",       2.0)

            if action == "apply_stress":  stress_level = min(500.0, stress_level + self.random.uniform(5.0, 20.0))
            elif action == "heat_treat":  temperature  = min(1000.0, temperature + self.random.uniform(50.0, 150.0))
            elif action == "add_porosity": porosity    = min(30.0, porosity + self.random.uniform(0.5, 2.0))
            elif action == "quench":      temperature  = max(-20.0, temperature - self.random.uniform(100.0, 300.0))
            elif action == "anneal":      temperature  = max(20.0, temperature - self.random.uniform(20.0, 80.0))

            # stress → strain (Hooke's law region, then plastic)
            E = max(1.0, yield_strength / 0.002)   # Young's modulus proxy
            elastic_strain = stress_level / E
            plastic_strain = max(0.0, (stress_level - yield_strength) / E * 0.1) if stress_level > yield_strength else 0.0
            strain = max(0.0, elastic_strain + plastic_strain + _noise(0.001))

            # strain → crack_growth (Paris law proxy)
            crack_growth = max(0.0, min(10.0, crack_growth + strain * 0.5 + _noise(0.01)))

            # temperature → conductivity (metals: conductivity ↓ with temperature)
            conductivity = max(0.1, min(20.0, 6.0 - (temperature - 25.0) * 0.005 + _noise(0.1)))

            # heat treatment: high temp raises hardness (martensite), slow cooling lowers it (annealing)
            if action == "quench":
                hardness      = min(100.0, hardness + self.random.uniform(5.0, 15.0))
                yield_strength = min(500.0, yield_strength + self.random.uniform(10.0, 30.0))
            elif action == "anneal":
                hardness      = max(20.0, hardness - self.random.uniform(5.0, 12.0))
                yield_strength = max(100.0, yield_strength - self.random.uniform(10.0, 25.0))
            else:
                # Thermal softening at high temperature
                thermal_soften = max(0.0, (temperature - 300.0) / 1000.0)
                hardness       = max(10.0, hardness - thermal_soften * 5.0 + _noise(0.3))
                yield_strength = max(50.0, yield_strength - thermal_soften * 20.0 + _noise(1.0))

            # porosity reduces strength and conductivity
            porosity_penalty = porosity / 30.0
            yield_strength   = max(50.0, yield_strength * (1.0 - porosity_penalty))
            conductivity      = max(0.1, conductivity  * (1.0 - porosity_penalty * 0.5))

            # temperature drift toward ambient
            temperature = max(20.0, temperature - (temperature - 25.0) * 0.05 + _noise(0.5))

            new_state.update({
                "temperature": round(temperature, 2), "stress_level": round(stress_level, 2),
                "strain": round(strain, 4), "hardness": round(hardness, 2),
                "yield_strength": round(yield_strength, 2), "conductivity": round(conductivity, 3),
                "crack_growth": round(crack_growth, 3), "porosity": round(porosity, 3),
            })

        # ── Fallback for unknown domains ───────────────────────────────────
        else:
            # Just tick forward with minor noise on whatever variables exist
            for key in list(new_state.keys()):
                val = new_state[key]
                if isinstance(val, float):
                    new_state[key] = round(val * (1.0 + self.random.gauss(0, 0.01)), 3)

        # Post-clamp: override any indirect drift through fixed variables
        if fix_actions:
            for fa in fix_actions:
                new_state[fa.variable] = fa.value

        deltas = {
            k: new_state.get(k, 0) - before.get(k, 0)
            for k in set(new_state) | set(before)
        }
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