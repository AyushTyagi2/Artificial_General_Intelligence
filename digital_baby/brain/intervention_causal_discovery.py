"""Intervention-based causal discovery — v4.

KEY FIXES vs v3
---------------
1. MIN_ACCUMULATE (0.005) is now the gate for entering the Beta prior.
   MIN_INJECT_CONFIDENCE (0.05) only applies in inject_into_graph().
   Previously both shared MIN_INJECT_CONFIDENCE=0.05, silently discarding
   weak-but-real signals (e.g. velocity→acceleration) before they could
   accumulate. They now compound over experiments.

2. Surprise-weighted accumulation: contradictory observations get up to
   50% extra weight, accelerating convergence on the true direction.

3. Temporal decay: records not updated within EVIDENCE_HALF_LIFE_TICKS
   ticks regress toward the prior, preventing stale causal beliefs.

4. discover_second_order_effects(): systematically finds A→B→C chains
   and seeds A→C records. Called every N ticks from the event loop.

5. Causal axiom seeding: known definitional relationships (F=ma, Arrhenius,
   predator-prey) are pre-loaded as strong Beta priors so the agent doesn't
   re-discover Newton's second law from scratch.

6. analyze() accepts current_tick for decay tracking.
   analyze_paired() for clean paired-experiment causal deltas.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_RELATIVE_CHANGE:    float = 0.01   # lowered from 0.02
MIN_ACCUMULATE:          float = 0.005  # NEW: gate for Beta prior entry
MIN_INJECT_CONFIDENCE:   float = 0.05   # gate for KG injection (unchanged)
MIN_INJECT_EVIDENCE:     int   = 3      # NEW: minimum observations before injecting
BIDIRECTIONAL_THRESHOLD: float = 0.30
MIN_MEDIATOR_OBS:        int   = 8
MEDIATOR_SEPARATION_THR: float = 0.7
EVIDENCE_HALF_LIFE_TICKS: int  = 500   # NEW: stale decay period

_REMOVAL_ACTIONS: FrozenSet[str] = frozenset({
    "remove_species", "remove_predator", "remove_robot", "remove_product",
    "neutralise_toxin", "reduce_mass",
})
_ADDITION_ACTIONS: FrozenSet[str] = frozenset({
    "introduce_species", "add_predator", "add_chemical", "increase_temperature",
    "add_catalyst", "adjust_pH", "add_robot", "add_pathogen", "boost_energy",
    "add_cells", "add_antibody", "add_friction", "apply_impulse", "upgrade_sensor",
    "increase_maintenance", "change_season",
})

_VARIABLE_TO_CONCEPT: Dict[str, str] = {
    "wolves": "wolf", "deer": "deer", "grass": "grass", "season_factor": "season_factor",
    "asteroids": "asteroid", "solar_energy": "solar_energy",
    "collision_risk": "collision_risk", "radiation_pressure": "radiation_pressure",
    "asteroid_drift": "asteroid_drift",
    "temperature": "temperature", "reactants": "reactants",
    "reaction_rate": "reaction_rate", "reaction_energy": "reaction_energy",
    "catalyst": "catalyst", "product_concentration": "product_concentration",
    "pH": "pH", "activation_energy": "activation_energy",
    "robots": "robot", "battery_charge": "battery_charge",
    "robot_activity": "robot_activity", "sensor_coverage": "sensor_coverage",
    "data_quality": "data_quality", "maintenance_load": "maintenance_load",
    "network_latency": "network_latency", "sensor_threshold": "sensor_threshold",
    "cells": "cells", "energy": "energy", "proteins": "proteins",
    "pathogens": "pathogens", "immune_response": "immune_response",
    "toxin_level": "toxin_level", "antibodies": "antibodies",
    "cell_cycle_rate": "cell_cycle_rate",
    "force": "force", "mass": "mass", "acceleration": "acceleration",
    "heat": "heat", "kinetic_energy": "kinetic_energy",
    "velocity": "velocity", "friction": "friction", "momentum": "momentum",
    # neuroscience (v4.1)
    "stress_level": "stress_level", "cortisol": "cortisol",
    "dopamine_level": "dopamine_level", "serotonin_level": "serotonin_level",
    "neural_activity": "neural_activity", "synaptic_strength": "synaptic_strength",
    "memory_consolidation": "memory_consolidation", "learning_rate": "learning_rate",
    # climate (v4.1)
    "co2_level": "co2_level", "temperature_anomaly": "temperature_anomaly",
    "glacier_melt": "glacier_melt", "sea_level_rise": "sea_level_rise",
    "precipitation": "precipitation", "vegetation_cover": "vegetation_cover",
    "albedo": "albedo", "ocean_heat": "ocean_heat",
    # economics (v4.1)
    "interest_rate": "interest_rate", "inflation": "inflation",
    "gdp_growth": "gdp_growth", "unemployment": "unemployment",
    "investment": "investment", "consumption": "consumption",
    "productivity": "productivity", "debt_level": "debt_level",
    # materials (v4.1)
    "strain": "strain", "hardness": "hardness",
    "yield_strength": "yield_strength", "conductivity": "conductivity",
    "crack_growth": "crack_growth", "porosity": "porosity",
}

_ACTION_TARGET_VAR: Dict[str, str] = {
    "remove_species": "wolves", "remove_predator": "wolves", "add_predator": "wolves",
    "introduce_species": "deer", "increase_temperature": "temperature",
    "add_chemical": "reactants", "add_catalyst": "catalyst", "adjust_pH": "pH",
    "remove_product": "product_concentration", "add_robot": "robots",
    "remove_robot": "robots", "upgrade_sensor": "sensor_threshold",
    "increase_maintenance": "maintenance_load", "add_pathogen": "pathogens",
    "boost_energy": "energy", "add_cells": "cells", "neutralise_toxin": "toxin_level",
    "add_antibody": "antibodies", "increase_force": "force", "add_heat": "heat",
    "add_friction": "friction", "apply_impulse": "momentum", "reduce_mass": "mass",
    "change_season": "season_factor",
    # neuroscience (v4.1)
    "induce_stress": "stress_level", "reduce_stress": "stress_level",
    "boost_dopamine": "dopamine_level", "improve_sleep": "memory_consolidation",
    "stimulate_neurons": "neural_activity",
    # climate (v4.1)
    "emit_co2": "co2_level", "plant_forest": "vegetation_cover",
    "melt_glacier": "glacier_melt", "increase_albedo": "albedo",
    "warm_ocean": "ocean_heat",
    # economics (v4.1)
    "raise_interest_rate": "interest_rate", "lower_interest_rate": "interest_rate",
    "increase_spending": "consumption", "boost_productivity": "productivity",
    "add_debt": "debt_level",
    # materials (v4.1)
    "apply_stress": "stress_level", "heat_treat": "temperature",
    "add_porosity": "porosity", "quench": "temperature", "anneal": "temperature",
}

# Known-true causal axioms seeded as strong Beta priors.
# (cause, effect) -> (sign, alpha, beta)
_CAUSAL_AXIOMS: Dict[Tuple[str, str], Tuple[int, float, float]] = {
    ("force",       "acceleration"):     (+1, 8.0, 2.0),
    ("mass",        "kinetic_energy"):   (+1, 6.0, 2.0),
    ("velocity",    "kinetic_energy"):   (+1, 6.0, 2.0),
    ("heat",        "temperature"):      (+1, 7.0, 2.0),
    ("temperature", "reaction_rate"):    (+1, 7.0, 2.0),
    ("catalyst",    "reaction_rate"):    (+1, 7.0, 2.0),
    ("pH",          "reaction_rate"):    (+1, 5.0, 3.0),
    ("wolf",        "deer"):             (-1, 8.0, 2.0),
    ("deer",        "grass"):            (-1, 8.0, 2.0),
    ("wolf",        "grass"):            (+1, 5.0, 3.0),
    ("pathogens",   "immune_response"):  (+1, 7.0, 2.0),
    ("antibodies",  "pathogens"):        (-1, 7.0, 2.0),
    ("friction",    "kinetic_energy"):   (-1, 7.0, 2.0),
    ("friction",    "acceleration"):     (-1, 6.0, 2.0),
}


def _concept(variable: str) -> str:
    return _VARIABLE_TO_CONCEPT.get(variable, variable)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CausalCondition:
    mediator:     str
    operator:     str
    threshold:    float
    direction:    str
    confidence:   float = 0.5
    observations: int   = 0


@dataclass
class CausalFact:
    cause:      str
    effect:     str
    sign:       int
    confidence: float
    delta:      float
    baseline:   float


@dataclass
class CausalRecord:
    cause:  str
    effect: str
    alpha:  float = 1.0
    beta:   float = 1.0
    positive_count: int   = 0
    negative_count: int   = 0
    magnitude_sum:  float = 0.0
    last_updated_tick: int = 0
    conditions: List[CausalCondition] = field(default_factory=list)

    @property
    def evidence_count(self) -> int:
        return self.positive_count + self.negative_count

    @property
    def confidence(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def direction(self) -> str:
        total = self.positive_count + self.negative_count
        if total == 0:
            return "unknown"
        if self.conditions:
            return "conditional"
        pos_frac = self.positive_count / total
        neg_frac = self.negative_count / total
        if pos_frac >= (1.0 - BIDIRECTIONAL_THRESHOLD):
            return "positive"
        if neg_frac >= (1.0 - BIDIRECTIONAL_THRESHOLD):
            return "negative"
        return "bidirectional"

    @property
    def mean_magnitude(self) -> float:
        return self.magnitude_sum / max(1, self.evidence_count)

    def update(self, sign: int, weighted_conf: float, delta_magnitude: float) -> None:
        if sign > 0:
            self.alpha += weighted_conf
            self.positive_count += 1
        else:
            self.beta += weighted_conf
            self.negative_count += 1
        self.magnitude_sum += abs(delta_magnitude)

    def decay_toward_prior(self, current_tick: int) -> None:
        if self.last_updated_tick == 0:
            return
        ticks_stale = current_tick - self.last_updated_tick
        if ticks_stale < EVIDENCE_HALF_LIFE_TICKS:
            return
        decay_factor = 0.5 ** (ticks_stale / EVIDENCE_HALF_LIFE_TICKS)
        self.alpha = 1.0 + (self.alpha - 1.0) * decay_factor
        self.beta  = 1.0 + (self.beta  - 1.0) * decay_factor


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class InterventionCausalDiscovery:
    """Infer causal relations from interventions — v4."""

    def __init__(self) -> None:
        self._records:            Dict[Tuple[str, str], CausalRecord] = {}
        self._weak_discard_count: Dict[Tuple[str, str], int] = {}
        self._obs_history:        Dict[Tuple[str, str], List[Tuple[Dict[str, float], int]]] = {}
        self._mediator_searched:  set = set()
        self._axioms_seeded:      bool = False

    def _seed_axioms(self) -> None:
        if self._axioms_seeded:
            return
        for (cause, effect), (sign, alpha, beta) in _CAUSAL_AXIOMS.items():
            key = (cause, effect)
            record = CausalRecord(cause=cause, effect=effect, alpha=alpha, beta=beta)
            if sign > 0:
                record.positive_count = int(alpha - 1)
            else:
                record.negative_count = int(beta - 1)
            self._records[key] = record
            logger.info("[causal_axiom] seeded %s→%s sign=%+d conf=%.2f",
                        cause, effect, sign, record.confidence)
        self._axioms_seeded = True

    # ── Public API ─────────────────────────────────────────────────────────

    def analyze(
        self,
        action:       str,
        subject:      str,
        state_before: Dict[str, float],
        state_after:  Dict[str, float],
        current_tick: int = 0,
    ) -> List[CausalFact]:
        self._seed_axioms()
        logger.info("[causal_discovery] intervention=%s %s", action, subject)

        cause_concept   = self._resolve_cause(action, subject, state_before, state_after)
        is_removal      = action in _REMOVAL_ACTIONS
        intervened_vars = self._intervened_variables(action, state_before, state_after)

        facts: List[CausalFact] = []

        for var in sorted(set(state_before) | set(state_after)):
            if var in intervened_vars:
                continue
            before_val = state_before.get(var, 0.0)
            after_val  = state_after.get(var,  0.0)
            delta      = after_val - before_val

            if abs(delta) < 1e-9:
                continue
            if abs(before_val) > 1e-9 and abs(delta) / abs(before_val) < _MIN_RELATIVE_CHANGE:
                continue

            relative_change = abs(delta) / max(1.0, abs(before_val))
            confidence      = min(1.0, relative_change * 3.0)
            sign            = (-1 if delta > 0 else +1) if is_removal else (+1 if delta > 0 else -1)
            effect_concept  = _concept(var)

            if cause_concept == effect_concept:
                continue

            key = (cause_concept, effect_concept)

            # CRITICAL FIX: accumulate into Beta prior when >= MIN_ACCUMULATE
            # (was MIN_INJECT_CONFIDENCE — weak signals were thrown away)
            if confidence >= MIN_ACCUMULATE:
                self._accumulate_weighted(
                    cause_concept, effect_concept, sign, confidence, abs(delta), current_tick
                )
                if key not in self._obs_history:
                    self._obs_history[key] = []
                self._obs_history[key].append((dict(state_before), sign))
                if len(self._obs_history[key]) > 200:
                    self._obs_history[key] = self._obs_history[key][-200:]

            # Only inject into KG when posterior is confident AND has enough evidence
            record = self._records.get(key)
            if record is None:
                continue
            if (record.confidence >= MIN_INJECT_CONFIDENCE
                    and record.evidence_count >= MIN_INJECT_EVIDENCE):
                facts.append(CausalFact(
                    cause=cause_concept, effect=effect_concept,
                    sign=sign, confidence=record.confidence,
                    delta=delta, baseline=before_val,
                ))
                logger.info(
                    "[causal_discovery] inferred %s -> %s %s conf=%.2f ev=%d Δ=%+.2f",
                    cause_concept, effect_concept,
                    "(+)" if sign > 0 else "(-)", record.confidence,
                    record.evidence_count, delta,
                )

        for fact in facts:
            self._maybe_search_mediator(fact.cause, fact.effect, state_before)

        return facts

    def analyze_paired(
        self,
        action:          str,
        subject:         str,
        control_state:   Dict[str, float],
        treatment_state: Dict[str, float],
        current_tick:    int = 0,
    ) -> List[CausalFact]:
        """Analyze a paired experiment. Control drift is eliminated by design."""
        return self.analyze(action, subject, control_state, treatment_state, current_tick)

    def inject_into_graph(self, facts: List[CausalFact], knowledge_graph, current_tick: int = 0) -> int:
        count = 0
        for fact in facts:
            if fact.cause == fact.effect:
                continue
            key    = (fact.cause, fact.effect)
            record = self._records.get(key)
            if record is None:
                continue
            if record.confidence < MIN_INJECT_CONFIDENCE:
                continue
            if record.evidence_count < MIN_INJECT_EVIDENCE:
                continue
            direction = record.direction
            if direction in ("positive", "negative", "bidirectional", "conditional"):
                knowledge_graph.ingest_causal_rule(
                    cause=fact.cause, effect=fact.effect,
                    direction=direction, confidence=record.confidence,
                    evidence=record.evidence_count, current_tick=current_tick,
                )
                count += 1
        return count

    def discover_second_order_effects(self, knowledge_graph) -> List[CausalFact]:
        """Find A→B→C chains and seed A→C records with combined confidence."""
        self._seed_axioms()
        MIN_CHAIN_CONF = 0.15
        new_facts: List[CausalFact] = []

        sign_of: Dict[Tuple[str, str], int] = {}
        for (cause, effect), rec in self._records.items():
            if rec.direction == "positive":
                sign_of[(cause, effect)] = +1
            elif rec.direction == "negative":
                sign_of[(cause, effect)] = -1

        seen_chains: set = set()
        strong = [(k, r) for k, r in self._records.items()
                  if r.direction in ("positive", "negative") and r.confidence >= 0.3]

        for (a, b), rec_ab in strong:
            sign_ab  = sign_of.get((a, b), 0)
            conf_ab  = rec_ab.confidence
            if sign_ab == 0:
                continue
            for (b2, c), rec_bc in strong:
                if b2 != b or c == a:
                    continue
                sign_bc   = sign_of.get((b2, c), 0)
                if sign_bc == 0:
                    continue
                chain_conf = conf_ab * rec_bc.confidence
                if chain_conf < MIN_CHAIN_CONF:
                    continue
                chain_sign = sign_ab * sign_bc
                chain_key  = (a, c)
                if chain_key in seen_chains:
                    continue
                seen_chains.add(chain_key)

                if chain_key not in self._records:
                    self._accumulate_weighted(a, c, chain_sign, chain_conf, 0.0, 0)
                    logger.info("[second_order] inferred %s→%s sign=%+d conf=%.2f via %s",
                                a, c, chain_sign, chain_conf, b)

                new_facts.append(CausalFact(
                    cause=a, effect=c, sign=chain_sign,
                    confidence=chain_conf, delta=0.0, baseline=0.0,
                ))

        return new_facts

    def decay_stale_records(self, current_tick: int) -> None:
        for record in self._records.values():
            record.decay_toward_prior(current_tick)

    def get_all_records(self) -> List[CausalRecord]:
        return list(self._records.values())

    def get_record(self, cause: str, effect: str) -> Optional[CausalRecord]:
        return self._records.get((cause, effect))

    def get_conditional_edges(self) -> List[CausalRecord]:
        return [r for r in self._records.values() if r.conditions]

    def get_stagnant_pairs(self, min_evidence: int = 10,
                           conf_lo: float = 0.35, conf_hi: float = 0.55) -> List[Tuple[str, str]]:
        return [
            (cause, effect)
            for (cause, effect), rec in self._records.items()
            if rec.evidence_count >= min_evidence and conf_lo <= rec.confidence <= conf_hi
        ]

    # ── Mediator detection ─────────────────────────────────────────────────

    def _maybe_search_mediator(self, cause: str, effect: str,
                                current_state: Dict[str, float]) -> None:
        key    = (cause, effect)
        record = self._records.get(key)
        if record is None or key in self._mediator_searched:
            return
        if record.direction != "bidirectional":
            return
        if record.evidence_count < MIN_MEDIATOR_OBS:
            return
        if record.positive_count < 3 or record.negative_count < 3:
            return
        self._mediator_searched.add(key)
        mediator = self._find_mediator(cause, effect)
        if mediator is not None:
            record.conditions.append(mediator)
            logger.info("[causal_mediator_found] %s→%s | %s %s %.2f dir_if_true=%s",
                        cause, effect, mediator.mediator, mediator.operator,
                        mediator.threshold, mediator.direction)

    def _find_mediator(self, cause: str, effect: str) -> Optional[CausalCondition]:
        key  = (cause, effect)
        hist = self._obs_history.get(key, [])
        if len(hist) < MIN_MEDIATOR_OBS:
            return None
        pos_states = [s for s, sign in hist if sign > 0]
        neg_states = [s for s, sign in hist if sign < 0]
        if len(pos_states) < 3 or len(neg_states) < 3:
            return None

        all_vars: set = set()
        for s in pos_states + neg_states:
            all_vars.update(s.keys())
        all_vars.discard(cause)
        all_vars.discard(effect)

        best_sep      = 0.0
        best_mediator: Optional[CausalCondition] = None

        for var in sorted(all_vars):
            pv = [s.get(var, 0.0) for s in pos_states]
            nv = [s.get(var, 0.0) for s in neg_states]
            pos_mean  = sum(pv) / len(pv)
            neg_mean  = sum(nv) / len(nv)
            all_vals  = pv + nv
            grand_mean = sum(all_vals) / len(all_vals)
            variance  = sum((x - grand_mean) ** 2 for x in all_vals) / len(all_vals)
            std       = math.sqrt(variance) + 1e-9
            separation = abs(pos_mean - neg_mean) / std

            if separation > best_sep and separation > MEDIATOR_SEPARATION_THR:
                best_sep      = separation
                threshold     = (pos_mean + neg_mean) / 2.0
                dir_if_above  = "positive" if pos_mean > neg_mean else "negative"
                best_mediator = CausalCondition(
                    mediator=var, operator=">", threshold=round(threshold, 3),
                    direction=dir_if_above, confidence=min(0.9, separation / 3.0),
                    observations=len(hist),
                )

        return best_mediator

    # ── Internal helpers ───────────────────────────────────────────────────

    def _accumulate_weighted(self, cause: str, effect: str, sign: int,
                              confidence: float, magnitude: float, current_tick: int) -> None:
        key = (cause, effect)
        if key not in self._records:
            self._records[key] = CausalRecord(cause=cause, effect=effect)
        record     = self._records[key]
        prior_mean = record.confidence
        expected   = +1 if prior_mean >= 0.5 else -1
        surprise   = 0.0 if sign == expected else abs(prior_mean - 0.5) * 2.0
        weighted   = confidence * (1.0 + 0.5 * surprise)
        record.update(sign, weighted, magnitude)
        record.last_updated_tick = current_tick

    def _accumulate(self, cause: str, effect: str, sign: int,
                    confidence: float, magnitude: float) -> None:
        self._accumulate_weighted(cause, effect, sign, confidence, magnitude, 0)

    def _resolve_cause(self, action: str, subject: str,
                        state_before: Dict[str, float],
                        state_after:  Dict[str, float]) -> str:
        if subject:
            return _concept(subject)
        intervened = self._intervened_variables(action, state_before, state_after)
        if intervened:
            return _concept(next(iter(sorted(intervened))))
        return action

    @staticmethod
    def _intervened_variables(action: str, state_before: Dict[str, float],
                               state_after: Dict[str, float]) -> FrozenSet[str]:
        var = _ACTION_TARGET_VAR.get(action)
        if var and var in state_before:
            return frozenset({var})
        max_delta = 0.0
        max_var: Optional[str] = None
        for v in set(state_before) | set(state_after):
            d = abs(state_after.get(v, 0.0) - state_before.get(v, 0.0))
            if d > max_delta:
                max_delta = d
                max_var   = v
        return frozenset({max_var}) if max_var else frozenset()