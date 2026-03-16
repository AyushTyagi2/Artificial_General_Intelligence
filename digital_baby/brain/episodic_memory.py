"""brain/episodic_memory.py — Episodic fact generator.

Converts raw (action, deltas, domain) observations from the world simulator
into structured episodic facts that the Learner / Memory pipeline can parse.

The key insight: the existing Reasoner.extract_relation() only matches strings
containing known relation tokens ("affects", "causes", "hunts", etc.).  The old
transition_facts like "wolves_delta is -5.87" matched "is" but produced a
useless (wolves_delta, is, -5.87) triple that was already known after tick 1
and never contributed a new_fact again.

This module generates facts like:
  "wolf negatively_affects deer"        ← causal direction from sign
  "wolf positively_affects grass"       ← directional observation
  "wolf affects grass"                  ← below directional threshold

These strings:
  1. match _KNOWN_RELATIONS in Reasoner → parsed as proper triplets
  2. vary per tick because the subject/object combination depends on which
     variables actually changed and by how much → new_facts > 0
  3. are deduplicated by (subject, relation, object) inside Memory so the
     same well-known fact reinforces confidence rather than cluttering memory

Public API
----------
generate_episodic_facts(action, subject, domain, deltas, tick) -> List[str]
    Returns a list of fact strings ready for Learner.learn_from_page().

record_episodic_observation(action, subject, domain,
                             state_before, state_after, tick) -> List[str]
    Convenience wrapper: computes deltas and calls generate_episodic_facts.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Minimum absolute delta to count as an observable effect.
# Mirrors _MIN_RELATIVE_CHANGE in intervention_causal_discovery.
_MIN_DELTA_ABS: float = 1e-3

# When abs(delta) >= this fraction of the variable's baseline OR abs(delta)
# exceeds _DIRECTIONAL_ABS_MIN, we emit a directional (positive/negative)
# rather than plain "affects" relation.
_DIRECTIONAL_THRESHOLD: float = 0.01   # 1% relative change
_DIRECTIONAL_ABS_MIN:   float = 0.10   # or 0.1 absolute units

# Maximum number of effect facts per observation (avoids flooding memory
# with every tiny co-moving variable on large state vectors).
_MAX_EFFECTS_PER_OBS: int = 8

# Maps action name → canonical cause concept (mirrors _ACTION_TARGET_VAR
# in intervention_causal_discovery but expressed as concept names).
_ACTION_TO_CAUSE: Dict[str, str] = {
    "introduce_species":   "deer",
    "remove_species":      "deer",
    "add_predator":        "wolf",
    "remove_predator":     "wolf",
    "change_season":       "season_factor",
    "increase_temperature":"temperature",
    "add_chemical":        "reactants",
    "add_catalyst":        "catalyst",
    "adjust_pH":           "ph",
    "remove_product":      "product_concentration",
    "add_robot":           "robot",
    "remove_robot":        "robot",
    "upgrade_sensor":      "sensor_threshold",
    "increase_maintenance":"maintenance_load",
    "add_pathogen":        "pathogens",
    "boost_energy":        "energy",
    "add_cells":           "cells",
    "neutralise_toxin":    "toxin_level",
    "add_antibody":        "antibodies",
    "increase_force":      "force",
    "add_heat":            "heat",
    "add_friction":        "friction",
    "apply_impulse":       "force",
    "reduce_mass":         "mass",
    "induce_stress":       "stress_level",
    "reduce_stress":       "stress_level",
    "boost_dopamine":      "dopamine_level",
    "improve_sleep":       "memory_consolidation",
    "stimulate_neurons":   "neural_activity",
    "emit_co2":            "co2_level",
    "plant_forest":        "vegetation_cover",
    "melt_glacier":        "glacier_melt",
    "increase_albedo":     "albedo",
    "warm_ocean":          "ocean_heat",
    "raise_interest_rate": "interest_rate",
    "lower_interest_rate": "interest_rate",
    "increase_spending":   "consumption",
    "boost_productivity":  "productivity",
    "add_debt":            "debt_level",
    "apply_stress":        "stress_level",
    "heat_treat":          "temperature",
    "add_porosity":        "porosity",
    "quench":              "temperature",
    "anneal":              "temperature",
}

# Variables that are the direct target of the action — don't emit them
# as *effects* of the cause concept (they ARE the cause, not a downstream
# effect).
_INTERVENTION_VARS: Dict[str, str] = {
    "introduce_species":   "deer",
    "remove_species":      "deer",
    "add_predator":        "wolves",
    "remove_predator":     "wolves",
    "change_season":       "season_factor",
    "increase_temperature":"temperature",
    "add_chemical":        "reactants",
    "add_catalyst":        "catalyst",
    "adjust_pH":           "ph",
    "remove_product":      "product_concentration",
    "add_robot":           "robots",
    "remove_robot":        "robots",
    "upgrade_sensor":      "sensor_threshold",
    "increase_maintenance":"maintenance_load",
    "add_pathogen":        "pathogens",
    "boost_energy":        "energy",
    "add_cells":           "cells",
    "neutralise_toxin":    "toxin_level",
    "add_antibody":        "antibodies",
    "increase_force":      "force",
    "add_heat":            "heat",
    "add_friction":        "friction",
    "apply_impulse":       "momentum",
    "reduce_mass":         "mass",
    "induce_stress":       "stress_level",
    "reduce_stress":       "stress_level",
    "boost_dopamine":      "dopamine_level",
    "improve_sleep":       "memory_consolidation",
    "stimulate_neurons":   "neural_activity",
    "emit_co2":            "co2_level",
    "plant_forest":        "vegetation_cover",
    "melt_glacier":        "glacier_melt",
    "increase_albedo":     "albedo",
    "warm_ocean":          "ocean_heat",
    "raise_interest_rate": "interest_rate",
    "lower_interest_rate": "interest_rate",
    "increase_spending":   "consumption",
    "boost_productivity":  "productivity",
    "add_debt":            "debt_level",
    "apply_stress":        "stress_level",
    "heat_treat":          "temperature",
    "add_porosity":        "porosity",
    "quench":              "temperature",
    "anneal":              "temperature",
}


def _normalise_var(var: str) -> str:
    """Collapse plural/alias forms to canonical concept names."""
    _MAP = {
        "wolves": "wolf", "robots": "robot", "asteroids": "asteroid",
        "cells_count": "cells", "ph": "ph",
    }
    return _MAP.get(var.lower(), var.lower())


def generate_episodic_facts(
    action:  str,
    subject: str,
    domain:  str,
    deltas:  Dict[str, float],
    tick:    int,
    state_before: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Convert a world-simulator observation into learnable fact strings.

    Returns a list of strings in one of these forms (all recognised by
    Reasoner.extract_relation via _KNOWN_RELATIONS):
      "<cause> negatively_affects <effect>"
      "<cause> positively_affects <effect>"
      "<cause> affects <effect>"

    Direction is chosen by: abs(delta)/max(baseline, 1e-6) >= 1% OR
    abs(delta) >= 0.1 → directional; otherwise plain "affects".

    Parameters
    ----------
    action       : the intervention name (e.g. "add_predator")
    subject      : canonical subject resolved by the event loop (e.g. "wolf")
    domain       : world domain (e.g. "ecosystem")
    deltas       : {variable: delta_value} from the world step
    tick         : current tick (used in debug logging only)
    state_before : pre-intervention state (optional; used for relative thresholds)
    """
    if not deltas:
        return []

    # Resolve the cause concept
    cause = subject.strip().lower() if subject.strip() else _ACTION_TO_CAUSE.get(action, action)
    cause = _normalise_var(cause)

    # The directly intervened variable — skip as an effect
    skip_var = _INTERVENTION_VARS.get(action, "")

    # Sort effects by abs(delta) descending; take top _MAX_EFFECTS_PER_OBS
    ranked = sorted(
        [(var, delta) for var, delta in deltas.items()
         if abs(delta) >= _MIN_DELTA_ABS
         and _normalise_var(var) != _normalise_var(skip_var)
         and _normalise_var(var) != cause],
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:_MAX_EFFECTS_PER_OBS]

    facts: List[str] = []
    seen: set = set()

    for var, delta in ranked:
        effect = _normalise_var(var)
        if effect == cause:
            continue

        # Choose relation based on direction and magnitude.
        # Use 0.0 as the default baseline for variables not present before the
        # intervention (a new variable appearing counts as fully novel).
        # Guard against zero-baseline division with max(..., 1e-6).
        baseline = abs(state_before.get(var, 0.0)) if state_before else 0.0
        relative = abs(delta) / max(baseline, 1e-6)

        if relative >= _DIRECTIONAL_THRESHOLD or abs(delta) >= _DIRECTIONAL_ABS_MIN:
            relation = "positively_affects" if delta > 0 else "negatively_affects"
        else:
            relation = "affects"

        fact = f"{cause} {relation} {effect}"
        if fact not in seen:
            seen.add(fact)
            facts.append(fact)

    logger.debug(
        "[episodic] tick=%d action=%s cause=%s facts=%d: %s",
        tick, action, cause, len(facts),
        "; ".join(facts[:3]) + ("..." if len(facts) > 3 else ""),
    )
    return facts


def record_episodic_observation(
    action:       str,
    subject:      str,
    domain:       str,
    state_before: Dict[str, float],
    state_after:  Dict[str, float],
    tick:         int,
) -> List[str]:
    """Compute deltas and generate episodic facts in one call.

    This is the primary entry point for the event loop.
    """
    deltas = {
        var: state_after.get(var, 0.0) - state_before.get(var, 0.0)
        for var in set(state_before) | set(state_after)
    }
    # Only keep variables that actually changed
    deltas = {var: d for var, d in deltas.items() if abs(d) >= _MIN_DELTA_ABS}

    return generate_episodic_facts(
        action=action,
        subject=subject,
        domain=domain,
        deltas=deltas,
        tick=tick,
        state_before=state_before,
    )