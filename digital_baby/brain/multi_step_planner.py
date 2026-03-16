"""Multi-Step Experiment Planner — Architecture v2.

Replaces single-tick random intervention selection with depth-2 experiment
sequences that maximally distinguish competing hypotheses.

Algorithm
---------
Given N candidate hypotheses H₁…Hₙ and a fixed action vocabulary A:

For each candidate sequence (a₁, a₂) ∈ A × A:
  1. For each hypothesis Hᵢ, simulate the predicted world state after
     applying a₁ then a₂.
  2. Compute pairwise divergence between predicted states.
  3. Score = avg pairwise divergence (higher = more discriminating).

Select the sequence with the highest score.

State simulation is deliberately lightweight — it uses the causal graph
confidence scores rather than a full simulator, making it ~10ms per plan
even with 81 candidate sequences (9 actions × 9 actions).

Integration
-----------
    planner = MultiStepPlanner()

    seq = planner.plan(
        hypotheses=ranked_hypotheses[:8],
        causal_rules=memory.get_causal_rules(),
        domain_state=generator.state_for_domain(domain),
        available_actions=generator.actions_by_domain.get(domain, []),
        current_tick=tick,
    )
    if seq:
        # Execute first action this tick
        preferred_action = seq.action_sequence[0]
        planner.record_execution(seq, tick)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_PLAN_DEPTH: int  = 2      # sequence length
MIN_DIVERGENCE: float = 0.05  # minimum predicted divergence to bother planning
PLAN_COOLDOWN_TICKS: int = 3  # don't replan more often than this


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExperimentSequence:
    """A planned multi-step experiment."""
    action_sequence:    List[str]
    target_hypotheses:  List[str]    # hypothesis rules being tested
    predicted_divergence: float
    domain:             str
    planned_tick:       int

    def __str__(self) -> str:
        acts = " → ".join(self.action_sequence)
        return f"Sequence({acts}) divergence={self.predicted_divergence:.3f}"


@dataclass
class _ActionEffect:
    """Lightweight prediction of how an action affects domain variables."""
    action:    str
    variable:  str
    delta:     float    # predicted signed delta


# ---------------------------------------------------------------------------
# Domain action → variable effect table
# ---------------------------------------------------------------------------
# This is a simplified causal model used for lookahead simulation.
# It is seeded from known domain structures and updated as causal rules
# are confirmed.

_BASE_EFFECTS: Dict[str, List[Tuple[str, float]]] = {
    # (action) → [(variable, expected_delta), ...]
    "add_predator":         [("wolves", +2.0), ("deer", -3.0)],
    "remove_predator":      [("wolves", -2.0), ("deer", +3.0)],
    "introduce_species":    [("deer",   +4.0), ("grass", -2.0)],
    "remove_species":       [("deer",   -4.0), ("grass", +2.0)],
    "add_robot":            [("robots", +1.0), ("sensor_coverage", +0.8), ("data_quality", +0.5)],
    "remove_robot":         [("robots", -1.0), ("sensor_coverage", -0.8), ("data_quality", -0.5)],
    "increase_temperature": [("temperature", +10.0), ("reaction_rate", +1.2), ("reaction_energy", +2.5)],
    "add_chemical":         [("reactants", +1.0), ("reaction_rate", +0.8), ("reaction_energy", +1.5)],
    "add_pathogen":         [("pathogens", +5.0), ("immune_response", +3.0), ("proteins", -2.0)],
    "boost_energy":         [("energy", +10.0), ("proteins", +2.0), ("pathogens", -1.0)],
    "add_cells":            [("cells", +10.0), ("immune_response", -2.0)],
    "increase_force":       [("force", +5.0), ("acceleration", +1.0), ("kinetic_energy", +3.0)],
    "add_heat":             [("heat", +5.0), ("kinetic_energy", -2.0), ("force", -1.0)],
}


def _simulate_action(
    state: Dict[str, float],
    action: str,
    causal_rules: List,    # CausalRuleRecord list
) -> Dict[str, float]:
    """Return a new state dict after applying one action."""
    new_state = dict(state)

    # Apply base effects
    for var, delta in _BASE_EFFECTS.get(action, []):
        if var in new_state:
            new_state[var] = new_state[var] + delta

    # Apply propagation via causal rules (one step)
    for rule in causal_rules:
        if rule.confidence < 0.25:
            continue
        if rule.cause in new_state and rule.effect in new_state:
            cause_delta = new_state[rule.cause] - state.get(rule.cause, 0.0)
            if abs(cause_delta) < 0.01:
                continue
            sign = 1.0 if rule.direction == "positive" else -1.0
            propagated = cause_delta * rule.confidence * sign * 0.3
            new_state[rule.effect] = new_state.get(rule.effect, 0.0) + propagated

    return new_state


def _state_distance(s1: Dict[str, float], s2: Dict[str, float]) -> float:
    """L1 normalised distance between two state vectors."""
    keys = set(s1) | set(s2)
    if not keys:
        return 0.0
    total = sum(abs(s1.get(k, 0.0) - s2.get(k, 0.0)) for k in keys)
    scale = sum(abs(s1.get(k, 0.0)) + abs(s2.get(k, 0.0)) for k in keys) + 1e-9
    return total / scale


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MultiStepPlanner:
    """Plans 2-step intervention sequences to maximally discriminate hypotheses.

    Usage
    -----
    Instantiate once in BabyEventLoop.__init__(), call plan() each tick
    when a stateful domain step is about to be taken.
    """

    def __init__(self, max_depth: int = MAX_PLAN_DEPTH) -> None:
        self.max_depth    = max_depth
        self._last_plan:  Optional[ExperimentSequence] = None
        self._last_plan_tick: int = -PLAN_COOLDOWN_TICKS
        self._execution_log: List[Dict] = []

    def plan(
        self,
        hypotheses:        List,              # Hypothesis objects
        causal_rules:      List,              # CausalRuleRecord list
        domain_state:      Dict[str, float],
        available_actions: List[str],
        current_tick:      int,
        domain:            str = "unknown",
    ) -> Optional[ExperimentSequence]:
        """Find the action sequence that best discriminates hypotheses.

        Returns None if:
        - fewer than 2 hypotheses (nothing to discriminate)
        - fewer than 2 actions available
        - cooldown period hasn't elapsed
        - predicted divergence is below MIN_DIVERGENCE
        """
        if current_tick - self._last_plan_tick < PLAN_COOLDOWN_TICKS:
            return self._last_plan

        if len(hypotheses) < 2 or len(available_actions) < 1:
            return None

        best_seq: Optional[ExperimentSequence] = None
        best_divergence = MIN_DIVERGENCE

        # Build candidate sequences
        actions = available_actions[:9]    # cap at 9 to limit 81-combo search
        candidates: List[List[str]]
        if self.max_depth == 1 or len(actions) == 1:
            candidates = [[a] for a in actions]
        else:
            candidates = [[a1, a2] for a1 in actions for a2 in actions]

        for action_seq in candidates:
            divergence = self._score_sequence(
                action_seq, hypotheses[:8], causal_rules, domain_state
            )
            if divergence > best_divergence:
                best_divergence = divergence
                best_seq = ExperimentSequence(
                    action_sequence=action_seq,
                    target_hypotheses=[
                        getattr(h, "rule", "?")[:50] for h in hypotheses[:3]
                    ],
                    predicted_divergence=divergence,
                    domain=domain,
                    planned_tick=current_tick,
                )

        if best_seq:
            logger.info(
                "[multi_step_planner] tick=%d planned=%s divergence=%.3f",
                current_tick, " → ".join(best_seq.action_sequence), best_divergence,
            )
            self._last_plan = best_seq
            self._last_plan_tick = current_tick

        return best_seq

    def record_execution(self, seq: ExperimentSequence, tick: int) -> None:
        """Log that a planned sequence was executed."""
        self._execution_log.append({
            "tick":       tick,
            "sequence":   seq.action_sequence,
            "domain":     seq.domain,
            "divergence": seq.predicted_divergence,
        })
        self._execution_log = self._execution_log[-200:]

    def stats(self) -> Dict:
        return {
            "plans_executed":   len(self._execution_log),
            "last_plan":        str(self._last_plan) if self._last_plan else None,
            "mean_divergence":  (
                sum(e["divergence"] for e in self._execution_log) /
                max(1, len(self._execution_log))
            ),
        }

    # ── Internal ───────────────────────────────────────────────────────────────

    def _score_sequence(
        self,
        action_seq:    List[str],
        hypotheses:    List,
        causal_rules:  List,
        initial_state: Dict[str, float],
    ) -> float:
        """Score a sequence by average pairwise divergence of predicted states."""
        # Each hypothesis has a slightly different confidence distribution,
        # which propagates differently through the causal rules.
        # We perturb the causal rules according to each hypothesis's confidence
        # to simulate what each "believes" will happen.

        predicted_states: List[Dict[str, float]] = []

        for hyp in hypotheses:
            state = dict(initial_state)
            hyp_conf = float(getattr(hyp, "confidence", 0.5))
            hyp_concepts = getattr(hyp, "concepts", [])

            # Build a perturbed rule set based on this hypothesis's confidence
            perturbed_rules = _PerturbedRuleSet(causal_rules, hyp_concepts, hyp_conf)

            for action in action_seq:
                state = _simulate_action(state, action, perturbed_rules.rules)

            predicted_states.append(state)

        # Average pairwise distance
        if len(predicted_states) < 2:
            return 0.0

        n = len(predicted_states)
        total = 0.0
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += _state_distance(predicted_states[i], predicted_states[j])
                pairs += 1

        return total / pairs if pairs > 0 else 0.0


class _PerturbedRuleSet:
    """Lightweight wrapper that adjusts rule confidence based on a hypothesis."""

    def __init__(self, base_rules: List, focal_concepts: List[str], hyp_conf: float):
        self.rules = []
        for rule in base_rules:
            if rule.cause in focal_concepts or rule.effect in focal_concepts:
                # Hypothesis claims this rule holds with hyp_conf strength
                # Use that as a weight modifier
                class _R:
                    def __init__(self, r, c):
                        self.cause     = r.cause
                        self.effect    = r.effect
                        self.direction = r.direction
                        self.confidence = c
                self.rules.append(_R(rule, hyp_conf))
            else:
                self.rules.append(rule)