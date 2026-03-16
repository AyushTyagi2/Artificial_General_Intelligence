"""Hypothesis-driven Experiment Planner — v1.

This module sits *above* the existing ExperimentPlanner (which selects
actions by information-gain scoring).  It adds a deliberate, hypothesis-
testing layer: given a hypothesis it designs a controlled experiment,
schedules it into the event loop's action queue, and evaluates the result.

Motivation
----------
The existing CuriosityModel + ExperimentPlanner already prevents naive
repetition of the same action.  But experiments are still chosen by an
implicit scoring function; they are not explicitly designed to answer a
specific hypothesis with controlled variables.

This module adds that intentional layer without replacing what exists.

Lifecycle
---------
1. ``select_hypothesis_to_test()`` picks the highest-priority ACTIVE hypothesis
   that does not yet have a running experiment.
2. ``design_plan(hypothesis)`` generates an ExperimentDesign: which variable to
   manipulate, which variables to hold fixed, what intervention action to use,
   and what outcome direction to predict.
3. ``claim_next_action(tick)`` is called by the event loop instead of the usual
   curiosity UCB1 path.  It returns (action_str, plan) when a designed
   experiment is pending, or (None, None) when none is queued.
4. ``evaluate(plan, before_state, after_state, tick)`` compares predicted vs.
   observed direction and updates hypothesis confidence.

Integration
-----------
In EventLoop.__init__:

    self.hyp_experiment_planner = HypothesisExperimentPlanner(
        hypothesis_engine=self.hypothesis_engine,
        knowledge_graph=self.knowledge_graph,
    )

In _select_action_v4() (highest priority branch, before UCB1):

    action, hep = self.hyp_experiment_planner.claim_next_action(tick)
    if action:
        self._pending_hep = hep
        return ActionSelection(action=action, reason="hypothesis_experiment")

    # If no pending experiment, maybe design one
    self.hyp_experiment_planner.maybe_design(tick)

After world step (where experiment result is known):

    if getattr(self, "_pending_hep", None):
        self.hyp_experiment_planner.evaluate(
            self._pending_hep, before_state, after_state, tick
        )
        self._pending_hep = None
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DESIGN_INTERVAL:        int   = 15    # ticks between design passes
MAX_QUEUED:             int   = 3     # maximum pending experiment designs
HYPOTHESIS_MIN_CONF:    float = 0.35  # only test hypotheses that are plausible
HYPOTHESIS_MAX_CONF:    float = 0.80  # skip near-confirmed hypotheses
CONFIDENCE_UPDATE_HIT:  float = +0.08 # confidence boost on correct prediction
CONFIDENCE_UPDATE_MISS: float = -0.06 # confidence penalty on wrong prediction
DELTA_THRESHOLD:        float = 0.01  # minimum state change to count as observed

# Maps known cause variables to actions that manipulate them
_CAUSE_TO_ACTIONS: Dict[str, List[str]] = {
    "temperature":      ["increase_temperature", "add_heat"],
    "pressure":         ["increase_pressure"],
    "concentration":    ["add_chemical"],
    "catalyst":         ["add_catalyst"],
    "ph":               ["adjust_pH"],
    "energy":           ["boost_energy"],
    "force":            ["increase_force", "apply_impulse"],
    "wolf":             ["add_predator", "remove_predator"],
    "deer":             ["introduce_species"],
    "cells":            ["add_cells"],
    "pathogens":        ["add_pathogen"],
    "antibodies":       ["add_antibody"],
    "toxin_level":      ["neutralise_toxin"],
    "robot":            ["add_robot"],
    "season_factor":    ["change_season"],
    "friction":         ["add_friction"],
    "momentum":         ["apply_impulse"],
    "maintenance_load": ["increase_maintenance"],
    "heat":             ["add_heat"],
    "neurons":          ["stimulate_neurons"],
    "dopamine":         ["stimulate_neurons"],
    "serotonin":        ["stimulate_neurons"],
}

_DEFAULT_ACTION = "stimulate_neurons"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExperimentDesign:
    """A fully specified, hypothesis-driven experiment."""
    hypothesis_rule:    str          # e.g. "temperature → reaction_rate"
    cause_variable:     str
    effect_variable:    str
    predicted_sign:     int          # +1 positive, -1 negative
    intervention:       str          # action string for the event loop
    control_variables:  List[str] = field(default_factory=list)
    tick_designed:      int = 0
    tick_executed:      int = -1
    outcome_sign:       Optional[int] = None   # +1/-1 after evaluation
    matched:            Optional[bool] = None  # True if prediction was correct

    @property
    def is_pending(self) -> bool:
        return self.tick_executed == -1

    @property
    def is_complete(self) -> bool:
        return self.matched is not None

    def __str__(self) -> str:
        return (
            f"ExperimentDesign(rule={self.hypothesis_rule!r}, "
            f"intervention={self.intervention!r}, "
            f"predicted={'↑' if self.predicted_sign > 0 else '↓'})"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HypothesisExperimentPlanner:
    """Designs controlled experiments to test specific hypotheses."""

    def __init__(self, hypothesis_engine, knowledge_graph) -> None:
        self._hengine  = hypothesis_engine   # HypothesisEngine from brain.hypothesis
        self._graph    = knowledge_graph     # KnowledgeGraph
        self._queue:   List[ExperimentDesign]      = []
        self._history: List[ExperimentDesign]      = []
        self._last_design_tick: int                = -DESIGN_INTERVAL
        # Track which hypothesis rules are already under test
        self._active_rules: Set[str]               = set()

    # ── Public API ─────────────────────────────────────────────────────────

    def maybe_design(self, tick: int) -> Optional[ExperimentDesign]:
        """Design a new experiment if the queue is not full and enough time has passed."""
        if tick - self._last_design_tick < DESIGN_INTERVAL:
            return None
        if len(self._queue) >= MAX_QUEUED:
            return None
        hyp = self._select_hypothesis()
        if hyp is None:
            return None
        plan = self._design_plan(hyp, tick)
        if plan:
            self._queue.append(plan)
            self._active_rules.add(plan.hypothesis_rule)
            self._last_design_tick = tick
            logger.info(
                "[hyp_experiment_planner] designed experiment rule=%r "
                "intervention=%r predicted_sign=%+d",
                plan.hypothesis_rule, plan.intervention, plan.predicted_sign,
            )
        return plan

    def claim_next_action(self, tick: int) -> Tuple[Optional[str], Optional[ExperimentDesign]]:
        """Return (action_str, design) for the next pending experiment, or (None, None)."""
        for plan in self._queue:
            if plan.is_pending:
                plan.tick_executed = tick
                return plan.intervention, plan
        return None, None

    def evaluate(
        self,
        plan: ExperimentDesign,
        before_state: dict,
        after_state: dict,
        tick: int,
    ) -> bool:
        """Compare predicted outcome vs. observed outcome; update hypothesis confidence.

        Returns True if prediction matched.
        """
        before_val = before_state.get(plan.effect_variable, 0.0)
        after_val  = after_state.get(plan.effect_variable, 0.0)
        delta      = after_val - before_val

        if abs(delta) < DELTA_THRESHOLD:
            logger.debug(
                "[hyp_experiment_planner] negligible_delta rule=%r delta=%.4f — skipping",
                plan.hypothesis_rule, delta,
            )
            return False

        observed_sign = +1 if delta > 0 else -1
        matched       = (observed_sign == plan.predicted_sign)

        plan.outcome_sign = observed_sign
        plan.matched      = matched

        # Update hypothesis confidence
        self._update_confidence(plan.hypothesis_rule, matched)

        result_sym = "✓" if matched else "✗"
        logger.info(
            "[hyp_experiment_planner] %s rule=%r predicted=%+d observed=%+d delta=%.4f",
            result_sym, plan.hypothesis_rule,
            plan.predicted_sign, observed_sign, delta,
        )

        # Move from queue to history
        if plan in self._queue:
            self._queue.remove(plan)
        self._active_rules.discard(plan.hypothesis_rule)
        self._history.append(plan)
        return matched

    @property
    def pending_count(self) -> int:
        return sum(1 for p in self._queue if p.is_pending)

    @property
    def history(self) -> List[ExperimentDesign]:
        return list(self._history)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _select_hypothesis(self):
        """Pick the most informative ACTIVE hypothesis that isn't already queued."""
        try:
            hypotheses = self._hengine.get_active_hypotheses()
        except AttributeError:
            # Fallback: try common alternative method name
            try:
                hypotheses = self._hengine.active_hypotheses
            except AttributeError:
                return None

        candidates = []
        for h in hypotheses:
            conf = getattr(h, "confidence", 0.5)
            rule = getattr(h, "rule", None) or getattr(h, "hypothesis_rule", "")
            if rule in self._active_rules:
                continue
            if not (HYPOTHESIS_MIN_CONF <= conf <= HYPOTHESIS_MAX_CONF):
                continue
            # Score: prefer mid-confidence hypotheses (most informative)
            score = 1.0 - abs(conf - 0.5) * 2.0
            candidates.append((score, h))

        if not candidates:
            return None
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]

    def _design_plan(self, hypothesis, tick: int) -> Optional[ExperimentDesign]:
        """Convert a Hypothesis object into a concrete ExperimentDesign."""
        # Extract cause and effect from the hypothesis
        cause  = getattr(hypothesis, "cause",  None)
        effect = getattr(hypothesis, "effect", None)
        rule   = getattr(hypothesis, "rule",   None) or getattr(hypothesis, "hypothesis_rule", "")

        if not cause or not effect:
            # Try to parse from rule string "X → Y" or "X affects Y"
            if rule:
                cause, effect = _parse_rule(rule)
            if not cause or not effect:
                return None

        # Determine predicted direction from the knowledge graph
        predicted_sign = self._predict_sign(cause, effect, hypothesis)

        # Select intervention action
        intervention = _choose_intervention(cause)

        # Identify control variables (graph neighbours of cause, excluding effect)
        control_vars = [
            n for n in self._graph.get_neighbors(cause)
            if n != effect
        ][:4]

        return ExperimentDesign(
            hypothesis_rule=rule or f"{cause} → {effect}",
            cause_variable=cause,
            effect_variable=effect,
            predicted_sign=predicted_sign,
            intervention=intervention,
            control_variables=control_vars,
            tick_designed=tick,
        )

    def _predict_sign(self, cause: str, effect: str, hypothesis) -> int:
        """Predict the direction of the causal effect."""
        # 1. Try the hypothesis object itself
        direction = getattr(hypothesis, "direction", None)
        if direction == "positive":
            return +1
        if direction == "negative":
            return -1

        # 2. Look at the knowledge graph edge
        for edge in self._graph.edges:
            if edge.source == cause and edge.target == effect:
                if "positive" in edge.relation:
                    return +1
                if "negative" in edge.relation:
                    return -1

        # 3. Default: positive (most common)
        return +1

    def _update_confidence(self, rule: str, matched: bool) -> None:
        """Nudge the hypothesis confidence based on the experiment result."""
        delta = CONFIDENCE_UPDATE_HIT if matched else CONFIDENCE_UPDATE_MISS
        try:
            self._hengine.update_confidence(rule, delta)
        except (AttributeError, TypeError):
            # HypothesisEngine API may vary; best-effort only
            logger.debug(
                "[hyp_experiment_planner] could not update confidence for %r: no compatible API",
                rule,
            )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _parse_rule(rule: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (cause, effect) from a rule string like 'X → Y' or 'X affects Y'."""
    # Arrow format
    m = __import__("re").search(r"(\w+)\s*[→->]+\s*(\w+)", rule)
    if m:
        return m.group(1).lower(), m.group(2).lower()
    # Verb format
    m = __import__("re").search(r"(\w+)\s+(?:affects|causes|influences)\s+(\w+)", rule, flags=2)
    if m:
        return m.group(1).lower(), m.group(2).lower()
    return None, None


def _choose_intervention(cause: str) -> str:
    """Return the best action to manipulate *cause*."""
    c = cause.lower()
    for key, actions in _CAUSE_TO_ACTIONS.items():
        if key in c or c in key:
            return random.choice(actions)
    return _DEFAULT_ACTION