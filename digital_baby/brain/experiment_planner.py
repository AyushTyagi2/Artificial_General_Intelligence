"""Information-gain + discriminating experiment planner — v4.

Changes from v3
---------------
1. EXPLICIT EXPERIMENT RECORDS:
   plan_experiment() returns an ExperimentPlan that captures the full
   intended experiment: which hypothesis is being tested, which action to
   execute, the predicted outcome sign, and which modality to use.  The event
   loop attaches the observed outcome and passes it back via record_outcome()
   so the planner can track prediction accuracy per hypothesis.

2. PAIRED EXPERIMENT SUPPORT:
   ExperimentPlan carries a 'use_paired' flag.  When True the event loop
   should run the world twice from the same base state (control + treatment)
   and pass the causal delta to InterventionCausalDiscovery.analyze_paired().
   Paired experiments are recommended for hypotheses that have been tested
   >= PAIRED_THRESHOLD times without resolution.

3. HYPOTHESIS-TARGETED ACTION SELECTION:
   select_action_for_hypothesis() maps from hypothesis cause variable to the
   best available action using CAUSE_TO_ACTIONS, filtered for anti-repetition.
   This replaces the previous loose coupling where the event loop picked actions
   independently of the hypothesis being tested.

4. STALENESS BUDGET:
   Each hypothesis gets a staleness budget: how many more ticks before it is
   forcibly suspended.  Hypotheses that consume their budget without resolving
   are marked for suspension in the next lifecycle pass.

5. PREDICTION ACCURACY TRACKING:
   record_outcome() logs whether the predicted sign matched the observed sign.
   The accuracy rate per hypothesis is used to boost/penalise the IG score:
   a hypothesis with 30% accuracy is probably wrong, while one with 80%
   accuracy is nearly confirmed.

Scoring formula (v4)
--------------------
    score(h) = α·IG(h) + β·curiosity(h) + γ·uncertainty(h)
             + δ·age_bonus(h) + ε·discriminating_score(h) + ζ·accuracy_boost(h)

Default weights: α=0.30  β=0.18  γ=0.17  δ=0.12  ε=0.13  ζ=0.10
"""

from __future__ import annotations

import logging
import math
import random as _random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

IG_WEIGHT:   float = 0.30
CUR_WEIGHT:  float = 0.18
UNC_WEIGHT:  float = 0.17
AGE_WEIGHT:  float = 0.12
DISC_WEIGHT: float = 0.13
ACC_WEIGHT:  float = 0.10

AGE_BONUS_SATURATION: int   = 50
RECENT_ACTION_WINDOW: int   = 5
REPEAT_PENALTY:       float = 0.30
PAIRED_THRESHOLD:     int   = 5      # tests before recommending paired experiment
STALENESS_BUDGET:     int   = 25     # ticks before a hypothesis is suspended

# Map from cause concept → list of actions that manipulate it
CAUSE_TO_ACTIONS: Dict[str, List[str]] = {
    "wolf":          ["add_predator", "remove_predator"],
    "deer":          ["introduce_species", "remove_species"],
    "force":         ["increase_force", "apply_impulse"],
    "temperature":   ["increase_temperature", "add_heat"],
    "reactants":     ["add_chemical"],
    "catalyst":      ["add_catalyst"],
    "pH":            ["adjust_pH"],
    "robot":         ["add_robot", "remove_robot"],
    "pathogens":     ["add_pathogen"],
    "energy":        ["boost_energy"],
    "cells":         ["add_cells"],
    "toxin_level":   ["neutralise_toxin"],
    "antibodies":    ["add_antibody"],
    "friction":      ["add_friction"],
    "momentum":      ["apply_impulse"],
    "mass":          ["reduce_mass"],
    "season_factor": ["change_season"],
    "heat":          ["add_heat"],
    "maintenance_load": ["increase_maintenance"],
    "sensor_threshold": ["upgrade_sensor"],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExperimentScore:
    """Scored hypothesis ready for selection."""
    hypothesis:           object
    total_score:          float
    information_gain:     float
    curiosity_weight:     float
    uncertainty_weight:   float
    age_bonus:            float
    discriminating_score: float
    accuracy_boost:       float
    last_tested_tick:     int
    staleness_remaining:  int    # v4 NEW

    def __repr__(self) -> str:
        rule = getattr(self.hypothesis, "rule", "?")
        return (
            f"ExperimentScore(rule={rule!r:.40}, total={self.total_score:.3f}, "
            f"ig={self.information_gain:.3f}, unc={self.uncertainty_weight:.3f}, "
            f"acc={self.accuracy_boost:.3f}, stale_left={self.staleness_remaining})"
        )


@dataclass
class ExperimentPlan:
    """v4 NEW: A fully specified experiment plan.

    The event loop executes this plan by:
    1. Selecting `action` in domain `domain`
    2. If `use_paired`: run world twice from same state (control + treatment)
       and call causal_discovery.analyze_paired()
       Else: run world once and call causal_discovery.analyze()
    3. Call planner.record_outcome(plan, observed_sign, tick)
    """
    hypothesis_rule:  str
    cause_concept:    str
    effect_concept:   str
    action:           str
    predicted_sign:   int       # +1 = predicted increase, -1 = predicted decrease
    domain:           str
    use_paired:       bool      # v4: recommend paired experiment
    tick_planned:     int
    modality:         str = "direct_intervention"


@dataclass
class OutcomeRecord:
    """v4 NEW: Tracks prediction accuracy per hypothesis."""
    predictions: int = 0
    correct:     int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / max(1, self.predictions)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ExperimentPlanner:
    """Ranks hypotheses by information gain + discriminating power — v4."""

    def __init__(
        self,
        ig_weight:     float = IG_WEIGHT,
        cur_weight:    float = CUR_WEIGHT,
        uncert_weight: float = UNC_WEIGHT,
        age_weight:    float = AGE_WEIGHT,
        disc_weight:   float = DISC_WEIGHT,
        acc_weight:    float = ACC_WEIGHT,
    ) -> None:
        total = ig_weight + cur_weight + uncert_weight + age_weight + disc_weight + acc_weight
        if abs(total - 1.0) > 0.02:
            logger.warning("[planner] weights sum to %.3f — normalising", total)
            ig_weight    /= total; cur_weight  /= total; uncert_weight /= total
            age_weight   /= total; disc_weight /= total; acc_weight    /= total

        self.ig_weight     = ig_weight
        self.cur_weight    = cur_weight
        self.uncert_weight = uncert_weight
        self.age_weight    = age_weight
        self.disc_weight   = disc_weight
        self.acc_weight    = acc_weight

        self._last_tested:     Dict[str, int]        = {}
        self._test_count:      Dict[str, int]        = defaultdict(int)
        self._recent_actions:  Deque[str]            = deque(maxlen=RECENT_ACTION_WINDOW)
        self._outcomes:        Dict[str, OutcomeRecord] = defaultdict(OutcomeRecord)

    # ── Public API ─────────────────────────────────────────────────────────

    def rank(
        self,
        hypotheses:       List,
        curiosity_model,
        current_tick:     int,
        topic:            str,
        domain_state:     Optional[Dict[str, float]] = None,
        causal_rules:     Optional[List] = None,
        preferred_action: Optional[str] = None,
    ) -> List[ExperimentScore]:
        """Score and rank hypotheses by expected learning value — v4."""
        if not hypotheses:
            return []

        # Only rank testable hypotheses
        testable = [h for h in hypotheses if getattr(h, "is_testable", True)]
        if not testable:
            return []

        pred_error = 0.0
        if curiosity_model is not None:
            pred_error = getattr(curiosity_model, "prediction_error_by_topic", {}).get(topic, 0.0)

        scores: List[ExperimentScore] = []
        top_k  = testable[:8]

        for hyp in testable:
            confidence = float(getattr(hyp, "confidence", 0.5))
            rule       = getattr(hyp, "rule", "")
            concepts   = getattr(hyp, "concepts", [])

            ig    = self._information_gain(confidence)
            cur   = self._curiosity_score(hyp, concepts, pred_error, curiosity_model)
            unc   = self._uncertainty(confidence)
            age   = self._age_bonus(rule, current_tick)
            disc  = self._discriminating_score(hyp, top_k, preferred_action, domain_state, causal_rules)
            acc   = self._accuracy_boost(rule)

            # Apply anti-repetition penalty
            cause = concepts[0] if concepts else ""
            actions = CAUSE_TO_ACTIONS.get(cause, [])
            rep_penalty = min(1.0, sum(
                self.action_repetition_penalty(a) for a in actions
            ) / max(1, len(actions)))

            total = (
                self.ig_weight     * ig
                + self.cur_weight  * cur
                + self.uncert_weight * unc
                + self.age_weight  * age
                + self.disc_weight * disc
                + self.acc_weight  * acc
            ) * (1.0 - 0.4 * rep_penalty)

            # Staleness: how many more ticks before forced suspension
            tests_done = self._test_count.get(rule, 0)
            staleness_remaining = max(0, STALENESS_BUDGET - tests_done)

            scores.append(ExperimentScore(
                hypothesis=hyp,
                total_score=total,
                information_gain=ig,
                curiosity_weight=cur,
                uncertainty_weight=unc,
                age_bonus=age,
                discriminating_score=disc,
                accuracy_boost=acc,
                last_tested_tick=self._last_tested.get(rule, 0),
                staleness_remaining=staleness_remaining,
            ))

            logger.debug(
                "[planner] hyp=%r total=%.3f ig=%.3f unc=%.3f age=%.3f disc=%.3f acc=%.3f stale=%d",
                rule[:40], total, ig, unc, age, disc, acc, staleness_remaining,
            )

        return sorted(scores, key=lambda s: s.total_score, reverse=True)

    def plan_experiment(
        self,
        hypothesis,
        current_tick:  int,
        domain:        str,
        causal_records: Optional[Dict] = None,
    ) -> Optional[ExperimentPlan]:
        """v4 NEW: Build a fully specified ExperimentPlan for a hypothesis.

        Selects the best action to test the hypothesis cause variable,
        predicts the expected outcome sign from accumulated causal evidence,
        and recommends paired experiment mode when the hypothesis has been
        tested several times without resolution.
        """
        concepts = getattr(hypothesis, "concepts", [])
        if len(concepts) < 2:
            return None

        cause, effect = concepts[0], concepts[1]
        rule          = hypothesis.rule

        # Select the best available action for this cause
        action = self.select_action_for_hypothesis(hypothesis)
        if action is None:
            return None

        # Predict the sign from accumulated causal records
        predicted_sign = self._predict_sign(cause, effect, hypothesis, causal_records or {})

        # Recommend paired experiment when this hypothesis has been tested
        # several times without resolving (reduces noise, improves signal)
        tests_done = self._test_count.get(rule, 0)
        use_paired = (tests_done >= PAIRED_THRESHOLD)

        return ExperimentPlan(
            hypothesis_rule=rule,
            cause_concept=cause,
            effect_concept=effect,
            action=action,
            predicted_sign=predicted_sign,
            domain=domain,
            use_paired=use_paired,
            tick_planned=current_tick,
        )

    def select_action_for_hypothesis(self, hypothesis) -> Optional[str]:
        """v4 NEW: Choose the best action to test a hypothesis's cause variable.

        Filters for non-recently-used actions (anti-repetition).
        Falls back to any available action if all have been used recently.
        """
        concepts = getattr(hypothesis, "concepts", [])
        if not concepts:
            return None
        cause   = concepts[0]
        actions = CAUSE_TO_ACTIONS.get(cause, [])
        if not actions:
            return None

        # Prefer actions not recently taken
        fresh = [a for a in actions if a not in self._recent_actions]
        if fresh:
            return _random.choice(fresh)
        # All recently used — pick one with least repetitions
        counts = {a: sum(1 for x in self._recent_actions if x == a) for a in actions}
        return min(counts, key=counts.get)

    def record_tested(self, rule: str, tick: int) -> None:
        """Record that a hypothesis was tested (resets age bonus, increments count)."""
        self._last_tested[rule] = tick
        self._test_count[rule]  = self._test_count.get(rule, 0) + 1

    def record_action(self, action: str) -> None:
        self._recent_actions.append(action)

    def record_outcome(
        self,
        plan:          ExperimentPlan,
        observed_sign: int,
        tick:          int,
    ) -> None:
        """v4 NEW: Record whether a prediction was correct.

        Updates the prediction accuracy tracker for the tested hypothesis.
        """
        outcome = self._outcomes[plan.hypothesis_rule]
        outcome.predictions += 1
        if observed_sign == plan.predicted_sign:
            outcome.correct += 1
        self.record_tested(plan.hypothesis_rule, tick)
        self.record_action(plan.action)

        logger.debug(
            "[planner] outcome rule=%r predicted=%+d observed=%+d correct=%s accuracy=%.2f",
            plan.hypothesis_rule[:40], plan.predicted_sign, observed_sign,
            observed_sign == plan.predicted_sign,
            outcome.accuracy,
        )

    def action_repetition_penalty(self, action: str) -> float:
        count = sum(1 for a in self._recent_actions if a == action)
        return min(1.0, count * REPEAT_PENALTY)

    def top_hypothesis(
        self,
        hypotheses:   List,
        curiosity_model,
        current_tick: int,
        topic:        str,
    ) -> Optional[object]:
        ranked = self.rank(hypotheses, curiosity_model, current_tick, topic)
        return ranked[0].hypothesis if ranked else None

    def get_stale_hypotheses(
        self, hypotheses: List, current_tick: int
    ) -> List[object]:
        """v4 NEW: Return hypotheses that have exhausted their staleness budget."""
        stale = []
        for hyp in hypotheses:
            rule  = getattr(hyp, "rule", "")
            tests = self._test_count.get(rule, 0)
            if tests >= STALENESS_BUDGET:
                stale.append(hyp)
        return stale

    def prediction_accuracy(self, rule: str) -> float:
        """v4 NEW: Return prediction accuracy for a hypothesis rule."""
        return self._outcomes[rule].accuracy if rule in self._outcomes else 0.5

    # ── Scoring components ─────────────────────────────────────────────────

    @staticmethod
    def _information_gain(confidence: float) -> float:
        """Expected reduction in binary entropy from one experiment."""
        p = max(1e-9, min(1.0 - 1e-9, confidence))

        def h(prob: float) -> float:
            prob = max(1e-9, min(1.0 - 1e-9, prob))
            return -prob * math.log2(prob) - (1.0 - prob) * math.log2(1.0 - prob)

        h_prior     = h(p)
        p_confirmed = p * 0.8 + 0.1
        p_refuted   = p * 0.2
        h_posterior = 0.5 * h(p_confirmed) + 0.5 * h(p_refuted)
        return max(0.0, h_prior - h_posterior)

    @staticmethod
    def _uncertainty(confidence: float) -> float:
        return 1.0 - abs(confidence - 0.5) * 2.0

    def _age_bonus(self, rule: str, current_tick: int) -> float:
        last = self._last_tested.get(rule, 0)
        return min(1.0, (current_tick - last) / AGE_BONUS_SATURATION)

    def _accuracy_boost(self, rule: str) -> float:
        """v4 NEW: Boost score for hypotheses with a clear accuracy signal.

        High accuracy (>0.75) → hypothesis is probably real, boost to confirm.
        Low accuracy (<0.25)  → hypothesis is probably wrong, boost to refute.
        Near 0.5              → no signal, no boost.
        """
        if rule not in self._outcomes:
            return 0.0   # no data yet
        acc = self._outcomes[rule].accuracy
        # Peaks at acc=1.0 and acc=0.0, zero at acc=0.5
        return abs(acc - 0.5) * 2.0

    @staticmethod
    def _curiosity_score(
        hyp,
        concepts:       List[str],
        pred_error:     float,
        curiosity_model,
    ) -> float:
        if curiosity_model is None:
            return 0.0
        try:
            visited    = getattr(curiosity_model, "topic_visits", {})
            unexplored = [c for c in concepts if not visited.get(c, 0)]
            raw        = curiosity_model.curiosity_score(
                unknown_concepts=unexplored,
                prediction_error=pred_error,
                unexplored_concepts=unexplored,
                new_relation=False,
                new_pattern=False,
            )
            return min(1.0, raw / 25.0)
        except Exception as exc:
            logger.debug("[planner] curiosity_score_failed error=%s", exc)
            return 0.0

    def _discriminating_score(
        self,
        hyp,
        pool:             List,
        preferred_action: Optional[str],
        domain_state:     Optional[Dict[str, float]],
        causal_rules:     Optional[List],
    ) -> float:
        if preferred_action is None or len(pool) < 2:
            return 0.0

        p_i = self._predict_outcome(hyp, preferred_action, domain_state, causal_rules)
        disc_values: List[float] = []

        for peer in pool:
            if peer is hyp:
                continue
            p_j      = self._predict_outcome(peer, preferred_action, domain_state, causal_rules)
            sep      = abs(p_i - p_j)
            peer_unc = self._uncertainty(float(getattr(peer, "confidence", 0.5)))
            disc_values.append(sep * peer_unc)

        if not disc_values:
            return 0.0
        return min(1.0, sum(disc_values) / len(disc_values))

    @staticmethod
    def _predict_outcome(
        hyp,
        action:       str,
        domain_state: Optional[Dict[str, float]],
        causal_rules: Optional[List],
    ) -> float:
        concepts = getattr(hyp, "concepts", [])
        if not concepts:
            return 0.5

        cause  = concepts[0] if concepts else ""
        effect = concepts[1] if len(concepts) > 1 else ""

        _ACTION_CAUSE = {
            "increase_temperature": "temperature",
            "add_chemical":         "reactants",
            "add_catalyst":         "catalyst",
            "adjust_pH":            "pH",
            "increase_force":       "force",
            "add_heat":             "heat",
            "add_friction":         "friction",
            "apply_impulse":        "force",
            "add_predator":         "wolf",
            "remove_predator":      "wolf",
            "introduce_species":    "deer",
            "remove_species":       "deer",
            "add_robot":            "robot",
            "remove_robot":         "robot",
            "add_pathogen":         "pathogens",
            "boost_energy":         "energy",
            "add_cells":            "cells",
            "neutralise_toxin":     "toxin_level",
        }

        action_cause = _ACTION_CAUSE.get(action, "")
        if action_cause != cause:
            return 0.5

        if causal_rules:
            for rule in causal_rules:
                rc = getattr(rule, "cause",  "")
                re = getattr(rule, "effect", "")
                rd = getattr(rule, "direction", "")
                if rc == cause and re == effect:
                    if rd == "positive":
                        return 0.85
                    if rd == "negative":
                        return 0.15
                    if rd in ("bidirectional", "mixed", "conditional"):
                        return 0.50
        return 0.5

    def _predict_sign(
        self,
        cause:          str,
        effect:         str,
        hypothesis,
        causal_records: Dict,
    ) -> int:
        """v4 NEW: Predict the expected sign (+1/-1) for a cause→effect intervention.

        Priority:
        1. Existing CausalRecord with clear direction
        2. Hypothesis confidence (>0.5 → positive, <=0.5 → negative)
        3. Default: +1 (predict positive)
        """
        record = causal_records.get((cause, effect))
        if record is not None:
            direction = getattr(record, "direction", "unknown")
            if direction == "positive":
                return +1
            if direction == "negative":
                return -1

        confidence = float(getattr(hypothesis, "confidence", 0.5))
        if "decreases" in hypothesis.rule.lower():
            return -1
        if "increases" in hypothesis.rule.lower():
            return +1
        return +1 if confidence >= 0.5 else -1