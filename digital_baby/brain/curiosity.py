"""Curiosity model for topic selection and novelty rewards — v4.

Changes from v3
---------------
1. UCB1-BASED DOMAIN SELECTION (replaces score_topics()):
   score_topics_ucb() implements Upper Confidence Bound (UCB1) which provides
   a theoretically grounded exploration-exploitation tradeoff:
       UCB_score(domain) = mean_entropy(domain) + C * sqrt(log(total_ticks+1) / (visits+1))
   This guarantees no domain is permanently neglected and automatically
   increases exploration bonus as a domain goes unvisited.

2. MANDATORY ACTION POLICY (select_action()):
   New method that implements a 5-priority cascade and NEVER returns None:
   Priority 1: Test top hypothesis with targeted intervention
   Priority 2: Intervene on most uncertain epistemic edge
   Priority 3: Explore least-visited domain with random intervention
   Priority 4: Ingest pending concepts from queue
   Priority 5: Synthesise new derivative variable (always available)

3. ACTION-TO-CAUSE MAPPING:
   CAUSE_TO_ACTIONS and ACTION_TO_CAUSE dicts allow the exploration policy
   to translate between graph concepts and executable interventions, ensuring
   interventions directly target hypothesis cause variables.

4. EPISTEMIC ENTROPY TRACKING per domain:
   The curiosity model now accepts entropy updates from EpistemicStateTracker
   via register_domain_entropy() so UCB1 scores reflect actual uncertainty
   remaining in each domain.

All v3 public API is preserved.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_ACCUMULATOR: float = 50.0

_EP_WEIGHTS = {
    "entropy_reduced":    0.35,
    "new_edges":          0.25,
    "validation_results": 0.15,
    "law_novel":          0.15,
    "mediator_found":     0.10,
}

# UCB1 exploration constant: higher = more exploration of unvisited domains.
# Raised from 0.7 → 1.4 to force stronger rotation away from high-pred-error
# domains that have accumulated error but aren't making progress.
UCB1_C: float = 1.4

# All domains with simulation support (v4.1: expanded from 6 → 10)
ALL_DOMAINS: Tuple[str, ...] = (
    "ecosystem", "physics", "chemistry", "biology", "technology", "astronomy",
    "neuroscience", "climate", "economics", "materials",
)

# Cause concept → actions that manipulate it (v4.1: extended for new domains)
CAUSE_TO_ACTIONS: Dict[str, List[str]] = {
    # ecosystem
    "wolf":          ["add_predator", "remove_predator"],
    "deer":          ["introduce_species", "remove_species"],
    "season_factor": ["change_season"],
    # physics
    "force":         ["increase_force", "apply_impulse"],
    "heat":          ["add_heat"],
    "friction":      ["add_friction"],
    "momentum":      ["apply_impulse"],
    "mass":          ["reduce_mass"],
    # chemistry
    "temperature":   ["increase_temperature", "add_heat"],
    "reactants":     ["add_chemical"],
    "catalyst":      ["add_catalyst"],
    "pH":            ["adjust_pH"],
    # biology
    "pathogens":     ["add_pathogen"],
    "energy":        ["boost_energy"],
    "cells":         ["add_cells"],
    "toxin_level":   ["neutralise_toxin"],
    "antibodies":    ["add_antibody"],
    # technology
    "robot":         ["add_robot", "remove_robot"],
    "maintenance_load": ["increase_maintenance"],
    "sensor_threshold": ["upgrade_sensor"],
    # neuroscience (v4.1)
    "stress_level":       ["induce_stress", "reduce_stress"],
    "dopamine_level":     ["boost_dopamine"],
    "neural_activity":    ["stimulate_neurons"],
    "memory_consolidation": ["improve_sleep"],
    "cortisol":           ["reduce_stress"],
    "learning_rate":      ["boost_dopamine"],
    # climate (v4.1)
    "co2_level":          ["emit_co2"],
    "temperature_anomaly": ["emit_co2"],
    "glacier_melt":       ["melt_glacier"],
    "vegetation_cover":   ["plant_forest"],
    "albedo":             ["increase_albedo"],
    "ocean_heat":         ["warm_ocean"],
    # economics (v4.1)
    "interest_rate":      ["raise_interest_rate", "lower_interest_rate"],
    "consumption":        ["increase_spending"],
    "productivity":       ["boost_productivity"],
    "investment":         ["lower_interest_rate"],
    "debt_level":         ["add_debt"],
    # materials (v4.1)
    "stress_level_mat":   ["apply_stress"],   # disambiguated from neuroscience
    "hardness":           ["heat_treat", "quench"],
    "yield_strength":     ["quench", "anneal"],
    "porosity":           ["add_porosity"],
    "conductivity":       ["heat_treat"],
    "crack_growth":       ["apply_stress"],
}

ACTION_TO_CAUSE: Dict[str, str] = {
    action: cause
    for cause, actions in CAUSE_TO_ACTIONS.items()
    for action in actions
}

# Domain → default fallback actions for exploration (v4.1: extended)
DOMAIN_DEFAULT_ACTIONS: Dict[str, List[str]] = {
    "ecosystem":    ["add_predator", "introduce_species", "change_season"],
    "physics":      ["increase_force", "add_friction", "apply_impulse"],
    "chemistry":    ["increase_temperature", "add_catalyst", "adjust_pH"],
    "biology":      ["add_pathogen", "boost_energy", "add_cells"],
    "technology":   ["add_robot", "upgrade_sensor", "increase_maintenance"],
    "astronomy":    ["introduce_species", "remove_species"],
    # new domains (v4.1)
    "neuroscience": ["induce_stress", "boost_dopamine", "stimulate_neurons"],
    "climate":      ["emit_co2", "plant_forest", "warm_ocean"],
    "economics":    ["raise_interest_rate", "boost_productivity", "increase_spending"],
    "materials":    ["apply_stress", "heat_treat", "quench"],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CuriositySignal:
    topic:  str
    score:  float
    reason: str


@dataclass
class ActionSelection:
    """v4: Result of the mandatory action policy."""
    action:    str           # the intervention to execute
    subject:   str           # target subject/entity
    priority:  int           # 1-5 (which priority tier selected this)
    reason:    str           # human-readable explanation
    hypothesis_rule: Optional[str] = None   # if testing a specific hypothesis


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CuriosityModel:
    """Tracks unknowns, prediction error, novelty, and discovered patterns — v4."""

    UNKNOWN_CONCEPT_BONUS  = 5.0
    PREDICTION_ERROR_BONUS = 10.0
    NEW_RELATION_BONUS     = 3.0
    NEW_PATTERN_BONUS      = 7.0
    ERROR_THRESHOLD        = 0.3

    def __init__(self) -> None:
        self.topic_visits:                    Dict[str, int]      = defaultdict(int)
        self.unknown_concepts_by_topic:       Dict[str, Set[str]] = defaultdict(set)
        self.pending_concepts:                Deque[str]          = deque()
        self.prediction_error_by_topic:       Dict[str, float]    = defaultdict(float)
        self.structural_novelty_by_topic:     Dict[str, float]    = defaultdict(float)
        self.hypothesis_uncertainty_by_topic: Dict[str, float]    = defaultdict(float)
        self.experiment_signal_by_topic:      Dict[str, float]    = defaultdict(float)
        self.new_patterns_by_topic:           Dict[str, int]      = defaultdict(int)

        # v4 NEW: entropy per domain (fed from EpistemicStateTracker)
        self.domain_entropy:                  Dict[str, float]    = defaultdict(float)

        # v4 NEW: total tick count for UCB1 denominator
        self._total_ticks: int = 0

        # v4 NEW: recent actions for anti-repetition
        self._recent_actions: Deque[str] = deque(maxlen=10)

        # v4.1 NEW: consecutive-visit counter per domain for stale-domain reset.
        # When the same domain is visited >= _STALE_DOMAIN_THRESHOLD times in a
        # row without structural novelty, prediction_error is hard-reset so UCB1
        # can route attention elsewhere.
        self._consecutive_domain_visits: Dict[str, int] = defaultdict(int)
        self._last_visited_domain: str = ""
        self._STALE_DOMAIN_THRESHOLD: int = 3

    # ── Signal registration (v3 unchanged) ────────────────────────────────

    def curiosity_score(
        self,
        unknown_concepts: Iterable[str],
        prediction_error: float,
        unexplored_concepts: Iterable[str],
        new_relation: bool,
        new_pattern: bool = False,
    ) -> float:
        unknown_count    = len([c for c in unknown_concepts if c])
        unexplored_count = len([c for c in unexplored_concepts if c])
        score = 0.0
        if unknown_count > 0:
            score += self.UNKNOWN_CONCEPT_BONUS
        if prediction_error > self.ERROR_THRESHOLD:
            score += prediction_error * self.PREDICTION_ERROR_BONUS
        if unexplored_count > 0:
            score += self.UNKNOWN_CONCEPT_BONUS + (unexplored_count - 1)
        if new_relation:
            score += self.NEW_RELATION_BONUS
        if new_pattern:
            score += self.NEW_PATTERN_BONUS
        return score

    def register_unknowns(self, topic: str, unknown_concepts: Iterable[str]) -> None:
        clean = [c.strip().lower() for c in unknown_concepts if c.strip()]
        self.unknown_concepts_by_topic[topic].update(clean)
        for concept in clean:
            if concept not in self.pending_concepts:
                self.pending_concepts.append(concept)

    def register_prediction_error(self, topic: str, error: float) -> None:
        v = self.prediction_error_by_topic[topic]
        self.prediction_error_by_topic[topic] = min(_MAX_ACCUMULATOR, v + max(0.0, error))

    def register_new_pattern(self, topic: str) -> None:
        self.new_patterns_by_topic[topic] += 1

    def register_structural_novelty(self, topic: str, novelty: float) -> None:
        v = self.structural_novelty_by_topic[topic]
        self.structural_novelty_by_topic[topic] = min(_MAX_ACCUMULATOR, v + max(0.0, novelty))

    def register_hypothesis_uncertainty(self, topic: str, uncertainty: float) -> None:
        v = self.hypothesis_uncertainty_by_topic[topic]
        self.hypothesis_uncertainty_by_topic[topic] = min(_MAX_ACCUMULATOR, v + max(0.0, uncertainty))

    def register_experiment_signal(self, topic: str, signal: float) -> None:
        v = self.experiment_signal_by_topic[topic]
        self.experiment_signal_by_topic[topic] = min(_MAX_ACCUMULATOR, v + max(0.0, signal))

    def register_domain_entropy(self, domain: str, entropy: float) -> None:
        """v4 NEW: Record epistemic entropy for a domain (from EpistemicStateTracker)."""
        self.domain_entropy[domain] = max(0.0, entropy)

    def tick(self) -> None:
        """v4 NEW: Increment total tick counter (call once per event loop tick)."""
        self._total_ticks += 1

    # ── Reward functions (v3 unchanged) ───────────────────────────────────

    def reward(self, novelty: float, conflict_bonus: float = 0.0) -> float:
        return min(100.0, max(0.0, novelty + conflict_bonus))

    def epistemic_reward(
        self,
        entropy_reduced: float,
        new_edges: int,
        validation_results: list,
        law_novel: bool,
        mediator_found: bool,
    ) -> float:
        entropy_term = min(1.0, max(0.0, entropy_reduced) / 2.0)
        edge_term    = min(1.0, max(0, new_edges) / 5.0)
        val_term     = min(1.0, len(validation_results) / 3.0)
        law_term     = 1.0 if law_novel else 0.0
        med_term     = 1.0 if mediator_found else 0.0
        weights      = list(_EP_WEIGHTS.values())
        components   = [entropy_term, edge_term, val_term, law_term, med_term]
        return round(sum(w * c for w, c in zip(weights, components)), 4)

    # ── Topic selection ────────────────────────────────────────────────────

    def pop_goal_concept(self) -> Optional[str]:
        if not self.pending_concepts:
            return None
        return self.pending_concepts.popleft()

    def score_topics(
        self,
        topics: Iterable[str],
        weak_fact_ratio_by_topic: Dict[str, float],
    ) -> List[CuriositySignal]:
        """v3 scoring (kept for backward compatibility). Use score_topics_ucb() for v4."""
        signals: List[CuriositySignal] = []
        for topic in topics:
            visits = self.topic_visits[topic]
            novelty      = 1.0 / (1.0 + visits)
            visit_factor = 1.0 / ((visits + 1) ** 0.5)

            unknown_weight         = min(5.0, len(self.unknown_concepts_by_topic.get(topic, set())) * 0.3)
            weak_bonus             = min(3.0, weak_fact_ratio_by_topic.get(topic, 0.0))
            prediction_error       = min(10.0, self.prediction_error_by_topic.get(topic, 0.0))
            structural             = min(5.0,  self.structural_novelty_by_topic.get(topic, 0.0))
            hypothesis_uncertainty = min(5.0,  self.hypothesis_uncertainty_by_topic.get(topic, 0.0))
            experiment_signal      = min(5.0,  self.experiment_signal_by_topic.get(topic, 0.0))
            pattern_bonus          = min(3.0,  self.new_patterns_by_topic.get(topic, 0) * 0.5)

            raw   = (novelty + unknown_weight + weak_bonus + prediction_error
                     + structural + hypothesis_uncertainty + experiment_signal + pattern_bonus)
            score = max(0.0, raw * visit_factor)

            reason = (
                f"novelty={novelty:.2f}, unknown={unknown_weight:.2f}, weak={weak_bonus:.2f}, "
                f"pred_err={prediction_error:.2f}, structural={structural:.2f}, "
                f"hyp_unc={hypothesis_uncertainty:.2f}, exp={experiment_signal:.2f}, "
                f"pattern={pattern_bonus:.2f}, visits={visits}, vfactor={visit_factor:.3f}"
            )
            signals.append(CuriositySignal(topic=topic, score=score, reason=reason))

        return sorted(signals, key=lambda s: s.score, reverse=True)

    def score_topics_ucb(
        self,
        topics: Iterable[str],
        weak_fact_ratio_by_topic: Dict[str, float],
        c: float = UCB1_C,
    ) -> List[CuriositySignal]:
        """v4 NEW: UCB1-based topic selection.

        UCB_score(topic) = exploitation_score(topic)
                         + C * sqrt(log(total_ticks+1) / (visits+1))

        exploitation_score = weighted sum of entropy, prediction error, etc.
        exploration_bonus  = UCB1 term that grows as a topic goes unvisited.

        This guarantees every topic gets periodically revisited, and ensures
        the agent shifts attention to high-entropy domains automatically.
        """
        signals: List[CuriositySignal] = []
        log_total = math.log(self._total_ticks + 2)  # +2 avoids log(1)=0 weirdness

        for topic in topics:
            visits = self.topic_visits[topic]

            # Exploitation: how much do we currently know we could learn here?
            entropy_term = min(1.0, self.domain_entropy.get(topic, 0.5))
            pred_err     = min(1.0, self.prediction_error_by_topic.get(topic, 0.0) / 10.0)
            hyp_unc      = min(1.0, self.hypothesis_uncertainty_by_topic.get(topic, 0.0) / 5.0)
            unknown      = min(1.0, len(self.unknown_concepts_by_topic.get(topic, set())) * 0.1)
            exp_signal   = min(1.0, self.experiment_signal_by_topic.get(topic, 0.0) / 5.0)

            exploitation = (
                0.35 * entropy_term
                + 0.25 * pred_err
                + 0.20 * hyp_unc
                + 0.10 * unknown
                + 0.10 * exp_signal
            )

            # Exploration: UCB1 bonus grows as the topic goes unvisited
            exploration = c * math.sqrt(log_total / (visits + 1))

            score = exploitation + exploration

            reason = (
                f"UCB1: exploit={exploitation:.3f} explore={exploration:.3f} "
                f"entropy={entropy_term:.2f} pred_err={pred_err:.2f} "
                f"hyp_unc={hyp_unc:.2f} visits={visits}"
            )
            signals.append(CuriositySignal(topic=topic, score=score, reason=reason))

        return sorted(signals, key=lambda s: s.score, reverse=True)

    # ── v4 NEW: Mandatory action policy ───────────────────────────────────

    def select_action(
        self,
        hypotheses:      List,   # List[Hypothesis]
        epistemic,               # EpistemicStateTracker
        experiment_planner,      # ExperimentPlanner
        current_tick:    int,
        available_domains: Tuple[str, ...] = ALL_DOMAINS,
    ) -> ActionSelection:
        """v4 NEW: Mandatory action selection — NEVER returns action=none.

        Priority cascade:
        1. Test top hypothesis with targeted intervention
        2. Intervene on most uncertain epistemic edge
        3. Explore least-visited domain with random intervention
        4. Ingest pending concept from queue
        5. Synthesise new derivative variable (always available)
        """
        import random

        # ── Priority 1: Test top testable hypothesis ───────────────────
        testable = [h for h in hypotheses if getattr(h, "is_testable", True)]
        if testable and experiment_planner is not None:
            try:
                ranked = experiment_planner.rank(
                    testable, self, current_tick,
                    topic=self._top_topic(available_domains),
                )
                if ranked:
                    top_hyp = ranked[0].hypothesis
                    concepts = getattr(top_hyp, "concepts", [])
                    if concepts:
                        cause   = concepts[0]
                        actions = CAUSE_TO_ACTIONS.get(cause, [])
                        # Filter recently used actions
                        fresh = [a for a in actions if a not in self._recent_actions]
                        action = (random.choice(fresh) if fresh
                                  else (random.choice(actions) if actions else None))
                        if action:
                            self._recent_actions.append(action)
                            return ActionSelection(
                                action=action, subject=cause, priority=1,
                                reason=f"testing_hypothesis:{top_hyp.rule[:40]}",
                                hypothesis_rule=top_hyp.rule,
                            )
            except Exception:
                pass

        # ── Priority 2: Intervene on most uncertain epistemic edge ─────
        if epistemic is not None:
            try:
                top_uncertain = epistemic.top_uncertain_causes(n=5)
                for cause in top_uncertain:
                    if epistemic.is_stale(cause, current_tick):
                        continue
                    actions = CAUSE_TO_ACTIONS.get(cause, [])
                    fresh   = [a for a in actions if a not in self._recent_actions]
                    action  = random.choice(fresh) if fresh else (random.choice(actions) if actions else None)
                    if action:
                        self._recent_actions.append(action)
                        return ActionSelection(
                            action=action, subject=cause, priority=2,
                            reason=f"epistemic_uncertainty:{cause}",
                        )
            except Exception:
                pass

        # ── Priority 3: Explore least-visited domain ───────────────────
        topic_scores = self.score_topics_ucb(available_domains, {})
        if topic_scores:
            # Pick the highest-UCB1 domain
            best_domain = topic_scores[0].topic
            domain_actions = DOMAIN_DEFAULT_ACTIONS.get(best_domain, [])
            fresh = [a for a in domain_actions if a not in self._recent_actions]
            if fresh:
                action = random.choice(fresh)
                cause  = ACTION_TO_CAUSE.get(action, action)
                self._recent_actions.append(action)
                return ActionSelection(
                    action=action, subject=cause, priority=3,
                    reason=f"explore_domain:{best_domain} ucb={topic_scores[0].score:.3f}",
                )

        # ── Priority 4: Ingest pending concept ────────────────────────
        if self.pending_concepts:
            concept = self.pending_concepts[0]  # peek without removing
            return ActionSelection(
                action="INGEST", subject=concept, priority=4,
                reason=f"ingest_pending_concept:{concept}",
            )

        # ── Priority 5: Synthesise (always available) ──────────────────
        # Generate a derivative variable from the most-visited domain
        synth_domain = topic_scores[-1].topic if topic_scores else "physics"
        return ActionSelection(
            action="SYNTHESISE", subject=synth_domain, priority=5,
            reason=f"synthesise_new_variable_in:{synth_domain}",
        )

    def _top_topic(self, domains: Tuple[str, ...]) -> str:
        scores = self.score_topics_ucb(domains, {})
        return scores[0].topic if scores else domains[0]

    # ── mark_visited (v3 unchanged) ───────────────────────────────────────

    def mark_visited(self, topic: str) -> None:
        visits = self.topic_visits[topic]
        self.topic_visits[topic] = visits + 1

        # ── Stale-domain consecutive-visit tracking ────────────────────────
        # If the same domain is visited >= threshold times in a row with no
        # structural novelty improvement, hard-reset its prediction_error so
        # UCB1 stops routing all attention to it.
        if topic == self._last_visited_domain:
            self._consecutive_domain_visits[topic] += 1
        else:
            # Reset counter for the previously visited domain
            if self._last_visited_domain:
                self._consecutive_domain_visits[self._last_visited_domain] = 0
            self._consecutive_domain_visits[topic] = 1
            self._last_visited_domain = topic

        if self._consecutive_domain_visits[topic] >= self._STALE_DOMAIN_THRESHOLD:
            # Hard-reset: zero out pred error and novelty so UCB1 must explore
            self.prediction_error_by_topic[topic]       = 0.0
            self.structural_novelty_by_topic[topic]     = 0.0
            self.hypothesis_uncertainty_by_topic[topic] = 0.0
            self._consecutive_domain_visits[topic]       = 0   # allow re-accumulation
        # ──────────────────────────────────────────────────────────────────

        if topic in self.unknown_concepts_by_topic:
            self.unknown_concepts_by_topic[topic].clear()
        decay = max(0.3, 0.8 - (visits * 0.05))
        self.prediction_error_by_topic[topic]       *= decay
        self.structural_novelty_by_topic[topic]     *= decay * 0.75
        self.hypothesis_uncertainty_by_topic[topic] *= decay
        self.experiment_signal_by_topic[topic]      *= decay
        self.new_patterns_by_topic[topic]            = max(0, self.new_patterns_by_topic[topic] - 1)