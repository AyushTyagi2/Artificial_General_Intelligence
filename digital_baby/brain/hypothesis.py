"""Hypothesis generation, scoring, lifecycle management — v4.

Changes from v3
---------------
1. HYPOTHESIS LIFECYCLE STATUS:
   Hypothesis now carries a 'status' field with values:
   CANDIDATE → ACTIVE → CONFIRMED / REFUTED / SUSPENDED / ABANDONED
   - CONFIRMED: confidence >= 0.70, evidence >= 5
   - REFUTED:   confidence <= 0.25, evidence >= 5
   - SUSPENDED: ambiguous (0.35–0.55) after 10+ tests
   - ABANDONED: ambiguous after 20+ tests
   The ExperimentPlanner and HypothesisValidator both respect these statuses
   to avoid wasting cycles on unresolvable relationships.

2. STAGNATION DETECTOR (HypothesisStagnationDetector):
   Tracks the confidence trajectory of each hypothesis over a rolling window.
   Returns True when confidence has moved less than STAGNATION_THRESHOLD over
   the last N tests, signalling that more of the same experiments will not help.

3. HYPOTHESIS MUTATION (mutate_abandoned):
   When a hypothesis is ABANDONED, generates mutated variants by:
   (a) swapping the cause to a graph neighbour of the same effect
   (b) inserting a potential mediator between cause and effect
   These are seeded at confidence=0.5 (agnostic) and added to the test queue.

4. RETRY POLICY (HypothesisRetryPolicy):
   Suspended hypotheses can be retried after RETRY_COOLDOWN_TICKS ticks,
   but only using a different experimental modality than the one that led to
   suspension.  Cycles through: direct_intervention → paired_experiment →
   natural_observation → direct_intervention.

All v3 public API is preserved.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .concepts import ConceptTypeSystem
from .memory import CausalRuleRecord
from .patterns import PatternRule


# ---------------------------------------------------------------------------
# Lifecycle constants
# ---------------------------------------------------------------------------

class HypothesisStatus(str, Enum):
    CANDIDATE  = "candidate"
    ACTIVE     = "active"
    CONFIRMED  = "confirmed"
    REFUTED    = "refuted"
    SUSPENDED  = "suspended"
    ABANDONED  = "abandoned"

CONFIRMED_CONFIDENCE:  float = 0.70
REFUTED_CONFIDENCE:    float = 0.25
MIN_DECISIVE_EVIDENCE: int   = 5
MIN_SUSPEND_TESTS:     int   = 10
MIN_ABANDON_TESTS:     int   = 20
AMBIGUOUS_LO:          float = 0.35
AMBIGUOUS_HI:          float = 0.55

# Stagnation: confidence must change by less than this over the window
STAGNATION_THRESHOLD: float = 0.02
STAGNATION_WINDOW:    int   = 10

# Retry policy
RETRY_COOLDOWN_TICKS: int   = 50
_MODALITY_CYCLE: Tuple[str, ...] = (
    "direct_intervention", "paired_experiment", "natural_observation"
)


# ---------------------------------------------------------------------------
# Hypothesis dataclass (v4: adds status, test_count, last_tested_tick)
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    rule:                   str
    concepts:               List[str]
    confidence:             float
    supporting_evidence:    int
    contradicting_evidence: int
    status:                 str  = HypothesisStatus.ACTIVE   # v4 NEW
    test_count:             int  = 0                          # v4 NEW
    last_tested_tick:       int  = 0                          # v4 NEW
    last_modality:          str  = "direct_intervention"      # v4 NEW

    @property
    def total_evidence(self) -> int:
        return self.supporting_evidence + self.contradicting_evidence

    def update_status(self) -> None:
        """v4: Recompute status from current confidence and evidence counts."""
        if self.total_evidence >= MIN_DECISIVE_EVIDENCE:
            if self.confidence >= CONFIRMED_CONFIDENCE:
                self.status = HypothesisStatus.CONFIRMED
                return
            if self.confidence <= REFUTED_CONFIDENCE:
                self.status = HypothesisStatus.REFUTED
                return

        if AMBIGUOUS_LO <= self.confidence <= AMBIGUOUS_HI:
            if self.test_count >= MIN_ABANDON_TESTS:
                self.status = HypothesisStatus.ABANDONED
                return
            if self.test_count >= MIN_SUSPEND_TESTS:
                self.status = HypothesisStatus.SUSPENDED
                return

        self.status = HypothesisStatus.ACTIVE

    @property
    def is_testable(self) -> bool:
        """Return True if this hypothesis should be considered for testing."""
        return self.status in (
            HypothesisStatus.CANDIDATE,
            HypothesisStatus.ACTIVE,
        )


# ---------------------------------------------------------------------------
# Stagnation detector
# ---------------------------------------------------------------------------

class HypothesisStagnationDetector:
    """Tracks confidence trajectory to detect stuck hypotheses.

    Usage:
        detector = HypothesisStagnationDetector()
        detector.record(hyp.rule, hyp.confidence)
        if detector.is_stagnant(hyp.rule):
            hyp.status = HypothesisStatus.SUSPENDED
    """

    def __init__(
        self,
        window:    int   = STAGNATION_WINDOW,
        threshold: float = STAGNATION_THRESHOLD,
    ) -> None:
        self._window    = window
        self._threshold = threshold
        self._history:  Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )

    def record(self, rule: str, confidence: float) -> None:
        self._history[rule].append(confidence)

    def is_stagnant(self, rule: str) -> bool:
        hist = self._history[rule]
        if len(hist) < self._window:
            return False
        spread = max(hist) - min(hist)
        return spread < self._threshold

    def stagnant_rules(self, rules: Iterable[str]) -> List[str]:
        return [r for r in rules if self.is_stagnant(r)]

    def reset(self, rule: str) -> None:
        """Call after a hypothesis is mutated or retired."""
        self._history.pop(rule, None)


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------

class HypothesisRetryPolicy:
    """Controls when and how suspended hypotheses may be retried.

    When a hypothesis is suspended, the next retry must use a different
    experimental modality (direct_intervention → paired_experiment →
    natural_observation → ...) to generate diverse evidence.
    """

    def __init__(self) -> None:
        self._suspension_tick: Dict[str, int] = {}
        self._retry_modality:  Dict[str, str] = {}

    def suspend(
        self, rule: str, current_tick: int, last_modality: str
    ) -> None:
        self._suspension_tick[rule] = current_tick
        self._retry_modality[rule]  = self._next_modality(last_modality)

    def is_eligible_for_retry(self, rule: str, current_tick: int) -> bool:
        suspended_at = self._suspension_tick.get(rule, 0)
        return (current_tick - suspended_at) >= RETRY_COOLDOWN_TICKS

    def required_modality(self, rule: str) -> str:
        return self._retry_modality.get(rule, _MODALITY_CYCLE[0])

    @staticmethod
    def _next_modality(current: str) -> str:
        idx = _MODALITY_CYCLE.index(current) if current in _MODALITY_CYCLE else 0
        return _MODALITY_CYCLE[(idx + 1) % len(_MODALITY_CYCLE)]


# ---------------------------------------------------------------------------
# Hypothesis mutation
# ---------------------------------------------------------------------------

def mutate_abandoned_hypothesis(
    hyp: Hypothesis,
    knowledge_graph,
    max_mutations: int = 3,
) -> List[Hypothesis]:
    """v4: Generate mutated variants of an abandoned hypothesis.

    Two mutation strategies:
    1. Cause swap: replace cause with a graph neighbour that also affects effect.
    2. Mediator insert: reformulate as a conditional hypothesis with a mediator.

    All mutations start at confidence=0.5 (agnostic) and test_count=0.
    """
    if len(hyp.concepts) < 2:
        return []

    cause, effect = hyp.concepts[0], hyp.concepts[1]
    mutations: List[Hypothesis] = []

    # Strategy 1: swap cause to a different concept that neighbours effect
    try:
        effect_neighbours = knowledge_graph.get_neighbors(effect)
        for alt_cause in effect_neighbours[:5]:
            if alt_cause == cause or alt_cause == effect:
                continue
            mutations.append(Hypothesis(
                rule=f"if {alt_cause} increases then {effect} changes",
                concepts=[alt_cause, effect],
                confidence=0.50,
                supporting_evidence=0,
                contradicting_evidence=0,
                status=HypothesisStatus.CANDIDATE,
                test_count=0,
            ))
            if len(mutations) >= max_mutations:
                return mutations
    except Exception:
        pass

    # Strategy 2: find a common neighbour as potential mediator
    try:
        cause_neighbours  = set(knowledge_graph.get_neighbors(cause))
        effect_neighbours = set(knowledge_graph.get_neighbors(effect))
        mediators = cause_neighbours & effect_neighbours - {cause, effect}
        for m in list(mediators)[:2]:
            mutations.append(Hypothesis(
                rule=f"if {cause} increases and {m} is present then {effect} increases",
                concepts=[cause, effect, m],
                confidence=0.50,
                supporting_evidence=0,
                contradicting_evidence=0,
                status=HypothesisStatus.CANDIDATE,
                test_count=0,
            ))
            if len(mutations) >= max_mutations:
                return mutations
    except Exception:
        pass

    return mutations


# ---------------------------------------------------------------------------
# HypothesisEngine (v4: integrates lifecycle management)
# ---------------------------------------------------------------------------

class HypothesisEngine:
    """Converts pattern observations into structured typed hypotheses — v4.

    v4 additions:
    - Generated hypotheses carry status=ACTIVE by default.
    - apply_lifecycle_updates() batch-updates status for all known hypotheses.
    - get_testable() filters to only CANDIDATE and ACTIVE hypotheses.
    - get_mutations() generates mutations for ABANDONED hypotheses.
    """

    semantic_templates: List[Tuple[str, str, str]] = [
        ("animal", "eats", "plant"),
        ("predator", "hunts", "prey"),
        ("animal", "lives_in", "ecosystem"),
        ("plant", "needs", "sunlight"),
    ]

    causal_templates: List[str] = [
        "if {x} decreases then {y} increases",
        "if {x} increases then {y} decreases",
        "{x} affects {y}",
        "{x} controls {y} population",
    ]

    def __init__(self) -> None:
        self.stagnation_detector = HypothesisStagnationDetector()
        self.retry_policy        = HypothesisRetryPolicy()

    def from_patterns(
        self,
        patterns: Iterable[PatternRule],
        concept_types: ConceptTypeSystem,
        triplets: Sequence[Tuple[str, str, str]],
    ) -> List[Hypothesis]:
        hypotheses: List[Hypothesis] = []
        by_relation: dict[str, list[Tuple[str, str]]] = {}
        for s, r, o in triplets:
            by_relation.setdefault(r, []).append((s, o))

        for pattern in patterns:
            pairs      = by_relation.get(pattern.relation, [])
            typed_rules = [concept_types.typed_relation(s, pattern.relation, o) for s, o in pairs]
            rule       = self._majority_rule(typed_rules) if typed_rules else f"unknown {pattern.relation} unknown"
            support    = max(1, pattern.count)
            contradict = len([r for r in typed_rules if r != rule])
            confidence = (support + 0.5) / (support + contradict + 1)
            concepts   = self._concepts_for_relation(pattern.relation)
            hypotheses.append(Hypothesis(
                rule=rule, concepts=concepts, confidence=confidence,
                supporting_evidence=support, contradicting_evidence=contradict,
            ))

        hypotheses.extend(self.generate_semantic_hypotheses(concept_types, triplets))
        hypotheses.extend(self.generate_causal_hypotheses(triplets))
        return self._deduplicate(hypotheses)

    def from_causal_rules(self, rules: Sequence[CausalRuleRecord]) -> List[Hypothesis]:
        hypotheses: List[Hypothesis] = []
        for rule in rules:
            if rule.direction == "positive":
                statement = f"if {rule.cause} increases then {rule.effect} will increase"
            elif rule.direction == "negative":
                statement = f"if {rule.cause} increases then {rule.effect} will decrease"
            else:
                statement = f"if {rule.cause} changes then {rule.effect} may change"

            hypotheses.append(Hypothesis(
                rule=statement,
                concepts=[rule.cause, rule.effect],
                confidence=max(0.2, min(0.99, rule.confidence)),
                supporting_evidence=max(1, rule.observations),
                contradicting_evidence=max(1, int((1.0 - rule.confidence) * max(1, rule.observations))),
            ))
        return hypotheses

    def generate_semantic_hypotheses(
        self,
        concept_types: ConceptTypeSystem,
        triplets: Sequence[Tuple[str, str, str]],
        max_hypotheses: int = 8,
    ) -> List[Hypothesis]:
        relation_set = {r for _, r, _ in triplets}
        hypotheses: List[Hypothesis] = []

        for subj_type, relation, obj_type in self.semantic_templates:
            support    = sum(1 for s, r, o in triplets if r == relation
                            and concept_types.get_type(s) == subj_type
                            and concept_types.get_type(o) == obj_type)
            contradict = sum(1 for _s, r, _o in triplets if r == relation) - support
            confidence = (support + 1) / max(1, support + contradict + 2)
            hypotheses.append(Hypothesis(
                rule=f"{subj_type} {relation} {obj_type}",
                concepts=[subj_type, obj_type],
                confidence=confidence,
                supporting_evidence=max(1, support),
                contradicting_evidence=max(1, contradict),
            ))
            if relation not in relation_set:
                hypotheses[-1].confidence *= 0.6

        return hypotheses[:max_hypotheses]

    _CAUSAL_VARIABLES: frozenset = frozenset({
        "wolves", "deer", "grass",
        "temperature", "reactants", "reaction_rate", "reaction_energy",
        "asteroids", "solar_energy", "collision_risk", "radiation_pressure", "asteroid_drift",
        "robots", "battery_charge", "robot_activity", "sensor_coverage", "data_quality",
        "cells", "energy", "proteins", "pathogens", "immune_response",
        "force", "mass", "acceleration", "heat", "kinetic_energy",
        "velocity", "friction", "momentum", "catalyst", "pH",
    })

    # Reinforcement threshold: if a tick produces >= this many reinforced
    # observations (known facts re-seen) but zero new facts, we still attempt
    # to generate hypotheses by recombining well-evidenced concept pairs.
    REINFORCEMENT_HYPOTHESIS_THRESHOLD: int = 3

    def from_reinforcement(
        self,
        reinforcement_count: int,
        memory,                        # Memory — used to sample high-evidence facts
        max_new: int = 4,
    ) -> List[Hypothesis]:
        """Generate hypotheses from reinforced (re-observed) facts.

        Called when new_facts == 0 but reinforcement >= threshold, so the
        hypothesis engine keeps producing candidates even during no-novelty
        stalls.  Uses well-evidenced causal-variable pairs already in memory.

        Parameters
        ----------
        reinforcement_count : number of reinforced facts this tick
        memory              : Memory instance (for causal rules)
        max_new             : cap on hypotheses produced per call
        """
        if reinforcement_count < self.REINFORCEMENT_HYPOTHESIS_THRESHOLD:
            return []

        # Draw from the top-evidenced causal rules already in memory as seeds
        causal_rules = sorted(
            memory.get_causal_rules(),
            key=lambda r: r.observations,
            reverse=True,
        )[:20]

        hypotheses: List[Hypothesis] = []
        seen: Set[str] = set()

        for i, rule_a in enumerate(causal_rules):
            for rule_b in causal_rules[i + 1:]:
                # Look for chains: A→B and B→C imply A might affect C
                if rule_a.effect != rule_b.cause:
                    continue
                chain_rule = (
                    f"if {rule_a.cause} changes then {rule_b.effect} may change"
                    f" via {rule_a.effect}"
                )
                if chain_rule in seen:
                    continue
                seen.add(chain_rule)
                # Confidence is the geometric mean of the two legs, discounted
                conf = (rule_a.confidence * rule_b.confidence) ** 0.5 * 0.8
                hypotheses.append(Hypothesis(
                    rule=chain_rule,
                    concepts=[rule_a.cause, rule_a.effect, rule_b.effect],
                    confidence=round(conf, 3),
                    supporting_evidence=1,
                    contradicting_evidence=1,
                ))
                if len(hypotheses) >= max_new:
                    return hypotheses

        return hypotheses

    def generate_causal_hypotheses(
        self, triplets: Sequence[Tuple[str, str, str]]
    ) -> List[Hypothesis]:
        present = sorted({
            s for s, _, _ in triplets if s in self._CAUSAL_VARIABLES
        } | {
            o for _, _, o in triplets if o in self._CAUSAL_VARIABLES
        })
        if len(present) < 2:
            return []

        hypotheses: List[Hypothesis] = []
        for i in range(min(4, len(present) - 1)):
            x, y = present[i], present[i + 1]
            for template in self.causal_templates:
                hypotheses.append(Hypothesis(
                    rule=template.format(x=x, y=y),
                    concepts=[x, y],
                    confidence=0.45,
                    supporting_evidence=1,
                    contradicting_evidence=1,
                ))
        return hypotheses[:12]

    # ── v4: Lifecycle management ───────────────────────────────────────────

    def apply_lifecycle_updates(
        self,
        hypotheses: List[Hypothesis],
        current_tick: int,
    ) -> Tuple[List[Hypothesis], List[Hypothesis]]:
        """Batch-update hypothesis statuses and return (active, abandoned) lists.

        Call this once per tick (or every N ticks) from the event loop.

        Returns
        -------
        active    : hypotheses with status CANDIDATE or ACTIVE
        abandoned : hypotheses with status ABANDONED (ready for mutation)
        """
        active:    List[Hypothesis] = []
        abandoned: List[Hypothesis] = []

        for hyp in hypotheses:
            # Record confidence trajectory for stagnation detection
            self.stagnation_detector.record(hyp.rule, hyp.confidence)

            # Check for stagnation → suspend
            if (hyp.status == HypothesisStatus.ACTIVE
                    and self.stagnation_detector.is_stagnant(hyp.rule)
                    and hyp.test_count >= MIN_SUSPEND_TESTS):
                hyp.status = HypothesisStatus.SUSPENDED
                self.retry_policy.suspend(hyp.rule, current_tick, hyp.last_modality)
                continue

            # Check if suspended hypothesis is ready for retry
            if (hyp.status == HypothesisStatus.SUSPENDED
                    and self.retry_policy.is_eligible_for_retry(hyp.rule, current_tick)):
                hyp.status     = HypothesisStatus.ACTIVE
                hyp.test_count = 0   # reset so retry is fair
                self.stagnation_detector.reset(hyp.rule)

            # Update status from current confidence/evidence
            hyp.update_status()

            if hyp.status == HypothesisStatus.ABANDONED:
                abandoned.append(hyp)
            elif hyp.is_testable:
                active.append(hyp)

        return active, abandoned

    def get_testable(self, hypotheses: List[Hypothesis]) -> List[Hypothesis]:
        """Return only hypotheses eligible for testing."""
        return [h for h in hypotheses if h.is_testable]

    def get_mutations(
        self,
        hypotheses: List[Hypothesis],
        knowledge_graph,
    ) -> List[Hypothesis]:
        """Generate mutations for all ABANDONED hypotheses."""
        mutations: List[Hypothesis] = []
        for hyp in hypotheses:
            if hyp.status == HypothesisStatus.ABANDONED:
                mutations.extend(
                    mutate_abandoned_hypothesis(hyp, knowledge_graph)
                )
        return mutations

    # ── Utilities (unchanged from v3) ─────────────────────────────────────

    @staticmethod
    def _deduplicate(hypotheses: Sequence[Hypothesis]) -> List[Hypothesis]:
        best: dict[str, Hypothesis] = {}
        for hypothesis in hypotheses:
            existing = best.get(hypothesis.rule)
            if existing is None or hypothesis.confidence > existing.confidence:
                best[hypothesis.rule] = hypothesis
        return list(best.values())

    @staticmethod
    def _majority_rule(rules: Sequence[str]) -> str:
        counts: dict[str, int] = {}
        for rule in rules:
            counts[rule] = counts.get(rule, 0) + 1
        return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]

    @staticmethod
    def _concepts_for_relation(relation: str) -> List[str]:
        mapping = {
            "hunts":     ["predator", "prey"],
            "orbits":    ["planet_or_moon", "star_or_planet"],
            "is":        ["child", "parent"],
            "reacts_with": ["reactant_a", "reactant_b"],
            "food_chain": ["predator", "prey", "producer"],
            "lives_in":  ["animal", "ecosystem"],
            "needs":     ["living_thing", "resource"],
            "eats":      ["consumer", "food"],
            "affects":   ["cause", "effect"],
            "controls":  ["controller", "controlled_population"],
            "causes":    ["cause", "effect"],
            "inhibits":  ["inhibitor", "target"],
            "promotes":  ["promoter", "target"],
            "catalyses": ["catalyst", "reaction"],
        }
        return mapping.get(relation, ["entity_a", "entity_b"])