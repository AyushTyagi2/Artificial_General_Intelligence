"""Mediator-blocking experiment planner.

Validates multi-hop causal chains by designing and evaluating experiments that
hold the intermediate node (mediator) constant while varying the cause.

Protocol for chain A → B → C
------------------------------
1. Free run:     vary A normally, record ΔC_free.
2. Blocking run: fix_variable(B, baseline_value), vary A identically, record ΔC_blocked.
3. Mediation ratio: MR = 1 − |ΔC_blocked| / |ΔC_free|
   MR ≥ MEDIATION_THRESHOLD  →  chain confirmed (B mediates A→C)
   MR <  MEDIATION_THRESHOLD  →  chain weakened  (B may not be the pathway)

The blocking run is executed by passing a FixVariableAction to
Experimenter.apply_environment_dynamics() in the event loop.

Integration
-----------
After CausalChainReasoner produces inferred_rules:

    from digital_baby.brain.mediator_blocking_planner import MediatorBlockingPlanner

    # In BabyEventLoop.__init__:
    self.mediator_planner = MediatorBlockingPlanner(self.knowledge_graph)

    # In the tick body, after chain reasoning:
    for rule in inferred_rules:
        self.mediator_planner.maybe_queue(rule, baseline_state)

    # Action selection: check for a pending blocking plan first (P0.5 priority):
    blocking_plan = self.mediator_planner.get_pending_plan()
    if blocking_plan:
        # tick T+1: execute blocking_plan.fix_actions alongside the same cause action
        ...
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MEDIATION_THRESHOLD: float = 0.70   # MR must reach this to confirm chain
MAX_QUEUE_SIZE:      int   = 8       # cap pending validations
MIN_CHAIN_CONF:      float = 0.25    # only queue chains above this confidence
BLOCKING_COOLDOWN:   int   = 50      # ticks before re-validating the same pair

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FixVariableAction:
    """Instruction to hold a variable at a fixed value during a world step."""
    variable: str
    value:    float


@dataclass
class MediatorBlockingPlan:
    """A two-tick experiment plan for validating a causal chain."""
    chain_source:   str
    chain_mediator: str
    chain_target:   str
    cause_action:   str              # action to vary the source
    fix_action:     FixVariableAction
    chain_conf:     float
    mechanism_name: str = ""
    planned_tick:   int = 0

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.chain_source, self.chain_mediator, self.chain_target)


@dataclass
class BlockingResult:
    plan:           MediatorBlockingPlan
    delta_free:     float   # |ΔC| in the unblocked run
    delta_blocked:  float   # |ΔC| in the blocking run
    mediation_ratio: float
    confirmed:      bool
    tick:           int


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MediatorBlockingPlanner:
    """Queues mediator-blocking experiments and processes their results.

    Parameters
    ----------
    knowledge_graph : KnowledgeGraph
    cause_to_actions : dict, optional
        Maps cause variable names to experiment actions.  Uses
        ExperimentPlanner.CAUSE_TO_ACTIONS by default.
    """

    def __init__(
        self,
        knowledge_graph,
        cause_to_actions: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self._graph    = knowledge_graph
        self._c2a      = cause_to_actions or self._default_cause_to_actions()
        self._queue:   List[MediatorBlockingPlan] = []
        self._pending: Optional[MediatorBlockingPlan] = None
        self._last_validated: Dict[Tuple[str,str,str], int] = {}
        self._results: List[BlockingResult] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def maybe_queue(
        self,
        inferred_rule,
        baseline_state: Dict[str, float],
        current_tick:   int = 0,
        mechanism_name: str = "",
    ) -> bool:
        """Attempt to queue a blocking experiment for *inferred_rule*.

        Returns True if the plan was added to the queue.
        """
        chain = inferred_rule.chain
        if len(chain.nodes) < 3:
            return False
        if inferred_rule.confidence < MIN_CHAIN_CONF:
            return False

        source   = chain.nodes[0]
        mediator = chain.nodes[1]
        target   = chain.nodes[-1]
        key      = (source, mediator, target)

        # Respect cooldown
        last = self._last_validated.get(key, -BLOCKING_COOLDOWN)
        if current_tick - last < BLOCKING_COOLDOWN:
            return False

        # Don't exceed queue cap
        if len(self._queue) >= MAX_QUEUE_SIZE:
            return False

        # Already queued?
        if any(p.key == key for p in self._queue):
            return False

        cause_action = self._pick_action(source)
        if cause_action is None:
            return False

        baseline_val = baseline_state.get(mediator, 0.0)

        plan = MediatorBlockingPlan(
            chain_source=source,
            chain_mediator=mediator,
            chain_target=target,
            cause_action=cause_action,
            fix_action=FixVariableAction(variable=mediator, value=baseline_val),
            chain_conf=inferred_rule.confidence,
            mechanism_name=mechanism_name,
            planned_tick=current_tick,
        )
        self._queue.append(plan)
        logger.info(
            "[mediator_planner] queued A=%s B=%s C=%s action=%s fix_val=%.2f",
            source, mediator, target, cause_action, baseline_val,
        )
        return True

    def get_pending_plan(self) -> Optional[MediatorBlockingPlan]:
        """Return the next plan to execute (if any), without dequeuing it.

        The event loop should call ``advance()`` once the two-tick experiment
        is complete.
        """
        if self._pending is not None:
            return self._pending
        if self._queue:
            self._pending = self._queue.pop(0)
            return self._pending
        return None

    def record_result(
        self,
        delta_free:    float,
        delta_blocked: float,
        current_tick:  int,
    ) -> Optional[BlockingResult]:
        """Record the outcome of the pending blocking experiment.

        Applies confidence update to the inferred edge in the knowledge graph.
        Returns the BlockingResult, then clears the pending plan.
        """
        plan = self._pending
        if plan is None:
            logger.warning("[mediator_planner] record_result called with no pending plan")
            return None

        mr = 1.0 - (abs(delta_blocked) / max(abs(delta_free), 1e-6))
        mr = max(0.0, min(1.0, mr))
        confirmed = mr >= MEDIATION_THRESHOLD

        result = BlockingResult(
            plan=plan,
            delta_free=delta_free,
            delta_blocked=delta_blocked,
            mediation_ratio=mr,
            confirmed=confirmed,
            tick=current_tick,
        )
        self._results.append(result)
        self._last_validated[plan.key] = current_tick
        self._pending = None

        # Update knowledge graph edge
        self._update_kg_edge(plan, confirmed, mr)

        logger.info(
            "[mediator_planner] result A=%s B=%s C=%s MR=%.2f confirmed=%s",
            plan.chain_source, plan.chain_mediator, plan.chain_target, mr, confirmed,
        )
        return result

    def advance(self) -> None:
        """Explicitly clear the pending plan (if record_result was not called)."""
        self._pending = None

    def queue_depth(self) -> int:
        return len(self._queue) + (1 if self._pending else 0)

    def get_recent_results(self, n: int = 10) -> List[BlockingResult]:
        return self._results[-n:]

    # ── Internal helpers ───────────────────────────────────────────────────

    def _update_kg_edge(self, plan: MediatorBlockingPlan, confirmed: bool, mr: float) -> None:
        """Update the inferred_affects edge confidence based on mediator test result."""
        for e in self._graph.edges:
            if (e.source == plan.chain_source
                    and e.target == plan.chain_target
                    and "inferred" in e.relation):
                if confirmed:
                    e.provenance = "mediator_confirmed"
                    e.confidence = min(0.95, e.confidence * (1.0 + mr * 0.5))
                    logger.info(
                        "[mediator_planner] edge_confirmed %s→%s new_conf=%.2f",
                        plan.chain_source, plan.chain_target, e.confidence,
                    )
                else:
                    e.confidence = e.confidence * 0.5
                    logger.info(
                        "[mediator_planner] edge_weakened %s→%s new_conf=%.2f",
                        plan.chain_source, plan.chain_target, e.confidence,
                    )
                # Update status tier
                from digital_baby.brain.knowledge_graph import _edge_status
                e.status = _edge_status(e.confidence)
                return

    def _pick_action(self, cause: str) -> Optional[str]:
        actions = self._c2a.get(cause, [])
        return actions[0] if actions else None

    @staticmethod
    def _default_cause_to_actions() -> Dict[str, List[str]]:
        from digital_baby.brain.experiment_planner import CAUSE_TO_ACTIONS
        return CAUSE_TO_ACTIONS