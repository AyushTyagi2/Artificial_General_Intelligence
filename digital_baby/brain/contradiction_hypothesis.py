"""Contradiction-driven hypothesis engine — Architecture v3 (new module).

When existing causal rules are bidirectional/mixed and no mediator has been
found yet, this engine generates new classes of hypotheses that cannot be
expressed in the standard 'if X increases then Y will change' template:

    - Threshold hypotheses:  'if X > T then X positively affects Y'
    - Mediator hypotheses:   'Z mediates the relationship between X and Y'
    - Reversal hypotheses:   'direction of X→Y reverses when Z changes sign'
    - Interaction hypotheses:'X and Z jointly determine the effect on Y'

It also proposes new synthetic mediator variables when a contradiction persists
with no mediator found (after MIN_CONTRADICTION_TICKS unresolved ticks).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_CONTRADICTION_OBS:   int   = 8     # observations before considering a rule contradictory
MIN_CONTRADICTION_CONF:  float = 0.35  # minimum confidence before contradiction matters
MIN_CONTRADICTION_TICKS: int   = 20    # ticks a contradiction must persist before variable proposal
CONTRADICTION_TEMPLATE_LIMIT: int = 4  # max hypotheses generated per contradictory pair


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProposedVariable:
    """A new synthetic variable proposed to explain a contradiction."""
    name:         str
    domain:       str
    parent_cause: str
    parent_effect: str
    description:  str
    synthesis_type: str  # 'mediator' | 'threshold_gate' | 'interaction'
    initial_value: float = 0.0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ContradictionHypothesisEngine:
    """Generates hypotheses and variable proposals from unresolved contradictions."""

    def __init__(self) -> None:
        self._contradiction_first_seen: Dict[Tuple[str, str], int] = {}
        self._proposed_variables: Dict[Tuple[str, str], str]       = {}  # pair → var name

    # ── Public API ─────────────────────────────────────────────────────────

    def run(
        self,
        causal_rules:  List,
        knowledge_graph,
        domain:        str,
        tick:          int,
    ) -> Tuple[List, List[ProposedVariable]]:
        """Scan causal rules for contradictions and generate hypotheses + variable proposals.

        Parameters
        ----------
        causal_rules    : List of CausalRule/CausalRecord objects (duck-typed).
        knowledge_graph : KnowledgeGraph for checking existing variable coverage.
        domain          : Current active domain.
        tick            : Current tick (for timing variable proposals).

        Returns
        -------
        (new_hypotheses, new_variables) — lists of Hypothesis-like objects
        and ProposedVariable objects.
        """
        from digital_baby.brain.hypothesis import Hypothesis  # lazy import to avoid cycles

        contradictions = [
            r for r in causal_rules
            if self._is_contradictory(r)
        ]

        new_hypotheses: List[Hypothesis]      = []
        new_variables:  List[ProposedVariable] = []

        for rule in contradictions:
            cause  = getattr(rule, "cause",  "")
            effect = getattr(rule, "effect", "")
            conf   = float(getattr(rule, "confidence", 0.5))
            key    = (cause, effect)

            # Track first-seen tick for this contradiction
            if key not in self._contradiction_first_seen:
                self._contradiction_first_seen[key] = tick

            # Generate hypothesis templates from this contradiction
            templates = self._build_hypotheses(cause, effect, conf, knowledge_graph)
            for t in templates[:CONTRADICTION_TEMPLATE_LIMIT]:
                new_hypotheses.append(Hypothesis(
                    rule=t,
                    concepts=[cause, effect],
                    confidence=0.35,
                    supporting_evidence=0,
                    contradicting_evidence=0,
                ))

            # Propose a new mediator variable if contradiction is old enough
            ticks_contradicting = tick - self._contradiction_first_seen[key]
            has_conditions = bool(getattr(rule, "conditions", []))
            already_proposed = key in self._proposed_variables

            if (ticks_contradicting >= MIN_CONTRADICTION_TICKS
                    and not has_conditions
                    and not already_proposed):
                proposal = self._propose_variable(cause, effect, domain, tick)
                if proposal:
                    new_variables.append(proposal)
                    self._proposed_variables[key] = proposal.name
                    logger.info(
                        "[contradiction_hyp] proposed_variable=%s for_pair=%s->%s tick=%d",
                        proposal.name, cause, effect, tick,
                    )

        if new_hypotheses:
            logger.debug(
                "[contradiction_hyp] tick=%d domain=%s contradictions=%d new_hyps=%d new_vars=%d",
                tick, domain, len(contradictions), len(new_hypotheses), len(new_variables),
            )

        return new_hypotheses, new_variables

    # ── Hypothesis template builders ───────────────────────────────────────

    def _build_hypotheses(
        self,
        cause:          str,
        effect:         str,
        confidence:     float,
        knowledge_graph,
    ) -> List[str]:
        """Generate the four contradiction hypothesis templates."""
        templates: List[str] = []

        # 1. Threshold hypothesis — guess that a threshold around current mean matters
        templates.append(
            f"if {cause} exceeds threshold then {cause} positively affects {effect}"
        )
        templates.append(
            f"if {cause} is below threshold then {cause} negatively affects {effect}"
        )

        # 2. Mediator hypothesis — name to be discovered by causal discovery
        templates.append(
            f"some variable mediates the relationship between {cause} and {effect}"
        )

        # 3. Reversal hypothesis — direction flips based on context
        templates.append(
            f"the direction of {cause} on {effect} reverses under different conditions"
        )

        # 4. Interaction hypothesis — look for a known adjacent variable
        adjacent = self._find_adjacent_variable(cause, effect, knowledge_graph)
        if adjacent:
            templates.append(
                f"{cause} and {adjacent} jointly determine the effect on {effect}"
            )

        return templates

    @staticmethod
    def _find_adjacent_variable(
        cause:          str,
        effect:         str,
        knowledge_graph,
    ) -> Optional[str]:
        """Find a variable that is causally adjacent to both cause and effect."""
        try:
            edges = knowledge_graph.edges
            cause_neighbours  = {e.target for e in edges if e.source == cause}
            effect_neighbours = {e.source for e in edges if e.target == effect}
            shared = (cause_neighbours & effect_neighbours) - {cause, effect}
            if shared:
                return sorted(shared)[0]
        except Exception:
            pass
        return None

    # ── Variable proposals ─────────────────────────────────────────────────

    def _propose_variable(
        self,
        cause:  str,
        effect: str,
        domain: str,
        tick:   int,
    ) -> Optional[ProposedVariable]:
        """Propose a new mediator variable to explain the contradiction."""
        # Naming: cond_<cause>_on_<effect> (readable, unique per pair)
        var_name = f"cond_{cause}_on_{effect}"

        # Limit name length for readability
        if len(var_name) > 40:
            var_name = f"med_{cause[:8]}_{effect[:8]}"

        return ProposedVariable(
            name          = var_name,
            domain        = domain,
            parent_cause  = cause,
            parent_effect = effect,
            description   = (
                f"Mediating context variable explaining why {cause} sometimes "
                f"positively and sometimes negatively affects {effect}. "
                f"Value 0.0 = inhibitory context, 1.0 = excitatory context."
            ),
            synthesis_type = "mediator",
            initial_value  = 0.5,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _is_contradictory(rule) -> bool:
        direction = getattr(rule, "direction", "")
        conf      = float(getattr(rule, "confidence", 0.0))
        obs       = getattr(rule, "evidence_count",
                    getattr(rule, "observations", 0))
        return (
            direction in ("bidirectional", "mixed")
            and conf >= MIN_CONTRADICTION_CONF
            and obs  >= MIN_CONTRADICTION_OBS
        )