"""Mechanism naming for discovered causal chains.

Maps multi-hop CausalChain objects to named scientific mechanisms by
comparing their abstract role sequence against a built-in template library.

Usage
-----
    matcher = MechanismMatcher(role_assigner)
    for rule in inferred_rules:
        match = matcher.match(rule.chain)
        if match:
            rule.mechanism_name = match.template.name
            research_agent.queue_query(match.research_query, priority="high")

Adding new templates
--------------------
Append a MechanismTemplate to MECHANISM_LIBRARY.  Each template is a list of
(abstract_role, edge_direction) pairs describing the chain from start to end.
edge_direction is "pos", "neg", or "" (any).  Roles must match those returned
by TopologicalRoleAssigner.get_role().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template library
# ---------------------------------------------------------------------------

@dataclass
class MechanismTemplate:
    name:            str
    description:     str
    # sequence of (abstract_role, expected_direction) for each hop
    # direction: "pos" | "neg" | "" (wildcard)
    role_chain:      List[Tuple[str, str]]
    min_match_score: float = 0.70


MECHANISM_LIBRARY: List[MechanismTemplate] = [
    MechanismTemplate(
        name="trophic_cascade",
        description="Predator suppresses prey, releasing resource from grazing pressure.",
        role_chain=[("predator", "neg"), ("prey", "neg"), ("resource", "")],
    ),
    MechanismTemplate(
        name="negative_feedback",
        description="Output feeds back to suppress its own driver.",
        role_chain=[("driver", "pos"), ("output", "neg"), ("driver", "")],
    ),
    MechanismTemplate(
        name="competitive_exclusion",
        description="Two consumers compete via a shared limiting resource.",
        role_chain=[("agent", "neg"), ("shared_resource", "neg"), ("agent", "")],
    ),
    MechanismTemplate(
        name="cascade_amplification",
        description="A trigger activates a relay that amplifies a downstream output.",
        role_chain=[("trigger", "pos"), ("relay", "pos"), ("output", "")],
    ),
    MechanismTemplate(
        name="arrhenius_activation",
        description="Temperature lowers activation barrier, accelerating reaction rate.",
        role_chain=[("driver", "pos"), ("barrier", "neg"), ("rate", "")],
    ),
    MechanismTemplate(
        name="resource_depletion",
        description="Consumer drives up consumption rate, depleting substrate.",
        role_chain=[("consumer", "pos"), ("consumption_rate", "neg"), ("resource", "")],
    ),
    MechanismTemplate(
        name="homeostatic_regulation",
        description="Error signal drives a corrective output that reduces the error.",
        role_chain=[("error", "pos"), ("corrective_output", "neg"), ("error", "")],
    ),
    MechanismTemplate(
        name="predator_mediated_coexistence",
        description="Predator preferentially suppresses dominant competitor, enabling weaker one.",
        role_chain=[("predator", "neg"), ("dominant_competitor", "neg"), ("weaker_competitor", "")],
        min_match_score=0.60,   # looser: role distinction is hard
    ),
    MechanismTemplate(
        name="positive_feedback_loop",
        description="Output reinforces its own driver, producing runaway growth.",
        role_chain=[("driver", "pos"), ("output", "pos"), ("driver", "")],
    ),
    MechanismTemplate(
        name="substrate_depletion_kinetics",
        description="Enzyme/catalyst depletes substrate, slowing its own reaction.",
        role_chain=[("catalyst", "pos"), ("rate", "neg"), ("substrate", "")],
    ),
]

# Role synonyms — roles that are treated as equivalent for matching
_ROLE_SYNONYMS: Dict[str, str] = {
    "suppressor":  "predator",
    "inhibitor":   "predator",
    "activator":   "driver",
    "amplifier":   "driver",
    "mediator":    "relay",
    "substrate":   "resource",
    "sink":        "resource",
    "effector":    "output",
    "terminal":    "output",
}

# Partial credit for related (non-identical, non-synonym) roles
_ROLE_PARTIAL_CREDIT: Dict[frozenset, float] = {
    frozenset({"predator", "consumer"}): 0.6,
    frozenset({"driver",   "trigger"}):  0.7,
    frozenset({"resource", "output"}):   0.5,
    frozenset({"relay",    "catalyst"}): 0.7,
}


def _role_score(actual: str, expected: str) -> float:
    """Score how well *actual* role matches *expected* role."""
    if actual == expected:
        return 1.0
    # Check synonym mapping
    canonical_actual   = _ROLE_SYNONYMS.get(actual,   actual)
    canonical_expected = _ROLE_SYNONYMS.get(expected, expected)
    if canonical_actual == canonical_expected:
        return 0.9
    # Check partial credit table
    pair = frozenset({canonical_actual, canonical_expected})
    if pair in _ROLE_PARTIAL_CREDIT:
        return _ROLE_PARTIAL_CREDIT[pair]
    return 0.0


def _direction_score(chain_direction: str, expected_direction: str) -> float:
    """1.0 if direction matches or expected is wildcard, else 0.5 (partial)."""
    if not expected_direction:          # wildcard
        return 1.0
    if chain_direction.startswith(expected_direction):
        return 1.0
    return 0.5


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MechanismMatch:
    template:       MechanismTemplate
    chain:          object                  # CausalChain
    match_score:    float                   # 0–1
    role_binding:   Dict[str, str]          # abstract_role → concrete_variable
    research_query: str                     # ready for ResearchAgent


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MechanismMatcher:
    """Matches CausalChain objects against the mechanism template library.

    Parameters
    ----------
    role_assigner : callable
        A function ``role_assigner(node: str) -> str`` that returns the
        abstract role of a graph node.  Typically
        ``TopologicalRoleAssigner.get_role``.
    library : list[MechanismTemplate], optional
        Override the default MECHANISM_LIBRARY (useful in tests).
    """

    def __init__(
        self,
        role_assigner: Callable[[str], str],
        library: Optional[List[MechanismTemplate]] = None,
    ) -> None:
        self._get_role = role_assigner
        self._library  = library if library is not None else MECHANISM_LIBRARY
        # Cache of already-matched (source, target) pairs to avoid log spam
        self._matched_pairs: Dict[Tuple[str, str], str] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def match(self, chain) -> Optional[MechanismMatch]:
        """Attempt to match *chain* against all library templates.

        Returns the best MechanismMatch above the template's min_match_score,
        or None if no template matches.
        """
        nodes = chain.nodes   # list[str]
        if len(nodes) < 3:
            return None       # only chains of length >= 2 hops can be named

        # Map nodes to roles
        roles = [self._get_role(n) for n in nodes]

        # Compute edge directions from the chain
        edge_dirs = []
        for edge in chain.edges:
            rel = getattr(edge, "relation", "")
            if "negative" in rel:
                edge_dirs.append("neg")
            elif "positive" in rel:
                edge_dirs.append("pos")
            else:
                edge_dirs.append("")

        best_match: Optional[MechanismMatch] = None
        best_score: float = 0.0

        for tmpl in self._library:
            if len(tmpl.role_chain) != len(nodes) - 1:
                continue
            # Score = mean of per-slot (role_score × direction_score)
            slot_scores = []
            for i, (expected_role, expected_dir) in enumerate(tmpl.role_chain):
                rs = _role_score(roles[i], expected_role)
                ds = _direction_score(edge_dirs[i] if i < len(edge_dirs) else "", expected_dir)
                slot_scores.append(rs * ds)
            # Also score the final (target) node's role against the template endpoint
            last_role     = tmpl.role_chain[-1][0]
            final_rs      = _role_score(roles[-1], last_role)
            slot_scores.append(final_rs)

            score = sum(slot_scores) / len(slot_scores)
            if score > best_score and score >= tmpl.min_match_score:
                best_score = score
                best_match = tmpl

        if best_match is None:
            return None

        role_binding = {
            best_match.role_chain[i][0]: nodes[i]
            for i in range(len(best_match.role_chain))
        }
        role_binding[best_match.role_chain[-1][0]] = nodes[-1]

        research_query = f"mechanisms of {best_match.name.replace('_', ' ')}"

        result = MechanismMatch(
            template=best_match,
            chain=chain,
            match_score=best_score,
            role_binding=role_binding,
            research_query=research_query,
        )

        key = (chain.source, chain.target)
        if self._matched_pairs.get(key) != best_match.name:
            self._matched_pairs[key] = best_match.name
            logger.info(
                "[mechanism] matched %s chain=%s score=%.2f binding=%s",
                best_match.name, chain.path_str(), best_score, role_binding,
            )

        return result

    def match_all(self, inferred_rules: list) -> Dict[Tuple[str, str], MechanismMatch]:
        """Run match() over a list of InferredRule objects.

        Returns a dict keyed by (source, target).
        """
        results: Dict[Tuple[str, str], MechanismMatch] = {}
        for rule in inferred_rules:
            m = self.match(rule.chain)
            if m:
                results[(rule.source, rule.target)] = m
        return results