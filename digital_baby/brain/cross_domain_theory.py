"""Cross-Domain Theory Formation — Architecture v2.

Detects causal sub-graph patterns that appear in multiple domains and:
1. Abstracts them into named schemas (e.g. "predation_pattern").
2. Generates cross-domain analogical hypotheses.
3. Prevents redundant intervention by flagging "already known" patterns.

Algorithm
---------
1. Represent each confirmed causal pair as a typed pattern
   (cause_role, direction, effect_role).
2. Use a compact fingerprint to detect when the same structural pattern
   occurs in ≥ MIN_DOMAINS domains.
3. Emit a CrossDomainTheory when a new isomorphism is found.
4. Generate a new Hypothesis for each domain that *lacks* the pattern.

Pattern vocabulary
------------------
Patterns are abstracted from raw variable names using a semantic role
dictionary. E.g. "wolf" → "predator", "deer" → "prey", "robot" → "agent".
This allows wolf→deer and robot→sensor to be recognised as the same
"agent_reduces_resource" pattern.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Semantic role mapping (raw variable → abstract role)
# ---------------------------------------------------------------------------

_ROLE: Dict[str, str] = {
    # Ecosystem
    "wolf":           "predator",
    "deer":           "prey",
    "grass":          "resource",
    "wolves":         "predator",
    # Biology
    "pathogens":      "agent",
    "immune_response":"response",
    "energy":         "resource",
    "proteins":       "resource",
    "cells":          "substrate",
    # Technology
    "robot":          "agent",
    "robots":         "agent",
    "sensor_coverage":"output",
    "data_quality":   "output",
    "robot_activity": "activity",
    "battery_charge": "resource",
    # Chemistry
    "temperature":    "driver",
    "reactants":      "substrate",
    "reaction_rate":  "output",
    "reaction_energy":"output",
    # Physics
    "force":          "driver",
    "acceleration":   "output",
    "heat":           "driver",
    "kinetic_energy": "output",
    "mass":           "substrate",
    # Astronomy
    "asteroid":       "agent",
    "asteroids":      "agent",
    "collision_risk": "output",
    "radiation_pressure": "driver",
}

_DOMAIN_FOR_VAR: Dict[str, str] = {
    "wolf": "ecosystem", "deer": "ecosystem", "grass": "ecosystem",
    "wolves": "ecosystem",
    "pathogens": "biology", "immune_response": "biology",
    "energy": "biology", "proteins": "biology", "cells": "biology",
    "robot": "technology", "robots": "technology",
    "sensor_coverage": "technology", "data_quality": "technology",
    "robot_activity": "technology", "battery_charge": "technology",
    "temperature": "chemistry", "reactants": "chemistry",
    "reaction_rate": "chemistry", "reaction_energy": "chemistry",
    "force": "physics", "acceleration": "physics",
    "heat": "physics", "kinetic_energy": "physics", "mass": "physics",
    "asteroid": "astronomy", "asteroids": "astronomy",
    "collision_risk": "astronomy", "radiation_pressure": "astronomy",
}

MIN_DOMAINS: int = 2    # pattern must appear in this many domains to be abstracted


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CausalPattern:
    """One (cause, direction, effect) typed as abstract roles."""
    cause_var:    str
    effect_var:   str
    direction:    str     # "positive" | "negative" | "mixed"
    confidence:   float
    domain:       str

    @property
    def cause_role(self) -> str:
        return _ROLE.get(self.cause_var, self.cause_var)

    @property
    def effect_role(self) -> str:
        return _ROLE.get(self.effect_var, self.effect_var)

    @property
    def fingerprint(self) -> str:
        """Domain-agnostic pattern fingerprint."""
        return f"{self.cause_role}_{self.direction[:3]}_{self.effect_role}"


@dataclass
class CrossDomainTheory:
    """An abstract pattern found in multiple domains."""
    name:             str
    fingerprint:      str
    cause_role:       str
    effect_role:      str
    direction:        str
    domains_found:    List[str]
    instances:        List[CausalPattern]
    discovered_tick:  int
    analogical_hyps:  List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Theory[{self.name}] ({self.cause_role} −{self.direction[:3]}→ {self.effect_role}) "
            f"found in {self.domains_found}"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CrossDomainTheoryEngine:
    """Detects cross-domain structural isomorphisms and generates analogical hypotheses.

    Usage (inside BabyEventLoop)
    -----------------------------
    self.theory_engine = CrossDomainTheoryEngine()

    # After causal discovery each tick:
    for rule in self.memory.get_causal_rules():
        self.theory_engine.register_pattern(
            cause=rule.cause, effect=rule.effect,
            direction=rule.direction, confidence=rule.confidence
        )

    new_theories, new_hyps = self.theory_engine.run(current_tick)
    """

    def __init__(self, min_domains: int = MIN_DOMAINS) -> None:
        self.min_domains = min_domains
        self._patterns:  List[CausalPattern] = []
        self._theories:  Dict[str, CrossDomainTheory] = {}   # fingerprint → theory
        self._seen_pairs: Set[Tuple[str, str]] = set()

    # ── Public API ─────────────────────────────────────────────────────────────

    def register_pattern(
        self,
        cause:      str,
        effect:     str,
        direction:  str,
        confidence: float,
    ) -> None:
        """Register one confirmed causal pair for theory detection."""
        pair = (cause, effect)
        if pair in self._seen_pairs:
            # Update confidence on existing pattern
            for p in self._patterns:
                if p.cause_var == cause and p.effect_var == effect:
                    p.confidence = max(p.confidence, confidence)
            return

        domain = (
            _DOMAIN_FOR_VAR.get(cause)
            or _DOMAIN_FOR_VAR.get(effect)
            or "unknown"
        )
        self._patterns.append(CausalPattern(
            cause_var=cause,
            effect_var=effect,
            direction=direction,
            confidence=confidence,
            domain=domain,
        ))
        self._seen_pairs.add(pair)

    def run(
        self, current_tick: int
    ) -> Tuple[List[CrossDomainTheory], List[str]]:
        """Detect new cross-domain theories and emit analogical hypotheses.

        Returns
        -------
        (new_theories, new_hypothesis_strings)
        """
        new_theories: List[CrossDomainTheory] = []
        new_hyp_strings: List[str] = []

        # Group patterns by fingerprint
        by_fp: Dict[str, List[CausalPattern]] = defaultdict(list)
        for p in self._patterns:
            by_fp[p.fingerprint].append(p)

        for fp, instances in by_fp.items():
            domains = list({p.domain for p in instances})
            if len(domains) < self.min_domains:
                continue

            if fp in self._theories:
                # Update existing theory with any new domains
                theory = self._theories[fp]
                added = [d for d in domains if d not in theory.domains_found]
                if not added:
                    continue
                theory.domains_found.extend(added)
                theory.instances = instances
                logger.info(
                    "[theory] extended %s → now in domains %s",
                    theory.name, theory.domains_found,
                )
            else:
                # New theory!
                name = self._name_theory(instances[0])
                theory = CrossDomainTheory(
                    name=name,
                    fingerprint=fp,
                    cause_role=instances[0].cause_role,
                    effect_role=instances[0].effect_role,
                    direction=instances[0].direction,
                    domains_found=domains,
                    instances=instances,
                    discovered_tick=current_tick,
                )
                self._theories[fp] = theory
                new_theories.append(theory)
                logger.info("[theory] new_theory=%s domains=%s", name, domains)

            # Generate analogical hypotheses for domains that LACK the pattern
            all_domains = set(_DOMAIN_FOR_VAR.values())
            missing_domains = all_domains - set(theory.domains_found) - {"unknown"}
            for missing in missing_domains:
                hyp = self._analogical_hypothesis(theory, missing)
                if hyp and hyp not in theory.analogical_hyps:
                    theory.analogical_hyps.append(hyp)
                    new_hyp_strings.append(hyp)
                    logger.info(
                        "[theory] analogical_hypothesis domain=%s hyp=%r",
                        missing, hyp[:80],
                    )

        return new_theories, new_hyp_strings

    def get_all_theories(self) -> List[CrossDomainTheory]:
        return list(self._theories.values())

    def theory_count(self) -> int:
        return len(self._theories)

    def summary(self) -> Dict:
        return {
            "theories": self.theory_count(),
            "patterns_registered": len(self._patterns),
            "theory_list": [
                {
                    "name": t.name,
                    "domains": t.domains_found,
                    "fingerprint": t.fingerprint,
                }
                for t in self._theories.values()
            ],
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _name_theory(pattern: CausalPattern) -> str:
        dir_word = {
            "positive": "amplifies",
            "negative": "suppresses",
            "mixed":    "modulates",
        }.get(pattern.direction, "affects")
        return f"{pattern.cause_role}_{dir_word}_{pattern.effect_role}"

    @staticmethod
    def _analogical_hypothesis(theory: CrossDomainTheory, target_domain: str) -> Optional[str]:
        """Generate a domain-specific hypothesis by analogy."""
        # Find domain-specific variables for the abstract roles
        role_to_var: Dict[str, List[str]] = defaultdict(list)
        for var, role in _ROLE.items():
            if _DOMAIN_FOR_VAR.get(var) == target_domain:
                role_to_var[role].append(var)

        causes  = role_to_var.get(theory.cause_role, [])
        effects = role_to_var.get(theory.effect_role, [])

        if not causes or not effects:
            return None

        cause_var  = causes[0]
        effect_var = effects[0]

        dir_phrase = {
            "positive": "increases",
            "negative": "decreases",
            "mixed":    "changes",
        }.get(theory.direction, "affects")

        return (
            f"[analogy:{theory.name}] if {cause_var} changes "
            f"then {effect_var} {dir_phrase} "
            f"(by analogy with {theory.domains_found})"
        )