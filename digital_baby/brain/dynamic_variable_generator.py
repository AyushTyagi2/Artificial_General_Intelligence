"""Dynamic Variable Generator — Architecture v3 (new module).

Synthesizes new state variables when a domain reaches learning saturation
so the causal simulator never permanently exhausts its novelty.

Three synthesis strategies
--------------------------
1. DERIVATIVE  : dX/dt rate-of-change variable for existing X.
                 Adds new edges: X → dX/dt (trivially positive) and
                 dX/dt → [downstream effects] (to be discovered).

2. THRESHOLD   : Boolean-like gate variable 0/1 that fires when X > threshold.
                 Enables conditional causal structures (feeds Module 2).

3. COMPOUND    : Cross-domain interaction variable combining two variables
                 from different domains.  Feeds CrossDomainTheoryEngine.

Saturation detection
---------------------
A domain is saturated when the mean ep_ig for all its recent ticks falls
below MIN_EP_IG_FOR_SATURATION for at least SATURATION_TICKS ticks.
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

MIN_EP_IG_FOR_SATURATION: float = 0.005   # bits — below this per tick is saturated
SATURATION_TICKS:         int   = 10      # ticks below threshold before triggering
MAX_VARIABLES_PER_DOMAIN: int   = 15      # cap — avoid unbounded state space
SYNTHESIS_COOLDOWN_TICKS: int   = 30      # minimum ticks between syntheses per domain


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SynthesizedVariable:
    """A new variable created by the generator."""
    name:              str
    domain:            str
    synthesis_type:    str            # 'derivative' | 'threshold' | 'compound'
    parent_vars:       List[str]
    initial_value:     float          = 0.0
    threshold:         Optional[float] = None  # for threshold type
    compound_domains:  List[str]      = field(default_factory=list)
    added_at_tick:     int            = 0
    causal_priors:     Dict[str, str] = field(default_factory=dict)
    # e.g. {"temperature": "positive"}  → temperature → this var is positive
    description:       str            = ""


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DynamicVariableGenerator:
    """Detects domain saturation and synthesizes new state variables."""

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self._rng = rng or random.Random()
        # domain → list of (tick, ep_ig) recent readings
        self._ep_ig_history:     Dict[str, List[Tuple[int, float]]] = {}
        # domain → last tick a variable was synthesized
        self._last_synthesis:    Dict[str, int]                     = {}
        # set of all synthesized variable names (to avoid duplicates)
        self._synthesized_names: Set[str]                           = set()
        # domain → list of SynthesizedVariable added so far
        self.synthesized:        Dict[str, List[SynthesizedVariable]] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def record_ep_ig(self, domain: str, ep_ig: float, tick: int) -> None:
        """Record the epistemic information gain for a domain tick."""
        if domain not in self._ep_ig_history:
            self._ep_ig_history[domain] = []
        self._ep_ig_history[domain].append((tick, ep_ig))
        # Keep only recent history
        self._ep_ig_history[domain] = self._ep_ig_history[domain][-50:]

    def maybe_synthesize(
        self,
        domain:        str,
        current_vars:  Dict[str, float],
        causal_rules:  List,
        tick:          int,
    ) -> Optional[SynthesizedVariable]:
        """Check if domain is saturated; if so, synthesize a new variable.

        Returns the new SynthesizedVariable if one was created, else None.
        """
        if not self._is_saturated(domain, tick):
            return None
        if len(current_vars) >= MAX_VARIABLES_PER_DOMAIN:
            logger.debug("[dynvar] domain=%s at variable cap (%d)", domain, MAX_VARIABLES_PER_DOMAIN)
            return None
        cooldown = self._last_synthesis.get(domain, 0)
        if tick - cooldown < SYNTHESIS_COOLDOWN_TICKS:
            return None

        var = self._select_and_synthesize(domain, current_vars, causal_rules, tick)
        if var:
            self._last_synthesis[domain] = tick
            self._synthesized_names.add(var.name)
            if domain not in self.synthesized:
                self.synthesized[domain] = []
            self.synthesized[domain].append(var)
            logger.info(
                "[dynvar] synthesized name=%s domain=%s type=%s parents=%s tick=%d",
                var.name, domain, var.synthesis_type, var.parent_vars, tick,
            )
        return var

    def inject_into_domain_states(
        self,
        var:          SynthesizedVariable,
        domain_states: Dict[str, Dict[str, float]],
    ) -> None:
        """Add the new variable to the domain's state dict."""
        if var.domain not in domain_states:
            domain_states[var.domain] = {}
        domain_states[var.domain][var.name] = var.initial_value

    def compute_derivative_update(
        self,
        parent_name:   str,
        current_value: float,
        prev_value:    float,
    ) -> float:
        """Compute the value of a derivative variable at this tick."""
        return round(current_value - prev_value, 4)

    def compute_threshold_update(
        self,
        parent_name:   str,
        current_value: float,
        threshold:     float,
    ) -> float:
        """Compute 0.0 or 1.0 for a threshold gate variable."""
        return 1.0 if current_value > threshold else 0.0

    # ── Saturation detection ───────────────────────────────────────────────

    def _is_saturated(self, domain: str, tick: int) -> bool:
        """Return True if this domain has been below the ep_ig threshold long enough."""
        hist = self._ep_ig_history.get(domain, [])
        if len(hist) < SATURATION_TICKS:
            return False
        recent = [ig for _, ig in hist[-SATURATION_TICKS:]]
        return max(recent) < MIN_EP_IG_FOR_SATURATION

    # ── Strategy selection ─────────────────────────────────────────────────

    def _select_and_synthesize(
        self,
        domain:       str,
        current_vars: Dict[str, float],
        causal_rules: List,
        tick:         int,
    ) -> Optional[SynthesizedVariable]:
        """Pick the best synthesis strategy for the current domain state."""
        existing = set(current_vars.keys()) | self._synthesized_names

        # Strategy priority: threshold > derivative > compound
        # Threshold first: generates conditional edge opportunities immediately
        var = self._synthesize_threshold(domain, current_vars, existing, tick)
        if var:
            return var
        var = self._synthesize_derivative(domain, current_vars, existing, tick)
        if var:
            return var
        return self._synthesize_compound(domain, current_vars, existing, tick)

    # ── Strategy 1: Derivative ────────────────────────────────────────────

    def _synthesize_derivative(
        self,
        domain:       str,
        current_vars: Dict[str, float],
        existing:     Set[str],
        tick:         int,
    ) -> Optional[SynthesizedVariable]:
        """Create a dX/dt variable for the most active variable in the domain."""
        # Pick the variable with highest variance (most active causal driver)
        candidates = [v for v in current_vars if f"{v}_rate" not in existing]
        if not candidates:
            return None

        # Prefer variables that are direct action targets (most observed)
        _PREFERRED = {
            "chemistry":  ["temperature", "reactants", "reaction_energy"],
            "physics":    ["force", "velocity", "kinetic_energy"],
            "biology":    ["pathogens", "cells", "energy"],
            "technology": ["robot_activity", "sensor_coverage", "data_quality"],
            "ecosystem":  ["deer", "wolves", "grass"],
            "astronomy":  ["asteroids", "radiation_pressure"],
        }
        preferred = [v for v in _PREFERRED.get(domain, []) if v in candidates]
        parent = preferred[0] if preferred else candidates[0]

        name = f"{parent}_rate"
        return SynthesizedVariable(
            name           = name,
            domain         = domain,
            synthesis_type = "derivative",
            parent_vars    = [parent],
            initial_value  = 0.0,
            added_at_tick  = tick,
            causal_priors  = {parent: "positive"},
            description    = (
                f"Rate of change of {parent} per tick. "
                f"Computed as Δ{parent}/Δt. "
                f"May causally affect downstream variables independently of {parent} itself."
            ),
        )

    # ── Strategy 2: Threshold gate ────────────────────────────────────────

    def _synthesize_threshold(
        self,
        domain:       str,
        current_vars: Dict[str, float],
        existing:     Set[str],
        tick:         int,
    ) -> Optional[SynthesizedVariable]:
        """Create a binary gate variable that fires when a key variable exceeds its mean."""
        # Target variables that appear in bidirectional causal records
        # (they are the most likely to benefit from a threshold split)
        _THRESHOLD_TARGETS = {
            "chemistry":  ("temperature",    50.0),
            "technology": ("sensor_coverage", 5.0),
            "biology":    ("pathogens",        5.0),
            "physics":    ("velocity",        20.0),
            "ecosystem":  ("grass",           50.0),
            "astronomy":  ("asteroids",      200.0),
        }
        target_spec = _THRESHOLD_TARGETS.get(domain)
        if not target_spec:
            return None
        parent_var, default_threshold = target_spec
        if parent_var not in current_vars:
            return None

        gate_name = f"{parent_var}_high"
        if gate_name in existing:
            return None

        # Use current value as threshold if it's in a reasonable range
        current_val = current_vars[parent_var]
        threshold   = current_val if current_val > 0 else default_threshold

        return SynthesizedVariable(
            name           = gate_name,
            domain         = domain,
            synthesis_type = "threshold",
            parent_vars    = [parent_var],
            initial_value  = 1.0 if current_val > threshold else 0.0,
            threshold      = round(threshold, 2),
            added_at_tick  = tick,
            causal_priors  = {parent_var: "positive"},
            description    = (
                f"Boolean gate: 1.0 when {parent_var} > {threshold:.2f}, else 0.0. "
                f"Enables conditional causality detection. "
                f"Will appear as mediating variable in conditional edges."
            ),
        )

    # ── Strategy 3: Compound (cross-domain) ──────────────────────────────

    def _synthesize_compound(
        self,
        domain:       str,
        current_vars: Dict[str, float],
        existing:     Set[str],
        tick:         int,
    ) -> Optional[SynthesizedVariable]:
        """Create a cross-domain interaction variable (e.g. thermal_bio_stress)."""
        _COMPOUND_SPECS: List[Tuple[str, str, str, str, str, float]] = [
            # (domain, var1, other_domain, var2, compound_name, initial)
            ("chemistry", "temperature", "biology",    "cells",          "thermal_cell_stress",   0.0),
            ("physics",   "heat",        "chemistry",  "reaction_rate",  "thermal_reaction_drive", 0.0),
            ("technology","sensor_coverage","astronomy","radiation_pressure","radiation_sensor_noise",0.0),
            ("biology",   "immune_response","chemistry","pH",            "immune_pH_coupling",     0.0),
            ("physics",   "force",       "technology", "robot_activity", "mechanical_robot_wear",  0.0),
            ("ecosystem", "grass",       "chemistry",  "reaction_energy","biomass_energy_proxy",   0.0),
        ]
        for spec in _COMPOUND_SPECS:
            d1, v1, d2, v2, cname, init = spec
            if d1 != domain:
                continue
            if cname in existing:
                continue
            if v1 not in current_vars:
                continue
            return SynthesizedVariable(
                name           = cname,
                domain         = domain,
                synthesis_type = "compound",
                parent_vars    = [v1, v2],
                initial_value  = init,
                compound_domains = [d1, d2],
                added_at_tick  = tick,
                causal_priors  = {v1: "positive", v2: "positive"},
                description    = (
                    f"Cross-domain interaction between {v1} ({d1}) and {v2} ({d2}). "
                    f"Computed as {v1} × {v2} (normalised). "
                    f"Feeds the CrossDomainTheoryEngine."
                ),
            )
        return None