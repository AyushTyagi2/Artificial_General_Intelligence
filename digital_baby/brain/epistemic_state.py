"""Epistemic State Tracker — Architecture v2.

Maintains a Beta distribution over the confidence of every edge in the
knowledge graph.  Surfaces:

  - per-edge entropy  H(Beta(α,β))
  - global entropy map (sorted by uncertainty)
  - information-gain estimate for a proposed intervention
  - diminishing-returns detector (returns True when an intervention on a
    given cause variable would produce < MIN_MARGINAL_IG bits)
  - epistemic reward signal (sum of entropy reduced this tick)

Beta(α, β) models our belief that a causal edge truly exists:
  α = prior_strength + confirmations
  β = prior_strength + refutations

Entropy of Beta(α,β) ≈ ln B(α,β) − (α−1)ψ(α) − (β−1)ψ(β) + (α+β−2)ψ(α+β)

where B is the Beta function and ψ is the digamma function.
We approximate digamma with a fast series expansion.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRIOR_STRENGTH: float = 1.0        # pseudo-counts added to both α and β
MIN_MARGINAL_IG: float = 0.02      # bits — below this an intervention is "stale"
MAX_HISTORY_LEN: int  = 500        # how many (cause, tick, ig_reduced) to keep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _digamma(x: float) -> float:
    """Fast digamma approximation (Lanczos / asymptotic series)."""
    if x <= 0:
        return 0.0
    # Shift x up until x >= 6 for the asymptotic expansion
    result = 0.0
    while x < 6:
        result -= 1.0 / x
        x += 1.0
    # Asymptotic series
    result += math.log(x) - 0.5 / x
    x2 = x * x
    result -= 1 / (12 * x2) - 1 / (120 * x2 * x2)
    return result


def _beta_entropy(alpha: float, beta: float) -> float:
    """Uncertainty measure for Beta(alpha, beta).

    We use the normalised variance Var/(max_var) which ranges 0–1:
      Var(Beta) = αβ / ((α+β)²(α+β+1))
      max_var   = 0.25  (achieved at α=β=1, uniform)

    This avoids the negative values of differential entropy at high
    concentration, while still peaking at maximum uncertainty and
    approaching 0 for near-certain beliefs.
    """
    if alpha <= 0 or beta <= 0:
        return 0.0
    ab = alpha + beta
    var = (alpha * beta) / (ab * ab * (ab + 1))
    return min(1.0, var / 0.25)   # normalised to [0, 1]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EdgeBelief:
    """Beta distribution over 'does this causal edge truly hold?'"""
    cause:     str
    effect:    str
    alpha:     float = field(default=PRIOR_STRENGTH)
    beta:      float = field(default=PRIOR_STRENGTH)
    last_tick: int   = 0

    @property
    def mean_confidence(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def entropy(self) -> float:
        return _beta_entropy(self.alpha, self.beta)

    def update(self, confirmed: bool, tick: int) -> float:
        """Bayesian update. Returns bits of entropy reduced."""
        h_before = self.entropy
        if confirmed:
            self.alpha += 1.0
        else:
            self.beta += 1.0
        self.last_tick = tick
        return max(0.0, h_before - self.entropy)

    def __repr__(self) -> str:
        return (
            f"EdgeBelief({self.cause}→{self.effect} "
            f"α={self.alpha:.1f} β={self.beta:.1f} "
            f"conf={self.mean_confidence:.3f} H={self.entropy:.3f})"
        )


@dataclass
class EpistemicReward:
    """Summary of epistemic progress this tick."""
    tick:          int
    entropy_reduced: float    # total bits reduced
    edges_updated: int
    novel_edges:   int        # edges seen for the first time

    @property
    def is_productive(self) -> bool:
        return self.entropy_reduced > MIN_MARGINAL_IG or self.novel_edges > 0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EpistemicStateTracker:
    """Tracks Bayesian uncertainty over every causal edge.

    Usage (inside BabyEventLoop)
    -----------------------------
    # Initialise once:
    self.epistemic = EpistemicStateTracker()

    # After each causal discovery:
    ig = self.epistemic.update_edge(cause, effect, confirmed=True, tick=tick)

    # Reward signal:
    reward = self.epistemic.tick_reward(tick)

    # Intervention selection: find most uncertain edges
    top = self.epistemic.top_uncertain_edges(n=5)

    # Diminishing-returns check:
    if self.epistemic.is_stale(cause, tick):
        choose_different_intervention()
    """

    def __init__(self) -> None:
        self._beliefs: Dict[Tuple[str, str], EdgeBelief] = {}
        self._tick_ig: Dict[int, float] = defaultdict(float)
        self._tick_novel: Dict[int, int] = defaultdict(int)
        self._tick_updated: Dict[int, int] = defaultdict(int)
        # intervention_staleness[cause] = cumulative observations with low IG
        self._stale_count: Dict[str, int] = defaultdict(int)

    # ── Public API ─────────────────────────────────────────────────────────────

    def update_edge(
        self,
        cause:     str,
        effect:    str,
        confirmed: bool,
        tick:      int,
        confidence_hint: Optional[float] = None,
    ) -> float:
        """Record one observation for (cause → effect) and return IG in nats."""
        key = (cause, effect)
        novel = key not in self._beliefs

        if novel:
            belief = EdgeBelief(cause=cause, effect=effect)
            if confidence_hint is not None:
                # Initialise from an external confidence estimate
                # α/(α+β) = c  →  α = c·strength, β = (1−c)·strength
                s = PRIOR_STRENGTH * 4
                belief.alpha = confidence_hint * s + PRIOR_STRENGTH
                belief.beta  = (1 - confidence_hint) * s + PRIOR_STRENGTH
            self._beliefs[key] = belief
            self._tick_novel[tick] += 1

        ig = self._beliefs[key].update(confirmed, tick)
        self._tick_ig[tick]      += ig
        self._tick_updated[tick] += 1

        if ig < MIN_MARGINAL_IG:
            self._stale_count[cause] = self._stale_count.get(cause, 0) + 1
        else:
            self._stale_count[cause] = 0   # reset on informative update

        return ig

    def register_edge_from_graph(
        self,
        cause:      str,
        effect:     str,
        confidence: float,
        evidence:   int,
        tick:       int = 0,
    ) -> None:
        """Seed belief from existing knowledge-graph edge (no IG generated)."""
        key = (cause, effect)
        if key in self._beliefs:
            return
        s = max(1, evidence)
        alpha = confidence * s + PRIOR_STRENGTH
        beta  = (1 - confidence) * s + PRIOR_STRENGTH
        self._beliefs[key] = EdgeBelief(
            cause=cause, effect=effect,
            alpha=alpha, beta=beta,
            last_tick=tick,
        )

    def tick_reward(self, tick: int) -> EpistemicReward:
        """Return the epistemic reward accumulated during this tick."""
        return EpistemicReward(
            tick=tick,
            entropy_reduced=self._tick_ig.get(tick, 0.0),
            edges_updated=self._tick_updated.get(tick, 0),
            novel_edges=self._tick_novel.get(tick, 0),
        )

    def top_uncertain_edges(self, n: int = 10) -> List[EdgeBelief]:
        """Return edges with highest entropy (most uncertain)."""
        return sorted(
            self._beliefs.values(),
            key=lambda b: b.entropy,
            reverse=True,
        )[:n]

    def top_uncertain_causes(self, n: int = 5) -> List[str]:
        """Return the cause-variables whose outgoing edges are most uncertain."""
        cause_entropy: Dict[str, float] = defaultdict(float)
        for belief in self._beliefs.values():
            cause_entropy[belief.cause] += belief.entropy
        return sorted(cause_entropy, key=lambda c: -cause_entropy[c])[:n]

    def is_stale(self, cause: str, current_tick: int, threshold: int = 6) -> bool:
        """Return True if repeated interventions on `cause` are yielding low IG.

        A cause is stale when:
        - Its outgoing edges have accumulated enough evidence (>threshold observations), AND
        - All outgoing edges have mean confidence > 0.70 (well-characterised)
        This detects the plateau pattern: repeated interventions on known relationships.
        """
        outgoing = [(c, e) for (c, e), b in self._beliefs.items() if c == cause]
        if len(outgoing) < 1:
            return False
        # Count total evidence across outgoing edges
        total_ev = sum(
            self._beliefs[(c, e)].alpha + self._beliefs[(c, e)].beta
            for c, e in outgoing
        )
        if total_ev < threshold * 2:  # need at least 2× threshold evidence
            return False
        # Check if all well-characterised (high confidence, low uncertainty)
        mean_h = sum(self._beliefs[(c, e)].entropy for c, e in outgoing) / len(outgoing)
        return mean_h < 0.10  # all edges are low-entropy (well-known)

    def expected_ig_for_cause(self, cause: str) -> float:
        """Estimate expected information gain from perturbing `cause`.

        = sum of entropy over all uncertain outgoing edges from this cause
        (proxy for how much we could learn by intervening on it).
        """
        total = 0.0
        for (c, _e), belief in self._beliefs.items():
            if c == cause:
                total += belief.entropy
        return total

    def get_belief(self, cause: str, effect: str) -> Optional[EdgeBelief]:
        return self._beliefs.get((cause, effect))

    def edge_count(self) -> int:
        return len(self._beliefs)

    def mean_entropy(self) -> float:
        if not self._beliefs:
            return 0.0
        return sum(b.entropy for b in self._beliefs.values()) / len(self._beliefs)

    def entropy_series(self, last_n: int = 100) -> List[Tuple[int, float]]:
        """Return (tick, cumulative_ig) pairs for the last N ticks with activity."""
        ticks = sorted(self._tick_ig.keys())[-last_n:]
        return [(t, self._tick_ig[t]) for t in ticks]

    def summary(self) -> Dict:
        """Compact summary for dashboard / logging."""
        beliefs = list(self._beliefs.values())
        if not beliefs:
            return {"edges": 0, "mean_entropy": 0.0, "max_entropy": 0.0}
        entropies = [b.entropy for b in beliefs]
        return {
            "edges":        len(beliefs),
            "mean_entropy": sum(entropies) / len(entropies),
            "max_entropy":  max(entropies),
            "stale_causes": [c for c, n in self._stale_count.items() if n >= 4],
            "top_uncertain": [
                {"edge": f"{b.cause}→{b.effect}", "H": round(b.entropy, 3)}
                for b in self.top_uncertain_edges(5)
            ],
        }