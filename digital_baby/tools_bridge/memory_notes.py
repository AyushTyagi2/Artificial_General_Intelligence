"""
tools_bridge.memory_notes
=========================
Writes important brain discoveries to the notes tool as durable long-term
records that survive process restarts and are human-readable.

What gets saved
---------------
- Confirmed hypotheses  (confidence >= 0.70, evidence >= 5)
- Discovered scientific laws  (r² >= 0.80)
- High-confidence causal rules  (confidence >= 0.75, observations >= 8)

The notes tool stores everything in  notes.json  at the project root.
Each note is tagged so the brain can later search by category.

Usage (from event_loop)
-----------------------
    from digital_baby.tools_bridge.memory_notes import maybe_record_discoveries

    # Call once per tick — internally rate-limited
    maybe_record_discoveries(
        hypotheses=hypotheses,
        causal_rules=memory.get_causal_rules(),
        discovered_laws=laws,          # list of LawRecord objects (optional)
        tick=tick,
        store_path="notes.json",       # optional override
    )
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

from digital_baby.tools_bridge import get_router

logger = logging.getLogger(__name__)

# How often (in ticks) to flush discoveries to notes
_FLUSH_INTERVAL = 50

# Track what we've already noted (in-process dedup)
_noted_hypotheses:   set = set()
_noted_causal_rules: set = set()
_noted_laws:         set = set()
_last_flush_tick:    int = -_FLUSH_INTERVAL


def maybe_record_discoveries(
    hypotheses:    List[Any],
    causal_rules:  List[Any],
    discovered_laws: Optional[List[Any]] = None,
    tick:          int = 0,
    store_path:    str = "notes.json",
) -> int:
    """
    Persist noteworthy discoveries to the notes tool.

    Returns the number of new notes written this call.
    """
    global _last_flush_tick
    if tick - _last_flush_tick < _FLUSH_INTERVAL:
        return 0
    _last_flush_tick = tick

    router = get_router()
    if "notes" not in getattr(router, "available_tools", []):
        return 0

    written = 0

    # ── Confirmed hypotheses ──────────────────────────────────────────────
    for h in hypotheses:
        rule       = getattr(h, "rule", "")
        confidence = getattr(h, "confidence", 0.0)
        evidence   = getattr(h, "supporting_evidence", 0)
        if not rule or rule in _noted_hypotheses:
            continue
        if confidence >= 0.70 and evidence >= 5:
            result = router.dispatch({
                "tool": "notes",
                "input": {
                    "action":     "create",
                    "store_path": store_path,
                    "title":      f"Confirmed: {rule[:80]}",
                    "content":    (
                        f"Rule: {rule}\n"
                        f"Confidence: {confidence:.3f}\n"
                        f"Supporting evidence: {evidence}\n"
                        f"Discovered at tick: {tick}"
                    ),
                    "tags": ["hypothesis", "confirmed"],
                },
            })
            if result["status"] == "success":
                _noted_hypotheses.add(rule)
                written += 1
                logger.info("[tools_bridge.notes] confirmed_hypothesis rule=%s", rule[:60])

    # ── High-confidence causal rules ──────────────────────────────────────
    for rule in causal_rules:
        cause       = getattr(rule, "cause", "")
        effect      = getattr(rule, "effect", "")
        confidence  = getattr(rule, "confidence", 0.0)
        observations = getattr(rule, "observations", 0)
        direction   = getattr(rule, "direction", "")
        key         = f"{cause}->{effect}"
        if not cause or not effect or key in _noted_causal_rules:
            continue
        if confidence >= 0.75 and observations >= 8:
            result = router.dispatch({
                "tool": "notes",
                "input": {
                    "action":     "create",
                    "store_path": store_path,
                    "title":      f"Causal law: {cause} {direction} {effect}",
                    "content":    (
                        f"Cause: {cause}\n"
                        f"Effect: {effect}\n"
                        f"Direction: {direction}\n"
                        f"Confidence: {confidence:.3f}\n"
                        f"Observations: {observations}\n"
                        f"Recorded at tick: {tick}"
                    ),
                    "tags": ["causal_rule", "high_confidence"],
                },
            })
            if result["status"] == "success":
                _noted_causal_rules.add(key)
                written += 1
                logger.info("[tools_bridge.notes] causal_rule %s->%s conf=%.2f",
                            cause, effect, confidence)

    # ── Scientific laws ───────────────────────────────────────────────────
    for law in (discovered_laws or []):
        eq  = getattr(law, "equation_str", "") or getattr(law, "name", "")
        r2  = getattr(law, "r_squared", 0.0)
        key = eq
        if not eq or key in _noted_laws:
            continue
        if r2 >= 0.80:
            result = router.dispatch({
                "tool": "notes",
                "input": {
                    "action":     "create",
                    "store_path": store_path,
                    "title":      f"Law discovered: {eq[:80]}",
                    "content":    (
                        f"Equation: {eq}\n"
                        f"R²: {r2:.4f}\n"
                        f"Data points: {getattr(law, 'n_datapoints', '?')}\n"
                        f"Discovered at tick: {tick}"
                    ),
                    "tags": ["scientific_law"],
                },
            })
            if result["status"] == "success":
                _noted_laws.add(key)
                written += 1
                logger.info("[tools_bridge.notes] law eq=%s r2=%.3f", eq[:60], r2)

    return written