"""
tools_bridge.working_memory
============================
Uses the context tool as the brain's working memory — a rolling window
of what the agent observed and did each tick.

This makes it easy to ask the brain "what happened recently?" and to feed
that context into the chatbot for richer reasoning.

Usage (from event_loop)
-----------------------
    from digital_baby.tools_bridge.working_memory import record_tick, get_recent

    # At the end of each tick:
    record_tick(
        tick=tick,
        topic=selected_topic,
        domain=domain,
        action=action_str,
        new_facts=total_new_facts,
        hypotheses=len(hypotheses),
        reward=ep_reward_score,
        causal_rules=new_causal_rules,
    )

    # To retrieve recent context (e.g. to pass to chatbot):
    recent = get_recent(limit=5)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from digital_baby.tools_bridge import get_router

logger = logging.getLogger(__name__)


def record_tick(
    tick:          int,
    topic:         str,
    domain:        str,
    action:        str,
    new_facts:     int   = 0,
    hypotheses:    int   = 0,
    reward:        float = 0.0,
    causal_rules:  Optional[List[str]] = None,
    extra:         Optional[Dict[str, Any]] = None,
) -> None:
    """
    Store a one-line summary of this tick in the context tool.
    Fires-and-forgets — never raises.
    """
    router = get_router()
    if "context" not in getattr(router, "available_tools", []):
        return

    parts = [f"tick={tick}", f"topic={topic}", f"domain={domain}",
             f"action={action}", f"new_facts={new_facts}",
             f"hypotheses={hypotheses}", f"reward={reward:.4f}"]
    if causal_rules:
        parts.append(f"new_causal={causal_rules[0][:40]}")
    if extra:
        for k, v in extra.items():
            parts.append(f"{k}={v}")

    text = " | ".join(parts)
    try:
        router.dispatch({
            "tool":  "context",
            "input": {"action": "add", "text": text, "type": "observation"},
        })
    except Exception as exc:
        logger.debug("[tools_bridge.context] record_tick failed: %s", exc)


def get_recent(limit: int = 10, entry_type: Optional[str] = None) -> List[str]:
    """
    Return the most recent *limit* context entries as plain strings.
    """
    router = get_router()
    if "context" not in getattr(router, "available_tools", []):
        return []

    inp: Dict[str, Any] = {"action": "get", "limit": limit}
    if entry_type:
        inp["type"] = entry_type

    result = router.dispatch({"tool": "context", "input": inp})
    if result["status"] != "success":
        return []
    return [e.get("text", "") for e in result["output"].get("entries", [])]


def get_stats() -> Dict[str, Any]:
    """Return context store stats (total entries, counts by type)."""
    router = get_router()
    if "context" not in getattr(router, "available_tools", []):
        return {}
    result = router.dispatch({"tool": "context", "input": {"action": "stats"}})
    return result["output"] if result["status"] == "success" else {}