"""
Tool: context  (short-term / working memory)
============================================
In-process rolling window of recent observations, thoughts, goals, and
errors.  Data lives only for the duration of the process — use 'notes'
for persistence across restarts.

Actions
-------
    add     – text (str), type (str, optional: observation|thought|goal|error)
    get     – limit (int, default 10), type (str, optional filter)
    search  – query (str), limit (int, default 5)
    clear   – (no extra fields)
    stats   – (no extra fields)

Output (success) varies by action — always inside
    {"status": "success", "output": {...}}
"""
from __future__ import annotations

import re
from collections import deque
from datetime import datetime
from typing import Dict, List

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from tools.base import ok, err

# ── Module-level store (singleton per process) ─────────────────────────────

_MAX = 200
_store: deque = deque(maxlen=_MAX)


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _infer_type(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(goal|want|plan|intend|objective|aim)\b", t):
        return "goal"
    if re.search(r"\b(observed|noticed|found|discovered|saw)\b", t):
        return "observation"
    if re.search(r"\b(error|failed|exception|broke|crash)\b", t):
        return "error"
    return "thought"


# ── Entry point ───────────────────────────────────────────────────────────

def run(input: dict) -> dict:
    action = input.get("action", "get")

    # ── add ───────────────────────────────────────────────────────────────
    if action == "add":
        text = (input.get("text") or "").strip()
        if not text:
            return err("'text' is required for add.")
        entry_type = input.get("type") or _infer_type(text)
        entry = {"text": text, "type": entry_type, "timestamp": _now()}
        _store.append(entry)
        return ok({"added": entry, "store_size": len(_store)})

    # ── get ───────────────────────────────────────────────────────────────
    elif action == "get":
        limit       = max(1, int(input.get("limit", 10)))
        filter_type = input.get("type")
        items = list(_store)
        if filter_type:
            items = [e for e in items if e["type"] == filter_type]
        result = items[-limit:]
        return ok({"entries": result, "count": len(result)})

    # ── search ────────────────────────────────────────────────────────────
    elif action == "search":
        query = (input.get("query") or "").strip()
        if not query:
            return err("'query' is required for search.")
        limit = max(1, int(input.get("limit", 5)))
        q = query.lower()
        matches = [e for e in _store if q in e["text"].lower()]
        return ok({"query": query, "entries": matches[-limit:],
                    "count": len(matches)})

    # ── clear ─────────────────────────────────────────────────────────────
    elif action == "clear":
        n = len(_store)
        _store.clear()
        return ok({"cleared": n, "message": f"Cleared {n} context entries."})

    # ── stats ─────────────────────────────────────────────────────────────
    elif action == "stats":
        counts: Dict[str, int] = {}
        for e in _store:
            counts[e["type"]] = counts.get(e["type"], 0) + 1
        return ok({"total": len(_store), "by_type": counts})

    else:
        return err(
            f"Unknown action '{action}'. "
            "Valid: add, get, search, clear, stats."
        )