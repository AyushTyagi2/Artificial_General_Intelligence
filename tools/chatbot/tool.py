"""
Tool: chatbot
=============
Conversational interface for reasoning, hypothesis generation, and
knowledge summarisation.

Uses the Anthropic Claude API when ANTHROPIC_API_KEY is set in the
environment.  Falls back to a lightweight rule-based responder otherwise
so the system runs offline without errors.

Input
-----
    message    : str         (required)
    system     : str         (optional — instruction for the model)
    history    : list[dict]  (optional — prior turns:
                              [{"role": "user"|"assistant", "text": "..."}])
    max_tokens : int         (optional, default 256, max 4096)

Output (success)
----------------
    {
        "reply":   "...",
        "backend": "claude" | "fallback",
        "tokens":  int | null
    }
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import List

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from tools.base import ok, err

# ── Fallback responder ─────────────────────────────────────────────────────

_PATTERNS = [
    (r"\bwhat is\b",                 "That is a definitional question worth exploring carefully."),
    (r"\bwhy\b",                      "Multiple causal factors are likely involved."),
    (r"\bhow\b",                      "The mechanism probably involves several interacting components."),
    (r"\bpredict\b|\bforecast\b",     "Based on observed patterns, a gradual trend seems likely."),
    (r"\bhypothes\w+",                "A good hypothesis should be falsifiable and grounded in evidence."),
    (r"\blearn\b|\bknowledge\b",      "Learning proceeds by forming associations and updating beliefs."),
    (r"\bmemory\b|\bremember\b",      "Memory consolidation strengthens frequently accessed patterns."),
    (r"\bcurious\b|\bcuriosity\b",    "Curiosity drives exploration toward areas of high uncertainty."),
    (r"\bsummari[sz]e\b|\bsummary\b", "A summary distils the most important points into compact form."),
]

_FALLBACK_DEFAULT = (
    "I've processed your message. "
    "Set the ANTHROPIC_API_KEY environment variable for full reasoning capability."
)


def _fallback(message: str) -> str:
    ml = message.lower()
    for pattern, reply in _PATTERNS:
        if re.search(pattern, ml):
            return reply
    return _FALLBACK_DEFAULT


# ── Claude API backend ─────────────────────────────────────────────────────

def _claude(message: str, system: str, history: List[dict],
            max_tokens: int, api_key: str) -> str:
    messages = []
    for h in history:
        role    = h.get("role", "user")
        content = h.get("text") or h.get("content") or ""
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    body = json.dumps({
        "model":      "claude-haiku-4-5-20251001",
        "max_tokens": max_tokens,
        "system":     system,
        "messages":   messages,
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    return data["content"][0]["text"]


# ── Entry point ───────────────────────────────────────────────────────────

_DEFAULT_SYSTEM = (
    "You are the reasoning module of a Digital Baby learning agent. "
    "Be concise, factual, and helpful."
)


def run(input: dict) -> dict:
    message = (input.get("message") or "").strip()
    if not message:
        return err("'message' is required and must not be empty.")

    system     = input.get("system") or _DEFAULT_SYSTEM
    history    = input.get("history") or []
    max_tokens = max(1, min(int(input.get("max_tokens", 256)), 4096))

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            reply = _claude(message, system, history, max_tokens, api_key)
            return ok({"reply": reply, "backend": "claude", "tokens": None})
        except Exception as exc:
            # Degrade gracefully — never hard-fail
            return ok({
                "reply":   _fallback(message),
                "backend": "fallback",
                "tokens":  None,
                "warning": f"Claude API error ({exc}); using fallback.",
            })

    return ok({"reply": _fallback(message), "backend": "fallback", "tokens": None})