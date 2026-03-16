"""
tools_bridge.concept_chat
==========================
Uses the chatbot tool to generate brief explanations and extract
additional relations for concepts the brain cannot find in Wikipedia.

Designed as a fallback enrichment layer:
  Wikipedia → (miss) → chatbot → parsed triplets added to knowledge graph

Also provides ``ask_brain`` — a simple question-answering entry point
that feeds recent working-memory context to the LLM.

Usage (from event_loop or brain modules)
-----------------------------------------
    from digital_baby.tools_bridge.concept_chat import (
        explain_concept,
        ask_brain,
    )

    # Get LLM-derived triplets for a concept Wikipedia missed
    triplets = explain_concept("activation_energy")

    # Ask the LLM a free-form question with recent context
    answer = ask_brain("Why does increased temperature raise reaction rate?")
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from digital_baby.tools_bridge import get_router

logger = logging.getLogger(__name__)

# Simple relation words to parse out of LLM responses
_REL_WORDS = re.compile(
    r"\b(causes?|affects?|increases?|decreases?|produces?|inhibits?|"
    r"requires?|enables?|leads?\s+to|results?\s+in|is|orbits?|eats?|hunts?)\b",
    re.I,
)

_MAX_CONCEPT_LEN = 40
_BLOCKED = {"the", "a", "an", "it", "its", "this", "that", "is", "are", "was"}


def _clean(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text).strip().lower()
    text = re.sub(r"\s+", "_", text)
    return text[:_MAX_CONCEPT_LEN] if 2 <= len(text) <= _MAX_CONCEPT_LEN else ""


def _parse_triplets(text: str, concept: str, max_n: int = 8) -> List[Tuple[str, str, str]]:
    """Very lightweight triplet extraction from free-form LLM text."""
    triplets: List[Tuple[str, str, str]] = []
    concept_clean = _clean(concept)

    for sentence in re.split(r"[.!?\n]", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        m = _REL_WORDS.search(sentence)
        if not m:
            continue

        before = sentence[:m.start()].strip()
        after  = sentence[m.end():].strip()
        rel    = re.sub(r"\s+", "_", m.group(0).lower().strip())

        subj = _clean(before) or concept_clean
        obj  = _clean(after.split()[0] if after.split() else "")

        if not subj or not obj or subj == obj:
            continue
        if subj in _BLOCKED or obj in _BLOCKED:
            continue
        triplets.append((subj, rel, obj))
        if len(triplets) >= max_n:
            break

    return triplets


def explain_concept(
    concept: str,
    max_triplets: int = 8,
) -> List[Tuple[str, str, str]]:
    """
    Ask the chatbot to explain *concept* and extract relation triplets.

    Used as a fallback when Wikipedia returns nothing.  Returns [] if the
    chatbot tool is unavailable or no ANTHROPIC_API_KEY is set (the fallback
    responder rarely produces parseable triplets for obscure concepts).
    """
    router = get_router()
    if "chatbot" not in getattr(router, "available_tools", []):
        return []

    prompt = (
        f"Briefly explain the concept '{concept.replace('_', ' ')}' "
        f"in 2-3 sentences focused on causal relationships. "
        f"For example: 'X causes Y', 'A increases B', 'C inhibits D'."
    )
    result = router.dispatch({
        "tool":  "chatbot",
        "input": {
            "message":    prompt,
            "system":     (
                "You are a knowledge extraction assistant. "
                "Respond with short factual sentences containing causal relationships. "
                "Do not use bullet points or headers."
            ),
            "max_tokens": 120,
        },
    })

    if result["status"] != "success":
        return []

    reply = result["output"].get("reply", "")
    if result["output"].get("backend") == "fallback":
        # The built-in fallback doesn't produce useful triplets
        return []

    triplets = _parse_triplets(reply, concept, max_triplets)
    if triplets:
        logger.info("[tools_bridge.chat] concept=%s triplets=%d", concept, len(triplets))
    return triplets


def ask_brain(
    question:       str,
    recent_context: Optional[List[str]] = None,
    max_tokens:     int = 200,
) -> str:
    """
    Ask the chatbot a free-form question, optionally enriched with recent
    working-memory context.

    Returns the reply string, or "" if unavailable.
    """
    router = get_router()
    if "chatbot" not in getattr(router, "available_tools", []):
        return ""

    history = []
    if recent_context:
        ctx_text = "\n".join(recent_context[-5:])
        history = [{"role": "user",
                    "text": f"Recent observations:\n{ctx_text}"}]

    result = router.dispatch({
        "tool":  "chatbot",
        "input": {
            "message":    question,
            "system":     (
                "You are the reasoning module of a Digital Baby learning agent. "
                "Be concise and scientifically accurate. "
                "Focus on causal mechanisms."
            ),
            "history":    history,
            "max_tokens": max_tokens,
        },
    })

    if result["status"] != "success":
        return ""
    return result["output"].get("reply", "")