"""
tools_bridge.wikipedia
======================
Wikipedia access via the tools bridge.

fetch_summary(concept)    — returns the article intro as a string
fetch_relations(concept)  — returns (subj, rel, obj) triplets
cooccurrence_check(a, b)  — returns (a_mentions_b, b_mentions_a)

Falls back gracefully to "" / [] on any error.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from digital_baby.tools_bridge import get_router

logger = logging.getLogger(__name__)


def fetch_summary(concept: str, full: bool = False) -> str:
    """
    Fetch a Wikipedia summary for *concept*.
    Returns "" on failure.
    """
    query = concept.replace("_", " ").strip()
    if not query:
        return ""

    router = get_router()
    result = router.dispatch({
        "tool":  "wikipedia",
        "input": {"query": query, "full_summary": full},
    })

    if result["status"] != "success":
        logger.debug("[tools_bridge.wikipedia] failed concept=%s err=%s",
                     concept, result["output"].get("error"))
        return ""

    return result["output"].get("summary", "")


def fetch_relations(
    concept: str,
    max_relations: int = 12,
) -> List[Tuple[str, str, str]]:
    """
    Fetch Wikipedia summary and extract (subj, rel, obj) triplets using the
    same _RELATION_PATTERNS as knowledge_ingestion (single source of truth).
    Returns [] on failure or when patterns find nothing.
    """
    summary = fetch_summary(concept, full=True)
    if not summary:
        return []

    # Reuse the battle-tested patterns from knowledge_ingestion directly
    try:
        from digital_baby.brain.knowledge_ingestion import (
            _RELATION_PATTERNS, _clean_concept, BLOCKED_ONTOLOGY_TYPES
        )
    except ImportError:
        logger.debug("[tools_bridge.wikipedia] could not import knowledge_ingestion patterns")
        return []

    triplets: List[Tuple[str, str, str]] = []
    for _label, relation, pattern in _RELATION_PATTERNS:
        for match in pattern.finditer(summary):
            try:
                subj = _clean_concept(match.group(1))
                obj  = _clean_concept(match.group(2))
            except IndexError:
                continue
            if not subj or not obj or subj == obj:
                continue
            if obj.replace("_", " ") in BLOCKED_ONTOLOGY_TYPES:
                continue
            triplets.append((subj, relation, obj))
            if len(triplets) >= max_relations:
                return triplets

    logger.debug("[tools_bridge.wikipedia] concept=%s relations=%d", concept, len(triplets))
    return triplets


def cooccurrence_check(
    concept_a: str,
    concept_b: str,
    timeout: float = 8.0,
) -> Tuple[bool, bool]:
    """
    Check whether concept_a and concept_b mention each other in Wikipedia.
    Returns (a_mentions_b, b_mentions_a).
    """
    a_text = fetch_summary(concept_a, full=True).lower()
    b_text = fetch_summary(concept_b, full=True).lower()

    b_clean = concept_b.replace("_", " ").lower()
    a_clean = concept_a.replace("_", " ").lower()

    return (b_clean in a_text), (a_clean in b_text)