"""
tools_bridge.knowledge_seeder
=============================
Injects fresh knowledge into the brain by searching for unknown concepts
and converting Wikipedia summaries into knowledge-page facts.

Solves the "new_facts=0" plateau: once the static JSON pages are exhausted,
this seeder discovers real content via the search and wikipedia tools and
feeds it back to the learner as new pages.

Usage (from event_loop)
-----------------------
    from digital_baby.tools_bridge.knowledge_seeder import maybe_seed_knowledge

    new_page = maybe_seed_knowledge(
        unknown_concepts=list(result.unknown_concepts),
        tick=tick,
        world_path=self.world_path,   # saves page to disk so it persists
    )
    if new_page:
        result2 = self.learner.learn_from_page(new_page)
        self.curiosity.register_unknowns(new_page['topic'], result2.unknown_concepts)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from digital_baby.tools_bridge import get_router

logger = logging.getLogger(__name__)

# Only seed once every N ticks, and only when facts are stagnant
_SEED_INTERVAL       = 15
_MIN_STAGNANT_TICKS  = 3   # need N consecutive new_facts=0 before seeding
_MAX_FACTS_PER_PAGE  = 20
_last_seed_tick      = -_SEED_INTERVAL
_stagnant_count      = 0
_seeded_concepts: set = set()


def _extract_facts_from_summary(concept: str, summary: str) -> List[str]:
    """
    Turn a Wikipedia summary into knowledge-page style fact strings.
    Uses knowledge_ingestion's patterns (single source of truth).
    """
    try:
        from digital_baby.brain.knowledge_ingestion import (
            _RELATION_PATTERNS, _clean_concept, BLOCKED_ONTOLOGY_TYPES
        )
    except ImportError:
        return []

    facts: List[str] = []
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
            facts.append(f"{subj} {relation} {obj}")
            if len(facts) >= _MAX_FACTS_PER_PAGE:
                return facts

    # Always include a basic "is" fact so the page isn't empty
    if not facts and concept:
        clean = concept.replace("_", " ")
        facts.append(f"{concept} is topic")

    return facts


def _build_page(concept: str, summary: str, domain: str = "general") -> Optional[Dict]:
    """Build a knowledge-page dict from a Wikipedia summary."""
    facts = _extract_facts_from_summary(concept, summary)
    if not facts:
        return None
    return {
        "topic":     concept,
        "domain":    domain,
        "facts":     sorted(set(facts)),
        "generated": True,
        "source":    "tools_bridge_seeder",
    }


def _infer_domain(concept: str, summary: str) -> str:
    """Guess domain from concept name and summary keywords."""
    text = (concept + " " + summary).lower()
    domain_keywords = {
        "physics":      ["force", "mass", "velocity", "energy", "momentum", "gravity"],
        "chemistry":    ["chemical", "reaction", "molecule", "atom", "compound", "acid"],
        "biology":      ["cell", "organism", "protein", "gene", "enzyme", "bacteria"],
        "ecology":      ["ecosystem", "predator", "prey", "habitat", "species", "food"],
        "astronomy":    ["star", "planet", "orbit", "galaxy", "solar", "comet"],
        "neuroscience": ["neuron", "brain", "synapse", "cortex", "dopamine", "memory"],
        "economics":    ["market", "price", "demand", "supply", "inflation", "gdp"],
        "climate":      ["temperature", "climate", "carbon", "atmosphere", "glacier"],
        "materials":    ["material", "metal", "alloy", "crystal", "polymer", "ceramic"],
        "technology":   ["robot", "computer", "sensor", "algorithm", "network", "data"],
    }
    scores: Dict[str, int] = {}
    for domain, keywords in domain_keywords.items():
        scores[domain] = sum(1 for kw in keywords if kw in text)
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else "general"


def maybe_seed_knowledge(
    unknown_concepts: List[str],
    tick:             int,
    new_facts_this_tick: int = 0,
    world_path:       Optional[Path] = None,
) -> Optional[Dict]:
    """
    Attempt to seed a new knowledge page from an unknown concept.

    Rate-limited to once every _SEED_INTERVAL ticks.
    Only fires when the brain has been finding new_facts=0 for several
    consecutive ticks (stagnant learning signal).

    Parameters
    ----------
    unknown_concepts     : concepts the brain asked about but couldn't find
    tick                 : current tick number
    new_facts_this_tick  : number of new facts found this tick (0 = stagnant)
    world_path           : if set, saves the page as a JSON file so the
                           brain loads it on future ticks

    Returns
    -------
    A knowledge-page dict, or None if nothing was seeded.
    """
    global _last_seed_tick, _stagnant_count

    # Track stagnation
    if new_facts_this_tick == 0:
        _stagnant_count += 1
    else:
        _stagnant_count = 0

    # Rate limit + stagnation gate
    if tick - _last_seed_tick < _SEED_INTERVAL:
        return None
    if _stagnant_count < _MIN_STAGNANT_TICKS:
        return None
    if not unknown_concepts:
        return None

    router = get_router()
    if "wikipedia" not in getattr(router, "available_tools", []):
        return None

    # Pick the best unseen concept to seed
    candidates = [c for c in unknown_concepts if c not in _seeded_concepts
                  and len(c) > 3 and "_" not in c[:2]]
    if not candidates:
        # All tried — reset so we can retry with new concepts
        _seeded_concepts.clear()
        candidates = [c for c in unknown_concepts if len(c) > 3]
    if not candidates:
        return None

    # Prefer concepts that look like real words (not synthetic like neuro_ecosystem)
    real = [c for c in candidates if not any(
        c.startswith(p) for p in ("neuro_", "alien_", "cond_", "synth_")
    )]
    concept = real[0] if real else candidates[0]
    _seeded_concepts.add(concept)

    # Fetch Wikipedia summary
    wiki_result = router.dispatch({
        "tool":  "wikipedia",
        "input": {"query": concept.replace("_", " "), "full_summary": True},
    })

    if wiki_result["status"] != "success":
        # Fallback: try search to find a related real article title
        search_result = router.dispatch({
            "tool":  "search",
            "input": {"query": concept.replace("_", " "), "max_results": 3},
        })
        if search_result["status"] == "success":
            hits = search_result["output"].get("results", [])
            if hits:
                # Use the first result's snippet as the summary
                summary = hits[0].get("snippet", "")
                title   = hits[0].get("title", concept)
                logger.info("[seeder] search_fallback concept=%s title=%s", concept, title)
            else:
                return None
        else:
            return None
    else:
        summary = wiki_result["output"].get("summary", "")
        title   = wiki_result["output"].get("title", concept)

    if not summary:
        return None

    domain = _infer_domain(concept, summary)
    page   = _build_page(concept, summary, domain)

    if not page or not page["facts"]:
        logger.debug("[seeder] no facts extracted for concept=%s", concept)
        return None

    _last_seed_tick = tick
    _stagnant_count = 0

    # Persist page to disk so it reloads on future runs
    if world_path is not None:
        try:
            safe_name = re.sub(r"[^\w]", "_", concept)[:40]
            page_path = Path(world_path) / f"seeded_{safe_name}.json"
            page_path.write_text(json.dumps(page, indent=2), encoding="utf-8")
            logger.info("[seeder] saved page=%s facts=%d domain=%s",
                        page_path.name, len(page["facts"]), domain)
        except OSError as exc:
            logger.debug("[seeder] could not save page: %s", exc)

    logger.info("[seeder] tick=%d concept=%s domain=%s facts=%d",
                tick, concept, domain, len(page["facts"]))
    return page