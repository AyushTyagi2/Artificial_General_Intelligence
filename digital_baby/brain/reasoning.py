"""Reasoning layer for relation extraction and contradiction detection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Conflict:
    """Represents a contradiction between two facts."""

    entity: str
    relation: str
    values: Tuple[str, str]


# ---------------------------------------------------------------------------
# Known relation tokens ordered longest-first so greedier matches win.
# E.g. "negatively_affects" must be tried before "affects".
# ---------------------------------------------------------------------------
_KNOWN_RELATIONS: List[str] = [
    "negatively_affects",
    "positively_affects",
    "negative_affects",
    "positive_affects",
    "mixed_affects",
    "inferred_affects",
    "causally_affects",
    "food_chain",
    "lives_in",
    "reacts_with",
    "controls",
    "affects",
    "orbits",
    "hunts",
    "needs",
    "eats",
    "has",
    "is",
]

# Build a single compiled regex:
#   ^(.+?)\s+(<relation>)\s+(.+)$
# The subject/object capture groups use lazy (.+?) so they don't swallow the
# relation keyword. Case-insensitive so "Is" and "IS" both match.
_RELATION_PATTERN: re.Pattern = re.compile(
    r"^(.+?)\s+(" + "|".join(re.escape(r) for r in _KNOWN_RELATIONS) + r")\s+(.+)$",
    re.IGNORECASE,
)


class Reasoner:
    """Simple symbolic reasoner with lightweight conflict checks."""

    def extract_relation(self, fact: str) -> Optional[Tuple[str, str, str]]:
        """Extract a (subject, relation, object) triplet from a fact string.

        Strategy
        --------
        1. Try the pattern-based matcher against all *known* relation tokens.
           This correctly handles multi-word subjects and objects, e.g.
           "carbon dioxide is greenhouse gas"  →  ("carbon dioxide", "is", "greenhouse gas")
        2. Fall back to the original whitespace-split heuristic only when no
           known relation is found, so novel relation words are still captured
           rather than silently dropped.

        Returns None if the fact has fewer than three tokens.
        """
        fact = fact.strip()

        # -- Pattern-based pass (preferred) --
        m = _RELATION_PATTERN.match(fact)
        if m:
            subject  = m.group(1).lower().strip()
            relation = m.group(2).lower().strip()
            obj      = m.group(3).lower().strip()
            # Strip trailing punctuation that world-pages sometimes include.
            obj = obj.rstrip(".,;:")
            return subject, relation, obj

        # -- Whitespace-split fallback for unknown relation tokens --
        tokens = fact.lower().split()
        if len(tokens) < 3:
            return None
        subject  = tokens[0]
        relation = tokens[1]
        obj      = " ".join(tokens[2:]).rstrip(".,;:")
        return subject, relation, obj

    def detect_contradictions(self, triplets: Iterable[Tuple[str, str, str]]) -> List[Conflict]:
        """Detect contradictions where one entity has competing values for a relation."""
        seen: Dict[Tuple[str, str], str] = {}
        conflicts: List[Conflict] = []

        for subject, relation, obj in triplets:
            key = (subject, relation)
            if key in seen and seen[key] != obj:
                conflicts.append(Conflict(entity=subject, relation=relation, values=(seen[key], obj)))
            else:
                seen[key] = obj

        return conflicts