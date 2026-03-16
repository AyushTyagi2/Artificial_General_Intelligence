"""Synthetic Concept Resolver — Architecture v2.

Handles the 30-90 second ingestion delays caused by the system injecting
synthetic compound terms (neo_species, nano_ecosystem, quantum_sensor, etc.)
that don't exist in Wikipedia/Wikidata.

Instead of wasting lookup time on failed disambiguation, this module:

1. Detects synthetic/compound concept names using heuristics.
2. Decomposes them into known base concepts.
3. Creates a placeholder knowledge node with inferred properties.
4. Optionally looks up the real Wikipedia pages for the base components.

Examples
--------
    "nano_ecosystem"   → components: ["nano", "ecosystem"]
                        real lookup:  "ecosystem"
                        inferred:     nano_ecosystem is_a ecosystem
                                      nano_ecosystem has_property nanoscale

    "quantum_sensor"   → components: ["quantum", "sensor"]
                        real lookup:  "sensor"
                        inferred:     quantum_sensor is_a sensor
                                      quantum_sensor has_property quantum

    "bio_habitat"      → components: ["bio", "habitat"]
                        real lookup:  "habitat"
                        inferred:     bio_habitat is_a habitat

Usage
-----
    resolver = SyntheticConceptResolver()

    # In KnowledgeIngestionPipeline.maybe_ingest() — before Wikidata lookup:
    concepts_to_lookup = []
    synthetic_edges    = []
    for concept in suggested_concepts:
        if resolver.is_synthetic(concept):
            edges = resolver.resolve(concept)
            synthetic_edges.extend(edges)
        else:
            concepts_to_lookup.append(concept)
    # Inject synthetic edges directly into the knowledge graph
    for (s, r, o) in synthetic_edges:
        knowledge_graph.add_edge(s, r, o, provenance="synthetic_resolved")
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known prefix → property mapping
# ---------------------------------------------------------------------------

_KNOWN_PREFIXES: Dict[str, List[Tuple[str, str]]] = {
    # prefix → [(relation, object), ...]
    "bio":     [("has_property", "biological"), ("domain", "biology")],
    "nano":    [("has_property", "nanoscale"),  ("has_property", "microscale")],
    "quantum": [("has_property", "quantum"),    ("domain", "physics")],
    "alien":   [("has_property", "extraterrestrial"), ("domain", "astronomy")],
    "eco":     [("has_property", "ecological"), ("domain", "ecosystem")],
    "neuro":   [("has_property", "neural"),     ("domain", "biology")],
    "cyber":   [("has_property", "digital"),    ("domain", "technology")],
    "hydro":   [("has_property", "aquatic"),    ("domain", "ecology")],
    "geo":     [("has_property", "geological"), ("domain", "physics")],
    "astro":   [("has_property", "astronomical"), ("domain", "astronomy")],
    "thermo":  [("has_property", "thermal"),    ("domain", "physics")],
    "micro":   [("has_property", "microscale"), ("domain", "biology")],
    "macro":   [("has_property", "macroscale")],
    "meta":    [("has_property", "higher_order")],
    "proto":   [("has_property", "primitive")],
    "ultra":   [("has_property", "extreme")],
    "hyper":   [("has_property", "extreme")],
    "neo":     [("has_property", "novel"),      ("has_property", "emergent")],
    "xeno":    [("has_property", "foreign"),    ("domain", "astronomy")],
    "synthetic": [("has_property", "artificial"), ("domain", "technology")],
}

# Known root concepts that map to real Wikipedia-searchable terms
_REAL_ROOTS: Set[str] = {
    "ecosystem", "habitat", "species", "sensor", "robot", "organism",
    "plant", "animal", "cell", "network", "system", "environment",
    "energy", "particle", "field", "process", "material", "structure",
    "colony", "organism", "agent", "reactor", "circuit", "computer",
}

# Patterns that identify synthetic compound concepts
_SYNTHETIC_PATTERNS = [
    re.compile(r"^(bio|nano|quantum|alien|eco|neuro|cyber|hydro|geo|astro|thermo|micro|macro|meta|proto|ultra|hyper|neo|xeno|synthetic)_(.+)$"),
    re.compile(r"^(.+)_(v\d+|mk\d+|\d+)$"),   # versioned concepts: sensor_v2, etc.
]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class SyntheticConceptResult:
    concept:         str
    is_synthetic:    bool
    prefix:          Optional[str]
    root:            Optional[str]
    inferred_edges:  List[Tuple[str, str, str]]   # (subject, relation, object)
    real_lookup_term: Optional[str]               # what to actually look up in Wikipedia


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SyntheticConceptResolver:
    """Resolves synthetic/compound concept names to avoid wasted Wikidata lookups.

    Completely stateless — safe to call on the same concept multiple times.
    """

    def __init__(self) -> None:
        self._resolved_cache: Dict[str, SyntheticConceptResult] = {}

    def is_synthetic(self, concept: str) -> bool:
        """Return True if this concept is likely synthetic/compound."""
        concept = concept.lower().strip()
        for pattern in _SYNTHETIC_PATTERNS:
            if pattern.match(concept):
                return True
        # Also synthetic if it contains numbers mid-word or double underscore
        if "__" in concept:
            return True
        if re.search(r"[a-z]\d[a-z]", concept):
            return True
        return False

    def resolve(self, concept: str) -> List[Tuple[str, str, str]]:
        """Decompose a synthetic concept into inferred knowledge-graph edges.

        Returns a list of (subject, relation, object) triples to inject.
        """
        concept = concept.lower().strip()

        if concept in self._resolved_cache:
            return self._resolved_cache[concept].inferred_edges

        result = self._analyse(concept)
        self._resolved_cache[concept] = result

        if result.is_synthetic:
            logger.debug(
                "[synthetic_resolver] %s → prefix=%s root=%s edges=%d lookup=%s",
                concept, result.prefix, result.root,
                len(result.inferred_edges), result.real_lookup_term,
            )

        return result.inferred_edges

    def resolve_full(self, concept: str) -> SyntheticConceptResult:
        """Full result including lookup term."""
        concept = concept.lower().strip()
        if concept not in self._resolved_cache:
            self._resolved_cache[concept] = self._analyse(concept)
        return self._resolved_cache[concept]

    def real_lookup_term(self, concept: str) -> Optional[str]:
        """Return the Wikipedia-searchable term for a synthetic concept, or None."""
        result = self.resolve_full(concept)
        if result.is_synthetic:
            return result.real_lookup_term
        return None

    def batch_classify(
        self,
        concepts: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Split concepts into (real, synthetic) lists.

        Returns (real_concepts_for_wikidata, synthetic_concepts_to_resolve).
        """
        real: List[str] = []
        synthetic: List[str] = []
        for c in concepts:
            if self.is_synthetic(c):
                synthetic.append(c)
            else:
                real.append(c)
        return real, synthetic

    # ── Internal ───────────────────────────────────────────────────────────────

    def _analyse(self, concept: str) -> SyntheticConceptResult:
        prefix: Optional[str] = None
        root:   Optional[str] = None
        edges:  List[Tuple[str, str, str]] = []
        lookup: Optional[str] = None

        for pattern in _SYNTHETIC_PATTERNS:
            m = pattern.match(concept)
            if m:
                prefix = m.group(1)
                root   = m.group(2)
                break

        if prefix is None:
            return SyntheticConceptResult(
                concept=concept,
                is_synthetic=False,
                prefix=None, root=None,
                inferred_edges=[],
                real_lookup_term=None,
            )

        # Base relationship: concept is_a root
        edges.append((concept, "is_a", root))

        # Prefix-specific properties
        for relation, obj in _KNOWN_PREFIXES.get(prefix, []):
            edges.append((concept, relation, obj))

        # If root is a real concept, flag it for lookup
        if root in _REAL_ROOTS:
            lookup = root
        else:
            # Try to find a real sub-root
            parts = root.split("_")
            for part in parts:
                if part in _REAL_ROOTS:
                    lookup = part
                    break

        return SyntheticConceptResult(
            concept=concept,
            is_synthetic=True,
            prefix=prefix,
            root=root,
            inferred_edges=edges,
            real_lookup_term=lookup,
        )