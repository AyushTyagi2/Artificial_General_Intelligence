"""Knowledge expansion helpers backed by Wikidata lookup.

This module keeps a local cache in `expanded_concepts.json` so previously
resolved concepts are not repeatedly queried from Wikidata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import quote
from urllib.request import urlopen
import json
import logging

LOGGER = logging.getLogger(__name__)

# Offline fallback taxonomy for environments without network access.
FALLBACK_PARENTS: Dict[str, str] = {
    "snake": "reptile",
    "reptile": "animal",
    "amphibian": "animal",
    "cat": "animal",
    "lion": "predator",
    "tiger": "predator",
    "wolf": "predator",
    "deer": "prey",
    "rabbit": "prey",
    "shrub": "plant",
    "asteroid": "celestial_body",
    "fungus": "organism",
}

WIKIDATA_RELATIONS: Dict[str, Tuple[str, str]] = {
    "P31": ("is", "instance_of"),
    "P279": ("is", "subclass_of"),
    "P361": ("part_of", "part_of"),
    "P131": ("located_in", "located_in"),
}


def _normalize(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def _safe_json_get(url: str, timeout: int = 6) -> dict:
    with urlopen(url, timeout=timeout) as response:  # nosec B310 - trusted https endpoint
        return json.loads(response.read().decode("utf-8"))


def _wikidata_entity_for_label(concept: str) -> Optional[str]:
    """Return best matching Wikidata entity id (Qxxx) for a label."""
    url = (
        "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&language=en"
        f"&type=item&search={quote(concept)}&limit=1"
    )
    try:
        payload = _safe_json_get(url)
    except Exception:
        return None
    results = payload.get("search", [])
    if not results:
        return None
    return results[0].get("id")


def _entity_payload(entity_id: str) -> dict:
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
    payload = _safe_json_get(url)
    return payload.get("entities", {}).get(entity_id, {})


def _entity_label(entity_id: str) -> Optional[str]:
    entity = _entity_payload(entity_id)
    return _normalize(entity.get("labels", {}).get("en", {}).get("value", "")) or None


class KnowledgeExpander:
    """Expands unknown concepts and persists discovered graph links."""

    def __init__(self, knowledge_path: str | Path) -> None:
        self.knowledge_path = Path(knowledge_path)
        self._entity_id_cache: Dict[str, str] = {}
        self._known_facts_cache: Optional[Set[str]] = None

    @property
    def expanded_path(self) -> Path:
        return self.knowledge_path / "expanded_concepts.json"

    def _load_expanded_page(self) -> dict:
        if self.expanded_path.exists():
            return json.loads(self.expanded_path.read_text(encoding="utf-8"))
        return {"topic": "expanded_concepts", "domain": "taxonomy", "facts": []}

    def _known_facts(self) -> Set[str]:
        if self._known_facts_cache is None:
            self._known_facts_cache = set(self._load_expanded_page().get("facts", []))
        return self._known_facts_cache

    def is_cached_concept(self, concept: str) -> bool:
        c = _normalize(concept)
        return any(fact.startswith(f"{c} ") for fact in self._known_facts())

    def expand_concept(self, concept: str) -> Optional[Tuple[str, str]]:
        """Expand one concept and persist `<concept> is <parent>`.

        Returns the inferred `(child, parent)` relation when successful.
        """
        child = _normalize(concept)
        if self.is_cached_concept(child):
            return None

        parent = self._lookup_parent(child)
        if not parent:
            return None
        self._persist_edge(child, "is", parent)
        return child, parent

    def expand_concept_graph(self, concept: str, depth: int = 2) -> List[Tuple[str, str, str]]:
        """Recursively expand concept graph via Wikidata.

        Relations: instance_of, subclass_of, part_of, located_in.
        """
        root = _normalize(concept)
        discovered: List[Tuple[str, str, str]] = []
        visited: Set[str] = set()

        def walk(node: str, remaining: int) -> None:
            if remaining < 0 or node in visited:
                return
            visited.add(node)
            for source, relation, target in self._related_edges(node):
                self._persist_edge(source, relation, target)
                discovered.append((source, relation, target))
                if relation == "is":
                    walk(target, remaining - 1)

        walk(root, depth)
        return discovered

    def _lookup_parent(self, concept: str) -> Optional[str]:
        try:
            entity_id = self._entity_id(concept)
            if entity_id:
                for _source, relation, target in self._related_edges(concept):
                    if relation == "is":
                        return target
        except Exception as exc:  # network and payload errors should not stop the loop
            LOGGER.debug("wikidata lookup failed for %s: %s", concept, exc)
        return FALLBACK_PARENTS.get(concept)

    def _entity_id(self, concept: str) -> Optional[str]:
        key = _normalize(concept)
        if key in self._entity_id_cache:
            return self._entity_id_cache[key]
        try:
            entity_id = _wikidata_entity_for_label(key)
        except Exception:
            entity_id = None
        if entity_id:
            self._entity_id_cache[key] = entity_id
        return entity_id

    def _related_edges(self, concept: str) -> List[Tuple[str, str, str]]:
        edges: List[Tuple[str, str, str]] = []
        entity_id = self._entity_id(concept)
        if not entity_id:
            parent = FALLBACK_PARENTS.get(_normalize(concept))
            if parent:
                edges.append((_normalize(concept), "is", parent))
            return edges

        try:
            entity = _entity_payload(entity_id)
            claims = entity.get("claims", {})
            for property_id, (relation, _label) in WIKIDATA_RELATIONS.items():
                for claim in claims.get(property_id, [])[:2]:
                    value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
                    target_id = value.get("id")
                    if not target_id:
                        continue
                    target = _entity_label(target_id) or _normalize(target_id)
                    edges.append((_normalize(concept), relation, target))
        except Exception as exc:
            LOGGER.debug("related edge expansion failed for %s: %s", concept, exc)

        if not edges and _normalize(concept) in FALLBACK_PARENTS:
            edges.append((_normalize(concept), "is", FALLBACK_PARENTS[_normalize(concept)]))
        return edges

    def _persist_edge(self, source: str, relation: str, target: str) -> None:
        self.knowledge_path.mkdir(parents=True, exist_ok=True)
        page = self._load_expanded_page()
        facts = self._known_facts()
        fact = f"{_normalize(source)} {relation} {_normalize(target)}"
        if fact not in facts:
            facts.add(fact)
            page["facts"] = sorted(facts)
            self.expanded_path.write_text(json.dumps(page, indent=2), encoding="utf-8")


def expand_concept(concept: str, knowledge_path: str | Path) -> Optional[Tuple[str, str]]:
    """Convenience function for one-off concept expansion."""
    return KnowledgeExpander(knowledge_path).expand_concept(concept)
