"""Knowledge expansion helpers backed by Wikidata lookup."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
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
}


def _normalize(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def _wikidata_entity_for_label(concept: str) -> Optional[str]:
    """Return best matching Wikidata entity id (Qxxx) for a label."""
    url = (
        "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&language=en"
        f"&type=item&search={quote(concept)}&limit=1"
    )
    with urlopen(url, timeout=6) as response:  # nosec B310 - trusted https endpoint
        payload = json.loads(response.read().decode("utf-8"))
    results = payload.get("search", [])
    if not results:
        return None
    return results[0].get("id")


def _entity_label(entity_id: str) -> Optional[str]:
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
    with urlopen(url, timeout=6) as response:  # nosec B310 - trusted https endpoint
        payload = json.loads(response.read().decode("utf-8"))
    entity = payload.get("entities", {}).get(entity_id, {})
    return _normalize(entity.get("labels", {}).get("en", {}).get("value", "")) or None


def _wikidata_parent_label(entity_id: str) -> Optional[str]:
    """Try to read parent class using subclass_of (P279) then instance_of (P31)."""
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
    with urlopen(url, timeout=6) as response:  # nosec B310 - trusted https endpoint
        payload = json.loads(response.read().decode("utf-8"))
    entity = payload.get("entities", {}).get(entity_id, {})
    claims = entity.get("claims", {})

    for prop in ("P279", "P31"):
        for claim in claims.get(prop, []):
            value = claim.get("mainsnak", {}).get("datavalue", {}).get("value", {})
            parent_id = value.get("id")
            if parent_id:
                return _entity_label(parent_id) or _normalize(parent_id)
    return None


class KnowledgeExpander:
    """Expands unknown concepts and persists discovered taxonomy links."""

    def __init__(self, knowledge_path: str | Path) -> None:
        self.knowledge_path = Path(knowledge_path)

    def expand_concept(self, concept: str) -> Optional[Tuple[str, str]]:
        """Expand one concept and persist `<concept> is <parent>` in a knowledge page.

        Returns the inferred `(child, parent)` relation when successful.
        """
        child = _normalize(concept)
        parent = self._lookup_parent(child)
        if not parent:
            return None
        self._persist_relation(child, parent)
        return child, parent

    def _lookup_parent(self, concept: str) -> Optional[str]:
        try:
            entity_id = _wikidata_entity_for_label(concept)
            if entity_id:
                parent = _wikidata_parent_label(entity_id)
                if parent:
                    return parent
        except Exception as exc:  # network and payload errors should not stop the loop
            LOGGER.debug("wikidata lookup failed for %s: %s", concept, exc)
        return FALLBACK_PARENTS.get(concept)

    def _persist_relation(self, child: str, parent: str) -> None:
        self.knowledge_path.mkdir(parents=True, exist_ok=True)
        file_path = self.knowledge_path / "expanded_concepts.json"
        if file_path.exists():
            page = json.loads(file_path.read_text(encoding="utf-8"))
        else:
            page = {"topic": "expanded_concepts", "domain": "taxonomy", "facts": []}

        fact = f"{child} is {parent}"
        facts = set(page.get("facts", []))
        if fact not in facts:
            facts.add(fact)
            page["facts"] = sorted(facts)
            file_path.write_text(json.dumps(page, indent=2), encoding="utf-8")


def expand_concept(concept: str, knowledge_path: str | Path) -> Optional[Tuple[str, str]]:
    """Convenience function for one-off concept expansion."""
    return KnowledgeExpander(knowledge_path).expand_concept(concept)
