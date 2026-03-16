"""Knowledge ingestion pipeline for digital_baby — v3.

Changes from v2
---------------
1. EXPANDED ALLOWED_ONTOLOGY_TYPES:
   Added ~40 new types covering physical quantities, chemical processes,
   biological processes, phenomena, properties, and abstract scientific
   concepts.  Many valid concepts were previously filtered because Wikidata
   returned types like 'physical quantity' or 'chemical reaction' which were
   not in the old whitelist.

2. DEFAULT-ACCEPT posture for unknown types:
   Previously: if no allowed type found → REJECT.
   Now: if no BLOCKED type found → ACCEPT with reduced initial confidence.
   It is better to admit a few irrelevant concepts (they score low and get
   pruned) than to miss relevant ones like 'activation_energy' or 'osmosis'.

3. HIERARCHICAL TYPE MATCHING:
   Pass 2 checks for common scientific suffix patterns (-tion, -sis, -ase,
   'process', 'reaction', etc.) so concepts like 'catalysis', 'phosphorylation',
   and 'hydrolysis' pass even if their Wikidata type is not in the whitelist.

4. EXPANDED RELATION PATTERNS:
   Added 20+ new causal/functional relation patterns (causes, inhibits,
   proportional_to, converts_to, metabolises, etc.) so Wikipedia summaries
   yield richer triplets.

5. GRAPH-SEEDED QUEUE EXPANSION:
   After ingesting a concept, its knowledge-graph neighbours that have not yet
   been ingested are automatically added to the queue, ensuring the ingestion
   frontier follows the structure of what the agent is already learning about.

6. INGESTION-TRIGGERED CURIOSITY:
   New concepts found via ingestion are registered to the curiosity model's
   pending queue so the event loop can prioritise their exploration.

All v2 public API is preserved.
"""

from __future__ import annotations

import json as _json
import logging
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ontology allow / block lists  (v3: significantly expanded ALLOWED set)
# ---------------------------------------------------------------------------

ALLOWED_ONTOLOGY_TYPES: Set[str] = {
    # --- v2 original ---
    "natural object", "physical object", "organism",
    "chemical substance", "chemical compound", "chemical element",
    "astronomical object", "celestial body", "star", "planet",
    "material", "natural material", "device",
    "geographic feature", "geographic location", "landform", "body of water",
    "natural phenomenon", "biological process", "taxon",
    "animal", "plant", "mineral", "rock", "metal",
    "gas", "liquid", "solid",
    "ecosystem", "biome", "habitat",
    "organ", "tissue", "cell", "molecule", "atom",
    "force", "energy", "radiation", "particle",
    "bird", "fish", "insect", "reptile", "mammal", "amphibian", "fungus",
    "accipitridae", "felidae", "canidae", "cervidae", "poaceae",
    "organisms known by a particular common name",
    "monotypic taxon", "infraspecific taxon",

    # --- v3 NEW: physical science ---
    "physical quantity", "physical property", "physical process",
    "mechanical property", "thermodynamic property", "thermodynamic process",
    "electromagnetic property", "wave", "field", "interaction",
    "scalar quantity", "vector quantity", "dimensionless quantity",
    "conservation law", "equation of state",

    # --- v3 NEW: chemistry ---
    "chemical reaction", "chemical process", "biochemical process",
    "biochemical reaction", "catalysis", "oxidation state",
    "chemical bond", "mixture", "solution", "colloid", "suspension",
    "acid", "base", "salt", "polymer", "monomer", "isomer",
    "organic compound", "inorganic compound",

    # --- v3 NEW: biology & ecology ---
    "ecological relationship", "physiological process", "metabolic process",
    "developmental process", "cellular process", "molecular process",
    "behaviour", "instinct", "adaptation", "evolutionary process",
    "symbiosis", "parasitism", "mutualism", "commensalism",
    "predation", "competition", "food web",

    # --- v3 NEW: abstract scientific ---
    "phenomenon", "property", "quantity", "measurement", "variable",
    "relationship", "process", "system", "cycle", "network",
    "principle", "law of nature", "effect", "mechanism",
    "model", "theory", "hypothesis", "scientific concept",

    # --- v3 NEW: structures & environments ---
    "environment", "microenvironment", "structure", "anatomy", "morphology",
    "population", "community", "habitat type",
}

BLOCKED_ONTOLOGY_TYPES: Set[str] = {
    "film", "television film", "television series",
    "video game", "software", "free software", "operating system",
    "application software", "music album", "album", "song", "single",
    "fictional character", "fictional entity", "comic book character",
    "superhero", "wikimedia disambiguation page", "wikimedia list article",
    "wikimedia template", "human name", "given name", "surname",
    "artwork", "painting", "sculpture", "book", "novel", "literary work",
    "brand", "company", "organization", "human", "person",
    "politician", "athlete", "award", "sports season", "sports team",
    "association football club",
}

# v3: Scientific suffix patterns for hierarchical type matching (Pass 2)
_SCIENCE_SUFFIXES: Tuple[str, ...] = (
    "tion", "sion", "sis", "ase", "ase", "ism", "ity", "ance", "ence",
    "process", "reaction", "phenomenon", "property", "quantity",
    "effect", "principle", "mechanism", "behaviour", "interaction",
    "cycle", "pathway", "system", "network",
)

# ---------------------------------------------------------------------------
# Seed concepts (extended in v3)
# ---------------------------------------------------------------------------

_SEED_CONCEPTS: List[str] = [
    "wolf", "deer", "oak", "moss", "fern", "salmon", "eagle", "bear",
    "carbon", "oxygen", "hydrogen", "nitrogen", "sodium", "iron",
    "star", "planet", "comet", "galaxy", "nebula", "asteroid",
    "transistor", "semiconductor", "photon", "electron", "neuron",
    "fungi", "bacteria", "virus", "algae", "coral",
    "river", "glacier", "volcano", "earthquake", "tectonic_plate",
    "enzyme", "protein", "dna", "rna", "mitochondria",
    "photosynthesis", "respiration", "osmosis", "diffusion",
    "water", "mountain", "ice", "snow", "forest",
    "predator", "prey", "symbiosis", "evolution", "adaptation",
    # v3 NEW seeds covering simulation domains
    "acceleration", "velocity", "momentum", "friction", "kinetic_energy",
    "temperature", "reaction_rate", "catalyst", "activation_energy",
    "immune_response", "pathogen", "antibody", "cell_cycle",
    "population_dynamics", "carrying_capacity", "trophic_cascade",
]

# ---------------------------------------------------------------------------
# Relation heuristics for Wikipedia text (v3: greatly expanded)
# ---------------------------------------------------------------------------

_RELATION_PATTERNS: List[Tuple[str, str, re.Pattern]] = [
    # --- v2 original ---
    ("is",         "is",         re.compile(r"\b(\w+)\s+is\s+a(?:n)?\s+(\w+)", re.I)),
    ("is",         "is",         re.compile(r"\b(\w+)\s+are\s+a(?:n)?\s+(\w+)", re.I)),
    ("eats",       "eats",       re.compile(r"\b(\w+)\s+(?:eats?|feeds?\s+on)\s+(\w+)", re.I)),
    ("hunts",      "hunts",      re.compile(r"\b(\w+)\s+hunts?\s+(\w+)", re.I)),
    ("part_of",    "part_of",    re.compile(r"\b(\w+)\s+is\s+part\s+of\s+(?:the\s+)?(\w+)", re.I)),
    ("located_in", "located_in", re.compile(r"\b(\w+)\s+(?:is\s+)?found\s+in\s+(?:the\s+)?(\w+)", re.I)),
    ("produces",   "produces",   re.compile(r"\b(\w+)\s+produces?\s+(\w+)", re.I)),
    ("contains",   "contains",   re.compile(r"\b(\w+)\s+contains?\s+(\w+)", re.I)),

    # --- v3 NEW: causal relations ---
    ("causes",     "causes",     re.compile(r"\b(\w+)\s+causes?\s+(\w+)", re.I)),
    ("triggers",   "triggers",   re.compile(r"\b(\w+)\s+triggers?\s+(\w+)", re.I)),
    ("promotes",   "promotes",   re.compile(r"\b(\w+)\s+promotes?\s+(\w+)", re.I)),
    ("inhibits",   "inhibits",   re.compile(r"\b(\w+)\s+inhibits?\s+(\w+)", re.I)),
    ("activates",  "activates",  re.compile(r"\b(\w+)\s+activates?\s+(\w+)", re.I)),
    ("suppresses", "suppresses", re.compile(r"\b(\w+)\s+suppresses?\s+(\w+)", re.I)),
    ("increases",  "increases",  re.compile(r"\b(\w+)\s+increases?\s+(\w+)", re.I)),
    ("decreases",  "decreases",  re.compile(r"\b(\w+)\s+decreases?\s+(\w+)", re.I)),
    ("leads_to",   "leads_to",   re.compile(r"\b(\w+)\s+leads?\s+to\s+(\w+)", re.I)),
    ("results_in", "results_in", re.compile(r"\b(\w+)\s+results?\s+in\s+(\w+)", re.I)),

    # --- v3 NEW: functional/structural ---
    ("converts",   "converts",   re.compile(r"\b(\w+)\s+converts?\s+(\w+)", re.I)),
    ("metabolises","metabolises",re.compile(r"\b(\w+)\s+metabolis(?:es?|izes?)\s+(\w+)", re.I)),
    ("absorbs",    "absorbs",    re.compile(r"\b(\w+)\s+absorbs?\s+(\w+)", re.I)),
    ("emits",      "emits",      re.compile(r"\b(\w+)\s+emits?\s+(\w+)", re.I)),
    ("requires",   "requires",   re.compile(r"\b(\w+)\s+requires?\s+(\w+)", re.I)),
    ("depends_on", "depends_on", re.compile(r"\b(\w+)\s+depends?\s+on\s+(\w+)", re.I)),
    ("regulates",  "regulates",  re.compile(r"\b(\w+)\s+regulates?\s+(\w+)", re.I)),
    ("catalyses",  "catalyses",  re.compile(r"\b(\w+)\s+catalys(?:es?|izes?)\s+(\w+)", re.I)),

    # --- v3 NEW: ecological ---
    ("competes_with","competes_with", re.compile(r"\b(\w+)\s+competes?\s+with\s+(\w+)", re.I)),
    ("parasitises", "parasitises",   re.compile(r"\b(\w+)\s+parasitis(?:es?|izes?)\s+(\w+)", re.I)),
    ("pollinates",  "pollinates",    re.compile(r"\b(\w+)\s+pollinates?\s+(\w+)", re.I)),
    ("decomposes",  "decomposes",    re.compile(r"\b(\w+)\s+decomposes?\s+(\w+)", re.I)),


    # --- v4 NEW: looser is-a for real Wikipedia prose ---
    # "wolf is mainly a carnivore", "wolf is also a predator"
    ("is", "is", re.compile(
        r"\b(\w+)\s+is\s+(?:mainly|primarily|chiefly|largely|also|"
        r"typically|generally|often|usually|considered)\s+a(?:n)?\s+(\w+)",
        re.I)),
    # "wolf is a large canine" -- adjective(s) before biological suffix
    ("is", "is", re.compile(
        r"\b(\w+)\s+is\s+a(?:n)?\s+(?:\w+\s+){0,2}(\w+(?:ivore|vore|ist|oid|ian|an)\b)",
        re.I)),
    # "member of the family Canidae"
    ("member_of", "member_of", re.compile(
        r"\b(\w+)\s+(?:is\s+(?:a\s+)?)?(?:\w+\s+)?member\s+of\s+(?:the\s+)?(?:family\s+|genus\s+|order\s+|class\s+)?(\w+)",
        re.I)),
    # "native to Eurasia"
    ("native_to", "native_to", re.compile(
        r"\b(\w+)\s+(?:is\s+)?native\s+to\s+(?:the\s+)?(\w+)",
        re.I)),
    # "feeds on large wild hoofed mammals", "preys on rabbits"
    ("feeds_on", "feeds_on", re.compile(
        r"\b(\w+)\s+(?:feeds?|preys?)\s+(?:primarily\s+|mainly\s+|mostly\s+)?on\s+(?:\w+\s+){0,3}(\w+)",
        re.I)),
    # "belongs to family X"
    ("belongs_to", "belongs_to", re.compile(
        r"\b(\w+)\s+belongs?\s+to\s+(?:the\s+)?(?:family\s+|genus\s+|order\s+)?(\w+)",
        re.I)),
    # "wolves live in forests", "wolf inhabits tundra"
    ("lives_in", "lives_in", re.compile(
        r"\b(\w+)\s+(?:lives?|inhabits?|dwells?)\s+in\s+(?:the\s+)?(\w+)",
        re.I)),
]

_STOP_WORDS: frozenset = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "can", "this",
    "that", "these", "those", "it", "its", "they", "their", "them",
    "also", "such", "most", "more", "many", "some", "other", "which",
    "when", "where", "how", "what", "who", "type", "form", "kind",
    "member", "part", "number", "group", "class", "species",
})


def _clean_concept(raw: str) -> Optional[str]:
    concept = raw.lower().strip().replace(" ", "_").replace("-", "_")
    concept = re.sub(r"[^a-z0-9_]", "", concept)
    if len(concept) < 3 or concept in _STOP_WORDS:
        return None
    if concept.replace("_", "").isdigit():
        return None
    return concept


# ---------------------------------------------------------------------------
# WikidataEntity
# ---------------------------------------------------------------------------

@dataclass
class WikidataEntity:
    qid:   str
    label: str
    types: List[str] = field(default_factory=list)

    @property
    def concept_name(self) -> str:
        return _clean_concept(self.label) or self.label.lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# Wikidata helpers (unchanged HTTP layer from v2)
# ---------------------------------------------------------------------------

_WIKIDATA_ENTITY_CACHE: Dict[str, Optional[WikidataEntity]] = {}


def _wikidata_get(url: str, timeout: float) -> Optional[dict]:
    import random, urllib.error
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "digital_baby/3.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return _json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503, 504):
                if attempt < max_attempts - 1:
                    delay = (2 ** attempt) + random.uniform(0.0, 1.0)
                    time.sleep(delay)
                    continue
            logger.debug("[ingestion] http_failed url=%s error=%s", url, exc)
            return None
        except Exception as exc:
            if attempt < max_attempts - 1:
                delay = (2 ** attempt) + random.uniform(0.0, 1.0)
                time.sleep(delay)
                continue
            logger.debug("[ingestion] http_failed url=%s error=%s", url, exc)
            return None
    return None


def _get_entity_types(qid: str, timeout: float) -> List[str]:
    sparql = (
        f"SELECT ?typeLabel WHERE {{"
        f" {{ wd:{qid} wdt:P31 ?type . }} UNION {{ wd:{qid} wdt:P279 ?type . }}"
        f" SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"en\" . }}"
        f"}} LIMIT 20"
    )
    url = (
        "https://query.wikidata.org/sparql?format=json&query="
        + urllib.parse.quote(sparql)
    )
    data = _wikidata_get(url, timeout)
    if not data:
        return []
    return [
        b.get("typeLabel", {}).get("value", "").lower().strip()
        for b in data.get("results", {}).get("bindings", [])
        if b.get("typeLabel", {}).get("value")
    ]


# Types that are clearly wrong disambiguation hits — military vessels, aircraft,
# administrative geography, sports/entertainment.  These appear when Wikidata
# returns a wrong QID for a natural-science term (e.g. antelope→frigate, or
# oak→ship).  Even under default-accept these should be rejected, because they
# add nonsense edges like "antelope instance_of frigate".
_NONSENSE_TYPES: frozenset = frozenset({
    "frigate", "warship", "ship", "aircraft", "fighter aircraft", "bomber",
    "missile", "weapon", "firearm", "sword", "cannon",
    "municipality", "commune", "administrative territorial entity",
    "census-designated place", "unincorporated community",
    "railway station", "airport", "bridge", "road",
    "sports club", "association football club", "sports season",
    "television channel", "radio station", "newspaper",
    "political party", "government agency", "military unit",
    "commune of france", "village in poland", "district",
})


def _is_allowed_type(types: List[str]) -> Tuple[bool, str]:
    """v3+hotfix: Four-pass type check.  Returns (allowed, reason).

    Hotfix (v4.1): Pass 4 now guards against nonsense disambiguation hits.
    'antelope instance_of frigate' was admitted because 'frigate' was not in
    BLOCKED_ONTOLOGY_TYPES and passed default-accept.  The _NONSENSE_TYPES
    set catches these before they pollute the graph.
    """
    types_set = {t.lower() for t in types}

    # Pass 1: hard block (existing blocked types)
    blocked_hit = types_set & BLOCKED_ONTOLOGY_TYPES
    if blocked_hit:
        return False, f"blocked_type:{next(iter(blocked_hit))}"

    # Pass 1b: nonsense-type guard (new in hotfix)
    nonsense_hit = types_set & _NONSENSE_TYPES
    if nonsense_hit:
        return False, f"nonsense_type:{next(iter(nonsense_hit))}"

    # Pass 2: explicit allow
    allowed_hit = types_set & ALLOWED_ONTOLOGY_TYPES
    if allowed_hit:
        return True, f"allowed_type:{next(iter(allowed_hit))}"

    # Pass 3: suffix / keyword match on any type string
    for t in types:
        t_lower = t.lower()
        if any(t_lower.endswith(s) or s in t_lower for s in _SCIENCE_SUFFIXES):
            return True, f"science_suffix_match:{t_lower}"

    # Pass 4: DEFAULT-ACCEPT when no blocking type found.
    # Only accept if the types look plausible (contain recognisable words)
    # rather than being a single opaque unrelated category like "frigate".
    if not types:
        return True, "no_types_returned_accept_unknown"

    # Check that at least one type contains a recognisable scientific word
    # (even if not in our explicit set).  This catches legitimate concepts
    # that Wikidata categorises under unusual labels.
    _PLAUSIBILITY_WORDS = {
        "species", "genus", "family", "order", "class", "phylum",
        "object", "substance", "compound", "element", "reaction",
        "process", "phenomenon", "property", "quantity", "system",
        "organism", "creature", "entity", "concept", "theory",
        "force", "field", "wave", "particle", "energy", "matter",
        "scientific", "natural", "biological", "chemical", "physical",
        "geological", "ecological", "astronomical", "mathematical",
    }
    for t in types:
        t_lower = t.lower()
        if any(word in t_lower for word in _PLAUSIBILITY_WORDS):
            return True, f"default_accept_plausible_type:{t_lower}"

    # Has types, none blocked/nonsense, none allowed, no suffix, no plausible word
    # → REJECT (was soft-accept before hotfix, which let frigate/warship through)
    return False, f"default_reject_unrecognised_types:{types[:2]}"


def _resolve_wikidata_entity(label: str, timeout: float = 8.0) -> Optional[WikidataEntity]:
    cache_key = label.lower().strip()
    if cache_key in _WIKIDATA_ENTITY_CACHE:
        return _WIKIDATA_ENTITY_CACHE[cache_key]

    search_url = (
        "https://www.wikidata.org/w/api.php"
        "?action=wbsearchentities"
        f"&search={urllib.parse.quote(label)}"
        "&language=en&limit=10&format=json"
    )
    data = _wikidata_get(search_url, timeout)
    if not data or not data.get("search"):
        _WIKIDATA_ENTITY_CACHE[cache_key] = None
        return None

    for candidate in data["search"]:
        qid = candidate.get("id", "")
        if not qid.startswith("Q"):
            continue

        description = candidate.get("description", "").lower()
        _NAME_SIGNALS = (
            "given name", "family name", "surname", "human name",
            "male given name", "female given name", "unisex given name",
            "patronymic", "übername", "jewish family name", "human",
            "non-governmental organization", "nonprofit", "foundation",
            "association", "municipality", "civil town", "county",
            "administrative", "populated place", "village", "city",
        )
        if any(sig in description for sig in _NAME_SIGNALS):
            logger.debug(
                "[ingestion] pre_filtered concept=%s qid=%s desc=%r",
                label, qid, candidate.get("description", ""),
            )
            continue

        types = _get_entity_types(qid, timeout)
        allowed, reason = _is_allowed_type(types)

        if not allowed:
            logger.info(
                "[ingestion] concept_filtered reason=%s concept=%s qid=%s",
                reason, label, qid,
            )
            continue

        entity = WikidataEntity(qid=qid, label=cache_key, types=types)
        logger.info(
            "[ingestion] entity_selected %s %s reason=%s",
            qid, label, reason,
        )
        _WIKIDATA_ENTITY_CACHE[cache_key] = entity
        return entity

    logger.info(
        "[ingestion] concept_filtered reason=all_candidates_rejected concept=%s",
        label,
    )
    _WIKIDATA_ENTITY_CACHE[cache_key] = None
    return None


# ---------------------------------------------------------------------------
# Source adapters
# ---------------------------------------------------------------------------

def _fetch_wikidata_relations(
    concept: str,
    timeout: float = 8.0,
    max_relations: int = 8,
) -> List[Tuple[str, str, str]]:
    entity = _resolve_wikidata_entity(concept.replace("_", " "), timeout)
    if entity is None:
        return []

    prop_map = {
        "P31":  "instance_of",
        "P279": "subclass_of",
        "P361": "part_of",
        "P131": "located_in",
        "P527": "has_part",
    }
    props_values = " ".join(f"wdt:{pid}" for pid in prop_map)

    sparql = (
        f"SELECT ?prop ?targetLabel WHERE {{"
        f" wd:{entity.qid} ?prop ?target ."
        f" VALUES ?prop {{ {props_values} }}"
        f" ?target rdfs:label ?targetLabel ."
        f" FILTER(LANG(?targetLabel) = \"en\")"
        f"}} LIMIT {max_relations * 3}"
    )
    url = (
        "https://query.wikidata.org/sparql?format=json&query="
        + urllib.parse.quote(sparql)
    )
    data = _wikidata_get(url, timeout)
    if not data:
        return []

    prop_url_to_pid = {
        f"http://www.wikidata.org/prop/direct/{pid}": pid
        for pid in prop_map
    }

    triplets: List[Tuple[str, str, str]] = []
    for binding in data.get("results", {}).get("bindings", []):
        prop_url   = binding.get("prop", {}).get("value", "")
        target_raw = binding.get("targetLabel", {}).get("value", "")
        pid = prop_url_to_pid.get(prop_url)
        if not pid:
            continue
        target_clean = _clean_concept(target_raw)
        if not target_clean:
            continue
        target_lower = target_raw.lower()
        if any(blocked in target_lower for blocked in BLOCKED_ONTOLOGY_TYPES):
            continue
        triplets.append((entity.concept_name, prop_map[pid], target_clean))
        if len(triplets) >= max_relations:
            break

    logger.info(
        "[ingestion] source=wikidata concept=%s qid=%s relations=%d",
        concept, entity.qid, len(triplets),
    )
    return triplets


def _fetch_wikipedia_relations(
    concept: str,
    timeout: float = 8.0,
    max_relations: int = 12,
) -> List[Tuple[str, str, str]]:
    """Fetch Wikipedia relations via the tools bridge (falls back to raw urllib)."""
    # ── Try tools bridge first ───────────────────────────────────────────
    try:
        from digital_baby.tools_bridge.wikipedia import fetch_relations as _bridge_fetch
        triplets = _bridge_fetch(concept, max_relations=max_relations)
        if triplets:
            logger.info("[ingestion] source=wikipedia(bridge) concept=%s relations=%d",
                        concept, len(triplets))
            return triplets
    except Exception:
        pass  # fall through to raw urllib below

    # ── Raw urllib fallback (original implementation) ────────────────────
    try:
        title = concept.replace("_", " ")
        url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + urllib.parse.quote(title)
        )
        req = urllib.request.Request(url, headers={"User-Agent": "digital_baby/3.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = _json.loads(resp.read().decode("utf-8"))

        if "disambiguation" in data.get("type", "").lower():
            logger.info("[ingestion] concept_filtered reason=disambiguation concept=%s", concept)
            return []

        summary = data.get("extract", "")
        if not summary:
            return []

        triplets = []
        for _label, relation, pattern in _RELATION_PATTERNS:
            for match in pattern.finditer(summary):
                subj = _clean_concept(match.group(1))
                obj  = _clean_concept(match.group(2))
                if not subj or not obj or subj == obj:
                    continue
                if obj.replace("_", " ") in BLOCKED_ONTOLOGY_TYPES:
                    continue
                triplets.append((subj, relation, obj))
                if len(triplets) >= max_relations:
                    break
            if len(triplets) >= max_relations:
                break

        logger.info("[ingestion] source=wikipedia concept=%s relations=%d",
                    concept, len(triplets))
        return triplets

    except Exception as exc:
        logger.debug("[ingestion] wikipedia_failed concept=%s error=%s", concept, exc)
        return []


def _fetch_wordnet_relations(
    concept: str,
    max_relations: int = 8,
) -> List[Tuple[str, str, str]]:
    try:
        from nltk.corpus import wordnet as wn  # type: ignore
        word    = concept.replace("_", " ")
        synsets = wn.synsets(word)
        if not synsets:
            return []

        triplets: List[Tuple[str, str, str]] = []
        synset = synsets[0]

        for hyper in synset.hypernyms()[:3]:
            name = _clean_concept(hyper.lemmas()[0].name())
            if name:
                triplets.append((concept, "subclass_of", name))
        for hypo in synset.hyponyms()[:3]:
            name = _clean_concept(hypo.lemmas()[0].name())
            if name:
                triplets.append((concept, "has_type", name))
        for mero in (synset.part_meronyms() + synset.substance_meronyms())[:4]:
            name = _clean_concept(mero.lemmas()[0].name())
            if name:
                triplets.append((concept, "has_part", name))

        triplets = triplets[:max_relations]
        logger.info(
            "[ingestion] source=wordnet concept=%s relations=%d",
            concept, len(triplets),
        )
        return triplets

    except Exception as exc:
        logger.debug("[ingestion] wordnet_failed concept=%s error=%s", concept, exc)
        return []


# ---------------------------------------------------------------------------
# Semantic proximity scoring (unchanged from v2)
# ---------------------------------------------------------------------------

def _score_concept_proximity(concept: str, graph_node_names: Set[str]) -> float:
    if not graph_node_names:
        return 0.0

    related: Set[str] = {concept, concept.rstrip("s")}
    entity = _WIKIDATA_ENTITY_CACHE.get(concept.replace("_", " "))
    if entity is not None:
        for t in entity.types:
            clean = _clean_concept(t)
            if clean:
                related.add(clean)
            for word in re.split(r"[\s_]+", t.lower()):
                w = _clean_concept(word)
                if w:
                    related.add(w)

    overlap   = len(related & graph_node_names)
    proximity = min(1.0, overlap / max(1, len(related)))
    return max(0.1, proximity)


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class KnowledgeIngestionPipeline:
    """Periodic, disambiguated ingestion of external knowledge — v3.

    v3 changes:
    - Default-accept for unrecognised Wikidata types (with reduced confidence).
    - Expanded relation patterns capture causal language from Wikipedia.
    - Graph-seeded queue expansion: neighbours of ingested concepts auto-queued.
    - New concepts registered to curiosity model's pending queue.

    Parameters
    ----------
    ingestion_interval        : ticks between runs (default 50)
    ingestion_batch_size      : concepts per run (default 5, max 20)
    max_relations_per_concept : cap per source per concept (default 10)
    request_timeout           : seconds per network call (default 8.0)
    enable_wikidata / enable_wikipedia / enable_wordnet : toggle sources
    unknown_concept_bonus     : curiosity boost per new concept (default 0.3)
    """

    def __init__(
        self,
        ingestion_interval: int = 50,
        ingestion_batch_size: int = 5,
        max_relations_per_concept: int = 10,   # v3: increased from 8
        request_timeout: float = 8.0,
        enable_wikidata: bool = True,
        enable_wikipedia: bool = True,
        enable_wordnet: bool = True,
        unknown_concept_bonus: float = 0.3,
        state_path=None,  # Path | None — persist ingestion state across restarts
    ) -> None:
        self.ingestion_interval        = max(1, ingestion_interval)
        self.ingestion_batch_size      = max(1, min(20, ingestion_batch_size))
        self.max_relations_per_concept = max(1, max_relations_per_concept)
        self.request_timeout           = request_timeout
        self.enable_wikidata           = enable_wikidata
        self.enable_wikipedia          = enable_wikipedia
        self.enable_wordnet            = enable_wordnet
        self.unknown_concept_bonus     = unknown_concept_bonus

        self._ingested_concepts: Dict[str, Tuple[float, Optional[str]]] = {}
        self._failed_concepts:   set = set()  # permanently barren concepts
        self._concept_queue: List[str] = []
        self._last_run_tick: int = 0
        self._state_path = state_path
        if state_path is not None:
            self._load_state(state_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def suggest_concepts(self, concepts: List[str]) -> None:
        for c in concepts:
            clean = _clean_concept(c)
            if (clean
                    and clean not in self._ingested_concepts
                    and clean not in self._failed_concepts
                    and clean not in self._concept_queue):
                self._concept_queue.append(clean)

    def maybe_ingest(
        self,
        tick: int,
        knowledge_graph,
        type_system,
        curiosity_model,
    ) -> int:
        if tick - self._last_run_tick < self.ingestion_interval:
            return 0
        self._last_run_tick = tick
        return self._run_ingestion(knowledge_graph, type_system, curiosity_model)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_state(self, path: "Path") -> None:
        """Load persisted ingestion state (survives process restarts)."""
        import json as _jj
        try:
            if not path.exists():
                return
            data = _jj.loads(path.read_text(encoding="utf-8"))
            for k, v in data.get("ingested", {}).items():
                self._ingested_concepts[k] = (float(v[0]), v[1])
            for c in data.get("failed", []):
                self._failed_concepts.add(c)
            logger.info("[ingestion] loaded_state ingested=%d failed=%d",
                        len(self._ingested_concepts), len(self._failed_concepts))
        except Exception as exc:
            logger.debug("[ingestion] state_load_failed: %s", exc)

    def _save_state(self, path: "Path") -> None:
        """Persist ingestion state to disk atomically."""
        import json as _jj
        try:
            data = {
                "ingested": {k: list(v) for k, v in self._ingested_concepts.items()},
                "failed":   sorted(self._failed_concepts),
            }
            tmp = path.with_suffix(".tmp")
            tmp.write_text(_jj.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except Exception as exc:
            logger.debug("[ingestion] state_save_failed: %s", exc)

    def _next_concepts(self, graph_node_names: Set[str]) -> List[str]:
        queue = [c for c in self._concept_queue
                 if c not in self._ingested_concepts
                 and c not in self._failed_concepts]
        scored_queue = sorted(
            [(c, _score_concept_proximity(c, graph_node_names)) for c in queue],
            key=lambda x: x[1],
            reverse=True,
        )
        batch = [c for c, _ in scored_queue[: self.ingestion_batch_size]]
        self._concept_queue = [c for c, _ in scored_queue[len(batch):]]

        remaining = self.ingestion_batch_size - len(batch)
        if remaining > 0:
            seeds = [
                s for s in _SEED_CONCEPTS
                if s not in self._ingested_concepts and s not in batch
            ]
            seed_scored = sorted(
                [(s, _score_concept_proximity(s, graph_node_names)) for s in seeds],
                key=lambda x: x[1],
                reverse=True,
            )
            batch.extend(s for s, _ in seed_scored[:remaining])

        return batch

    def _fetch_triplets_for_concept(self, concept: str) -> List[Tuple[str, str, str]]:
        all_triplets: List[Tuple[str, str, str]] = []

        if self.enable_wikidata:
            all_triplets.extend(
                _fetch_wikidata_relations(
                    concept, timeout=self.request_timeout,
                    max_relations=self.max_relations_per_concept,
                )
            )
        if self.enable_wikipedia:
            all_triplets.extend(
                _fetch_wikipedia_relations(
                    concept, timeout=self.request_timeout,
                    max_relations=self.max_relations_per_concept,
                )
            )
        if self.enable_wordnet:
            all_triplets.extend(
                _fetch_wordnet_relations(
                    concept, max_relations=self.max_relations_per_concept,
                )
            )

        seen: set = set()
        unique = []
        for t in all_triplets:
            if t not in seen:
                seen.add(t)
                unique.append(t)

        return unique[: self.max_relations_per_concept * 2]

    def _ingest_triplets_into_graph(
        self,
        triplets: List[Tuple[str, str, str]],
        knowledge_graph,
        type_system,
        is_default_accept: bool = False,
    ) -> int:
        from digital_baby.brain.knowledge_graph import GraphEdge

        # v3: slightly lower initial confidence for default-accept concepts
        base_confidence = 0.40 if is_default_accept else 0.52

        new_edge_count = 0
        existing: Dict[Tuple[str, str, str], int] = {
            (e.source, e.relation, e.target): i
            for i, e in enumerate(knowledge_graph.edges)
        }

        for subj, rel, obj in triplets:
            knowledge_graph.add_node(subj)
            knowledge_graph.add_node(obj)

            key = (subj, rel, obj)
            if key in existing:
                idx = existing[key]
                knowledge_graph.edges[idx].evidence    += 1
                knowledge_graph.edges[idx].confidence   = min(
                    0.99, knowledge_graph.edges[idx].confidence + 0.01
                )
                knowledge_graph.tick_stats.updated_edges += 1
                logger.debug(
                    "[graph] edge_updated %s %s %s evidence=%d",
                    subj, rel, obj, knowledge_graph.edges[idx].evidence,
                )
            else:
                edge = GraphEdge(
                    source=subj, target=obj, relation=rel,
                    confidence=base_confidence, evidence=1,
                )
                knowledge_graph.edges.append(edge)
                existing[key] = len(knowledge_graph.edges) - 1
                knowledge_graph.tick_stats.new_edges += 1
                new_edge_count += 1
                logger.info("[graph] edge_added %s %s %s", subj, rel, obj)

            type_system.infer_from_triplets([(subj, rel, obj)])

        return new_edge_count

    def _expand_queue_from_graph(
        self, ingested_concept: str, knowledge_graph
    ) -> List[str]:
        """v3 NEW: Add unvisited graph neighbours to the ingestion queue."""
        neighbours = knowledge_graph.get_neighbors(ingested_concept)
        new_seeds  = []
        for n in neighbours:
            if n not in self._ingested_concepts and n not in self._concept_queue:
                new_seeds.append(n)
                self._concept_queue.append(n)
        return new_seeds

    def _run_ingestion(
        self,
        knowledge_graph,
        type_system,
        curiosity_model,
    ) -> int:
        graph_node_names: Set[str] = set(knowledge_graph.nodes.keys())
        concepts = self._next_concepts(graph_node_names)

        if not concepts:
            logger.debug("[ingestion] no concepts available this cycle")
            return 0

        total_new_edges   = 0
        new_concepts_found: List[str] = []
        filtered_count    = 0

        for concept in concepts:
            triplets = self._fetch_triplets_for_concept(concept)

            if not triplets:
                filtered_count += 1
                self._ingested_concepts[concept] = (time.time(), None)
                self._failed_concepts.add(concept)  # never re-fetch barren concepts
                continue

            # v3: check if this concept was a default-accept (reduced confidence)
            entity  = _WIKIDATA_ENTITY_CACHE.get(concept.replace("_", " "))
            is_default = (entity is not None and not (set(entity.types) & ALLOWED_ONTOLOGY_TYPES))

            new_edges = self._ingest_triplets_into_graph(
                triplets, knowledge_graph, type_system, is_default_accept=is_default
            )
            total_new_edges += new_edges

            qid = entity.qid if entity else None
            self._ingested_concepts[concept] = (time.time(), qid)

            for subj, _rel, obj in triplets:
                for c in (subj, obj):
                    if c != concept and c not in self._ingested_concepts:
                        new_concepts_found.append(c)
                        if c not in self._concept_queue:
                            self._concept_queue.append(c)

            # v3 NEW: also expand queue from graph neighbours
            self._expand_queue_from_graph(concept, knowledge_graph)

        if new_concepts_found and curiosity_model is not None:
            unique_new = list(set(new_concepts_found))
            try:
                curiosity_model.register_unknowns("ingestion", unique_new)
            except Exception as exc:
                logger.debug("[ingestion] curiosity_update_failed error=%s", exc)

        logger.info(
            "[ingestion] cycle_complete new_edges=%d new_concepts=%d"
            " filtered=%d queued=%d",
            total_new_edges,
            len(set(new_concepts_found)),
            filtered_count,
            len(self._concept_queue),
        )


        if self._state_path is not None:
            self._save_state(self._state_path)
        return total_new_edges