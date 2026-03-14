"""Procedural world generator for open-ended topic creation.

This generator supports continuous novelty injection by mixing:
- template-based domain facts,
- external concept discovery from Wikidata,
- synthetic concept mutation,
- random relation discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple
from urllib.parse import quote
from urllib.request import urlopen
import json
import random

from digital_baby.brain.knowledge_expander import KnowledgeExpander


@dataclass
class DomainTemplate:
    name: str
    entity_pools: Dict[str, Sequence[str]]
    schema: List[Tuple[str, str, str]]


class KnowledgeGenerator:
    """Generates new knowledge inside stable domains using a topic registry."""

    def __init__(self, world_path: str | Path, seed: int | None = None) -> None:
        self.world_path = Path(world_path)
        self.random = random.Random(seed)
        self.generated_signatures: Set[str] = set()
        self.topic_registry: Dict[str, Dict] = {}
        self.expander = KnowledgeExpander(self.world_path)
        self.relation_pool = ["part_of", "located_in", "interacts_with", "depends_on"]
        self.discovery_queries = ["animal", "species", "planet", "ecosystem", "technology", "fungus", "robot"]
        self.mutation_prefixes = ["bio", "nano", "quantum", "alien", "eco", "neuro"]
        self.mutation_roots = ["robot", "plant", "ecosystem", "species", "habitat", "sensor"]

        self.domains = {
            "ecosystem": DomainTemplate(
                "ecosystem",
                {
                    "predator": ["wolf", "lion", "tiger", "lynx", "eagle"],
                    "prey": ["deer", "rabbit", "zebra", "mouse", "antelope"],
                    "plant": ["grass", "bush", "shrub", "fern", "tree"],
                    "soil": ["soil", "wetland", "meadow"],
                },
                [("predator", "hunts", "prey"), ("prey", "eats", "plant"), ("plant", "grows_in", "soil")],
            ),
            "astronomy": DomainTemplate(
                "astronomy",
                {
                    "planet": ["mercury", "venus", "earth", "mars", "kepler_22b"],
                    "star": ["sun", "sirius", "vega", "proxima_centauri"],
                    "moon": ["luna", "phobos", "deimos", "europa", "titan"],
                    "radiation": ["light", "radiation", "infrared", "ultraviolet"],
                },
                [("planet", "orbits", "star"), ("moon", "orbits", "planet"), ("star", "emits", "radiation")],
            ),
            "chemistry": DomainTemplate(
                "chemistry",
                {
                    "acid": ["hydrochloric_acid", "sulfuric_acid", "acetic_acid"],
                    "base": ["sodium_hydroxide", "ammonia", "potassium_hydroxide"],
                    "salt": ["sodium_chloride", "potassium_sulfate", "ammonium_acetate"],
                    "molecule": ["water", "glucose", "ethanol", "methane"],
                    "atom": ["carbon", "oxygen", "hydrogen", "nitrogen"],
                },
                [("acid", "reacts_with", "base"), ("reaction", "produces", "salt"), ("molecule", "contains", "atom")],
            ),
            "technology": DomainTemplate(
                "technology",
                {
                    "sensor": ["camera", "lidar", "thermometer", "microphone"],
                    "signal": ["image", "distance", "temperature", "audio"],
                    "processor": ["cpu", "gpu", "microcontroller", "asic"],
                    "algorithm": ["filter", "classifier", "planner", "optimizer"],
                    "battery": ["lithium_pack", "supercapacitor", "fuel_cell"],
                    "robot": ["drone", "rover", "arm_bot", "submarine_bot"],
                },
                [("sensor", "measures", "signal"), ("processor", "executes", "algorithm"), ("battery", "powers", "robot")],
            ),
        }

    def _materialize_facts(self, template: DomainTemplate) -> List[str]:
        bindings = {k: self.random.choice(list(v)) for k, v in template.entity_pools.items()}
        return [f"{bindings.get(s, s)} {r} {bindings.get(o, o)}" for s, r, o in template.schema]

    def _signature(self, domain: str, facts: List[str]) -> str:
        return domain + "::" + "|".join(sorted(facts))

    def _safe_json_get(self, url: str) -> dict:
        with urlopen(url, timeout=6) as response:  # nosec B310 - trusted https endpoint
            return json.loads(response.read().decode("utf-8"))

    def _wikidata_random_entity(self) -> str | None:
        """Fetch a random concept from a broad query bucket."""
        query = self.random.choice(self.discovery_queries)
        url = (
            "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&language=en&type=item"
            f"&search={quote(query)}&limit=20"
        )
        try:
            payload = self._safe_json_get(url)
            items = payload.get("search", [])
            if not items:
                return None
            chosen = self.random.choice(items)
            label = chosen.get("label") or ""
            return label.strip().lower().replace(" ", "_") or None
        except Exception:
            return None

    def _mutated_concept(self) -> str:
        return f"{self.random.choice(self.mutation_prefixes)}_{self.random.choice(self.mutation_roots)}"

    def _random_relation_fact(self, concepts: Sequence[str]) -> str | None:
        if len(concepts) < 2:
            return None
        subject = self.random.choice(list(concepts))
        obj = self.random.choice([c for c in concepts if c != subject] or list(concepts))
        relation = self.random.choice(self.relation_pool)
        return f"{subject} {relation} {obj}"

    def inject_open_world_novelty(self, max_entities: int = 3) -> Tuple[List[str], List[str]]:
        """Inject 1..N novel entities and optional random relations.

        Returns (new_facts, new_concepts).
        """
        new_facts: List[str] = []
        new_concepts: List[str] = []
        entity_count = self.random.randint(1, max(1, max_entities))

        for _ in range(entity_count):
            concept = None
            if self.random.random() < 0.65:
                concept = self._wikidata_random_entity()
            if not concept:
                concept = self._mutated_concept()

            if self.expander.is_cached_concept(concept):
                continue

            new_concepts.append(concept)
            edges = self.expander.expand_concept_graph(concept, depth=2)
            if edges:
                for source, relation, target in edges:
                    new_facts.append(f"{source} {relation} {target}")
            else:
                new_facts.append(f"{concept} is unknown")

        relation_fact = self._random_relation_fact(new_concepts)
        if relation_fact:
            new_facts.append(relation_fact)
        return sorted(set(new_facts)), sorted(set(new_concepts))

    def generate_topic(self, persist: bool = True, domain: str | None = None) -> Dict:
        """Generate/update a domain topic with combinatorial and open-world facts."""
        if domain is None:
            domain = self.random.choice(sorted(self.domains.keys()))
        template = self.domains.get(domain)
        if template is None:
            domain = self.random.choice(sorted(self.domains.keys()))
            template = self.domains[domain]

        for _ in range(25):
            facts = self._materialize_facts(template)
            # occasional concept discovery and mutation.
            if self.random.random() < 0.15:
                novelty_facts, _new_concepts = self.inject_open_world_novelty(max_entities=2)
                facts.extend(novelty_facts)
            signature = self._signature(domain, facts)
            if signature not in self.generated_signatures:
                self.generated_signatures.add(signature)
                break

        info = self.topic_registry.setdefault(domain, {"generations": 0, "last_signature": ""})
        info["generations"] += 1
        info["last_signature"] = signature

        page = {
            "topic": domain,
            "facts": sorted(set(facts)),
            "generated": True,
            "domain": domain,
            "generation": info["generations"],
        }

        if persist:
            self.world_path.mkdir(parents=True, exist_ok=True)
            (self.world_path / f"generated_{domain}.json").write_text(json.dumps(page, indent=2), encoding="utf-8")

        return page
