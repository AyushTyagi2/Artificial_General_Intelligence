"""Procedural world generator for open-ended topic creation.

This generator supports continuous novelty injection and a lightweight
stateful world simulator used for causal learning.
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
    """Generates open-world facts and simulates stateful domains."""

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
        self.domain_states: Dict[str, Dict[str, float]] = {
            "ecosystem": {"wolves": 5, "deer": 20, "grass": 100},
            "astronomy": {"planets": 8, "asteroids": 200, "solar_energy": 1000},
            "chemistry": {"temperature": 25, "reactants": 2, "reaction_energy": 0},
            "technology": {"robots": 4, "sensors": 10, "battery_charge": 80},
        }

        self.actions_by_domain: Dict[str, List[str]] = {
            "ecosystem": ["remove_predator", "add_predator", "introduce_species", "remove_species"],
            "astronomy": ["introduce_species", "remove_species"],
            "chemistry": ["increase_temperature", "add_chemical"],
            "technology": ["introduce_species", "remove_species"],
        }

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

    def state_for_domain(self, domain: str) -> Dict[str, float]:
        return dict(self.domain_states.setdefault(domain, {}))

    def update_domain_state(self, domain: str, new_state: Dict[str, float]) -> None:
        self.domain_states[domain] = dict(new_state)

    def choose_action(self, domain: str) -> str:
        return self.random.choice(self.actions_by_domain.get(domain, ["introduce_species"]))

    def _materialize_facts(self, template: DomainTemplate) -> List[str]:
        bindings = {k: self.random.choice(list(v)) for k, v in template.entity_pools.items()}
        return [f"{bindings.get(s, s)} {r} {bindings.get(o, o)}" for s, r, o in template.schema]

    def _signature(self, domain: str, facts: List[str]) -> str:
        return domain + "::" + "|".join(sorted(facts))

    def _safe_json_get(self, url: str) -> dict:
        with urlopen(url, timeout=6) as response:  # nosec B310 - trusted https endpoint
            return json.loads(response.read().decode("utf-8"))

    def _wikidata_random_entity(self) -> str | None:
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

    def state_snapshot_facts(self, domain: str) -> List[str]:
        state = self.state_for_domain(domain)
        return [f"{key} is {int(value) if isinstance(value, (int, float)) else value}" for key, value in sorted(state.items())]

    def inject_open_world_novelty(self, max_entities: int = 3) -> Tuple[List[str], List[str]]:
        new_facts: List[str] = []
        new_concepts: List[str] = []
        entity_count = self.random.randint(1, max(1, max_entities))

        for _ in range(entity_count):
            concept = self._wikidata_random_entity() if self.random.random() < 0.65 else None
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
        if domain is None:
            domain = self.random.choice(sorted(self.domains.keys()))
        template = self.domains.get(domain) or self.domains[self.random.choice(sorted(self.domains.keys()))]

        for _ in range(25):
            facts = self._materialize_facts(template)
            facts.extend(self.state_snapshot_facts(template.name))
            if self.random.random() < 0.15:
                novelty_facts, _ = self.inject_open_world_novelty(max_entities=2)
                facts.extend(novelty_facts)
            signature = self._signature(template.name, facts)
            if signature not in self.generated_signatures:
                self.generated_signatures.add(signature)
                break

        info = self.topic_registry.setdefault(template.name, {"generations": 0, "last_signature": ""})
        info["generations"] += 1
        info["last_signature"] = signature

        page = {
            "topic": template.name,
            "facts": sorted(set(facts)),
            "generated": True,
            "domain": template.name,
            "generation": info["generations"],
        }

        if persist:
            self.world_path.mkdir(parents=True, exist_ok=True)
            (self.world_path / f"generated_{template.name}.json").write_text(json.dumps(page, indent=2), encoding="utf-8")

        return page
