"""Procedural world generator for open-ended topic creation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple
import json
import random
import time


@dataclass
class DomainTemplate:
    """Template domain with entity pools and relation schemas."""

    name: str
    entity_pools: Dict[str, Sequence[str]]
    schema: List[Tuple[str, str, str]]


class KnowledgeGenerator:
    """Generates unbounded, combinatorial knowledge pages and persists them as JSON."""

    def __init__(self, world_path: str | Path, seed: int | None = None) -> None:
        self.world_path = Path(world_path)
        self.random = random.Random(seed)
        self.counter = 0
        self.generated_signatures: Set[str] = set()
        self.domains = [
            DomainTemplate(
                name="ecosystem",
                entity_pools={
                    "predator": ["wolf", "lion", "tiger", "lynx", "eagle"],
                    "prey": ["deer", "rabbit", "zebra", "mouse", "antelope"],
                    "plant": ["grass", "bush", "shrub", "fern", "tree"],
                    "soil": ["soil", "wetland", "meadow"],
                },
                schema=[
                    ("predator", "hunts", "prey"),
                    ("prey", "eats", "plant"),
                    ("plant", "grows_in", "soil"),
                ],
            ),
            DomainTemplate(
                name="planetary",
                entity_pools={
                    "planet": ["mercury", "venus", "earth", "mars", "kepler_22b"],
                    "star": ["sun", "sirius", "vega", "proxima_centauri"],
                    "moon": ["luna", "phobos", "deimos", "europa", "titan"],
                    "radiation": ["light", "radiation", "infrared", "ultraviolet"],
                },
                schema=[
                    ("planet", "orbits", "star"),
                    ("moon", "orbits", "planet"),
                    ("star", "emits", "radiation"),
                ],
            ),
            DomainTemplate(
                name="chemistry",
                entity_pools={
                    "acid": ["hydrochloric_acid", "sulfuric_acid", "acetic_acid"],
                    "base": ["sodium_hydroxide", "ammonia", "potassium_hydroxide"],
                    "salt": ["sodium_chloride", "potassium_sulfate", "ammonium_acetate"],
                    "molecule": ["water", "glucose", "ethanol", "methane"],
                    "atom": ["carbon", "oxygen", "hydrogen", "nitrogen"],
                },
                schema=[
                    ("acid", "reacts_with", "base"),
                    ("reaction", "produces", "salt"),
                    ("molecule", "contains", "atom"),
                ],
            ),
            DomainTemplate(
                name="technology",
                entity_pools={
                    "sensor": ["camera", "lidar", "thermometer", "microphone"],
                    "signal": ["image", "distance", "temperature", "audio"],
                    "processor": ["cpu", "gpu", "microcontroller", "asic"],
                    "algorithm": ["filter", "classifier", "planner", "optimizer"],
                    "battery": ["lithium_pack", "supercapacitor", "fuel_cell"],
                    "robot": ["drone", "rover", "arm_bot", "submarine_bot"],
                },
                schema=[
                    ("sensor", "measures", "signal"),
                    ("processor", "executes", "algorithm"),
                    ("battery", "powers", "robot"),
                ],
            ),
            DomainTemplate(
                name="predator_prey",
                entity_pools={
                    "predator": ["lion", "wolf", "hawk", "fox", "orca"],
                    "prey": ["zebra", "deer", "rabbit", "rodent", "seal"],
                    "prey2": ["antelope", "hare", "salmon", "goat", "buffalo"],
                    "plant": ["grass", "kelp", "berries", "lichen"],
                },
                schema=[
                    ("predator", "hunts", "prey"),
                    ("predator", "hunts", "prey2"),
                    ("prey", "eats", "plant"),
                    ("prey2", "eats", "plant"),
                ],
            ),
        ]

    def _materialize_facts(self, domain: DomainTemplate) -> List[str]:
        bindings: Dict[str, str] = {}
        for pool_name, values in domain.entity_pools.items():
            bindings[pool_name] = self.random.choice(list(values))

        facts: List[str] = []
        for subj_key, relation, obj_key in domain.schema:
            subj = bindings.get(subj_key, subj_key)
            obj = bindings.get(obj_key, obj_key)
            facts.append(f"{subj} {relation} {obj}")
        return facts

    def _signature(self, domain_name: str, facts: List[str]) -> str:
        return domain_name + "::" + "|".join(sorted(facts))

    def generate_topic(self, persist: bool = True, max_attempts: int = 20) -> Dict:
        """Generate a new synthetic topic page with combinatorial fact diversity."""
        for _ in range(max_attempts):
            domain = self.random.choice(self.domains)
            facts = self._materialize_facts(domain)
            signature = self._signature(domain.name, facts)
            if signature not in self.generated_signatures:
                self.generated_signatures.add(signature)
                break
        else:
            # Fallback if all combinations sampled in current process.
            domain = self.random.choice(self.domains)
            facts = self._materialize_facts(domain)

        self.counter += 1
        suffix = int(time.time() * 1000) % 100000 + self.counter
        topic = f"{domain.name}_{suffix}"
        page = {"topic": topic, "facts": facts, "generated": True, "domain": domain.name}

        if persist:
            self.world_path.mkdir(parents=True, exist_ok=True)
            (self.world_path / f"{topic}.json").write_text(json.dumps(page, indent=2), encoding="utf-8")

        return page
