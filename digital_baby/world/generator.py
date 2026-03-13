"""Procedural world generator for open-ended topic creation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple
import json
import random


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

    def generate_topic(self, persist: bool = True, domain: str | None = None) -> Dict:
        """Generate/update a domain topic with new combinatorial facts.

        Topic names are stable domain keys (e.g. `chemistry`, `astronomy`).
        """
        if domain is None:
            domain = self.random.choice(sorted(self.domains.keys()))
        template = self.domains.get(domain)
        if template is None:
            domain = self.random.choice(sorted(self.domains.keys()))
            template = self.domains[domain]

        for _ in range(25):
            facts = self._materialize_facts(template)
            signature = self._signature(domain, facts)
            if signature not in self.generated_signatures:
                self.generated_signatures.add(signature)
                break

        info = self.topic_registry.setdefault(domain, {"generations": 0, "last_signature": ""})
        info["generations"] += 1
        info["last_signature"] = signature

        page = {
            "topic": domain,
            "facts": facts,
            "generated": True,
            "domain": domain,
            "generation": info["generations"],
        }

        if persist:
            self.world_path.mkdir(parents=True, exist_ok=True)
            # overwrite domain-specific generated page for registry stability
            (self.world_path / f"generated_{domain}.json").write_text(json.dumps(page, indent=2), encoding="utf-8")

        return page
