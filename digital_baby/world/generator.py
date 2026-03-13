"""Procedural world generator for open-ended topic creation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import json
import random
import time


@dataclass
class DomainTemplate:
    """Template domain with relation triples for procedural page synthesis."""

    name: str
    triples: List[Tuple[str, str, str]]


class KnowledgeGenerator:
    """Generates unbounded knowledge pages and optionally persists them as JSON."""

    def __init__(self, world_path: str | Path, seed: int | None = None) -> None:
        self.world_path = Path(world_path)
        self.random = random.Random(seed)
        self.counter = 0
        self.domains = [
            DomainTemplate("ecosystem", [("wolf", "hunts", "deer"), ("deer", "eats", "plants"), ("plants", "grow_in", "soil")]),
            DomainTemplate("planetary", [("planet", "orbits", "star"), ("moon", "orbits", "planet"), ("star", "emits", "radiation")]),
            DomainTemplate("chemistry", [("acid", "reacts_with", "base"), ("reaction", "produces", "salt"), ("molecule", "contains", "atom")]),
            DomainTemplate("plant_biology", [("root", "absorbs", "water"), ("leaf", "performs", "photosynthesis"), ("plant", "needs", "sunlight")]),
            DomainTemplate("technology", [("sensor", "measures", "signal"), ("processor", "executes", "algorithm"), ("battery", "powers", "robot")]),
            DomainTemplate("predator_prey", [("lion", "hunts", "zebra"), ("hawk", "hunts", "rabbit"), ("fox", "hunts", "rodent")]),
        ]

    def generate_topic(self, persist: bool = True) -> Dict:
        """Generate a new synthetic topic page with unique topic name."""
        domain = self.random.choice(self.domains)
        self.counter += 1
        suffix = int(time.time() * 1000) % 100000 + self.counter
        topic = f"{domain.name}_{suffix}"
        facts = [f"{s} {r} {o}" for s, r, o in domain.triples]
        page = {"topic": topic, "facts": facts, "generated": True, "domain": domain.name}

        if persist:
            self.world_path.mkdir(parents=True, exist_ok=True)
            (self.world_path / f"{topic}.json").write_text(json.dumps(page, indent=2), encoding="utf-8")

        return page
