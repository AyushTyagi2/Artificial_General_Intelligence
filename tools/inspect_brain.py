"""Inspect saved digital baby memory state from the terminal."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from digital_baby.brain.concepts import ConceptTypeSystem
from digital_baby.brain.memory import Memory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect digital baby memory state.")
    parser.add_argument("--memory", default="digital_baby/world/memory_store.json", help="Path to memory JSON file.")
    parser.add_argument("--max-facts", type=int, default=20, help="Max number of facts to print.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    memory = Memory(Path(args.memory))

    facts = sorted(memory.facts.values(), key=lambda f: f.confidence, reverse=True)
    conflicts = memory.conflicting_relations_with_evidence()

    type_system = ConceptTypeSystem()
    type_system.infer_from_triplets(memory.relation_triplets())

    print("=== DIGITAL BABY BRAIN INSPECTION ===")
    print(f"Facts: {len(memory.facts)}")
    print(f"Unique fact index: {len(memory.fact_index)}")
    print(f"Concepts: {len(memory.all_entities())}")
    print(f"Typed entities: {len(type_system.entity_types)}")
    print(f"Patterns: {len(memory.patterns)}")
    print(f"Rules: {len(memory.world_model.get('rules', []))}")
    print(f"Predictions logged: {len(memory.world_model.get('predictions', []))}")
    print(f"Conflicts: {len(conflicts)}")
    print()

    print("-- Top Facts --")
    for fact in facts[: args.max_facts]:
        marker = " [compressed]" if fact.compressed else ""
        print(f"[{fact.confidence:.3f}] ev={fact.evidence} {fact.statement}{marker}")
    print()

    print("-- Sample Entity Types --")
    for entity, t in sorted(type_system.entity_types.items())[:20]:
        print(f"{entity}: {t} ({type_system.get_type_confidence(entity):.2f})")
    print()

    print("-- Belief States (conflicts resolved) --")
    if not conflicts:
        print("<none>")
    else:
        for subject, relation, _ranked in conflicts:
            belief = memory.resolve_belief(subject, relation)
            if belief:
                print(f"{subject} {relation}: best={belief.best} alternatives={belief.alternatives} confidence={belief.confidence:.2f}")


if __name__ == "__main__":
    main()
