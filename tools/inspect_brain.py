"""Inspect saved digital baby memory state from the terminal."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from digital_baby.brain.memory import Memory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect digital baby memory state.")
    parser.add_argument(
        "--memory",
        default="digital_baby/world/memory_store.json",
        help="Path to memory JSON file.",
    )
    parser.add_argument(
        "--max-facts",
        type=int,
        default=20,
        help="Max number of facts to print.",
    )
    parser.add_argument(
        "--export-triplets",
        default=None,
        help="Optional path to export graph triplets TSV.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    memory_path = Path(args.memory)

    if not memory_path.exists():
        print(f"Memory file not found: {memory_path}")
        return

    memory = Memory(memory_path)
    facts = sorted(memory.facts.values(), key=lambda f: f.confidence, reverse=True)
    entities = sorted(memory.all_entities())
    conflicts = memory.conflicting_relations()

    print("=== DIGITAL BABY BRAIN INSPECTION ===")
    print(f"Memory file: {memory_path}")
    print(f"Stored facts: {len(facts)}")
    print(f"Known concepts/entities: {len(entities)}")
    print(f"Conflicting relations: {len(conflicts)}")
    print()

    print("-- Top Facts (by confidence) --")
    for fact in facts[: args.max_facts]:
        print(f"[{fact.confidence:.3f}] ({fact.source_topic}) {fact.statement}")
    print()

    print("-- Concept List --")
    print(", ".join(entities[:80]) if entities else "<empty>")
    if len(entities) > 80:
        print(f"... and {len(entities) - 80} more")
    print()

    print("-- Relation Graph --")
    for subject in sorted(memory.entity_relations):
        rel_map = memory.entity_relations[subject]
        for relation in sorted(rel_map):
            for obj in sorted(rel_map[relation]):
                print(f"{subject} -> {relation} -> {obj}")
    print()

    print("-- Conflicts --")
    if not conflicts:
        print("<none>")
    else:
        for subject, relation, values in conflicts:
            print(f"{subject} -> {relation} -> {values}")

    if args.export_triplets:
        export_path = Path(args.export_triplets)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["subject\trelation\tobject"]
        for subject, relation, obj in memory.relation_triplets():
            lines.append(f"{subject}\t{relation}\t{obj}")
        export_path.write_text("\n".join(lines), encoding="utf-8")
        print()
        print(f"Exported triplets to: {export_path}")


if __name__ == "__main__":
    main()
