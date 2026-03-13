"""Inspect saved digital baby memory state from the terminal."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from digital_baby.brain.concepts import ConceptHierarchy
from digital_baby.brain.memory import Memory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect digital baby memory state.")
    parser.add_argument("--memory", default="digital_baby/world/memory_store.json", help="Path to memory JSON file.")
    parser.add_argument("--max-facts", type=int, default=20, help="Max number of facts to print.")
    parser.add_argument("--export-triplets", default=None, help="Optional path to export graph triplets TSV.")
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
    triplets = memory.relation_triplets()
    relation_count = len(triplets)
    conflicts = memory.conflicting_relations_with_evidence()

    hierarchy = ConceptHierarchy()
    hierarchy.ingest_triplets(triplets)
    top_concepts = hierarchy.top_concepts_by_connectivity(top_n=10)

    print("=== DIGITAL BABY BRAIN INSPECTION ===")
    print(f"Memory file: {memory_path}")
    print(f"Facts: {len(facts)}")
    print(f"Concepts: {len(entities)}")
    print(f"Relations: {relation_count}")
    print(f"Patterns: {len(memory.patterns)}")
    print(f"Conflicts: {len(conflicts)}")
    print()

    print("-- Top Concepts by Connectivity --")
    if not top_concepts:
        print("<none>")
    else:
        for concept, score in top_concepts:
            print(f"{concept}: {score}")
    print()

    print("-- Top Facts (by confidence) --")
    for fact in facts[: args.max_facts]:
        marker = " [compressed]" if fact.compressed else ""
        print(f"[{fact.confidence:.3f}] ev={fact.evidence} ({fact.source_topic}) {fact.statement}{marker}")
    print()

    print("-- Patterns --")
    if not memory.patterns:
        print("<none>")
    else:
        for p in memory.patterns:
            print(f"{p.template} | support={p.support} | label={p.label}")
    print()

    print("-- Conflicts (with evidence) --")
    if not conflicts:
        print("<none>")
    else:
        for subject, relation, ranked in conflicts:
            print(f"{subject} -> {relation} -> {ranked}")

    if args.export_triplets:
        export_path = Path(args.export_triplets)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["subject\trelation\tobject"]
        for subject, relation, obj in triplets:
            lines.append(f"{subject}\t{relation}\t{obj}")
        export_path.write_text("\n".join(lines), encoding="utf-8")
        print()
        print(f"Exported triplets to: {export_path}")


if __name__ == "__main__":
    main()
