"""Inspect saved digital baby memory state from the terminal.

Usage examples
--------------
# Default: top 20 facts sorted by confidence
python tools/inspect_brain.py --memory digital_baby/world/memory_store.json

# Show top 40 facts sorted by raw evidence count
python tools/inspect_brain.py --max-facts 40 --sort evidence

# Run the built-in confidence model verification suite
python tools/inspect_brain.py --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from digital_baby.brain.concepts import ConceptTypeSystem
from digital_baby.brain.memory import (
    CONFIDENCE_PRIOR_K,
    STALE_DECAY_RATE,
    STALE_TICKS,
    Memory,
    evidence_to_confidence,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 20) -> str:
    """Return a compact ASCII progress bar for a 0–1 value."""
    filled = round(value * width)
    return "[" + "█" * filled + "·" * (width - filled) + "]"


def _conf_label(conf: float) -> str:
    if conf >= 0.90:
        return "very-high"
    if conf >= 0.70:
        return "high"
    if conf >= 0.50:
        return "medium"
    if conf >= 0.20:
        return "low"
    return "very-low"


# ── Verification suite ────────────────────────────────────────────────────────

def run_verification() -> bool:
    """Run all confidence model invariant checks.  Returns True if all pass."""
    print("=" * 60)
    print("CONFIDENCE MODEL VERIFICATION")
    print(f"  formula : evidence / (evidence + k),  k = {CONFIDENCE_PRIOR_K}")
    print(f"  stale   : penalty after {STALE_TICKS} ticks, rate = {STALE_DECAY_RATE}/tick")
    print("=" * 60)

    failures: list[str] = []

    # ── 1. Monotonicity ──────────────────────────────────────────────────────
    print("\n1. Monotonicity — confidence must strictly increase with evidence")
    prev = -1.0
    mono_ok = True
    for ev in range(1, 201):
        c = evidence_to_confidence(ev)
        if c <= prev:
            failures.append(f"  FAIL monotonicity at evidence={ev}: {c:.6f} <= {prev:.6f}")
            mono_ok = False
        prev = c
    status = "PASS" if mono_ok else "FAIL"
    print(f"   {status}  (checked evidence 1..200)")

    # ── 2. Boundary values ───────────────────────────────────────────────────
    print("\n2. Key evidence thresholds")
    cases = [
        (1,    0.05, 0.15,  "single observation"),
        (10,   0.45, 0.55,  "10 observations → ~0.50"),
        (100,  0.88, 0.95,  "100 observations → ~0.91"),
        (1000, 0.98, 1.00,  "1000 observations → ~0.99"),
    ]
    for ev, lo, hi, label in cases:
        c = evidence_to_confidence(ev)
        ok = lo <= c <= hi
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failures.append(f"  FAIL threshold ev={ev}: {c:.4f} not in [{lo}, {hi}]")
        print(f"   {mark}  ev={ev:5d}  conf={c:.4f}  {_bar(c, 16)}  {label}")

    # ── 3. Re-observation restores confidence ────────────────────────────────
    print("\n3. Re-observation after stale decay restores full confidence")
    import tempfile, json, os

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = f.name
        json.dump({}, f)

    try:
        mem = Memory(tmp_path)
        mem.current_tick = 1
        mem.upsert_fact("wolf hunts deer", 0.0, "test", evidence_increment=50)
        fact = mem.get_fact("wolf hunts deer")
        base_conf = evidence_to_confidence(50)
        assert fact is not None

        # Simulate 200 ticks of staleness
        mem.current_tick = 201
        mem.apply_stale_decay()
        stale_conf = fact.confidence
        stale_ok = stale_conf < base_conf
        print(f"   {'PASS' if stale_ok else 'FAIL'}  stale penalty applied: {base_conf:.4f} → {stale_conf:.4f}")
        if not stale_ok:
            failures.append("  FAIL stale decay did not reduce confidence")

        # Re-observe — confidence must snap back to evidence-based value
        mem.current_tick = 202
        mem.upsert_fact("wolf hunts deer", 0.0, "test", evidence_increment=1)
        restored_conf = fact.confidence
        expected_conf = evidence_to_confidence(51)
        snap_ok = abs(restored_conf - expected_conf) < 1e-9
        print(f"   {'PASS' if snap_ok else 'FAIL'}  re-observation snaps back: {restored_conf:.4f} (expected {expected_conf:.4f})")
        if not snap_ok:
            failures.append(f"  FAIL snap-back: got {restored_conf:.4f}, expected {expected_conf:.4f}")

        # Evidence was never modified by the stale decay
        evidence_ok = fact.evidence == 51
        print(f"   {'PASS' if evidence_ok else 'FAIL'}  evidence count untouched by decay: {fact.evidence}")
        if not evidence_ok:
            failures.append(f"  FAIL evidence modified by decay: {fact.evidence}")

    finally:
        os.unlink(tmp_path)

    # ── 4. Migration of old store ────────────────────────────────────────────
    print("\n4. Old store migration — stored confidence overwritten by evidence formula")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_path = f.name
        # Write a store that mimics the old model: ev=1006, conf=0.0 (decayed to zero)
        json.dump({
            "facts": [
                {"statement": "cat is mammal", "confidence": 0.0, "source_topic": "animals",
                 "timestamp": 0.0, "evidence": 1006, "compressed": False}
            ],
            "entity_relations": {}, "relation_evidence": [], "fact_index": [],
            "patterns": [], "causal_rules": [],
            "world_model": {"rules": [], "predictions": [], "experiments": []},
        }, f)

    try:
        mem2 = Memory(tmp_path)
        migrated = mem2.get_fact("cat is mammal")
        assert migrated is not None
        expected = evidence_to_confidence(1006)
        mig_ok = abs(migrated.confidence - expected) < 1e-9
        print(f"   {'PASS' if mig_ok else 'FAIL'}  ev=1006, old_conf=0.000 → migrated_conf={migrated.confidence:.4f} (expected {expected:.4f})")
        if not mig_ok:
            failures.append(f"  FAIL migration: got {migrated.confidence:.4f}, expected {expected:.4f}")
    finally:
        os.unlink(tmp_path)

    # ── 5. Bounded in (0, 1) ────────────────────────────────────────────────
    print("\n5. Output always in (0, 1)")
    edge_cases = [1, 2, 5, 10, 100, 10_000, 1_000_000]
    bounds_ok = all(0.0 < evidence_to_confidence(e) < 1.0 for e in edge_cases)
    print(f"   {'PASS' if bounds_ok else 'FAIL'}  checked {edge_cases}")
    if not bounds_ok:
        failures.append("  FAIL output out of (0, 1) bounds")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S)")
        for f in failures:
            print(f)
        return False
    else:
        print("RESULT: ALL CHECKS PASSED")
        return True


# ── Main report ───────────────────────────────────────────────────────────────

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
        "--sort",
        choices=["confidence", "evidence"],
        default="confidence",
        help="Sort top facts by confidence (default) or raw evidence count.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run the confidence model verification suite and exit.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.verify:
        ok = run_verification()
        sys.exit(0 if ok else 1)

    memory = Memory(Path(args.memory))
    conflicts = memory.conflicting_relations_with_evidence()

    type_system = ConceptTypeSystem()
    type_system.infer_from_triplets(memory.relation_triplets())

    # ── Header ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("DIGITAL BABY BRAIN INSPECTION")
    print("=" * 60)
    print(f"Facts             : {len(memory.facts)}")
    print(f"Unique fact index : {len(memory.fact_index)}")
    print(f"Concepts          : {len(memory.all_entities())}")
    print(f"Typed entities    : {len(type_system.entity_types)}")
    print(f"Patterns          : {len(memory.patterns)}")
    print(f"Causal rules      : {len(memory.causal_rules)}")
    print(f"Hypotheses        : {len(memory.world_model.get('rules', []))}")
    print(f"Predictions logged: {len(memory.world_model.get('predictions', []))}")
    print(f"Experiments logged: {len(memory.world_model.get('experiments', []))}")
    print(f"Conflicts         : {len(conflicts)}")
    print()

    # ── Confidence model info ─────────────────────────────────────────────────
    print("── Confidence model ──────────────────────────────────────")
    print(f"  formula : evidence / (evidence + k),  k = {CONFIDENCE_PRIOR_K}")
    print(f"  stale   : penalty after {STALE_TICKS} unseen ticks,  rate = {STALE_DECAY_RATE}/tick")
    print(f"  ev=1 → {evidence_to_confidence(1):.3f}  "
          f"ev=10 → {evidence_to_confidence(10):.3f}  "
          f"ev=100 → {evidence_to_confidence(100):.3f}  "
          f"ev=1000 → {evidence_to_confidence(1000):.3f}")
    print()

    # ── Top facts ─────────────────────────────────────────────────────────────
    sort_key = (lambda f: f.confidence) if args.sort == "confidence" else (lambda f: f.evidence)
    facts_sorted = sorted(memory.facts.values(), key=sort_key, reverse=True)

    print(f"── Top {args.max_facts} facts (sorted by {args.sort}) ─────────────────────────")
    print(f"  {'conf':>6}  {'ev':>6}  bar                   label      statement")
    print(f"  {'------':>6}  {'------':>6}  {'--------------------':20}  {'--------':9}  ---------")
    for fact in facts_sorted[: args.max_facts]:
        # Recompute the pure evidence-based confidence for display, separate
        # from the stored value which may include a stale penalty.
        base_conf = evidence_to_confidence(fact.evidence)
        stale_note = ""
        if abs(fact.confidence - base_conf) > 0.001:
            stale_note = f" [stale -{base_conf - fact.confidence:.3f}]"
        compressed_note = " [compressed]" if fact.compressed else ""
        label = _conf_label(fact.confidence)
        print(
            f"  {fact.confidence:6.3f}  {fact.evidence:6d}  {_bar(fact.confidence)}"
            f"  {label:9s}  {fact.statement}{stale_note}{compressed_note}"
        )
    print()

    # ── Causal rules ──────────────────────────────────────────────────────────
    if memory.causal_rules:
        print("── Causal rules ──────────────────────────────────────────")
        print(f"  {'conf':>6}  {'obs':>6}  direction    cause → effect")
        print(f"  {'------':>6}  {'------':>6}  -----------  -----------------")
        for rule in sorted(memory.causal_rules, key=lambda r: r.confidence, reverse=True):
            arrow = {"positive": "─(+)→", "negative": "─(-)→", "mixed": "─(?)→"}.get(rule.direction, "─────→")
            print(f"  {rule.confidence:6.3f}  {rule.observations:6d}  {rule.direction:11s}  {rule.cause} {arrow} {rule.effect}")
        print()

    # ── Entity types ──────────────────────────────────────────────────────────
    print("── Sample entity types ───────────────────────────────────")
    for entity, t in sorted(type_system.entity_types.items())[:20]:
        conf = type_system.get_type_confidence(entity)
        print(f"  {entity}: {t} ({conf:.2f})")
    print()

    # ── Belief states ─────────────────────────────────────────────────────────
    print("── Belief states (conflicts resolved) ────────────────────")
    if not conflicts:
        print("  <none>")
    else:
        for subject, relation, _ranked in conflicts:
            belief = memory.resolve_belief(subject, relation)
            if belief:
                alts = ", ".join(belief.alternatives) if belief.alternatives else "—"
                print(
                    f"  {subject} {relation}: "
                    f"best={belief.best}  conf={belief.confidence:.2f}  "
                    f"alternatives=[{alts}]"
                )
    print()

    # ── Evidence distribution ─────────────────────────────────────────────────
    all_ev = [f.evidence for f in memory.facts.values()]
    if all_ev:
        buckets = {
            "1":       sum(1 for e in all_ev if e == 1),
            "2–9":     sum(1 for e in all_ev if 2 <= e <= 9),
            "10–49":   sum(1 for e in all_ev if 10 <= e <= 49),
            "50–199":  sum(1 for e in all_ev if 50 <= e <= 199),
            "200+":    sum(1 for e in all_ev if e >= 200),
        }
        print("── Evidence distribution ─────────────────────────────────")
        for label, count in buckets.items():
            bar_width = round(count / len(all_ev) * 30)
            print(f"  ev {label:>6}  {count:5d}  {'█' * bar_width}")
        avg_conf = sum(evidence_to_confidence(e) for e in all_ev) / len(all_ev)
        print(f"\n  Total facts : {len(all_ev)}")
        print(f"  Avg confidence (evidence-based) : {avg_conf:.3f}")
    print()


if __name__ == "__main__":
    main()