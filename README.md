# Digital Baby: Open-Ended Curiosity Engine (V3)

`digital_baby` is an extensible developmental learning prototype.

The agent continuously:

1. explores world topics,
2. learns facts into persistent graph memory,
3. tracks evidence and contradictions,
4. discovers recurring relational patterns,
5. forms concept hierarchies,
6. compresses redundant memories,
7. procedurally generates new topics when learning stalls.

This enables open-ended behavior instead of stopping after static pages are exhausted.

## Project Structure

```text
digital_baby/
  brain/
    memory.py
    curiosity.py
    reasoning.py
    learner.py
    questions.py
    patterns.py
    concepts.py
  world/
    knowledge_pages/
      *.json
    generator.py
    memory_store.json   # generated at runtime
  engine/
    event_loop.py
  main.py

tools/
  inspect_brain.py
```

## Key V3 Features

- **Procedural world generation** (`world/generator.py`) for unbounded new topics.
- **Evidence-aware memory** (`evidence` counts per fact/relation).
- **Conflict comparison with evidence** (competing values ranked by support).
- **Pattern discovery** (`brain/patterns.py`) from recurring relation types.
- **Concept hierarchies** (`brain/concepts.py`) inferred from `is` relations.
- **Memory compression** for repeated structures (e.g. summarized `hunts` facts).
- **Prediction-error curiosity** (contradictions raise exploration pressure).
- **Improved logging** with topic reason, new facts, patterns, compression, memory size.
- **Brain inspection tool** including concepts/relations/pattern/conflict summaries.

## Knowledge Page Format

```json
{
  "topic": "animals",
  "facts": [
    "cat is mammal",
    "bird is animal"
  ]
}
```

You can keep adding static pages under `digital_baby/world/knowledge_pages/`, and the
agent can also generate procedural pages automatically.

## Run the Agent

```bash
python -m digital_baby.main
```

Useful options:

```bash
python -m digital_baby.main --ticks 20 --sleep 0.2
python -m digital_baby.main --world digital_baby/world/knowledge_pages --memory digital_baby/world/memory_store.json
```

## Inspect the Brain

```bash
python tools/inspect_brain.py --memory digital_baby/world/memory_store.json
```

Optional export:

```bash
python tools/inspect_brain.py --export-triplets artifacts/brain_triplets.tsv
```

Example metrics printed:

- Facts
- Concepts
- Relations
- Patterns
- Conflicts
- Top concepts by connectivity

## Tick Lifecycle (V3)

1. Discover pages.
2. Select by concept-goal or curiosity score.
3. If low novelty/stalled, generate a new topic procedurally.
4. Learn facts + update evidence.
5. Detect contradiction evidence and raise prediction error.
6. Discover patterns and concept hierarchy.
7. Compress redundant memory structures.
8. Persist memory and continue.
