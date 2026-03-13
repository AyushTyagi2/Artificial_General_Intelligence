# Digital Baby: Curiosity-Driven Learning Agent (V2)

This repository contains a minimal but extensible prototype of a curiosity-driven
"digital baby" agent.

In V2, the agent is now both curiosity-driven and goal-directed:

- It discovers all world pages automatically from `digital_baby/world/knowledge_pages/`.
- It detects unknown concepts and turns them into exploration goals.
- It generates questions from unknowns/conflicts to drive future exploration.
- It persists memory and maintains a simple knowledge graph with conflict lookup.
- It includes a terminal brain inspection tool.

## Project Structure

```text
digital_baby/
  brain/
    memory.py
    curiosity.py
    reasoning.py
    learner.py
    questions.py
  world/
    knowledge_pages/
      *.json
    memory_store.json   # generated at runtime
  engine/
    event_loop.py
  main.py

tools/
  inspect_brain.py
```

## V2 Features

- **Scalable world discovery**: no code changes needed when adding new JSON pages.
- **Persistent memory** of facts with confidence scores in `[0, 1]`.
- **Confidence decay** over time to model uncertainty drift.
- **Knowledge graph representation**: `entity -> relation -> value`.
- **Graph utilities**: relation retrieval, entity querying, conflict extraction.
- **Topic exploration penalty** to reduce repeated topic looping.
- **Goal-directed exploration** via unknown concept queue and topic matching.
- **Question generation** from unknown concepts and contradictions.
- **Conflict investigation behavior** that boosts concept exploration pressure.
- **Readable per-tick logs** including topic, reason, unknowns, and reward.

## Knowledge Page Format

Each page is a JSON object:

```json
{
  "topic": "animals",
  "facts": [
    "cat is mammal",
    "bird is animal"
  ]
}
```

Add as many pages as needed under `digital_baby/world/knowledge_pages/`.

## Run the Agent

From repository root:

```bash
python -m digital_baby.main
```

Useful options:

```bash
python -m digital_baby.main --ticks 20 --sleep 0.2
python -m digital_baby.main --world digital_baby/world/knowledge_pages --memory digital_baby/world/memory_store.json
```

## Inspect the Brain

After running the agent:

```bash
python tools/inspect_brain.py --memory digital_baby/world/memory_store.json
```

Optional graph export:

```bash
python tools/inspect_brain.py --export-triplets artifacts/brain_triplets.tsv
```

## Tick Lifecycle

1. Observe all discovered world pages.
2. Select topic:
   - first by concept goals (direct/fuzzy match),
   - otherwise by curiosity scoring.
3. Read facts and update memory.
4. Update unknown concepts/relations.
5. Investigate conflicts and generate questions.
6. Decay confidence and persist memory.
7. Log tick summary and sleep.

## Extending Further

- Improve parser beyond `<subject> <relation> <object>`.
- Add richer confidence updates (Bayesian or evidence-based).
- Add topic embeddings for stronger concept-to-topic matching.
- Add long-horizon planning over generated question queues.
