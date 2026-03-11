# Digital Baby: Curiosity-Driven Learning Agent

This repository contains a minimal prototype of a curiosity-driven "digital baby" agent.
It explores a simple world of knowledge pages, learns facts with confidence scores,
persists memory to disk, detects contradictions, and uses curiosity to choose what to
explore next.

## Project Structure

```text
digital_baby/
  brain/
    memory.py
    curiosity.py
    reasoning.py
    learner.py
  world/
    knowledge_pages/
      animals.json
      biology.json
      physics.json
    memory_store.json   # generated at runtime
  engine/
    event_loop.py
  main.py
```

## Features

- **Persistent memory** of facts with confidence scores in `[0, 1]`.
- **Confidence decay** over time so unattended facts become weaker.
- **Simple knowledge graph** of `(subject, relation, object)` triplets.
- **Curiosity model** that prioritizes novelty and weak-confidence topics.
- **Reasoning checks** for contradictions on shared `(entity, relation)` pairs.
- **Continuous event loop** that can run forever or for a fixed number of ticks.

## Run

From repository root:

```bash
python -m digital_baby.main
```

Useful options:

```bash
python -m digital_baby.main --ticks 10 --sleep 0.2
python -m digital_baby.main --world digital_baby/world/knowledge_pages --memory digital_baby/world/memory_store.json
```

## Life Cycle per Tick

1. Observe available knowledge pages.
2. Score topics with curiosity.
3. Select a topic and read facts.
4. Learn facts and relation triplets.
5. Detect contradictions.
6. Decay confidence and persist memory.
7. Log activity and sleep briefly.

## Extending the Prototype

- Add more JSON pages in `digital_baby/world/knowledge_pages/`.
- Enrich fact parser in `reasoning.py` to handle richer grammar.
- Add episodic memory and temporal reasoning.
- Replace rule-based curiosity with learned intrinsic motivation.
