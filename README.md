# Digital Baby: World-Model Curiosity Engine

`digital_baby` is a developmental learning prototype that now supports long-term
stability and proto-scientific behavior.

## Core Capabilities

- Curiosity-driven topic exploration with novelty and prediction-error signals.
- Procedural knowledge generation with **stable domain registry** (e.g. chemistry, astronomy).
- Evidence-weighted, deduplicated memory using unique fact index `(entity, relation, value)`.
- Belief resolution for conflicts (`best` + alternatives + confidence).
- Pattern discovery and concept hierarchy extraction.
- Hypothesis formation and prediction generation/testing.
- Memory compression and persistent world model storage.

## Project Structure

```text
digital_baby/
  brain/
    memory.py
    reasoning.py
    learner.py
    curiosity.py
    patterns.py
    concepts.py
    questions.py
    hypothesis.py
    predictor.py
  engine/
    event_loop.py
  world/
    generator.py
    knowledge_pages/*.json
  main.py

tools/
  inspect_brain.py
```

## Stability Improvements

- Duplicate facts are not re-added as separate entries; evidence increments instead.
- Belief resolution summarizes conflicts by strongest evidence.
- Topic selection uses safe lookup and regeneration fallback.
- Generator updates existing domains (topic registry) instead of drifting to random IDs.

## Run

```bash
python -m digital_baby.main
python -m digital_baby.main --ticks 30 --sleep 0.0
```

## Inspect Brain State

```bash
python tools/inspect_brain.py --memory digital_baby/world/memory_store.json
```
