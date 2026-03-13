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
- Self-generated experiments for active hypothesis testing.
- Concept type system with typed hypotheses and relation constraints.
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
- Prediction filtering rejects type-invalid facts before storage.
- Belief resolution summarizes conflicts by strongest evidence.
- Topic selection uses safe lookup and regeneration fallback.
- Generator updates existing domains (topic registry) instead of drifting to random IDs.

## Active Learning Loop

observe -> hypothesize -> experiment -> update belief

## Run

```bash
python -m digital_baby.main
python -m digital_baby.main --ticks 30 --sleep 0.0
```

## Inspect Brain State

```bash
python tools/inspect_brain.py --memory digital_baby/world/memory_store.json
```


## Visualization Dashboard

A read-only dashboard is available under `digital_baby/dashboard/`:

- `log_parser.py` parses conflicts, prediction failures, and reward history from runtime logs.
- `graph_builder.py` builds an interactive NetworkX/PyVis graph.
- `server.py` serves Flask routes:
  - `/` dashboard page
  - `/graph` interactive knowledge graph

Run (with your log path):

```bash
DIGITAL_BABY_LOG=/path/to/agent.log python -m digital_baby.dashboard.server
```

Then open:

- `http://localhost:5055/`
- `http://localhost:5055/graph`


## Dashboard Architecture (Live AI Brain Monitor)

```text
Agent Log + Memory JSON
        |
        v
  dashboard/log_parser.py -----> dashboard/state_manager.py -----> API layer (Flask)
                                        |                           |                                        |                           | \__ /api/metrics
                                        |                           |____ /api/graph
                                        |                           |____ /api/state
                                        |                           |____ /ws/stream
                                        v
                             dashboard/graph_builder.py
                                        |
                                        v
                                 Frontend (index.html)
                              - vis-network graph
                              - ECharts metrics
                              - live event feed
```

The dashboard is read-only and does not alter the learning engine.
