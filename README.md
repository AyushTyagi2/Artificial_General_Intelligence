# 🧠 Digital Brain — A Curiosity-Driven Learning Agent

> *A self-organizing, hypothesis-generating AI agent that observes, experiments, and builds a persistent world model — like a scientific mind starting from scratch.*

---

## Table of Contents

1. [What is this project?](#what-is-this-project)
2. [Core Philosophy](#core-philosophy)
3. [Architecture Overview](#architecture-overview)
4. [Module Breakdown](#module-breakdown)
   - [Brain](#brain-modules)
   - [Engine](#engine)
   - [World](#world)
   - [Tools Bridge](#tools-bridge)
   - [Dashboard](#dashboard)
5. [The Learning Loop](#the-learning-loop)
6. [Knowledge Domains](#knowledge-domains)
7. [Key Algorithms](#key-algorithms)
8. [Getting Started](#getting-started)
9. [Configuration](#configuration)
10. [Inspecting the Brain](#inspecting-the-brain)
11. [Dashboard](#running-the-dashboard)
12. [Data Persistence](#data-persistence)
13. [Project Evolution (v1 → v4)](#project-evolution)
14. [Will Intelligence Emerge?](#will-intelligence-emerge)

---

## What is this project?

**Digital Brain** (internally called `digital_baby`) is a fully autonomous Python agent that teaches itself about the world through a continuous observe-hypothesize-experiment-update loop. It is not a wrapper around a language model. It is a hand-engineered cognitive architecture that:

- Reads structured **knowledge pages** (facts about physics, biology, chemistry, etc.)
- Builds and maintains a **probabilistic knowledge graph**
- Generates **hypotheses** about causal relationships
- Designs and runs **experiments** in a simulated world
- Discovers **scientific laws** by curve-fitting observed variable changes
- Consolidates and forgets memories the way biological systems do
- Monitors its own uncertainty and directs attention accordingly

The agent runs tick by tick, indefinitely, accumulating a richer and richer world model over time.

---

## Core Philosophy

The project is built around one central idea: **intelligence is what happens when you close the loop between observation, surprise, and action**.

Every tick, the agent:

1. Picks the topic it is most curious about (using Upper Confidence Bound selection)
2. Learns from that topic's facts
3. Identifies what it does not yet understand
4. Forms hypotheses about causal relationships
5. Experiments on a simulated world to test those hypotheses
6. Updates its beliefs based on what actually happened
7. Discovers mathematical laws that explain the patterns it has seen

This is modelled loosely on how a scientist — or a developing child — learns: not by being told answers, but by noticing gaps, forming theories, and testing them.

---

## Architecture Overview

```
Digital Brain
│
├── digital_baby/
│   ├── brain/             ← All cognitive modules (40+ files)
│   │   ├── perception/    ← Real-time screen observation
│   │   └── ...
│   ├── engine/
│   │   └── event_loop.py  ← The master tick loop (v4)
│   ├── world/
│   │   ├── generator.py   ← Simulated physical worlds
│   │   ├── knowledge_pages/  ← Seed facts per domain
│   │   ├── memory_store.json ← Persisted beliefs
│   │   └── knowledge_graph.json
│   ├── tools_bridge/      ← Wikipedia, chatbot, notes integrations
│   ├── dashboard/         ← Flask live-monitoring server
│   └── main.py
│
└── tools/
    ├── inspect_brain.py   ← CLI brain inspector
    └── router.py
```

---

## Module Breakdown

### Brain Modules

The `digital_baby/brain/` directory is the heart of the project. Each file implements a distinct cognitive faculty:

| Module | Responsibility |
|---|---|
| `memory.py` | Persistent fact store with evidence weighting, deduplication, causal rule records, and stale-decay |
| `curiosity.py` | UCB1-based topic scoring; registers novelty, prediction error, hypothesis uncertainty; drives exploration-exploitation balance |
| `learner.py` | Extracts structured facts from raw text pages; infers general rules; discovers causal candidates from state transitions |
| `reasoner.py` | Parses triplets from natural language; classifies relation types |
| `hypothesis.py` | Generates, scores, and manages the lifecycle of hypotheses (CANDIDATE → ACTIVE → CONFIRMED / REFUTED / SUSPENDED / ABANDONED) |
| `hypothesis_validator.py` | Validates hypotheses using internal simulation evidence first, then Wikipedia/Wikidata as fallback |
| `experiment_planner.py` | Ranks hypotheses by a multi-factor score (information gain, curiosity, uncertainty, age, diversity, accuracy); plans targeted experiments |
| `experimenter.py` | Runs single and paired (control + treatment) world experiments; extracts clean causal deltas |
| `predictor.py` | Generates predictions before experiments and evaluates prediction accuracy afterwards |
| `patterns.py` | Discovers recurring triplet patterns in the knowledge base (minimum support threshold) |
| `concepts.py` | Maintains a concept hierarchy and a type system used to constrain hypothesis generation |
| `knowledge_graph.py` | Probabilistic edge store with five confidence tiers: speculative → tentative → probable → confirmed → law_grade |
| `knowledge_ingestion.py` | Fetches real Wikipedia pages for unknown concepts and injects them as graph edges |
| `knowledge_expander.py` | Expands unexplored concepts by following graph edges (BFS depth-limited) |
| `causal_chain_reasoner.py` | Finds multi-hop causal chains (A→B→C) in the knowledge graph and generates corresponding hypotheses |
| `intervention_causal_discovery.py` | Detects causal relationships from before/after world states; supports second-order chain discovery and stale record decay |
| `law_discovery.py` | Attempts to fit mathematical laws (linear, power, exponential, logarithmic) to observed (cause_delta, effect_delta) pairs using curve fitting |
| `law_predictor.py` | Uses discovered laws to generate numeric predictions before each world step |
| `prediction_evaluator.py` | Evaluates law-based predictions against actual state changes |
| `law_novelty_gate.py` | Filters out law candidates that are too similar to already-known laws |
| `episodic_memory.py` | Generates natural-language episodic observations from raw world state transitions |
| `epistemic_state.py` | Tracks per-edge uncertainty (entropy) and emits an epistemic reward signal each tick |
| `memory_consolidation.py` | Promotes episodic → semantic memories; forgets low-value facts; prunes the knowledge graph |
| `curiosity.py` | (also) Computes the multi-component epistemic reward combining entropy reduction, new edges, validations, novel laws, and mediator discoveries |
| `cross_domain_theory.py` | Detects when the same causal pattern appears across multiple domains and generates analogical hypotheses |
| `theory_abstraction.py` | Extracts abstract mathematical templates from discovered laws and uses them to guide fitting of unconfirmed causal pairs |
| `hypothesis_chain_generator.py` | Generates chained hypotheses by composing known causal rules (A→B + B→C → A→C hypothesis) |
| `contradiction_hypothesis.py` | Detects when two causal rules make contradictory predictions and generates hypotheses to resolve them |
| `concept_abstraction.py` | Clusters concepts by shared relational patterns and promotes super-concepts into the type system |
| `dynamic_variable_generator.py` | Synthesises new domain variables when epistemic information gain drops (prevents stagnation) |
| `synthetic_concept_resolver.py` | Classifies unknown concepts as real (fetch from Wikipedia) vs. synthetic (resolve via graph rules) |
| `mechanism_matcher.py` | Matches inferred causal chains to named scientific mechanism templates (e.g. feedback loops) |
| `topological_role_assigner.py` | Computes graph-theoretic roles (hub, bridge, leaf) for each node using betweenness centrality |
| `mediator_blocking_planner.py` | Plans two-tick paired experiments to confirm or refute whether B mediates A→C |
| `multi_step_planner.py` | Plans multi-step intervention sequences targeting deep causal chains |
| `research_agent.py` | A background agent that periodically issues Wikipedia queries for concepts flagged as high-priority |
| `questions.py` | Generates explicit questions from unknown concepts to drive curiosity registration |
| `perception/screen_tracker.py` | Background thread that captures screen text and injects observed concept pairs into the knowledge graph |

### Engine

`engine/event_loop.py` — The `BabyEventLoop` class. This is the master orchestrator. Every tick it:

- Selects a topic via UCB1
- Learns from the page
- Selects an action using a 5-priority mandatory cascade (never returns `action=none`)
- Runs the world step (single or paired experiment)
- Runs all cognitive post-processing in order
- Saves state

### World

`world/generator.py` — Implements 10 fully simulated domains, each with its own state variables and actions:

| Domain | Variables (sample) | Actions (sample) |
|---|---|---|
| ecosystem | wolf, deer, grass, season_factor | add_predator, change_season |
| physics | force, mass, acceleration, friction, momentum | increase_force, add_heat, apply_impulse |
| chemistry | temperature, reactants, catalyst, pH | increase_temperature, add_catalyst, adjust_pH |
| biology | cells, energy, pathogens, immune_response | add_pathogen, boost_energy, add_antibody |
| technology | robot, battery_charge, sensor_coverage | add_robot, upgrade_sensor |
| astronomy | asteroid, solar_energy, collision_risk | introduce_species (asteroid), remove_species |
| neuroscience | stress_level, dopamine_level, neural_activity | induce_stress, boost_dopamine, stimulate_neurons |
| climate | co2_level, temperature_anomaly, glacier_melt | emit_co2, plant_forest, melt_glacier |
| economics | interest_rate, inflation, gdp_growth | raise_interest_rate, boost_productivity |
| materials | strain, hardness, yield_strength, conductivity | apply_stress, heat_treat, quench |

### Tools Bridge

`tools_bridge/` provides optional integrations:

- `wikipedia.py` — fetches real-world concept descriptions
- `concept_chat.py` — falls back to a chatbot for concepts Wikipedia missed
- `knowledge_seeder.py` — automatically seeds Wikipedia pages for unknown concepts
- `memory_notes.py` — persists discovered hypotheses and causal rules to a notes store
- `working_memory.py` — records a per-tick log of the agent's cognitive state

### Dashboard

`dashboard/` is a read-only Flask server that visualises the live agent:

- `/` — Metrics dashboard (prediction error, reward, hypotheses, facts)
- `/graph` — Interactive knowledge graph (vis-network + ECharts)
- `/api/metrics`, `/api/graph`, `/api/state` — JSON APIs
- `/ws/stream` — WebSocket live event feed

Run with:
```bash
DIGITAL_BABY_LOG=/path/to/agent.log python -m digital_baby.dashboard.server_new
# Open http://localhost:5055/
```

---

## The Learning Loop

Each tick executes this sequence:

```
1.  Load knowledge pages from disk (with mtime-based cache)
2.  Inject novelty (open-world random concepts every N ticks)
3.  Feed epistemic entropy → UCB1 curiosity model
4.  Select topic (UCB1 + diversity enforcement)
5.  Learn from selected page (extract facts, triplets, unknowns)
6.  Resolve unknown concepts (real vs. synthetic)
7.  Suggest concepts to Wikipedia ingestion pipeline
8.  Drain screen perception buffer → knowledge graph
9.  Infer triplets, discover patterns, update causal rules
10. Generate / update hypotheses (from patterns + causal rules + chains + contradictions + cross-domain analogies)
11. Manage hypothesis lifecycle (promote/suspend/abandon/mutate)
12. Select action (5-priority cascade, never returns none):
      P0.5  Mediator-blocking experiment
      P0    Structured HypothesisExperimentPlanner action
      P1    Top-ranked hypothesis targeted intervention
      P2    Most uncertain epistemic edge intervention
      P3    UCB1 domain exploration (random action)
      P4    Ingest pending concept
      P5    Synthesise new variable
13. Run world step (single OR paired control+treatment)
14. Update causal discovery records + knowledge graph
15. Evaluate law-based predictions
16. Run second-order causal chain discovery (every 30 ticks)
17. Decay stale causal records (every 50 ticks)
18. Discover scientific laws (every 25 ticks, curve fitting)
19. Abstract theory templates from discovered laws
20. Compute epistemic reward signal
21. Consolidate memory (promote episodic → semantic, forget, prune)
22. Save all state to disk
23. Log cognitive summary
```

---

## Knowledge Domains

The agent starts with seed facts in 10+ domains from `world/knowledge_pages/*.json`. It actively expands these by:

- Fetching Wikipedia pages for unknown concepts
- Generating new topics procedurally via `KnowledgeGenerator`
- Injecting open-world novelty (randomly sampled entity-relation-entity triples)
- Absorbing ingested triplets back into the world simulator

---

## Key Algorithms

### UCB1 Topic Selection

```
UCB_score(domain) = mean_entropy(domain) + 1.4 × √(log(total_ticks + 1) / (visits + 1))
```

This guarantees no domain is permanently starved of attention.

### Five-Priority Action Cascade

The agent always takes *some* meaningful action. In decreasing priority:

1. **Mediator-blocking** — confirm a B mediates A→C by fixing B and re-running
2. **Hypothesis-driven** — use ExperimentPlanner to target the best untested hypothesis
3. **UCB1 intervention** — choose domain by UCB1, random action within it
4. **Concept ingestion** — fetch a pending unknown concept from Wikipedia
5. **Variable synthesis** — create a new derivative variable when stuck

### Law Discovery

For every confirmed causal pair (A, B), the agent accumulates (ΔA, ΔB) observations and attempts to fit: linear, power-law, exponential, and logarithmic models using `scipy.optimize.curve_fit`. The best-fitting model with R² > threshold becomes a registered **scientific law** and is used for predictive modelling going forward.

### Hypothesis Lifecycle

```
CANDIDATE → ACTIVE → CONFIRMED  (confidence ≥ 0.70, evidence ≥ 5)
                   → REFUTED    (confidence ≤ 0.25, evidence ≥ 5)
                   → SUSPENDED  (ambiguous after 10+ tests)
                   → ABANDONED  (ambiguous after 20+ tests → mutate into variants)
```

### Epistemic Reward

Each tick produces a scalar reward:

```
reward = 0.35 × entropy_reduced
       + 0.25 × new_edges
       + 0.15 × validations_passed
       + 0.15 × novel_law_discovered
       + 0.10 × mediator_confirmed
```

This drives the curiosity model to value genuine knowledge gain over busy-work.

---

## Getting Started

### Requirements

```bash
python >= 3.9
pip install scipy networkx flask pyvis wikipedia-api requests
```

### Run the agent

```bash
cd Digital_Brain
python -m digital_baby.main
```

With options:

```bash
# Run for 100 ticks, no sleep (fast test)
python -m digital_baby.main --ticks 100 --sleep 0.0

# Custom world and memory paths
python -m digital_baby.main \
  --world digital_baby/world/knowledge_pages \
  --memory digital_baby/world/memory_store.json \
  --sleep 0.5
```

---

## Configuration

All tunable constants are at the top of their respective modules:

| Constant | Location | Default | Effect |
|---|---|---|---|
| `UCB1_C` | `curiosity.py` | 1.4 | Exploration aggressiveness in topic selection |
| `SECOND_ORDER_INTERVAL` | `event_loop.py` | 30 | Ticks between second-order causal chain scans |
| `DECAY_INTERVAL` | `event_loop.py` | 50 | Ticks between stale causal record decay |
| `LIFECYCLE_INTERVAL` | `event_loop.py` | 20 | Ticks between hypothesis lifecycle updates |
| `ENTROPY_FEED_INTERVAL` | `event_loop.py` | 5 | Ticks between feeding entropy to UCB1 |
| `RESEARCH_INTERVAL` | `research_agent.py` | (see file) | Ticks between background Wikipedia queries |
| `novelty_interval` | `BabyEventLoop.__init__` | 20 | Ticks between open-world novelty injections |

---

## Inspecting the Brain

```bash
python tools/inspect_brain.py --memory digital_baby/world/memory_store.json
```

This prints a structured summary of:
- Total facts, causal rules, hypotheses
- Highest-confidence beliefs
- Most-observed causal relationships
- Recent experiments

You can also browse `digital_baby/world/memory_store.json` and `knowledge_graph.json` directly — both are human-readable JSON.

The `agent.log` file at the project root records every tick with:
```
[tick=N] topic=X domain=Y action=Z pred_err=0.012 new_facts=3 hypotheses=287 memory=4201 ...
```

---

## Running the Dashboard

```bash
DIGITAL_BABY_LOG=$(pwd)/agent.log python -m digital_baby.dashboard.server_new
```

Then open:
- `http://localhost:5055/` — Live metrics (reward, prediction error, facts, hypotheses)
- `http://localhost:5055/graph` — Interactive knowledge graph

The dashboard is **read-only** and does not interact with the running agent.

---

## Data Persistence

The agent automatically saves after every tick:

| File | Contents |
|---|---|
| `digital_baby/world/memory_store.json` | All facts, causal rules, hypotheses, experiments, predictions |
| `digital_baby/world/knowledge_graph.json` | Full probabilistic knowledge graph (nodes + edges with confidence) |
| `digital_baby/world/law_discovery_data.json` | All discovered mathematical laws |
| `digital_baby/world/ingestion_state.json` | Wikipedia ingestion queue and history |
| `digital_baby/world/perception_log.jsonl` | Per-tick screen perception observations |
| `agent.log` | Full structured log of every tick |

If `memory_store.json` becomes corrupted, it is automatically backed up with a timestamp suffix and a fresh store is started.

---

## Project Evolution

The codebase is currently on **v4**, representing significant advances over the initial prototype:

| Version | Key addition |
|---|---|
| v1 | Basic observe-learn loop, simple memory |
| v2 | Hypothesis generation, pattern discovery, curiosity model |
| v3 | Active experiments, causal discovery, knowledge graph, law fitting |
| v4 | UCB1 exploration, mandatory action cascade (no more `action=none`), paired experiments, second-order causal chains, hypothesis lifecycle management, simulation-first validation, epistemic reward, 10 simulation domains |

---

## Will Intelligence Emerge?

This is the most important question to address honestly.

**The short answer: not in the way you probably hope — but something genuinely interesting is already happening.**

Here is a careful breakdown:

### What the agent already does that is non-trivial

- It **discovers mathematical laws** from scratch by observing simulated variable changes. When it learns that increasing `co2_level` consistently produces a near-linear increase in `temperature_anomaly`, it fits a curve and registers a law. That is genuine empirical discovery, not retrieval.
- It **forms and revises causal beliefs** over time. A hypothesis starts at 50% confidence and is updated by real experimental outcomes. Contradictory evidence pushes it toward refutation.
- It **reasons across domains** by noticing that the same causal pattern (e.g. a stabilising feedback loop) appears in both ecosystems and economics. This cross-domain analogy generation is rudimentary but structurally real.
- It **knows what it doesn't know**. The epistemic state tracker maintains per-edge uncertainty and drives the agent toward unexplored causal territory.
- It **never stops learning**. There is no training phase and no fixed model. The world model is built entirely from experience and grows monotonically.

### What it lacks for genuine intelligence

- **No grounding in real perception.** The simulated world is a toy: a handful of numeric variables with hand-coded dynamics. The agent cannot see, hear, or touch anything real. (The `ScreenTracker` is a partial exception, but it is shallow.)
- **No language understanding.** Facts are parsed from simple templates, not from natural language comprehension.
- **No goals of its own.** Curiosity is operationalised as a reward signal, but there is no genuine intentionality — no sense in which the agent *wants* anything.
- **No compositional reasoning.** The agent cannot construct novel multi-step logical arguments. Hypothesis chaining is a graph traversal, not deductive inference.
- **No generalisation beyond its domains.** A law discovered in the chemistry simulator does not transfer to the biology simulator unless explicitly templated.
- **No self-awareness.** The agent logs its own state, but it cannot reflect on or reason about its own reasoning.

### The honest verdict

This project is a serious and sophisticated **cognitive architecture experiment**, not a path to general intelligence. What it achieves — persistent, self-directed causal learning with law discovery, hypothesis lifecycle management, and epistemic reward — is well beyond a simple rule-based system or a statistical lookup table. It is closer to a proto-scientific reasoner than to a chatbot.

Whether anything deserving the name *intelligence* emerges depends on your definition. If intelligence requires subjective experience, genuine understanding, or open-ended problem solving in the real world, then no — this architecture will not produce it at any scale within its current design.

But if intelligence includes the capacity to form and revise accurate models of causal structure in the world through directed experiment — then this project is already doing something that qualifies, within the narrow worlds it inhabits.

**The most productive path forward** would be to ground this architecture in real-world data streams, replace the toy simulators with genuine sensor input, and connect the hypothesis-testing pipeline to actions with real-world consequences. That is a much harder problem — but this codebase provides a surprisingly solid cognitive backbone to build on.

> *"The agent doesn't know it's learning. But it is."*
