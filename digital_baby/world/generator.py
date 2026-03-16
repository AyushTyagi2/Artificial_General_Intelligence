"""Procedural world generator — v3.

Changes from v2
---------------
domain_states   : expanded with new variables for chemistry (+4), technology (+3),
                  biology (+3), physics (+3), ecosystem (+1).  Total variables per
                  domain: chemistry=8, technology=8, biology=8, physics=8, ecosystem=4.

actions_by_domain : expanded with new actions for all domains.

absorb_ingested_knowledge : unchanged (entity pool expansion logic).
All other methods unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple
from urllib.parse import quote
from urllib.request import urlopen
import json
import random

from digital_baby.brain.knowledge_expander import KnowledgeExpander


@dataclass
class DomainTemplate:
    name: str
    entity_pools: Dict[str, Sequence[str]]
    schema: List[Tuple[str, str, str]]


class KnowledgeGenerator:
    """Generates open-world facts and simulates stateful domains — v3."""

    def __init__(self, world_path: str | Path, seed: int | None = None) -> None:
        self.world_path = Path(world_path)
        self.random     = random.Random(seed)
        self.generated_signatures: Set[str]       = set()
        self.topic_registry:       Dict[str, Dict] = {}
        self.expander   = KnowledgeExpander(self.world_path)

        self.relation_pool     = ["part_of", "located_in", "interacts_with", "depends_on"]
        self.discovery_queries = ["animal", "species", "planet", "ecosystem", "technology", "fungus", "robot"]
        self.mutation_prefixes = ["bio", "nano", "quantum", "alien", "eco", "neuro"]
        self.mutation_roots    = ["robot", "plant", "ecosystem", "species", "habitat", "sensor"]

        # ── v3: expanded domain states ────────────────────────────────────────
        self.domain_states: Dict[str, Dict[str, float]] = {

            # Ecosystem: wolves/deer/grass trophic chain + season modifier
            "ecosystem": {
                "wolves":        5.0,
                "deer":         20.0,
                "grass":       100.0,
                # NEW v3: season_factor modulates grass regrowth rate
                # Enables: change_season → season_factor → grass_growth (new causal edge)
                "season_factor": 1.0,
            },

            # Astronomy: asteroid belt + radiation causal chain (unchanged — not saturated)
            "astronomy": {
                "asteroids":          200.0,
                "solar_energy":      1000.0,
                "collision_risk":      10.0,
                "radiation_pressure":   2.0,
                "asteroid_drift":       3.0,
            },

            # Chemistry: was 4 variables → now 8
            # New causal chains:
            #   add_catalyst     → catalyst↑ → reaction_rate↑ (multiplicative)
            #   reaction_rate    → product_concentration↑
            #   product_concentration → reaction_rate↓ (Le Chatelier inhibition)
            #   adjust_pH        → pH↑/↓    → reaction_rate (peak at pH 7)
            #   temperature      → activation_energy↓ → reaction_rate↑ (Arrhenius Ea)
            "chemistry": {
                "temperature":            25.0,
                "reactants":               2.0,
                "reaction_rate":           0.0,
                "reaction_energy":         0.0,
                # NEW v3
                "catalyst":                0.0,   # add_catalyst action
                "product_concentration":   0.0,   # accumulates, inhibits rate
                "pH":                      7.0,   # adjust_pH action; optimal at 7.0
                "activation_energy":      50.0,   # decreases with temperature
            },

            # Technology: was 5 variables → now 8
            # New causal chains:
            #   robots → maintenance_load↑ (fleet maintenance burden)
            #   maintenance_load → robot_activity↓ (efficiency penalty)
            #   sensor_coverage → network_latency↑ (congestion)
            #   CONDITIONAL EDGE: sensor_coverage → data_quality
            #     sign depends on whether coverage >= sensor_threshold
            #     (this resolves the observed robot→data_quality bidirectional edge)
            "technology": {
                "robots":            4.0,
                "battery_charge":   80.0,
                "robot_activity":    0.0,
                "sensor_coverage":   0.0,
                "data_quality":      0.0,
                # NEW v3
                "maintenance_load":  2.0,   # grows with robot count
                "network_latency":   5.0,   # congestion — degrades data_quality
                "sensor_threshold":  5.0,   # threshold for data_quality sign flip
            },

            # Biology: was 5 variables → now 8
            # New causal chains:
            #   pathogens → toxin_level↑ → cells↓         (toxin pathway)
            #   immune_response → antibodies↑ → pathogens↓ (antibody clearance)
            #   energy + proteins → cell_cycle_rate↑ → cells↑ (growth pathway)
            "biology": {
                "cells":            100.0,
                "energy":            50.0,
                "proteins":          20.0,
                "pathogens":          0.0,
                "immune_response":    0.0,
                # NEW v3
                "toxin_level":        0.0,   # produced by pathogens
                "antibodies":         0.0,   # produced by immune response
                "cell_cycle_rate":    0.5,   # affected by energy and proteins
            },

            # Physics: was 5 variables → now 8
            # New causal chains:
            #   force - friction → net_force → acceleration (friction opposes motion)
            #   acceleration → velocity↑ → momentum↑ (integration)
            #   apply_impulse → momentum (direct impulse intervention)
            #   reduce_mass → mass↓ → acceleration↑ (for same force)
            "physics": {
                "force":          10.0,
                "mass":            5.0,
                "acceleration":    2.0,
                "heat":           25.0,
                "kinetic_energy": 50.0,
                # NEW v3
                "velocity":        0.0,   # integrates from acceleration
                "friction":        2.0,   # add_friction action
                "momentum":        0.0,   # mass × velocity
            },

            # Neuroscience domain — new in v4
            # Causal chains:
            #   stress_level → cortisol↑ → hippocampus_activity↓ → memory_formation↓
            #   dopamine_level → reward_seeking↑ → learning_rate↑
            #   sleep_quality → memory_consolidation↑
            #   neural_activity → synaptic_strength↑ (plasticity)
            "neuroscience": {
                "stress_level":         5.0,
                "cortisol":             3.0,
                "dopamine_level":       5.0,
                "serotonin_level":      5.0,
                "neural_activity":      6.0,
                "synaptic_strength":    4.0,
                "memory_consolidation": 3.0,
                "learning_rate":        2.0,
            },

            # Climate domain — new in v4
            # Causal chains:
            #   co2_level → temperature_anomaly↑ → glacier_melt↑ → sea_level↑
            #   deforestation → co2_level↑ and albedo↑
            #   precipitation → vegetation_cover↑ → co2_absorption↑
            "climate": {
                "co2_level":          415.0,
                "temperature_anomaly":  1.2,
                "glacier_melt":         3.0,
                "sea_level_rise":       0.3,
                "precipitation":       60.0,
                "vegetation_cover":    40.0,
                "albedo":               0.3,
                "ocean_heat":          80.0,
            },

            # Economics domain — new in v4
            # Causal chains:
            #   interest_rate → investment↓ → gdp_growth↓
            #   inflation → purchasing_power↓ → consumption↓
            #   productivity → wages↑ → consumption↑ → gdp_growth↑
            "economics": {
                "interest_rate":     3.0,
                "inflation":         2.5,
                "gdp_growth":        2.0,
                "unemployment":      5.0,
                "investment":       20.0,
                "consumption":      60.0,
                "productivity":      3.0,
                "debt_level":       80.0,
            },

            # Materials domain — new in v4
            # Causal chains:
            #   temperature → conductivity↑, strength↓
            #   stress → strain↑ → crack_growth↑ → material_failure↑
            #   heat_treatment → hardness↑ → yield_strength↑
            "materials": {
                "temperature":      25.0,
                "stress_level":      5.0,
                "strain":            0.1,
                "hardness":         60.0,
                "yield_strength":  250.0,
                "conductivity":      6.0,
                "crack_growth":      0.0,
                "porosity":          2.0,
            },
        }

        # ── v3: expanded actions ──────────────────────────────────────────────
        self.actions_by_domain: Dict[str, List[str]] = {
            "ecosystem": [
                "remove_predator", "add_predator",
                "introduce_species", "remove_species",
                "change_season",           # NEW: modulates season_factor
            ],
            "astronomy": [
                "introduce_species", "remove_species",
            ],
            "chemistry": [
                "increase_temperature", "add_chemical",
                "add_catalyst",            # NEW: raises catalyst
                "adjust_pH",               # NEW: changes pH ±1
                "remove_product",          # NEW: removes product_concentration
            ],
            "technology": [
                "add_robot", "remove_robot",
                "upgrade_sensor",          # NEW: raises sensor_threshold
                "increase_maintenance",    # NEW: raises maintenance_load directly
            ],
            "biology": [
                "add_pathogen", "boost_energy", "add_cells",
                "neutralise_toxin",        # NEW: reduces toxin_level
                "add_antibody",            # NEW: direct antibody injection
            ],
            "physics": [
                "increase_force", "add_heat",
                "add_friction",            # NEW: raises friction
                "apply_impulse",           # NEW: direct momentum boost
                "reduce_mass",             # NEW: lowers mass
            ],
            "neuroscience": [
                "induce_stress",           # raises stress_level → cortisol↑
                "reduce_stress",           # lowers stress_level
                "boost_dopamine",          # raises dopamine_level → learning_rate↑
                "improve_sleep",           # raises memory_consolidation
                "stimulate_neurons",       # raises neural_activity → synaptic_strength↑
            ],
            "climate": [
                "emit_co2",                # raises co2_level → temperature_anomaly↑
                "plant_forest",            # raises vegetation_cover → co2_absorption↑
                "melt_glacier",            # raises glacier_melt → sea_level↑
                "increase_albedo",         # raises albedo → reduces temperature_anomaly
                "warm_ocean",              # raises ocean_heat → precipitation↑
            ],
            "economics": [
                "raise_interest_rate",     # raises interest_rate → investment↓
                "lower_interest_rate",     # lowers interest_rate → investment↑
                "increase_spending",       # raises consumption → gdp_growth↑
                "boost_productivity",      # raises productivity → wages↑
                "add_debt",                # raises debt_level → constrains investment
            ],
            "materials": [
                "apply_stress",            # raises stress_level → strain↑
                "heat_treat",              # raises temperature → hardness changes
                "add_porosity",            # raises porosity → strength↓
                "quench",                  # rapid cooling → hardness↑
                "anneal",                  # slow cooling → hardness↓, ductility↑
            ],
        }

        # ── Domain templates (text fact generation — unchanged from v2) ───────
        self.domains = {
            "ecosystem": DomainTemplate(
                "ecosystem",
                {
                    "predator": ["wolf", "lion", "tiger", "lynx", "eagle"],
                    "prey":     ["deer", "rabbit", "zebra", "mouse", "antelope"],
                    "plant":    ["grass", "bush", "shrub", "fern", "tree"],
                    "soil":     ["soil", "wetland", "meadow"],
                },
                [("predator", "hunts", "prey"), ("prey", "eats", "plant"), ("plant", "grows_in", "soil")],
            ),
            "astronomy": DomainTemplate(
                "astronomy",
                {
                    "planet":    ["mercury", "venus", "earth", "mars", "kepler_22b"],
                    "star":      ["sun", "sirius", "vega", "proxima_centauri"],
                    "moon":      ["luna", "phobos", "deimos", "europa", "titan"],
                    "radiation": ["light", "radiation", "infrared", "ultraviolet"],
                },
                [("planet", "orbits", "star"), ("moon", "orbits", "planet"), ("star", "emits", "radiation")],
            ),
            "chemistry": DomainTemplate(
                "chemistry",
                {
                    "acid":     ["hydrochloric_acid", "sulfuric_acid", "acetic_acid"],
                    "base":     ["sodium_hydroxide", "ammonia", "potassium_hydroxide"],
                    "salt":     ["sodium_chloride", "potassium_sulfate", "ammonium_acetate"],
                    "molecule": ["water", "glucose", "ethanol", "methane"],
                    "atom":     ["carbon", "oxygen", "hydrogen", "nitrogen"],
                },
                [("acid", "reacts_with", "base"), ("reaction", "produces", "salt"), ("molecule", "contains", "atom")],
            ),
            "technology": DomainTemplate(
                "technology",
                {
                    "sensor":    ["camera", "lidar", "thermometer", "microphone"],
                    "signal":    ["image", "distance", "temperature", "audio"],
                    "processor": ["cpu", "gpu", "microcontroller", "asic"],
                    "algorithm": ["filter", "classifier", "planner", "optimizer"],
                    "battery":   ["lithium_pack", "supercapacitor", "fuel_cell"],
                    "robot":     ["drone", "rover", "arm_bot", "submarine_bot"],
                },
                [("sensor", "measures", "signal"), ("processor", "executes", "algorithm"), ("battery", "powers", "robot")],
            ),
            "neuroscience": DomainTemplate(
                "neuroscience",
                {
                    "neurotransmitter": ["dopamine", "serotonin", "glutamate", "gaba", "acetylcholine"],
                    "hormone":          ["cortisol", "adrenaline", "oxytocin", "melatonin"],
                    "brain_region":     ["hippocampus", "amygdala", "prefrontal_cortex", "cerebellum"],
                    "process":          ["memory_consolidation", "neural_plasticity", "neurogenesis"],
                },
                [("neurotransmitter", "activates", "brain_region"), ("hormone", "modulates", "process"), ("brain_region", "enables", "process")],
            ),
            "climate": DomainTemplate(
                "climate",
                {
                    "greenhouse_gas": ["co2", "methane", "nitrous_oxide", "water_vapor"],
                    "system":         ["atmosphere", "ocean", "cryosphere", "biosphere"],
                    "phenomenon":     ["el_nino", "monsoon", "arctic_amplification", "albedo_feedback"],
                    "biome":          ["rainforest", "tundra", "savanna", "coral_reef"],
                },
                [("greenhouse_gas", "warms", "atmosphere"), ("system", "interacts_with", "system"), ("biome", "absorbs", "greenhouse_gas")],
            ),
            "economics": DomainTemplate(
                "economics",
                {
                    "market":    ["stock_market", "bond_market", "commodity_market", "forex"],
                    "actor":     ["consumer", "firm", "government", "central_bank"],
                    "commodity": ["oil", "gold", "wheat", "copper"],
                    "currency":  ["dollar", "euro", "yen", "yuan"],
                },
                [("actor", "participates_in", "market"), ("commodity", "traded_in", "market"), ("currency", "denominates", "commodity")],
            ),
            "materials": DomainTemplate(
                "materials",
                {
                    "metal":   ["steel", "aluminum", "titanium", "copper", "iron"],
                    "alloy":   ["bronze", "brass", "stainless_steel", "inconel"],
                    "polymer": ["polyethylene", "nylon", "kevlar", "epoxy"],
                    "process": ["annealing", "quenching", "sintering", "welding"],
                },
                [("metal", "forms", "alloy"), ("process", "transforms", "metal"), ("alloy", "has_property", "strength")],
            ),
        }

    # ── State access ───────────────────────────────────────────────────────────

    def state_for_domain(self, domain: str) -> Dict[str, float]:
        return dict(self.domain_states.setdefault(domain, {}))

    def update_domain_state(self, domain: str, new_state: Dict[str, float]) -> None:
        self.domain_states[domain] = dict(new_state)

    def choose_action(self, domain: str, preferred_action: str | None = None) -> str:
        actions = self.actions_by_domain.get(domain, ["introduce_species"])
        if preferred_action and preferred_action in actions:
            return preferred_action
        return self.random.choice(actions)

    # ── Knowledge ingestion feedback loop (unchanged from v2) ─────────────────

    def absorb_ingested_knowledge(
        self,
        triplets: List[Tuple[str, str, str]],
    ) -> int:
        """Wire ingested graph knowledge into the world simulator.

        Maps concept subclass/instance relationships to domain entity pools
        and adds new state variables when a concept is found to be part_of
        a known domain.

        Returns the number of entity pool or state variable expansions made.
        """
        ROLE_SIGNALS = {
            "predator":  ("ecosystem", "predator"),
            "prey":      ("ecosystem", "prey"),
            "herbivore": ("ecosystem", "prey"),
            "carnivore": ("ecosystem", "predator"),
            "plant":     ("ecosystem", "plant"),
            "mammal":    ("ecosystem", "predator"),
            "organism":  ("ecosystem", "prey"),
            "element":   ("chemistry", "atom"),
            "compound":  ("chemistry", "molecule"),
            "molecule":  ("chemistry", "molecule"),
            "atom":      ("chemistry", "atom"),
            "acid":      ("chemistry", "acid"),
            "base":      ("chemistry", "base"),
            "star":      ("astronomy", "star"),
            "planet":    ("astronomy", "planet"),
            "moon":      ("astronomy", "moon"),
            "satellite": ("astronomy", "moon"),
            "sensor":    ("technology", "sensor"),
            "robot":     ("technology", "robot"),
            "processor": ("technology", "processor"),
            # Neuroscience
            "neurotransmitter": ("neuroscience", "neurotransmitter"),
            "hormone":          ("neuroscience", "hormone"),
            "neuron":           ("neuroscience", "neuron"),
            # Climate
            "greenhouse_gas":   ("climate", "greenhouse_gas"),
            "glacier":          ("climate", "glacier"),
            "biome":            ("climate", "biome"),
            # Economics
            "commodity":        ("economics", "commodity"),
            "currency":         ("economics", "currency"),
            "market":           ("economics", "market"),
            # Materials
            "metal":            ("materials", "metal"),
            "alloy":            ("materials", "alloy"),
            "polymer":          ("materials", "polymer"),
        }

        expansions = 0
        for subj, rel, obj in triplets:
            if rel in ("subclass_of", "instance_of", "is_instance_of", "is"):
                target_role = ROLE_SIGNALS.get(obj.lower())
                if target_role:
                    domain_name, pool_name = target_role
                    template = self.domains.get(domain_name)
                    if template and pool_name in template.entity_pools:
                        pool       = list(template.entity_pools[pool_name])
                        subj_clean = subj.lower().replace(" ", "_")
                        if subj_clean not in pool:
                            pool.append(subj_clean)
                            template.entity_pools[pool_name] = pool  # type: ignore[assignment]
                            expansions += 1
                            import logging as _log
                            _log.getLogger(__name__).info(
                                "[world] entity_pool_expanded domain=%s pool=%s entity=%s",
                                domain_name, pool_name, subj_clean,
                            )

            if rel == "part_of" and obj in self.domain_states:
                state      = self.domain_states[obj]
                subj_clean = subj.lower().replace(" ", "_")
                if subj_clean not in state:
                    state[subj_clean] = float(self.random.randint(1, 20))
                    expansions += 1

        return expansions

    # ── Topic generation (unchanged from v2) ──────────────────────────────────

    def _materialize_facts(self, template: DomainTemplate) -> List[str]:
        bindings = {k: self.random.choice(list(v)) for k, v in template.entity_pools.items()}
        return [f"{bindings.get(s, s)} {r} {bindings.get(o, o)}" for s, r, o in template.schema]

    def _signature(self, domain: str, facts: List[str]) -> str:
        return domain + "::" + "|".join(sorted(facts))

    def _safe_json_get(self, url: str) -> dict:
        with urlopen(url, timeout=6) as response:  # nosec B310
            return json.loads(response.read().decode("utf-8"))

    def _wikidata_random_entity(self) -> str | None:
        query = self.random.choice(self.discovery_queries)
        url = (
            "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json"
            f"&language=en&type=item&search={quote(query)}&limit=20"
        )
        try:
            payload = self._safe_json_get(url)
            items   = payload.get("search", [])
            if not items:
                return None
            chosen = self.random.choice(items)
            label  = chosen.get("label") or ""
            return label.strip().lower().replace(" ", "_") or None
        except Exception:
            return None

    def _mutated_concept(self) -> str:
        return f"{self.random.choice(self.mutation_prefixes)}_{self.random.choice(self.mutation_roots)}"

    def _random_relation_fact(self, concepts: Sequence[str]) -> str | None:
        if len(concepts) < 2:
            return None
        subject  = self.random.choice(list(concepts))
        obj      = self.random.choice([c for c in concepts if c != subject] or list(concepts))
        relation = self.random.choice(self.relation_pool)
        return f"{subject} {relation} {obj}"

    def state_snapshot_facts(self, domain: str) -> List[str]:
        state = self.state_for_domain(domain)
        return [
            f"{key} is {int(value) if isinstance(value, (int, float)) else value}"
            for key, value in sorted(state.items())
        ]

    def inject_open_world_novelty(self, max_entities: int = 3) -> Tuple[List[str], List[str]]:
        new_facts:    List[str] = []
        new_concepts: List[str] = []
        entity_count  = self.random.randint(1, max(1, max_entities))

        for _ in range(entity_count):
            concept = self._wikidata_random_entity() if self.random.random() < 0.65 else None
            if not concept:
                concept = self._mutated_concept()
            if self.expander.is_cached_concept(concept):
                continue

            new_concepts.append(concept)
            edges = self.expander.expand_concept_graph(concept, depth=2)
            if edges:
                for source, relation, target in edges:
                    new_facts.append(f"{source} {relation} {target}")
            else:
                new_facts.append(f"{concept} is unknown")

        relation_fact = self._random_relation_fact(new_concepts)
        if relation_fact:
            new_facts.append(relation_fact)
        return sorted(set(new_facts)), sorted(set(new_concepts))

    def generate_topic(self, persist: bool = True, domain: str | None = None) -> Dict:
        if domain is None:
            domain = self.random.choice(sorted(self.domains.keys()))
        template = self.domains.get(domain) or self.domains[self.random.choice(sorted(self.domains.keys()))]

        for _ in range(25):
            facts     = self._materialize_facts(template)
            facts.extend(self.state_snapshot_facts(template.name))
            if self.random.random() < 0.15:
                novelty_facts, _ = self.inject_open_world_novelty(max_entities=2)
                facts.extend(novelty_facts)
            signature = self._signature(template.name, facts)
            if signature not in self.generated_signatures:
                self.generated_signatures.add(signature)
                break

        info = self.topic_registry.setdefault(template.name, {"generations": 0, "last_signature": ""})
        info["generations"]    += 1
        info["last_signature"]  = signature

        page = {
            "topic":      template.name,
            "facts":      sorted(set(facts)),
            "generated":  True,
            "domain":     template.name,
            "generation": info["generations"],
        }

        if persist:
            self.world_path.mkdir(parents=True, exist_ok=True)
            import tempfile as _tf, os as _os, shutil as _sh
            _dst = self.world_path / f"generated_{template.name}.json"
            _fd, _tmp = _tf.mkstemp(dir=str(self.world_path), suffix=".tmp")
            try:
                with _os.fdopen(_fd, "w", encoding="utf-8") as _fh:
                    json.dump(page, _fh, indent=2)
                try:
                    _os.replace(_tmp, str(_dst))
                except PermissionError:
                    _sh.copy2(_tmp, str(_dst))
                    _os.unlink(_tmp)
            except Exception:
                try:
                    _os.unlink(_tmp)
                except OSError:
                    pass
                raise

        return page