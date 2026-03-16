"""Main life-cycle loop for the digital baby agent — v4.

What changed from v3
--------------------
1. NO MORE action=none
   _select_action_v4() implements a 5-priority mandatory cascade using
   CuriosityModel.select_action().  The fallback chain is:
     P1 test top hypothesis (targeted action via ExperimentPlanner.plan_experiment)
     P2 intervene on most uncertain epistemic edge
     P3 explore highest-UCB1 domain
     P4 ingest pending concept
     P5 synthesise new variable (always available)

2. PAIRED EXPERIMENTS
   When ExperimentPlanner recommends use_paired=True (after N failed tests),
   _run_paired_world_step() runs the world twice from the same base state and
   extracts a clean causal delta free of baseline drift.

3. SECOND-ORDER CAUSAL DISCOVERY
   Every SECOND_ORDER_INTERVAL ticks causal_discovery.discover_second_order_effects()
   finds A→B→C chains and seeds A→C records.

4. HYPOTHESIS LIFECYCLE MANAGEMENT
   apply_lifecycle_updates() is called every LIFECYCLE_INTERVAL ticks.
   SUSPENDED/ABANDONED hypotheses are filtered from the test queue.
   Abandoned hypotheses are mutated into variants via mutate_abandoned_hypothesis().

5. SIMULATION-FIRST VALIDATION
   HypothesisValidator now receives the causal_records dict so it can
   validate from internal simulation evidence before calling Wikipedia/Wikidata.

6. DECAY + AXIOM SEEDING
   causal_discovery.decay_stale_records() is called every DECAY_INTERVAL ticks.
   Axioms are seeded on first analyze() call automatically.

7. UCB1 DOMAIN SELECTION
   _select_topic feeds domain entropy from EpistemicStateTracker back into
   CuriosityModel.register_domain_entropy() every tick.

8. EXPERIMENT OUTCOME RECORDING
   experiment_planner.record_outcome() is called after every world step so the
   accuracy-boost scoring component accumulates real prediction data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
import math
import time


def _safe_json_load(path, default=None):
    import json, shutil, time, logging
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return default
    try:
        raw = p.read_text(encoding="utf-8").strip().lstrip("\x00")
    except OSError:
        return default
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        backup = p.with_suffix(f".corrupted.{int(time.time())}.json")
        try:
            shutil.move(str(p), str(backup))
        except OSError:
            pass
        logging.getLogger(__name__).warning(
            "Corrupt JSON file %s — backed up to %s, using default.", p, backup
        )
        return default


from digital_baby.brain.concepts import ConceptHierarchy, ConceptTypeSystem
from digital_baby.brain.concept_abstraction import ConceptAbstractionEngine
from digital_baby.brain.causal_chain_reasoner import CausalChainReasoner
from digital_baby.brain.curiosity import CuriosityModel, ActionSelection
from digital_baby.brain.experimenter import Experimenter
from digital_baby.brain.experiment_planner import ExperimentPlanner, ExperimentPlan
from digital_baby.brain.hypothesis import Hypothesis, HypothesisEngine, mutate_abandoned_hypothesis
from digital_baby.brain.hypothesis_validator import HypothesisValidator
from digital_baby.brain.knowledge_expander import KnowledgeExpander
from digital_baby.brain.knowledge_graph import KnowledgeGraph
from digital_baby.brain.knowledge_ingestion import KnowledgeIngestionPipeline
from digital_baby.brain.intervention_causal_discovery import InterventionCausalDiscovery
from digital_baby.brain.episodic_memory import record_episodic_observation
from digital_baby.brain.law_discovery import LawDiscovery
from digital_baby.brain.learner import Learner
from digital_baby.brain.memory import ExperimentRecord, HypothesisRecord, Memory, PatternRecord, PredictionRecord
from digital_baby.brain.memory_consolidation import MemoryConsolidator
from digital_baby.brain.patterns import PatternDiscoverer
from digital_baby.brain.predictor import Predictor
from digital_baby.brain.questions import QuestionGenerator
from digital_baby.brain.reasoning import Reasoner
from digital_baby.world.generator import KnowledgeGenerator

from digital_baby.brain.epistemic_state import EpistemicStateTracker
from digital_baby.brain.law_novelty_gate import LawNoveltyGate
from digital_baby.brain.cross_domain_theory import CrossDomainTheoryEngine
from digital_baby.brain.multi_step_planner import MultiStepPlanner
from digital_baby.brain.hypothesis_chain_generator import HypothesisChainGenerator
from digital_baby.brain.synthetic_concept_resolver import SyntheticConceptResolver
from digital_baby.brain.dynamic_variable_generator import DynamicVariableGenerator
from digital_baby.brain.contradiction_hypothesis import ContradictionHypothesisEngine
from digital_baby.brain.theory_abstraction import TheoryAbstraction
from digital_baby.brain.research_agent import ResearchAgent, RESEARCH_INTERVAL
from digital_baby.brain.perception import ScreenTracker, WikiReader
from digital_baby.brain.mechanism_matcher import MechanismMatcher
from digital_baby.brain.topological_role_assigner import TopologicalRoleAssigner
from digital_baby.brain.mediator_blocking_planner import MediatorBlockingPlanner
from digital_baby.brain.hypothesis_experiment_planner import HypothesisExperimentPlanner
from digital_baby.brain.law_predictor import LawPredictor
from digital_baby.brain.prediction_evaluator import PredictionEvaluator
# ── Dashboard registry (graceful no-op if dashboard not running) ────────────
try:
    from digital_baby.dashboard.server_new import register_screen_tracker as _reg_screen
    _DASHBOARD_AVAILABLE = True
except ImportError:
    _DASHBOARD_AVAILABLE = False
    def _reg_screen(_t): pass   # no-op

# ── Tools bridge (graceful no-op if tools/ is absent) ──────────────────────
try:
    from digital_baby.tools_bridge.memory_notes    import maybe_record_discoveries
    from digital_baby.tools_bridge.working_memory   import record_tick
    from digital_baby.tools_bridge.concept_chat     import explain_concept
    from digital_baby.tools_bridge.knowledge_seeder import maybe_seed_knowledge
    from digital_baby.tools_bridge                  import get_router   # ← ADD THIS
    _TOOLS_BRIDGE_AVAILABLE = True
except ImportError:
    _TOOLS_BRIDGE_AVAILABLE = False
    def maybe_record_discoveries(*a, **kw): return 0
    def record_tick(*a, **kw): pass
    def explain_concept(concept, **kw): return []
    def maybe_seed_knowledge(*a, **kw): return None
    def get_router(): return None  # ← ADD THIS fallback too


# ---------------------------------------------------------------------------
# v4 intervals
# ---------------------------------------------------------------------------

SECOND_ORDER_INTERVAL: int = 30   # ticks between second-order chain scans
DECAY_INTERVAL:        int = 50   # ticks between causal record decay passes
LIFECYCLE_INTERVAL:    int = 20   # ticks between hypothesis lifecycle updates
ENTROPY_FEED_INTERVAL: int = 5    # ticks between feeding entropy to curiosity

# All domains that have a world simulator (apply_environment_dynamics branch).
# Expanding from 6 → 10 fixes the action=none pattern on neuroscience, climate,
# economics, and materials topics, which have domain_states + actions_by_domain
# in the generator but were previously excluded from world-step execution.
_SIMULATION_DOMAINS = frozenset({
    "ecosystem", "astronomy", "chemistry", "technology", "biology", "physics",
    "neuroscience", "climate", "economics", "materials",   # NEW v4.1
})


class BabyEventLoop:
    """Tick-based continuous loop that observes, learns, and updates beliefs — v4."""

    def __init__(
        self,
        world_path: str | Path,
        memory_path: str | Path,
        tick_sleep_seconds: float = 1.0,
        novelty_interval: int = 20,
    ) -> None:
        self.world_path = Path(world_path)
        self.memory     = Memory(memory_path)
        self.reasoner   = Reasoner()
        self.learner    = Learner(self.memory, self.reasoner)
        self.curiosity  = CuriosityModel()
        self.questions  = QuestionGenerator()
        self.patterns   = PatternDiscoverer()
        self.hypothesis_engine = HypothesisEngine()
        self.predictor  = Predictor()
        self.experimenter = Experimenter()
        self.causal_discovery = InterventionCausalDiscovery()
        self.concepts   = ConceptHierarchy()
        self.type_system = ConceptTypeSystem()
        self.expander   = KnowledgeExpander(self.world_path)
        self.knowledge_graph = KnowledgeGraph(self.world_path.parent / "knowledge_graph.json")
        self.generator  = KnowledgeGenerator(self.world_path)
        self.tick_sleep_seconds = tick_sleep_seconds
        self.novelty_interval   = max(1, novelty_interval)
        self.theory_abstraction     = TheoryAbstraction()
        # Perception module — writes to a JSONL file the dashboard reads
        _perception_log = self.world_path.parent / "perception_log.jsonl"
        self.screen_tracker = ScreenTracker(
            knowledge_graph=self.knowledge_graph,
            capture_interval=5.0,
            perception_log_path=_perception_log,
        )
        self.screen_tracker.start()
        # Register with dashboard so the toggle API can reach it
        _reg_screen(self.screen_tracker)

        # Wikipedia perception channel — clean text, higher confidence than OCR
        _wiki_log = self.world_path.parent / "wiki_log.jsonl"
        self.wiki_reader = WikiReader(
            knowledge_graph=self.knowledge_graph,
            wiki_log_path=_wiki_log,
        )
        self.wiki_reader.start()

        self.research_agent         = ResearchAgent(
            knowledge_graph=self.knowledge_graph,
            tool_router=get_router(),
        )

        self.hyp_experiment_planner = HypothesisExperimentPlanner(
    hypothesis_engine=self.hypothesis_engine,
    knowledge_graph=self.knowledge_graph,
)
        self._pending_hep = None
        self.law_predictor        = LawPredictor()
        self.prediction_evaluator = PredictionEvaluator()
        self.ingestion = KnowledgeIngestionPipeline(
            ingestion_interval=20,
            ingestion_batch_size=5,
            max_relations_per_concept=10,
            request_timeout=8.0,
            state_path=self.world_path.parent / "ingestion_state.json",
        )

        self.abstraction_engine    = ConceptAbstractionEngine(min_cluster_size=2)
        self.causal_chain_reasoner = CausalChainReasoner(max_depth=3, min_chain_confidence=0.15)
        self.role_assigner       = TopologicalRoleAssigner(self.knowledge_graph)
        self.mechanism_matcher   = MechanismMatcher(self.role_assigner.get_role)
        self.mediator_planner    = MediatorBlockingPlanner(self.knowledge_graph)

        # v4: simulation-first + axiom cache + soft deltas
        self.hypothesis_validator = HypothesisValidator(
            validation_interval=75,
            batch_size=3,
            request_timeout=8.0,
            enable_simulation=True,
            enable_axioms=True,
        )

        # v4: ExperimentPlanner with accuracy tracking + staleness budget
        self.experiment_planner = ExperimentPlanner(
            ig_weight=0.30, cur_weight=0.18, uncert_weight=0.17,
            age_weight=0.12, disc_weight=0.13, acc_weight=0.10,
        )

        self.law_discovery       = LawDiscovery()
        self._law_discovery_path = self.world_path.parent / "law_discovery_data.json"
        self.law_discovery.load(self._law_discovery_path)
        # Also re-register any loaded laws with the predictor
        for law in self.law_discovery.get_all_laws():
            self.law_predictor.register_law(law)
        self.memory_consolidator = MemoryConsolidator(
            consolidation_interval=10, max_episodic=500, max_semantic=3000,
        )

        self.epistemic         = EpistemicStateTracker()
        self.law_novelty_gate  = LawNoveltyGate()
        self.theory_engine     = CrossDomainTheoryEngine(min_domains=2)
        self.multi_step_planner = MultiStepPlanner(max_depth=2)
        self.hyp_chain_gen     = HypothesisChainGenerator()
        self.synthetic_resolver = SyntheticConceptResolver()
        self.dynvar_gen        = DynamicVariableGenerator()
        self.contradiction_engine = ContradictionHypothesisEngine()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.stall_ticks = 0
        self.state_transition_history: List[Dict[str, float]] = []
        self.last_selected_topic: Optional[str] = None
        self.consecutive_topic_count = 0
        self.last_forced_diversity_tick = 0

        # v4: track last action plan for outcome recording
        self._last_experiment_plan: Optional[ExperimentPlan] = None
        self._last_ep_ig_by_domain: Dict[str, float] = {}

    # ── Wikipedia topic selection ─────────────────────────────────────────────

    def _pick_wiki_topic(self) -> Optional[str]:
        """Choose a Wikipedia lookup topic from available curiosity signals.

        Priority order:
        1. Pending concepts from the ingestion pipeline
        2. KG nodes with lowest out-degree (most unknown)
        3. High-prediction-error variables
        4. Random KG node as fallback
        """
        import random

        # 1. Pending ingestion concepts
        pending = list(getattr(self.ingestion, "pending_concepts", []))
        if not pending:
            try:
                pending = list(self.ingestion._pending_concepts)
            except AttributeError:
                pass
        if pending:
            topic = random.choice(pending[:10])
            return topic.replace("_", " ")

        # 2. Low out-degree KG nodes (most unexplored)
        try:
            nodes = list(self.knowledge_graph.nodes)
            if nodes:
                # Prefer nodes that appear as objects but not well-explored as subjects
                out_degrees = {n: 0 for n in nodes}
                for edge in self.knowledge_graph.edges:
                    out_degrees[edge.source] = out_degrees.get(edge.source, 0) + 1
                low_degree = sorted(nodes, key=lambda n: out_degrees.get(n, 0))[:20]
                if low_degree:
                    return random.choice(low_degree[:5]).replace("_", " ")
        except Exception:
            pass

        # 3. High prediction-error variable
        try:
            error_topics = sorted(
                self.curiosity.prediction_error_by_topic.items(),
                key=lambda x: x[1], reverse=True,
            )
            if error_topics:
                return error_topics[0][0].replace("_", " ")
        except Exception:
            pass

        # 4. Random KG node
        try:
            nodes = list(self.knowledge_graph.nodes)
            if nodes:
                return random.choice(nodes).replace("_", " ")
        except Exception:
            pass

        return None

    # ── World helpers ─────────────────────────────────────────────────────────

    def _load_pages(self) -> List[Dict]:
        import os as _os
        pages = []
        if not hasattr(self, "_page_cache"):
            self._page_cache:  Dict[str, Dict]  = {}
            self._page_mtimes: Dict[str, float] = {}
        for file_path in sorted(self.world_path.glob("*.json")):
            key = str(file_path)
            try:
                mtime = _os.path.getmtime(file_path)
            except OSError:
                continue
            if key not in self._page_cache or self._page_mtimes.get(key) != mtime:
                page = _safe_json_load(file_path)
                if page is not None:
                    self._page_cache[key]  = page
                    self._page_mtimes[key] = mtime
                elif key in self._page_cache:
                    del self._page_cache[key]
            if key in self._page_cache:
                pages.append(self._page_cache[key])
        return pages

    def _safe_topic_lookup(self, topic: str, pages: List[Dict]) -> Dict:
        page = next((p for p in pages if p["topic"] == topic), None)
        if page is None:
            if topic.endswith("_state"):
                return pages[0] if pages else {"topic": topic, "facts": [], "domain": "unknown"}
            self.logger.warning("topic_missing: %s", topic)
            domain = topic.split("_")[0]
            page   = self.generator.generate_topic(persist=True, domain=domain)
            pages.append(page)
        return page

    @staticmethod
    def _page_keywords(page: Dict) -> List[str]:
        words = {page.get("topic", "").lower(), page.get("domain", "").lower()}
        for fact in page.get("facts", []):
            for token in fact.lower().split():
                words.add(token.strip())
        return sorted(words)

    def _find_topic_for_concept(self, concept: str, pages: List[Dict]) -> Optional[Tuple[str, str]]:
        lowered = concept.lower().strip()
        for page in pages:
            if lowered in {page.get("topic", "").lower(), page.get("domain", "").lower()}:
                return page["topic"], f"goal_direct_match:{lowered}"
        for page in pages:
            if lowered in self._page_keywords(page):
                return page["topic"], f"goal_keyword_match:{lowered}"
        memory_topics = self.memory.find_topics_for_concept(lowered)
        if memory_topics:
            return memory_topics[0], f"goal_memory_match:{lowered}"
        return None

    # ── v4: feed domain entropy into curiosity for UCB1 ───────────────────────

    def _feed_domain_entropy_to_curiosity(self) -> None:
        """Push per-domain entropy from EpistemicStateTracker into CuriosityModel."""
        domain_entropy: Dict[str, float] = {}
        for belief in self.epistemic._beliefs.values():
            cause  = belief.cause
            domain = self._concept_to_domain(cause)
            domain_entropy[domain] = domain_entropy.get(domain, 0.0) + belief.entropy

        for domain, entropy in domain_entropy.items():
            self.curiosity.register_domain_entropy(domain, entropy)

    @staticmethod
    def _concept_to_domain(concept: str) -> str:
        _MAP = {
            # ecosystem
            "wolf": "ecosystem", "deer": "ecosystem", "grass": "ecosystem",
            "season_factor": "ecosystem",
            # physics
            "force": "physics", "mass": "physics", "acceleration": "physics",
            "velocity": "physics", "friction": "physics", "momentum": "physics",
            "kinetic_energy": "physics", "heat": "physics",
            # chemistry
            "temperature": "chemistry", "reactants": "chemistry",
            "reaction_rate": "chemistry", "catalyst": "chemistry",
            "pH": "chemistry", "activation_energy": "chemistry",
            "product_concentration": "chemistry", "reaction_energy": "chemistry",
            # biology
            "cells": "biology", "energy": "biology", "pathogens": "biology",
            "immune_response": "biology", "antibodies": "biology",
            "toxin_level": "biology", "proteins": "biology",
            "cell_cycle_rate": "biology",
            # technology
            "robot": "technology", "battery_charge": "technology",
            "sensor_coverage": "technology", "data_quality": "technology",
            "maintenance_load": "technology", "network_latency": "technology",
            "sensor_threshold": "technology", "robot_activity": "technology",
            # astronomy
            "asteroid": "astronomy", "solar_energy": "astronomy",
            "collision_risk": "astronomy", "radiation_pressure": "astronomy",
            "asteroid_drift": "astronomy",
            # neuroscience (v4.1)
            "stress_level": "neuroscience", "cortisol": "neuroscience",
            "dopamine_level": "neuroscience", "serotonin_level": "neuroscience",
            "neural_activity": "neuroscience", "synaptic_strength": "neuroscience",
            "memory_consolidation": "neuroscience", "learning_rate": "neuroscience",
            # climate (v4.1)
            "co2_level": "climate", "temperature_anomaly": "climate",
            "glacier_melt": "climate", "sea_level_rise": "climate",
            "precipitation": "climate", "vegetation_cover": "climate",
            "albedo": "climate", "ocean_heat": "climate",
            # economics (v4.1)
            "interest_rate": "economics", "inflation": "economics",
            "gdp_growth": "economics", "unemployment": "economics",
            "investment": "economics", "consumption": "economics",
            "productivity": "economics", "debt_level": "economics",
            # materials (v4.1)
            "strain": "materials", "hardness": "materials",
            "yield_strength": "materials", "conductivity": "materials",
            "crack_growth": "materials", "porosity": "materials",
        }
        return _MAP.get(concept, "physics")

    # ── Topic selection ───────────────────────────────────────────────────────

    def _balanced_topic_scores(
        self, topics: List[str], weak_by_topic: Dict[str, float]
    ) -> List[Tuple[str, float, str]]:
        # v4: use UCB1-based scoring when available, fall back to v3 formula
        try:
            ucb_signals = {s.topic: s for s in
                           self.curiosity.score_topics_ucb(topics, weak_by_topic)}
            scored: List[Tuple[str, float, str]] = []
            for topic in topics:
                sig = ucb_signals.get(topic)
                if sig:
                    scored.append((topic, sig.score, sig.reason))
                else:
                    scored.append((topic, 0.0, "no_signal"))
            return sorted(scored, key=lambda x: x[1], reverse=True)
        except AttributeError:
            pass

        # Fallback to v3 balanced scoring
        curiosity_signals = {s.topic: s for s in
                             self.curiosity.score_topics(topics, weak_by_topic)}
        raw_scores  = {t: (curiosity_signals[t].score if t in curiosity_signals else 0.0)
                       for t in topics}
        max_raw     = max(raw_scores.values()) if raw_scores else 1.0
        min_raw     = min(raw_scores.values()) if raw_scores else 0.0
        score_range = max_raw - min_raw if max_raw != min_raw else 1.0

        scored = []
        for topic in topics:
            base_score    = raw_scores[topic]
            normalised    = (base_score - min_raw) / score_range
            visits        = self.curiosity.topic_visits[topic]
            visit_penalty = math.log(visits + 1)
            explore_bonus = 1.0 / math.sqrt(visits + 1)
            score         = normalised + explore_bonus - visit_penalty
            reason = f"base={base_score:.2f} norm={normalised:.3f} exp={explore_bonus:.2f} pen={visit_penalty:.2f}"
            scored.append((topic, score, reason))
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def _record_topic_selection(self, topic: str) -> None:
        if topic == self.last_selected_topic:
            self.consecutive_topic_count += 1
        else:
            self.last_selected_topic     = topic
            self.consecutive_topic_count = 1

    def _select_topic(
        self, pages: List[Dict], weak_by_topic: Dict[str, float], tick: int
    ) -> Tuple[Dict, str, float]:
        topics = [page["topic"] for page in pages]
        goal   = self.curiosity.pop_goal_concept()
        while goal is not None:
            matched = self._find_topic_for_concept(goal, pages)
            if matched:
                topic, reason = matched
                min_visits = min(self.curiosity.topic_visits[t] for t in topics) if topics else 0
                if self.curiosity.topic_visits[topic] <= min_visits + 1:
                    balanced       = self._balanced_topic_scores(topics, weak_by_topic)
                    selected_topic = topic
                    selected_score = next((s for t, s, _ in balanced if t == topic), 0.0)
                    if self.consecutive_topic_count > 10 and self.last_selected_topic == selected_topic:
                        alts = [(t, s, r) for t, s, r in balanced if t != selected_topic]
                        if alts:
                            selected_topic, selected_score, reason = alts[0]
                            self.last_forced_diversity_tick = tick
                    page = self._safe_topic_lookup(selected_topic, pages)
                    return page, reason, selected_score
            goal = self.curiosity.pop_goal_concept()

        balanced = self._balanced_topic_scores(topics, weak_by_topic)
        if not balanced:
            page = self._safe_topic_lookup(topics[0], pages)
            return page, "fallback:first_topic", 0.0

        selected_topic, selected_score, selected_reason = balanced[0]

        _STREAK_CAP = 5
        if (self.consecutive_topic_count >= _STREAK_CAP
                and self.last_selected_topic == selected_topic):
            alts = [(t, s, r) for t, s, r in balanced if t != selected_topic]
            if alts:
                selected_topic, selected_score, selected_reason = alts[0]
                self.last_forced_diversity_tick = tick
                self.logger.info("[diversity] streak_cap=%d → %s", _STREAK_CAP, selected_topic)
        elif (tick - self.last_forced_diversity_tick >= 20
              and self.last_selected_topic is not None):
            alts = [(t, s, r) for t, s, r in balanced if t != self.last_selected_topic]
            if alts:
                selected_topic, selected_score, selected_reason = alts[0]
                self.last_forced_diversity_tick = tick

        page = self._safe_topic_lookup(selected_topic, pages)
        return page, f"balanced:{selected_reason}", selected_score

    def _maybe_generate_topic(
        self, pages: List[Dict], best_score: float, test_domain: str | None = None
    ) -> Optional[Dict]:
        if self.stall_ticks >= 2 or best_score < 0.2 or test_domain is not None:
            page = self.generator.generate_topic(persist=True, domain=test_domain)
            pages.append(page)
            self.logger.info("[world] generated domain=%s", page.get("domain"))
            self.stall_ticks = 0
            return page
        return None

    # ── v4: mandatory action selection ───────────────────────────────────────

    def _select_action_v4(
        self,
        hypotheses: List[Hypothesis],
        domain:     str,
        tick:       int,
    ) -> Tuple[str, str, Optional[ExperimentPlan]]:
        """Return (action, subject, plan) — NEVER returns action='none'.

        Uses CuriosityModel.select_action() which implements the 5-priority
        cascade: test hypothesis → epistemic uncertainty → explore domain →
        ingest concept → synthesise variable.
        """
        # P0.5: mediator-blocking experiment takes precedence — resolves multiple
        # hypotheses at once by confirming/refuting a full causal chain
        _blocking = self.mediator_planner.get_pending_plan()
        if _blocking is not None:
            self.logger.debug(
                "[mediator_planner] P0.5 A=%s B=%s C=%s action=%s",
                _blocking.chain_source, _blocking.chain_mediator,
                _blocking.chain_target, _blocking.cause_action,
            )
            return _blocking.cause_action, _blocking.chain_source, None

        # Try structured plan from ExperimentPlanner first
        hep_action, hep_plan = self.hyp_experiment_planner.claim_next_action(tick)
        if hep_action:
            self._pending_hep = hep_plan
            self.logger.debug(
                "[hyp_exp_planner] P0 action=%s rule=%r",
                hep_action, hep_plan.hypothesis_rule,
            )
            return hep_action, hep_plan.cause_variable, None
        # Design a new experiment if the queue has space
        self.hyp_experiment_planner.maybe_design(tick)
        testable = self.hypothesis_engine.get_testable(hypotheses)
        if testable:
            ranked = self.experiment_planner.rank(
                testable[:20], self.curiosity, tick, domain,
                domain_state=self.generator.state_for_domain(domain),
                causal_rules=self.memory.get_causal_rules(),
            )
            if ranked:
                top_hyp   = ranked[0].hypothesis
                plan      = self.experiment_planner.plan_experiment(
                    top_hyp, tick, domain,
                    causal_records={
                        (r.cause, r.effect): r
                        for r in self.causal_discovery.get_all_records()
                    },
                )
                if plan is not None:
                    self.logger.debug(
                        "[planner_v4] P1_hypothesis action=%s cause=%s effect=%s paired=%s",
                        plan.action, plan.cause_concept, plan.effect_concept, plan.use_paired,
                    )
                    return plan.action, plan.cause_concept, plan

        # Fall through to CuriosityModel cascade (P2–P5)
        try:
            sel: ActionSelection = self.curiosity.select_action(
                hypotheses=testable,
                epistemic=self.epistemic,
                experiment_planner=self.experiment_planner,
                current_tick=tick,
            )
            self.logger.debug(
                "[curiosity_cascade] P%d action=%s subject=%s reason=%s",
                sel.priority, sel.action, sel.subject, sel.reason,
            )
            if sel.action in ("INGEST", "SYNTHESISE"):
                # These are handled in special branches — fall back to domain exploration
                actions = self.generator.actions_by_domain.get(domain, [])
                if actions:
                    import random
                    action = random.choice(actions)
                    return action, "", None
            return sel.action, sel.subject, None
        except Exception as exc:
            self.logger.debug("[curiosity_cascade] failed: %s", exc)

        # Hard fallback — always return something
        actions = self.generator.actions_by_domain.get(domain, [])
        if actions:
            import random
            return random.choice(actions), "", None

        return "increase_force", "force", None  # physics always available

    # ── v4: paired world step ─────────────────────────────────────────────────

    def _run_paired_world_step(
        self, domain: str, action: str, plan: ExperimentPlan
    ) -> Dict:
        """Run control + treatment from same base state; return clean causal delta."""
        base_state = self.experimenter.capture_state(
            self.generator.state_for_domain(domain)
        )
        paired = self.experimenter.run_paired(domain, base_state, action)

        # Update world state to treatment outcome
        self.generator.update_domain_state(domain, paired.treatment_state)

        # Extract clean causal deltas
        clean_deltas = self.experimenter.extract_causal_delta(paired)
        causal_subject = plan.cause_concept if plan else ""

        causal_facts = self.causal_discovery.analyze_paired(
            action=action,
            subject=causal_subject,
            control_state=paired.control_state,
            treatment_state=paired.treatment_state,
            current_tick=0,
        )
        causal_injected = self.causal_discovery.inject_into_graph(
            causal_facts, self.knowledge_graph
        )

        if causal_injected:
            self.logger.info("[paired_experiment] action=%s injected=%d clean_vars=%d",
                             action, causal_injected, len(clean_deltas))

        return {
            "prediction_error": 0.0,
            "deltas":           clean_deltas,
            "action":           action,
            "causal_facts":     causal_facts,
            "state_before":     paired.control_state,
            "state_after":      paired.treatment_state,
            "paired":           True,
        }

    # ── Cognitive helpers ─────────────────────────────────────────────────────

    def _update_hypotheses(
        self, discovered_rules: List[PatternRecord], triplets: List[Tuple[str, str, str]]
    ) -> List[Hypothesis]:
        from digital_baby.brain.patterns import PatternRule
        pattern_rules = [
            PatternRule(relation=r.relation, count=r.support, label=r.label, template=r.template)
            for r in discovered_rules
        ]
        hypotheses = self.hypothesis_engine.from_patterns(pattern_rules, self.type_system, triplets)
        self.memory.set_world_model_rules([
            HypothesisRecord(
                rule=h.rule, concepts=h.concepts, confidence=h.confidence,
                supporting_evidence=h.supporting_evidence,
                contradicting_evidence=h.contradicting_evidence,
            )
            for h in hypotheses
        ])
        return hypotheses

    def _expand_unknown_concepts(self, concepts: List[str]) -> None:
        for concept in concepts:
            if self.type_system.is_known(concept):
                continue
            edges = self.expander.expand_concept_graph(concept, depth=2)
            if edges:
                path = " -> ".join([edges[0][0]] + [edge[2] for edge in edges[:2]])
                self.logger.debug("[world] expanded=%s", path)

    def _inject_novelty(self, tick: int) -> None:
        if tick % self.novelty_interval != 0:
            return
        novelty_facts, new_concepts = self.generator.inject_open_world_novelty(max_entities=3)
        self.logger.info("[world] novelty_injection tick=%s concepts=%s", tick, new_concepts)
        if novelty_facts:
            novelty_page = {
                "topic": "open_world_novelty", "domain": "open_world",
                "facts": novelty_facts, "generated": True,
            }
            result = self.learner.learn_from_page(novelty_page)
            self.curiosity.register_unknowns(result.topic, result.unknown_concepts)
            self.curiosity.register_unknowns(result.topic, result.unknown_relations)

    def _preferred_action_for_hypothesis(self, hypothesis: Hypothesis, domain: str) -> Optional[str]:
        _CAUSE_TO_ACTION: Dict[str, Dict[str, str]] = {
            # chemistry
            "temperature":           {"chemistry":   "increase_temperature", "materials": "heat_treat"},
            "reactants":             {"chemistry":   "add_chemical"},
            "catalyst":              {"chemistry":   "add_catalyst"},
            "pH":                    {"chemistry":   "adjust_pH"},
            "product_concentration": {"chemistry":   "remove_product"},
            # ecosystem
            "wolf":                  {"ecosystem":   "add_predator"},
            "deer":                  {"ecosystem":   "introduce_species"},
            "grass":                 {"ecosystem":   "introduce_species"},
            "season_factor":         {"ecosystem":   "change_season"},
            # astronomy
            "asteroid":              {"astronomy":   "introduce_species"},
            # technology
            "robot":                 {"technology":  "add_robot"},
            "battery_charge":        {"technology":  "add_robot"},
            "sensor_coverage":       {"technology":  "upgrade_sensor"},
            "sensor_threshold":      {"technology":  "upgrade_sensor"},
            "maintenance_load":      {"technology":  "increase_maintenance"},
            # biology
            "pathogens":             {"biology":     "add_pathogen"},
            "energy":                {"biology":     "boost_energy"},
            "cells":                 {"biology":     "add_cells"},
            "toxin_level":           {"biology":     "neutralise_toxin"},
            "antibodies":            {"biology":     "add_antibody"},
            # physics
            "force":                 {"physics":     "increase_force"},
            "heat":                  {"physics":     "add_heat", "materials": "heat_treat"},
            "acceleration":          {"physics":     "increase_force"},
            "friction":              {"physics":     "add_friction"},
            "momentum":              {"physics":     "apply_impulse"},
            "mass":                  {"physics":     "reduce_mass"},
            # neuroscience (v4.1)
            "stress_level":          {"neuroscience": "induce_stress", "materials": "apply_stress"},
            "cortisol":              {"neuroscience": "induce_stress"},
            "dopamine_level":        {"neuroscience": "boost_dopamine"},
            "neural_activity":       {"neuroscience": "stimulate_neurons"},
            "memory_consolidation":  {"neuroscience": "improve_sleep"},
            "learning_rate":         {"neuroscience": "boost_dopamine"},
            # climate (v4.1)
            "co2_level":             {"climate": "emit_co2"},
            "temperature_anomaly":   {"climate": "emit_co2"},
            "glacier_melt":          {"climate": "melt_glacier"},
            "vegetation_cover":      {"climate": "plant_forest"},
            "albedo":                {"climate": "increase_albedo"},
            "ocean_heat":            {"climate": "warm_ocean"},
            # economics (v4.1)
            "interest_rate":         {"economics": "raise_interest_rate"},
            "investment":            {"economics": "lower_interest_rate"},
            "consumption":           {"economics": "increase_spending"},
            "gdp_growth":            {"economics": "boost_productivity"},
            "productivity":          {"economics": "boost_productivity"},
            "debt_level":            {"economics": "add_debt"},
            # materials (v4.1)
            "strain":                {"materials": "apply_stress"},
            "hardness":              {"materials": "heat_treat"},
            "yield_strength":        {"materials": "quench"},
            "conductivity":          {"materials": "heat_treat"},
            "crack_growth":          {"materials": "apply_stress"},
            "porosity":              {"materials": "add_porosity"},
        }
        for concept in hypothesis.concepts[:2]:
            domain_map = _CAUSE_TO_ACTION.get(concept, {})
            if domain in domain_map:
                return domain_map[domain]
        return None

    def _sync_knowledge_graph(self) -> None:
        self.knowledge_graph.ingest_triplets(
            self.memory.relation_triplets(), default_confidence=0.58
        )
        for rule in self.memory.get_causal_rules():
            self.knowledge_graph.ingest_causal_rule(
                cause=rule.cause, effect=rule.effect,
                direction=rule.direction, confidence=rule.confidence,
                evidence=rule.observations,
            )

    def _run_stateful_world_step(
        self, domain: str,
        preferred_action: Optional[str] = None,
        target: Optional[str] = None,
        current_tick: int = 0,
        fix_actions=None,
    ) -> Dict:
        """Run one world step.  Pass fix_actions to clamp mediator variables."""
        state  = self.generator.state_for_domain(domain)
        action = self.generator.choose_action(domain, preferred_action=preferred_action)
        self.logger.debug("[world] action=%s domain=%s target=%s fix=%s",
                          action, domain, target,
                          [f.variable for f in fix_actions] if fix_actions else None)

        prediction = self.predictor.predict_state_transition(domain, action, state)
        new_state, deltas = self.experimenter.apply_environment_dynamics(
            domain, state, action, fix_actions=fix_actions)
        self.generator.update_domain_state(domain, new_state)

        eval_result = self.predictor.evaluate_state_prediction(prediction, deltas)

        _DOMAIN_ACTION_SUBJECT = {
            ("ecosystem",     "remove_species"):        "deer",
            ("ecosystem",     "introduce_species"):     "deer",
            ("ecosystem",     "remove_predator"):       "wolf",
            ("ecosystem",     "add_predator"):          "wolf",
            ("ecosystem",     "change_season"):         "season_factor",
            ("astronomy",     "remove_species"):        "asteroid",
            ("astronomy",     "introduce_species"):     "asteroid",
            ("chemistry",     "increase_temperature"):  "temperature",
            ("chemistry",     "add_chemical"):          "reactants",
            ("chemistry",     "add_catalyst"):          "catalyst",
            ("chemistry",     "adjust_pH"):             "pH",
            ("chemistry",     "remove_product"):        "product_concentration",
            ("technology",    "add_robot"):             "robot",
            ("technology",    "remove_robot"):          "robot",
            ("technology",    "upgrade_sensor"):        "sensor_threshold",
            ("technology",    "increase_maintenance"):  "maintenance_load",
            ("biology",       "add_pathogen"):          "pathogens",
            ("biology",       "boost_energy"):          "energy",
            ("biology",       "add_cells"):             "cells",
            ("biology",       "neutralise_toxin"):      "toxin_level",
            ("biology",       "add_antibody"):          "antibodies",
            ("physics",       "increase_force"):        "force",
            ("physics",       "add_heat"):              "heat",
            ("physics",       "add_friction"):          "friction",
            ("physics",       "apply_impulse"):         "force",
            ("physics",       "reduce_mass"):           "mass",
            # v4.1 NEW: neuroscience
            ("neuroscience",  "induce_stress"):         "stress_level",
            ("neuroscience",  "reduce_stress"):         "stress_level",
            ("neuroscience",  "boost_dopamine"):        "dopamine_level",
            ("neuroscience",  "improve_sleep"):         "memory_consolidation",
            ("neuroscience",  "stimulate_neurons"):     "neural_activity",
            # v4.1 NEW: climate
            ("climate",       "emit_co2"):              "co2_level",
            ("climate",       "plant_forest"):          "vegetation_cover",
            ("climate",       "melt_glacier"):          "glacier_melt",
            ("climate",       "increase_albedo"):       "albedo",
            ("climate",       "warm_ocean"):            "ocean_heat",
            # v4.1 NEW: economics
            ("economics",     "raise_interest_rate"):   "interest_rate",
            ("economics",     "lower_interest_rate"):   "interest_rate",
            ("economics",     "increase_spending"):     "consumption",
            ("economics",     "boost_productivity"):    "productivity",
            ("economics",     "add_debt"):              "debt_level",
            # v4.1 NEW: materials
            ("materials",     "apply_stress"):          "stress_level",
            ("materials",     "heat_treat"):            "temperature",
            ("materials",     "add_porosity"):          "porosity",
            ("materials",     "quench"):                "temperature",
            ("materials",     "anneal"):                "temperature",
            # v4.1: neuroscience (previously fell through to '' — _ACTION_TO_CAUSE fallback)
            ("neuroscience",  "induce_stress"):         "stress_level",
            ("neuroscience",  "reduce_stress"):         "stress_level",
            ("neuroscience",  "boost_dopamine"):        "dopamine_level",
            ("neuroscience",  "improve_sleep"):         "memory_consolidation",
            ("neuroscience",  "stimulate_neurons"):     "neural_activity",
            # v4.1: climate
            ("climate",       "emit_co2"):              "co2_level",
            ("climate",       "plant_forest"):          "vegetation_cover",
            ("climate",       "melt_glacier"):          "glacier_melt",
            ("climate",       "increase_albedo"):       "albedo",
            ("climate",       "warm_ocean"):            "ocean_heat",
            # v4.1: economics
            ("economics",     "raise_interest_rate"):   "interest_rate",
            ("economics",     "lower_interest_rate"):   "interest_rate",
            ("economics",     "increase_spending"):     "consumption",
            ("economics",     "boost_productivity"):    "productivity",
            ("economics",     "add_debt"):              "debt_level",
        }
        causal_subject = _DOMAIN_ACTION_SUBJECT.get((domain, action), "")
        causal_facts   = self.causal_discovery.analyze(
            action=action, subject=causal_subject,
            state_before=state, state_after=new_state,
            current_tick=current_tick,
        )
        causal_injected = self.causal_discovery.inject_into_graph(
            causal_facts, self.knowledge_graph, current_tick=current_tick
        )
        if causal_injected:
            self.logger.debug("[causal_discovery] injected %d edges", causal_injected)

        for cf in causal_facts:
            record     = self.causal_discovery.get_record(cf.cause, cf.effect)
            _direction = record.direction if record else ("negative" if cf.sign == -1 else "positive")
            if _direction == "bidirectional":
                _direction = "mixed"
            self.memory.upsert_causal_rule(
                cause=cf.cause, effect=cf.effect,
                direction=_direction, observations_increment=1,
            )
            self.epistemic.update_edge(
                cause=cf.cause, effect=cf.effect,
                confirmed=(cf.sign > 0),
                tick=current_tick,
                confidence_hint=record.confidence if record else None,
            )

        for cause_var, delta_cause in deltas.items():
            if abs(delta_cause) < 1e-6:
                continue
            for effect_var, delta_effect in deltas.items():
                if cause_var != effect_var:
                    # Use delta_cause (not abs state) as x_val so the fitter sees
                    # the marginal change, not the absolute level contaminated by
                    # other simultaneously-changing variables.
                    self.law_discovery.record_observation(cause_var, effect_var, delta_cause, delta_effect)

        # ── Episodic fact generation ──────────────────────────────────────────
        # Old code produced "wolves_delta is -5.87" which matched the "is"
        # relation and became a known fact after tick 1 → new_facts=0 forever.
        # New code produces "wolf negatively_affects deer" etc. — strings that
        # Reasoner.extract_relation() parses into meaningful (s, r, o) triplets
        # that vary per observation and contribute to new_facts > 0.
        episodic_facts = record_episodic_observation(
            action=action,
            subject=causal_subject,
            domain=domain,
            state_before=state,
            state_after=new_state,
            tick=current_tick,
        )
        if episodic_facts:
            self.learner.learn_from_page({
                "topic": f"{domain}_episodic",
                "facts": episodic_facts,
            })

        self.state_transition_history.append(deltas)
        self.state_transition_history = self.state_transition_history[-120:]

        return {
            "prediction_error": eval_result.prediction_error,
            "deltas":           deltas,
            "action":           action,
            "causal_facts":     causal_facts,
            "state_before":     state,
            "state_after":      new_state,
            "paired":           False,
        }

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, max_ticks: Optional[int] = None) -> None:
        tick = 0
        while True:
            tick += 1
            self.memory.current_tick = tick
            self.knowledge_graph.reset_tick_stats()
            self.curiosity.tick()   # v4: increment UCB1 counter

            pages = self._load_pages()
            if not pages:
                self.logger.warning("No knowledge pages found; generating one.")
                pages = [self.generator.generate_topic(persist=True)]

            self._inject_novelty(tick)

            topics = [page["topic"] for page in pages]
            weak_by_topic = {
                topic: len([r for r in self.memory.facts.values()
                            if r.source_topic == topic and r.confidence < 0.5])
                       / max(1, len([r for r in self.memory.facts.values()
                                     if r.source_topic == topic]))
                for topic in topics
            }
            if tick % RESEARCH_INTERVAL == 0:
                self.research_agent.run_cycle(current_tick=tick)
            # v4: feed epistemic entropy into curiosity for UCB1
            if tick % ENTROPY_FEED_INTERVAL == 0:
                self._feed_domain_entropy_to_curiosity()

            selected_page, selection_reason, selection_score = self._select_topic(
                pages, weak_by_topic, tick
            )
            selected_topic = selected_page["topic"]
            self._record_topic_selection(selected_topic)

            result = self.learner.learn_from_page(selected_page)
            self._expand_unknown_concepts(sorted(result.unknown_concepts))
            self.type_system.registry.refresh()

            domain     = selected_page.get("domain") or selected_topic
            transition: Optional[Dict] = None

            structural_novelty = (
                len(result.new_entities)       * 0.12
                + len(result.new_relation_types) * 0.2
                + result.new_facts              * 0.08
                + result.reinforcement          * 0.01   # weak novelty from reinforcement
            )
            self.curiosity.register_structural_novelty(result.topic, structural_novelty)
            self.curiosity.register_unknowns(result.topic, result.unknown_concepts)
            self.curiosity.register_unknowns(result.topic, result.unknown_relations)
            self.curiosity.mark_visited(result.topic)

            if tick % 5 == 0:
                for _t in list(self.curiosity.prediction_error_by_topic.keys()):
                    self.curiosity.prediction_error_by_topic[_t] *= 0.9
                for _t in list(self.curiosity.structural_novelty_by_topic.keys()):
                    self.curiosity.structural_novelty_by_topic[_t] *= 0.85

            generated_questions = self.questions.from_unknown_concepts(result.unknown_concepts)
            generated_questions.extend(self.questions.from_unknown_concepts(result.unknown_relations))
            self.curiosity.register_unknowns(result.topic, [q.target_concept for q in generated_questions])

            # ── External knowledge ingestion ──────────────────────────────────
            _raw_unknowns = list(result.unknown_concepts) + list(result.unknown_relations)
            _real_concepts, _synthetic_concepts = self.synthetic_resolver.batch_classify(_raw_unknowns)

            for _sc in _synthetic_concepts:
                _syn_edges = self.synthetic_resolver.resolve(_sc)
                for (_s, _r, _o) in _syn_edges:
                    try:
                        self.knowledge_graph.add_edge(_s, _r, _o, provenance="synthetic_resolved", confidence=0.65)
                    except Exception:
                        pass
                _root = self.synthetic_resolver.real_lookup_term(_sc)
                if _root:
                    _real_concepts.append(_root)

            self.ingestion.suggest_concepts(_real_concepts)
            ingested_new_edges = self.ingestion.maybe_ingest(
                tick,
                knowledge_graph=self.knowledge_graph,
                type_system=self.type_system,
                curiosity_model=self.curiosity,
            )
            if ingested_new_edges:
                self.logger.info("[ingestion] tick=%s new_edges=%d", tick, ingested_new_edges)

            # ── Perception: drain screen-observed triples into KG ──────────────
            _perc_triples = self.screen_tracker.drain_pending()
            _perc_new = 0
            for _pt in _perc_triples:
                self.knowledge_graph.add_or_update_edge(
                    _pt.subject, _pt.object, _pt.relation,
                    confidence=_pt.confidence,
                    provenance="screen",
                    current_tick=tick,
                )
                self.ingestion.suggest_concepts([_pt.subject, _pt.object])
                _perc_new += 1
            if _perc_new:
                self.screen_tracker.total_perception_edges += _perc_new
                self.logger.info(
                    "[perception] tick=%d ingested=%d total_edges=%d",
                    tick, _perc_new, self.screen_tracker.total_perception_edges,
                )

            # ── Wikipedia perception channel ───────────────────────────────────
            # Every ~50 ticks, select a topic from curiosity signals and queue it.
            _wiki_new = 0
            if tick % 50 == 0:
                _wiki_topic = self._pick_wiki_topic()
                if _wiki_topic:
                    self.wiki_reader.ingest_topic(_wiki_topic, current_tick=tick)
                    self.logger.info("[wiki] queued_topic=%s tick=%d", _wiki_topic, tick)

            _wiki_triples = self.wiki_reader.drain_pending()
            for _wt in _wiki_triples:
                self.knowledge_graph.add_or_update_edge(
                    _wt.subject, _wt.object, _wt.relation,
                    confidence=_wt.confidence,
                    provenance="wikipedia",
                    current_tick=tick,
                )
                self.ingestion.suggest_concepts([_wt.subject, _wt.object])
                _wiki_new += 1
            if _wiki_new:
                self.wiki_reader.total_wiki_edges += _wiki_new
                self.logger.info(
                    "[wiki] tick=%d ingested=%d total_edges=%d",
                    tick, _wiki_new, self.wiki_reader.total_wiki_edges,
                )

            # ── Chatbot fallback for concepts Wikipedia missed ──────────────────
            if _TOOLS_BRIDGE_AVAILABLE and tick % 10 == 0:
                _missed = [c for c in sorted(result.unknown_concepts)[:3]
                           if not self.type_system.is_known(c)]
                for _mc in _missed:
                    _chat_triplets = explain_concept(_mc)
                    if _chat_triplets:
                        self.knowledge_graph.ingest_triplets(
                            _chat_triplets, default_confidence=0.45
                        )
                        self.logger.debug(
                            "[tools_bridge] chat_enriched concept=%s triplets=%d",
                            _mc, len(_chat_triplets),
                        )

            # ── Knowledge seeder: fetch real pages for unknown concepts ─────────
            if _TOOLS_BRIDGE_AVAILABLE:
                _seeded_page = maybe_seed_knowledge(
                    unknown_concepts=sorted(result.unknown_concepts)[:5],
                    tick=tick,
                    new_facts_this_tick=result.new_facts,
                    world_path=self.world_path,
                )
                if _seeded_page:
                    _seed_result = self.learner.learn_from_page(_seeded_page)
                    self.curiosity.register_unknowns(
                        _seeded_page["topic"], _seed_result.unknown_concepts
                    )
                    self.curiosity.mark_visited(_seeded_page["topic"])
                    self.logger.info(
                        "[tools_bridge] seeded topic=%s new_facts=%d",
                        _seeded_page["topic"], _seed_result.new_facts,
                    )

            triplets = self.memory.relation_triplets()
            self.knowledge_graph.ingest_triplets(triplets, default_confidence=0.58)
            self.type_system.infer_from_triplets(triplets)
            discovered     = self.patterns.discover(triplets, min_support=2)
            learned_rules  = self.learner.infer_general_rules(triplets)
            causal_candidates = self.learner.discover_causal_candidates(self.state_transition_history)

            new_causal_rules: List[str] = []
            for candidate in causal_candidates:
                record, created = self.memory.upsert_causal_rule(
                    cause=candidate.cause, effect=candidate.effect,
                    direction=candidate.direction, observations_increment=1,
                )
                relation_label = f"{record.cause} {record.direction}_affects {record.effect}"
                if created:
                    new_causal_rules.append(relation_label)
                    self.logger.info("[causal_rule] %s conf=%.2f", relation_label, record.confidence)
                self.curiosity.register_new_pattern(result.topic)
                self.knowledge_graph.ingest_causal_rule(
                    record.cause, record.effect, record.direction,
                    record.confidence, record.observations
                )
                discovered.append(PatternRecord(
                    template=relation_label, relation="affects",
                    support=max(1, record.observations), label="causal_rule",
                    timestamp=time.time(),
                ))

            for rule in learned_rules:
                discovered.append(PatternRecord(
                    template=rule, relation="eats", support=2,
                    label="learner_inferred", timestamp=time.time(),
                ))

            normalized_patterns: List[PatternRecord] = []
            for item in discovered:
                if isinstance(item, PatternRecord):
                    normalized_patterns.append(item)
                else:
                    normalized_patterns.append(PatternRecord(
                        template=item.template, relation=item.relation,
                        support=item.count, label=item.label, timestamp=time.time(),
                    ))
            self.memory.update_patterns(normalized_patterns)
            self.concepts.ingest_triplets(triplets)

            hypotheses = self._update_hypotheses(self.memory.patterns, triplets)
            causal_hypotheses = self.hypothesis_engine.from_causal_rules(
                self.memory.get_causal_rules()
            )
            hypotheses = hypotheses + causal_hypotheses

            # ── Reinforcement-driven hypothesis generation ─────────────────────
            # When new_facts == 0 but the learner re-observed many known facts
            # (reinforcement), generate chained hypotheses from well-evidenced
            # causal pairs so the hypothesis pool keeps growing during stalls.
            if result.new_facts == 0 and result.reinforcement >= \
                    self.hypothesis_engine.REINFORCEMENT_HYPOTHESIS_THRESHOLD:
                reinf_hyps = self.hypothesis_engine.from_reinforcement(
                    result.reinforcement, self.memory
                )
                if reinf_hyps:
                    hypotheses = hypotheses + reinf_hyps
                    self.logger.debug(
                        "[reinforcement] tick=%d reinf=%d generated_hyps=%d",
                        tick, result.reinforcement, len(reinf_hyps),
                    )

            # ── v4: Hypothesis lifecycle management ────────────────────────────
            # Only run every LIFECYCLE_INTERVAL ticks.
            # FIX: apply_lifecycle_updates previously replaced the full hypothesis
            # list with only active ones (dropping 300→79 then regenerating 306),
            # causing the oscillating count shown in the logs.  Now we:
            #  1. Update status fields in-place on existing hypotheses
            #  2. Only filter out ABANDONED ones (SUSPENDED stay in list but are
            #     skipped by the ExperimentPlanner via is_testable=False)
            #  3. Add mutations for ABANDONED hypotheses to the end of the list
            if tick % LIFECYCLE_INTERVAL == 0:
                active_hyps, abandoned_hyps = self.hypothesis_engine.apply_lifecycle_updates(
                    hypotheses, current_tick=tick
                )
                if abandoned_hyps:
                    self.logger.info(
                        "[lifecycle] tick=%d abandoned=%d generating_mutations",
                        tick, len(abandoned_hyps),
                    )
                    mutations = self.hypothesis_engine.get_mutations(
                        abandoned_hyps, self.knowledge_graph
                    )
                    # Keep active + suspended hypotheses; replace abandoned with mutations
                    suspended_hyps = [h for h in hypotheses
                                      if getattr(h, "status", "active") == "suspended"]
                    hypotheses = active_hyps + suspended_hyps + mutations
                # If no abandoned: leave hypotheses unchanged (just statuses updated in-place)

            uncertainty = self.memory.hypothesis_uncertainty()
            self.curiosity.register_hypothesis_uncertainty(result.topic, uncertainty)

            # ── Cross-domain theory engine ─────────────────────────────────────
            for _rule in self.memory.get_causal_rules():
                self.theory_engine.register_pattern(
                    cause=_rule.cause, effect=_rule.effect,
                    direction=_rule.direction, confidence=_rule.confidence,
                )
            new_theories, theory_hyp_strings = self.theory_engine.run(tick)
            if new_theories:
                self.logger.info("[theory] tick=%d new=%d %s",
                                 tick, len(new_theories), [t.name for t in new_theories])
            for _ths in theory_hyp_strings:
                # Extract concepts from the analogy string (words around "if" and "then")
                _concepts = []
                _parts = _ths.lower().split()
                for _pi, _pw in enumerate(_parts):
                    if _pw in ("if", "then") and _pi + 1 < len(_parts):
                        _concepts.append(_parts[_pi + 1].strip(".,"))
                _hyp = Hypothesis(
                    rule=_ths, concepts=_concepts, confidence=0.3,
                    supporting_evidence=1, contradicting_evidence=0,
                )
                hypotheses.append(_hyp)
                # Feed unexplored cross-domain concepts to curiosity model
                if _concepts:
                    self.curiosity.register_unknowns("cross_domain_analogy", _concepts)

            # ── Hypothesis chain generator ─────────────────────────────────────
            chain_hyps = self.hyp_chain_gen.run(
                causal_rules=self.memory.get_causal_rules(),
                existing_hypotheses=hypotheses,
                knowledge_graph=self.knowledge_graph,
                current_tick=tick,
            )
            if chain_hyps:
                hypotheses = hypotheses + [
                    Hypothesis(
                        rule=h.rule, concepts=h.concepts, confidence=h.confidence,
                        supporting_evidence=h.supporting_evidence,
                        contradicting_evidence=h.contradicting_evidence,
                    )
                    for h in chain_hyps
                ]

            # ── Contradiction hypothesis engine ───────────────────────────────
            contradiction_hyps, proposed_vars = self.contradiction_engine.run(
                causal_rules=self.memory.get_causal_rules(),
                knowledge_graph=self.knowledge_graph,
                domain=domain,
                tick=tick,
            )
            if contradiction_hyps:
                hypotheses = hypotheses + contradiction_hyps
            for pvar in proposed_vars:
                if pvar.domain in self.generator.domain_states:
                    if pvar.name not in self.generator.domain_states[pvar.domain]:
                        self.generator.domain_states[pvar.domain][pvar.name] = pvar.initial_value
                        self.logger.info("[contradiction_hyp] injected_variable=%s domain=%s",
                                         pvar.name, pvar.domain)

            # ── Dynamic variable generator ────────────────────────────────────
            _prev_ep_ig = self._last_ep_ig_by_domain.get(domain, 0.0)
            self.dynvar_gen.record_ep_ig(domain, _prev_ep_ig, tick)
            new_synth_var = self.dynvar_gen.maybe_synthesize(
                domain=domain,
                current_vars=self.generator.state_for_domain(domain),
                causal_rules=self.memory.get_causal_rules(),
                tick=tick,
            )
            if new_synth_var:
                self.dynvar_gen.inject_into_domain_states(
                    new_synth_var, self.generator.domain_states
                )
                self.curiosity.register_structural_novelty(selected_topic, 5.0)
                self.logger.info("[dynvar] new=%s domain=%s type=%s tick=%d",
                                 new_synth_var.name, domain,
                                 new_synth_var.synthesis_type, tick)

            # ── v4: Action selection — NEVER returns action=none ───────────────
            # Map text-only topics to a related simulation domain so the agent
            # always runs a world step even on non-simulation topics.
            # e.g. "animals"→"ecosystem", "microbiology"→"biology", "plants"→"ecosystem"
            _TOPIC_TO_SIM_DOMAIN: Dict[str, str] = {
                "animals":        "ecosystem",
                "plants":         "ecosystem",
                "ecology":        "ecosystem",
                "microbiology":   "biology",
                "taxonomy":       "ecosystem",
                "materials":      "materials",
                "neuroscience":   "neuroscience",
                "climate":        "climate",
                "economics":      "economics",
            }
            effective_domain = domain if domain in _SIMULATION_DOMAINS else \
                _TOPIC_TO_SIM_DOMAIN.get(domain, _TOPIC_TO_SIM_DOMAIN.get(selected_topic, ""))

            if effective_domain in _SIMULATION_DOMAINS:
                action_str, target_concept, exp_plan = self._select_action_v4(
                    hypotheses, effective_domain, tick
                )
                self._last_experiment_plan = exp_plan
                law_predictions = self.law_predictor.predict_all(
                state=self.generator.state_for_domain(effective_domain),
                tick=tick,
            )

                # v4: use paired experiment when planner recommends it
                # Detect active mediator-blocking plan and build fix_actions list
                _active_block = self.mediator_planner.get_pending_plan()
                _fix_actions  = ([_active_block.fix_action] if _active_block is not None
                                 else None)

                if exp_plan is not None and exp_plan.use_paired:
                    self.logger.info(
                        "[paired_exp] tick=%d action=%s cause=%s effect=%s domain=%s",
                        tick, exp_plan.action, exp_plan.cause_concept,
                        exp_plan.effect_concept, effective_domain,
                    )
                    transition = self._run_paired_world_step(effective_domain, action_str, exp_plan)
                else:
                    transition = self._run_stateful_world_step(
                        effective_domain, preferred_action=action_str,
                        target=target_concept, current_tick=tick,
                        fix_actions=_fix_actions,
                    )

                # Two-tick mediator result resolution
                if _active_block is not None and transition is not None:
                    _tvar = _active_block.chain_target
                    _delta = abs(transition.get("deltas", {}).get(_tvar, 0.0))
                    if not hasattr(_active_block, "_free_delta"):
                        # Tick T — free run: store ΔC
                        _active_block._free_delta = _delta
                        self.logger.debug(
                            "[mediator_planner] tick=%d free_delta=%.4f target=%s",
                            tick, _delta, _tvar)
                    else:
                        # Tick T+1 — blocking run: compute MR and record
                        _mr_result = self.mediator_planner.record_result(
                            delta_free=_active_block._free_delta,
                            delta_blocked=_delta,
                            current_tick=tick,
                        )
                        if _mr_result:
                            self.logger.info(
                                "[mediator_planner] confirmed=%s MR=%.2f %s→%s→%s",
                                _mr_result.confirmed, _mr_result.mediation_ratio,
                                _active_block.chain_source,
                                _active_block.chain_mediator,
                                _active_block.chain_target,
                            )

                self.curiosity.register_prediction_error(
                    result.topic, transition["prediction_error"]
                )
                self.experiment_planner.record_action(transition["action"])
                if law_predictions:
                    eval_result = self.prediction_evaluator.evaluate_all(
                        predictions=law_predictions,
                        state_before=transition.get("state_before") or {},
                        state_after=transition.get("state_after") or {},
                        knowledge_graph=self.knowledge_graph,
                        tick=tick,
                    )
                    if eval_result.hits > 0 or eval_result.misses > 0:
                        self.logger.info(
                            "[law_eval] tick=%d hits=%d misses=%d mean_acc=%.3f",
                            tick, eval_result.hits, eval_result.misses,
                            eval_result.mean_accuracy,
                        )
                if self._pending_hep is not None:
                    before_s = transition.get("before_state") or {}
                    after_s  = transition.get("after_state")  or transition.get("state") or {}
                    self.hyp_experiment_planner.evaluate(
                        self._pending_hep, before_s, after_s, tick
                    )
                    self._pending_hep = None
                # v4: record prediction outcome for accuracy tracking
                if exp_plan is not None:
                    causal_facts = transition.get("causal_facts") or []
                    observed_sign = None
                    for cf in causal_facts:
                        if cf.cause == exp_plan.cause_concept and cf.effect == exp_plan.effect_concept:
                            observed_sign = cf.sign
                            break
                    if observed_sign is not None:
                        self.experiment_planner.record_outcome(exp_plan, observed_sign, tick)

                for cf in (transition.get("causal_facts") or []):
                    rec = self.causal_discovery.get_record(cf.cause, cf.effect)
                    self.epistemic.update_edge(
                        cause=cf.cause, effect=cf.effect,
                        confirmed=(cf.sign > 0),
                        tick=tick,
                        confidence_hint=rec.confidence if rec else None,
                    )

                # Experiment evaluation
                candidate = None
                if exp_plan is not None:
                    for h in hypotheses:
                        if h.rule == exp_plan.hypothesis_rule:
                            candidate = h
                            break
                if candidate is None and hypotheses:
                    ranked = self.experiment_planner.rank(
                        hypotheses[:20], self.curiosity, tick, domain
                    )
                    candidate = ranked[0].hypothesis if ranked else hypotheses[0]

                if candidate is not None:
                    candidate_score = 1.0
                    predictions = self.predictor.predict(
                        [candidate], self.memory.all_entities(), self.type_system
                    )
                    prediction = predictions[0] if predictions else None

                    if self.experimenter.should_schedule(
                        candidate,
                        self.curiosity.prediction_error_by_topic.get(result.topic, 0.0),
                    ):
                        experiment = self.experimenter.generate(
                            candidate, self.memory.all_entities(), self.type_system, n=3
                        )
                        _sb = transition.get("state_before")
                        _sa = transition.get("state_after")
                        outcome = self.experimenter.evaluate(
                            experiment, triplets, self.type_system,
                            state_before=_sb, state_after=_sa,
                        )
                        actual_success    = outcome.supported >= outcome.contradicted
                        predicted_success = bool(prediction and prediction.valid)
                        evaluation        = self.predictor.evaluate_outcome(
                            predicted_success=predicted_success,
                            actual_success=actual_success,
                        )
                        self.curiosity.register_prediction_error(
                            result.topic, float(evaluation.prediction_error)
                        )
                        self.curiosity.register_experiment_signal(
                            result.topic, candidate_score / 10.0
                        )
                        self.experiment_planner.record_tested(
                            getattr(candidate, "rule", ""), tick
                        )
                        self.memory.add_experiment(ExperimentRecord(
                            rule=outcome.rule, name=experiment.name,
                            supported=outcome.supported, contradicted=outcome.contradicted,
                            timestamp=time.time(),
                        ))
                        self.memory.update_hypothesis_evidence(
                            outcome.rule, outcome.supported, outcome.contradicted
                        )
                        if prediction:
                            self.memory.add_prediction(PredictionRecord(
                                rule=prediction.rule, statement=prediction.statement,
                                success=evaluation.success, timestamp=time.time(),
                            ))

            # ── Betweenness precompute (every 500 ticks) ─────────────────────
            if tick % 500 == 0:
                self.role_assigner.precompute_betweenness()
                self.logger.info("[role_assigner] betweenness_refreshed tick=%d", tick)

            # ── v4: Second-order causal discovery ─────────────────────────────
            if tick % SECOND_ORDER_INTERVAL == 0:
                second_order = self.causal_discovery.discover_second_order_effects(
                    self.knowledge_graph
                )
                if second_order:
                    injected = self.causal_discovery.inject_into_graph(
                        second_order, self.knowledge_graph, current_tick=tick
                    )
                    self.logger.info(
                        "[second_order] tick=%d chains_found=%d injected=%d",
                        tick, len(second_order), injected,
                    )

            # ── v4: Stale causal record decay ──────────────────────────────────
            if tick % DECAY_INTERVAL == 0:
                self.causal_discovery.decay_stale_records(tick)

            # ── Idle-tick work (domain not in simulation set) ──────────────────
            if transition is None:
                if hypotheses:
                    weakest_candidate = hypotheses[-1]
                    idle_experiment   = self.experimenter.generate(
                        weakest_candidate, self.memory.all_entities(), self.type_system, n=2
                    )
                    idle_outcome = self.experimenter.evaluate(
                        idle_experiment, triplets, self.type_system
                    )
                    self.memory.add_experiment(ExperimentRecord(
                        rule=idle_outcome.rule, name=idle_experiment.name,
                        supported=idle_outcome.supported, contradicted=idle_outcome.contradicted,
                        timestamp=time.time(),
                    ))
                    self.memory.update_hypothesis_evidence(
                        idle_outcome.rule, idle_outcome.supported, idle_outcome.contradicted
                    )
                if result.unknown_concepts:
                    for concept in sorted(result.unknown_concepts)[:2]:
                        edges = self.expander.expand_concept_graph(concept, depth=1)
                        if edges:
                            self.logger.debug("[idle_tick] expanded concept=%s edges=%d",
                                             concept, len(edges))

            # ── new_causal_count ───────────────────────────────────────────────
            new_causal_count = 0
            if transition:
                for cf in (transition.get("causal_facts") or []):
                    record = self.causal_discovery.get_record(cf.cause, cf.effect)
                    if record is not None and record.evidence_count == 1:
                        new_causal_count += 1

            total_new_facts = result.new_facts + new_causal_count
            # A tick with zero new facts but significant reinforcement is not a
            # true stall — it means the agent is consolidating existing knowledge.
            is_stall = (total_new_facts == 0
                        and result.reinforcement < self.hypothesis_engine.REINFORCEMENT_HYPOTHESIS_THRESHOLD)
            self.stall_ticks = self.stall_ticks + 1 if is_stall else 0

            test_domain = None
            if hypotheses and len(hypotheses[0].rule.split()) >= 3:
                rel = hypotheses[0].rule.split()[1]
                rel_to_domain = {
                    "hunts": "ecosystem", "eats": "ecosystem",
                    "orbits": "astronomy", "emits": "astronomy",
                    "reacts_with": "chemistry", "produces": "chemistry",
                    "affects": "biology", "fights": "biology",
                    "causes": "physics", "attracts": "physics",
                }
                test_domain = rel_to_domain.get(rel)

            self._maybe_generate_topic(
                pages, selection_score,
                test_domain=test_domain if tick % 3 == 0 else None,
            )

            # ── Epistemic reward ───────────────────────────────────────────────
            ep_reward_obj  = self.epistemic.tick_reward(tick)
            novelty_score  = self.curiosity.score_topics([selected_topic], weak_by_topic)[0].score
            _any_novel_law = False

            stale_count = self.memory.apply_stale_decay()
            self.logger.debug("[memory] stale_penalised=%s tick=%s", stale_count, tick)

            if tick % 50 == 0:
                for rule in self.memory.get_causal_rules():
                    if rule.confidence < 0.7:
                        rule.confidence = max(0.05, rule.confidence * 0.85)

            # ── Multi-step causal chain reasoning ─────────────────────────────
            inferred_rules, chain_hypotheses = self.causal_chain_reasoner.run(
                self.knowledge_graph, self.hypothesis_engine,
            )
            if inferred_rules:
                self.logger.info("[reasoning] tick=%s inferred=%d", tick, len(inferred_rules))
            if chain_hypotheses:
                hypotheses = hypotheses + chain_hypotheses

            # ── Mechanism naming + mediator-blocking queue ─────────────────
            mechanism_matches = self.mechanism_matcher.match_all(inferred_rules)
            for rule in inferred_rules:
                _mm = mechanism_matches.get((rule.source, rule.target))
                if _mm is not None:
                    rule.mechanism_name = _mm.template.name
                    self.research_agent.queue_query(_mm.research_query, priority="high")
            _current_baseline = self.generator.state_for_domain(
                result.domain if hasattr(result, "domain") else selected_topic.split("_")[0]
            ) if hasattr(self.generator, "state_for_domain") else {}
            for rule in inferred_rules:
                mname = mechanism_matches.get((rule.source, rule.target), None)
                self.mediator_planner.maybe_queue(
                    rule, _current_baseline, current_tick=tick,
                    mechanism_name=mname.template.name if mname else "",
                )

            # ── Concept abstraction ───────────────────────────────────────────
            new_abstractions = self.abstraction_engine.run(
                self.memory.relation_triplets(), self.knowledge_graph,
                self.type_system, self.curiosity, current_tick=tick,
            )
            if new_abstractions:
                self.logger.info("[abstraction] tick=%s new=%s", tick,
                                 [c.name for c in new_abstractions])

            # ── Scientific law discovery every 25 ticks ───────────────────────
            if tick % 25 == 0:
                _confirmed = {
                    (e.source, e.target)
                    for e in self.knowledge_graph.edges
                    if e.relation in {"positive_affects", "negative_affects",
                                      "affects", "conditional_affects"}
                    and e.confidence >= 0.3
                }
                for causal_rule in self.memory.get_causal_rules():
                    if (causal_rule.cause, causal_rule.effect) not in _confirmed:
                        continue
                    law = self.law_discovery.attempt_fit(causal_rule.cause, causal_rule.effect)
                    if law:
                        law_eval = self.law_novelty_gate.evaluate(law, current_tick=tick)
                        # absorb_law runs for ALL fitted laws (updates template param stats)
                        self.theory_abstraction.absorb_law(law)
                        if law_eval.is_novel:
                            self.knowledge_graph.ingest_law(law)
                            self.law_predictor.register_law(law)
                            _any_novel_law = True
                            self.logger.info("[law_discovered] %s r2=%.3f n=%d",
                                            law.equation_str, law.r_squared, law.n_datapoints)

            # ── Theory template → guided law-fitting for unconfirmed pairs ─────
            if tick % 25 == 0:
                for _crule in self.memory.get_causal_rules():
                    if (_crule.cause, _crule.effect) in _confirmed:
                        continue  # already being fit
                    _tmpl = self.theory_abstraction.predict_template_for(
                        _crule.cause, _crule.effect
                    )
                    if _tmpl is not None:
                        _tseed = (
                            f"{_crule.cause} follows {_tmpl.name} "
                            f"pattern affecting {_crule.effect}"
                        )
                        if not any(getattr(h, "rule", "") == _tseed for h in hypotheses):
                            hypotheses.append(Hypothesis(
                                rule=_tseed,
                                concepts=[_crule.cause, _crule.effect],
                                confidence=0.45,
                                supporting_evidence=_tmpl.n_laws,
                                contradicting_evidence=0,
                            ))
                        self.logger.debug(
                            "[theory_seed] template=%s for %s→%s",
                            _tmpl.name, _crule.cause, _crule.effect,
                        )

            # ── Feedback: ingested knowledge → world simulator ────────────────
            recent_triplets = [
                (e.source, e.relation, e.target)
                for e in self.knowledge_graph.edges
                if e.provenance in ("ingestion", "abstraction")
            ]
            if recent_triplets:
                expansions = self.generator.absorb_ingested_knowledge(recent_triplets)
                if expansions:
                    self.logger.info("[world] feedback_loop tick=%s expansions=%d",
                                     tick, expansions)

            # ── v4: Hypothesis validation with simulation-first ────────────────
            causal_records_dict = {
                (r.cause, r.effect): r
                for r in self.causal_discovery.get_all_records()
            }
            validation_results = self.hypothesis_validator.maybe_validate(
                tick, hypotheses, self.memory,
                causal_records=causal_records_dict,   # v4: pass internal evidence
            )
            for vr in validation_results:
                self.logger.info(
                    "[validator] tick=%s rule=%r verdict=%s delta=%+.2f source=%s",
                    tick, vr.hypothesis_rule[:55], vr.verdict.value,
                    vr.confidence_delta, vr.source,
                )

            # ── Sync, consolidate, persist ────────────────────────────────────
            self._sync_knowledge_graph()

            consolidation_stats = self.memory_consolidator.run(
                self.knowledge_graph, current_tick=tick
            )
            if consolidation_stats:
                self.logger.info(
                    "[consolidation] tick=%s promoted=%d forgotten=%d evicted=%d spec_pruned=%d graph=%d",
                    tick, consolidation_stats.promoted, consolidation_stats.forgotten,
                    consolidation_stats.evicted,
                    getattr(consolidation_stats, "speculative_pruned", 0),
                    len(self.knowledge_graph.edges),
                )

            self.memory.save()
            self.knowledge_graph.save()
            self.law_discovery.save(self._law_discovery_path)

            # ── v3/v4: Epistemic reward ────────────────────────────────────────
            _mediator_found = bool(
                transition and any(
                    bool(self.causal_discovery.get_record(cf.cause, cf.effect) and
                         self.causal_discovery.get_record(cf.cause, cf.effect).conditions)
                    for cf in (transition.get("causal_facts") or [])
                )
            )

            ep_ig_val = ep_reward_obj.entropy_reduced
            ep_reward_score = self.curiosity.epistemic_reward(
                entropy_reduced    = ep_ig_val,
                new_edges          = self.knowledge_graph.tick_stats.new_edges,
                validation_results = validation_results,
                law_novel          = _any_novel_law,
                mediator_found     = _mediator_found,
            )
            self._last_ep_ig_by_domain[domain] = ep_ig_val

            # ── Cognitive summary ──────────────────────────────────────────────
            action_str_log = transition["action"] if transition else "none"
            pred_err       = transition["prediction_error"] if transition else 0.0
            kg             = self.knowledge_graph

            self.logger.info(
                "[tick=%s] topic=%s domain=%s action=%s "
                "pred_err=%.3f new_facts=%d reinf=%d hypotheses=%d "
                "memory=%d reward=%.4f "
                "graph nodes=%d edges=%d new=%d updated=%d"
                "%s%s%s%s%s%s%s%s%s",
                tick, selected_topic, domain, action_str_log,
                pred_err, total_new_facts, result.reinforcement, len(hypotheses),
                len(self.memory.facts), ep_reward_score,
                len(kg.nodes), len(kg.edges),
                kg.tick_stats.new_edges, kg.tick_stats.updated_edges,
                f"  causal={new_causal_rules[0]}" if new_causal_rules else "",
                f"  abstractions={len(new_abstractions)}" if new_abstractions else "",
                f"  inferred={len(inferred_rules)}" if inferred_rules else "",
                f"  theories={self.theory_engine.theory_count()}" if self.theory_engine.theory_count() > 0 else "",
                f"  validated={len(validation_results)}({sum(1 for v in validation_results if v.verdict.value == 'supported')}✓)" if validation_results else "",
                f"  ep_ig={ep_ig_val:.3f}" if ep_ig_val > 0.001 else "",
                f"  mediator=1" if _mediator_found else "",
                f"  new_var={new_synth_var.name}" if new_synth_var else "",
                f"  perception={_perc_new}" if _perc_new else "",
            )

            if max_ticks is not None and tick >= max_ticks:
                self.logger.info("Reached max_ticks=%s; stopping.", max_ticks)
                break

            # ── Tools bridge: record tick state + persist discoveries ──────────
            if _TOOLS_BRIDGE_AVAILABLE:
                record_tick(
                    tick=tick, topic=selected_topic, domain=domain,
                    action=action_str_log, new_facts=total_new_facts,
                    hypotheses=len(hypotheses), reward=ep_reward_score,
                    causal_rules=new_causal_rules if new_causal_rules else None,
                )
                maybe_record_discoveries(
                    hypotheses=hypotheses,
                    causal_rules=self.memory.get_causal_rules(),
                    tick=tick,
                )

            time.sleep(self.tick_sleep_seconds)