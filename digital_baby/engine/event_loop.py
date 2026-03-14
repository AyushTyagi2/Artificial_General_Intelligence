"""Main life-cycle loop for the digital baby agent."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
import math
import time

from digital_baby.brain.concepts import ConceptHierarchy, ConceptTypeSystem
from digital_baby.brain.curiosity import CuriosityModel
from digital_baby.brain.experimenter import Experimenter
from digital_baby.brain.hypothesis import Hypothesis, HypothesisEngine
from digital_baby.brain.knowledge_expander import KnowledgeExpander
from digital_baby.brain.knowledge_graph import KnowledgeGraph
from digital_baby.brain.learner import Learner
from digital_baby.brain.memory import ExperimentRecord, HypothesisRecord, Memory, PatternRecord, PredictionRecord
from digital_baby.brain.patterns import PatternDiscoverer
from digital_baby.brain.predictor import Predictor
from digital_baby.brain.questions import QuestionGenerator
from digital_baby.brain.reasoning import Reasoner
from digital_baby.world.generator import KnowledgeGenerator


class BabyEventLoop:
    """Tick-based continuous loop that observes, learns, and updates beliefs."""

    def __init__(
        self,
        world_path: str | Path,
        memory_path: str | Path,
        tick_sleep_seconds: float = 1.0,
        novelty_interval: int = 20,
    ) -> None:
        self.world_path = Path(world_path)
        self.memory = Memory(memory_path)
        self.reasoner = Reasoner()
        self.learner = Learner(self.memory, self.reasoner)
        self.curiosity = CuriosityModel()
        self.questions = QuestionGenerator()
        self.patterns = PatternDiscoverer()
        self.hypothesis_engine = HypothesisEngine()
        self.predictor = Predictor()
        self.experimenter = Experimenter()
        self.concepts = ConceptHierarchy()
        self.type_system = ConceptTypeSystem()
        self.expander = KnowledgeExpander(self.world_path)
        self.knowledge_graph = KnowledgeGraph(self.world_path.parent / "knowledge_graph.json")
        self.generator = KnowledgeGenerator(self.world_path)
        self.tick_sleep_seconds = tick_sleep_seconds
        self.novelty_interval = max(1, novelty_interval)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stall_ticks = 0
        self.state_transition_history: List[Dict[str, float]] = []
        self.last_selected_topic: Optional[str] = None
        self.consecutive_topic_count = 0
        self.last_forced_diversity_tick = 0

    def _load_pages(self) -> List[Dict]:
        pages = []
        for file_path in sorted(self.world_path.glob("*.json")):
            pages.append(json.loads(file_path.read_text(encoding="utf-8")))
        return pages

    def _safe_topic_lookup(self, topic: str, pages: List[Dict]) -> Dict:
        page = next((p for p in pages if p["topic"] == topic), None)
        if page is None:
            self.logger.warning("topic_missing: %s", topic)
            domain = topic.split("_")[0]
            page = self.generator.generate_topic(persist=True, domain=domain)
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

    def _balanced_topic_scores(self, topics: List[str], weak_by_topic: Dict[str, float]) -> List[Tuple[str, float, str]]:
        curiosity_signals = {s.topic: s for s in self.curiosity.score_topics(topics, weak_by_topic)}
        scored: List[Tuple[str, float, str]] = []
        for topic in topics:
            base_signal = curiosity_signals.get(topic)
            base_score = base_signal.score if base_signal else 0.0
            visits = self.curiosity.topic_visits[topic]
            visit_penalty = math.log(visits + 1)
            exploration_bonus = 1.0 / math.sqrt(visits + 1)
            score = base_score + exploration_bonus - visit_penalty
            reason = (
                f"base={base_score:.2f}, exploration_bonus={exploration_bonus:.2f}, "
                f"visit_penalty={visit_penalty:.2f}, visits={visits}"
            )
            self.logger.info("[exploration] domain=%s visits=%s score=%.2f", topic, visits, score)
            scored.append((topic, score, reason))
        return sorted(scored, key=lambda item: item[1], reverse=True)

    def _record_topic_selection(self, topic: str) -> None:
        if topic == self.last_selected_topic:
            self.consecutive_topic_count += 1
        else:
            self.last_selected_topic = topic
            self.consecutive_topic_count = 1

    def _select_topic(self, pages: List[Dict], weak_by_topic: Dict[str, float], tick: int) -> Tuple[Dict, str, float]:
        topics = [page["topic"] for page in pages]
        goal = self.curiosity.pop_goal_concept()
        while goal is not None:
            matched = self._find_topic_for_concept(goal, pages)
            if matched:
                topic, reason = matched
                min_visits = min(self.curiosity.topic_visits[t] for t in topics) if topics else 0
                if self.curiosity.topic_visits[topic] <= min_visits + 1:
                    balanced = self._balanced_topic_scores(topics, weak_by_topic)
                    selected_topic = topic
                    selected_reason = reason
                    selected_score = next((s for t, s, _r in balanced if t == topic), 0.0)

                    if self.consecutive_topic_count > 10 and self.last_selected_topic == selected_topic:
                        alternatives = [item for item in balanced if item[0] != selected_topic]
                        if alternatives:
                            selected_topic, selected_score, alt_reason = alternatives[0]
                            self.last_forced_diversity_tick = tick
                            selected_reason = f"forced_after_streak:{alt_reason}"

                    if tick - self.last_forced_diversity_tick >= 20 and self.last_selected_topic is not None:
                        alternatives = [item for item in balanced if item[0] != self.last_selected_topic]
                        if alternatives:
                            selected_topic, selected_score, alt_reason = alternatives[0]
                            self.last_forced_diversity_tick = tick
                            selected_reason = f"forced_every_20_ticks:{alt_reason}"

                    page = self._safe_topic_lookup(selected_topic, pages)
                    return page, selected_reason, selected_score
            goal = self.curiosity.pop_goal_concept()

        balanced = self._balanced_topic_scores(topics, weak_by_topic)
        if not balanced:
            page = self._safe_topic_lookup(topics[0], pages)
            return page, "fallback:first_topic", 0.0

        selected_topic, selected_score, selected_reason = balanced[0]

        # Hard lock-in prevention: if same domain chosen >10 times, force a different one.
        if self.consecutive_topic_count > 10 and self.last_selected_topic == selected_topic:
            alternatives = [item for item in balanced if item[0] != selected_topic]
            if alternatives:
                selected_topic, selected_score, selected_reason = alternatives[0]
                self.last_forced_diversity_tick = tick
                selected_reason = f"forced_after_streak:{selected_reason}"

        # Diversity guarantee: force a different topic at least every 20 ticks.
        if tick - self.last_forced_diversity_tick >= 20 and self.last_selected_topic is not None:
            alternatives = [item for item in balanced if item[0] != self.last_selected_topic]
            if alternatives:
                selected_topic, selected_score, selected_reason = alternatives[0]
                self.last_forced_diversity_tick = tick
                selected_reason = f"forced_every_20_ticks:{selected_reason}"

        page = self._safe_topic_lookup(selected_topic, pages)
        return page, f"balanced:{selected_reason}", selected_score

    def _maybe_generate_topic(self, pages: List[Dict], best_score: float, test_domain: str | None = None) -> Optional[Dict]:
        if self.stall_ticks >= 2 or best_score < 0.2 or test_domain is not None:
            page = self.generator.generate_topic(persist=True, domain=test_domain)
            pages.append(page)
            self.logger.info("generated domain=%s generation=%s", page.get("domain"), page.get("generation"))
            self.stall_ticks = 0
            return page
        return None

    def _update_hypotheses(self, discovered_rules: List[PatternRecord], triplets: List[Tuple[str, str, str]]) -> List[Hypothesis]:
        from digital_baby.brain.patterns import PatternRule

        pattern_rules = [PatternRule(relation=r.relation, count=r.support, label=r.label, template=r.template) for r in discovered_rules]
        hypotheses = self.hypothesis_engine.from_patterns(pattern_rules, self.type_system, triplets)
        self.memory.set_world_model_rules(
            [
                HypothesisRecord(
                    rule=h.rule,
                    concepts=h.concepts,
                    confidence=h.confidence,
                    supporting_evidence=h.supporting_evidence,
                    contradicting_evidence=h.contradicting_evidence,
                )
                for h in hypotheses
            ]
        )
        return hypotheses

    def _expand_unknown_concepts(self, concepts: List[str]) -> None:
        for concept in concepts:
            if self.type_system.is_known(concept):
                continue
            edges = self.expander.expand_concept_graph(concept, depth=2)
            if edges:
                path = " -> ".join([edges[0][0]] + [edge[2] for edge in edges[:2]])
                self.logger.info("[world] expanded=%s", path)

    def _curiosity_rank_hypotheses(self, topic: str, hypotheses: List[Hypothesis]) -> List[Tuple[Hypothesis, float]]:
        ranked: List[Tuple[Hypothesis, float]] = []
        prediction_error = self.curiosity.prediction_error_by_topic.get(topic, 0.0)
        for hypothesis in hypotheses:
            tokens = hypothesis.rule.split()
            rel = tokens[1] if len(tokens) >= 3 else "unknown"
            is_causal = "if" in hypothesis.rule or "affects" in hypothesis.rule or "controls" in hypothesis.rule
            unexplored = [c for c in hypothesis.concepts if not self.type_system.is_known(c)]
            score = self.curiosity.curiosity_score(
                unknown_concepts=unexplored,
                prediction_error=prediction_error,
                unexplored_concepts=unexplored,
                new_relation=rel not in {"is", "eats", "hunts", "orbits", "reacts_with", "contains"},
                new_pattern=is_causal,
            )
            ranked.append((hypothesis, score + (1.0 - hypothesis.confidence)))
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def _inject_novelty(self, tick: int) -> None:
        if tick % self.novelty_interval != 0:
            return
        novelty_facts, new_concepts = self.generator.inject_open_world_novelty(max_entities=3)
        self.logger.info("[world] novelty_injection=true tick=%s count=%s", tick, len(new_concepts))
        for concept in new_concepts:
            self.logger.info("[world] new_concept=%s", concept)
        for fact in novelty_facts:
            self.logger.info("[world] relation_added=%s", fact)

        if novelty_facts:
            novelty_page = {"topic": "open_world_novelty", "domain": "open_world", "facts": novelty_facts, "generated": True}
            result = self.learner.learn_from_page(novelty_page)
            self.curiosity.register_unknowns(result.topic, result.unknown_concepts)
            self.curiosity.register_unknowns(result.topic, result.unknown_relations)

    @staticmethod
    def _preferred_action_for_hypothesis(hypothesis: Hypothesis, domain: str) -> Optional[str]:
        rule = hypothesis.rule.lower()
        if domain == "chemistry" and "temperature" in rule:
            return "increase_temperature"
        if domain == "chemistry" and "reactants" in rule:
            return "add_chemical"
        if domain == "ecosystem" and "wolves" in rule and ("decrease" in rule or "will decrease" in rule):
            return "remove_predator"
        if domain == "ecosystem" and "wolves" in rule and ("increase" in rule or "will increase" in rule):
            return "add_predator"
        if domain == "ecosystem" and "deer" in rule and "increase" in rule:
            return "remove_predator"
        return None

    def _sync_knowledge_graph(self) -> None:
        """Sync observed relations and causal rules into persistent graph."""
        self.knowledge_graph.ingest_triplets(self.memory.relation_triplets(), default_confidence=0.58)
        for rule in self.memory.get_causal_rules():
            self.knowledge_graph.ingest_causal_rule(
                cause=rule.cause,
                effect=rule.effect,
                direction=rule.direction,
                confidence=rule.confidence,
                evidence=rule.observations,
            )

    def _run_stateful_world_step(self, domain: str, preferred_action: Optional[str] = None, target: Optional[str] = None) -> Dict[str, float]:
        state = self.generator.state_for_domain(domain)
        action = self.generator.choose_action(domain, preferred_action=preferred_action)
        if target:
            self.logger.info("[hypothesis_test] action=%s target=%s", action, target)
        self.logger.info("[world] action=%s domain=%s", action, domain)
        self.logger.info("[world] state_before=%s", state)

        prediction = self.predictor.predict_state_transition(domain, action, state)
        new_state, deltas = self.experimenter.apply_environment_dynamics(domain, state, action)
        self.generator.update_domain_state(domain, new_state)
        self.logger.info("[world] state_after=%s", new_state)
        for key, delta in sorted(deltas.items()):
            if abs(delta) > 0:
                self.logger.info("[world] state_change %s=%+.2f", key, delta)

        eval_result = self.predictor.evaluate_state_prediction(prediction, deltas)
        self.logger.info("[curiosity] prediction_error=%.3f", eval_result.prediction_error)

        transition_facts = [f"{k}_delta is {int(v) if abs(v-int(v)) < 1e-8 else round(v,2)}" for k, v in deltas.items() if abs(v) > 0]
        if transition_facts:
            self.learner.learn_from_page({"topic": f"{domain}_state", "facts": transition_facts})

        self.state_transition_history.append(deltas)
        self.state_transition_history = self.state_transition_history[-120:]
        return {"prediction_error": eval_result.prediction_error, "deltas": deltas, "action": action}

    def run(self, max_ticks: Optional[int] = None) -> None:
        tick = 0
        while True:
            tick += 1
            pages = self._load_pages()
            if not pages:
                self.logger.warning("No knowledge pages found in %s; generating one.", self.world_path)
                pages = [self.generator.generate_topic(persist=True)]

            self._inject_novelty(tick)

            topics = [page["topic"] for page in pages]
            weak_by_topic = {
                topic: len([r for r in self.memory.facts.values() if r.source_topic == topic and r.confidence < 0.5])
                / max(1, len([r for r in self.memory.facts.values() if r.source_topic == topic]))
                for topic in topics
            }

            selected_page, selection_reason, selection_score = self._select_topic(pages, weak_by_topic, tick)
            selected_topic = selected_page["topic"]
            self._record_topic_selection(selected_topic)
            self.logger.info("[tick=%s] selected_topic=%s reason=%s", tick, selected_topic, selection_reason)

            result = self.learner.learn_from_page(selected_page)
            self._expand_unknown_concepts(sorted(result.unknown_concepts))
            self.type_system.registry.refresh()

            domain = selected_page.get("domain") or selected_topic
            transition: Optional[Dict[str, float]] = None

            self.stall_ticks = self.stall_ticks + 1 if result.new_facts == 0 else 0
            structural_novelty = (len(result.new_entities) * 0.12) + (len(result.new_relation_types) * 0.2) + (result.new_facts * 0.08)
            self.curiosity.register_structural_novelty(result.topic, structural_novelty)
            self.curiosity.register_unknowns(result.topic, result.unknown_concepts)
            self.curiosity.register_unknowns(result.topic, result.unknown_relations)
            self.curiosity.mark_visited(result.topic)

            generated_questions = self.questions.from_unknown_concepts(result.unknown_concepts)
            generated_questions.extend(self.questions.from_unknown_concepts(result.unknown_relations))
            self.curiosity.register_unknowns(result.topic, [q.target_concept for q in generated_questions])

            triplets = self.memory.relation_triplets()
            self.knowledge_graph.ingest_triplets(triplets, default_confidence=0.58)
            self.type_system.infer_from_triplets(triplets)
            discovered = self.patterns.discover(triplets, min_support=2)
            learned_rules = self.learner.infer_general_rules(triplets)
            causal_candidates = self.learner.discover_causal_candidates(self.state_transition_history)
            for candidate in causal_candidates:
                record, created = self.memory.upsert_causal_rule(
                    cause=candidate.cause,
                    effect=candidate.effect,
                    direction=candidate.direction,
                    observations_increment=1,
                )
                relation_label = f"{record.cause} {record.direction}_affects {record.effect}"
                if created:
                    self.logger.info("[causal_rule_discovered] %s confidence=%.2f", relation_label, record.confidence)
                else:
                    self.logger.info("[causal_rule_updated] %s confidence=%.2f", relation_label, record.confidence)
                self.curiosity.register_new_pattern(result.topic)
                self.knowledge_graph.ingest_causal_rule(record.cause, record.effect, record.direction, record.confidence, record.observations)
                discovered.append(PatternRecord(template=relation_label, relation="affects", support=max(1, record.observations), label="causal_rule", timestamp=time.time()))

            for rule in learned_rules:
                discovered.append(PatternRecord(template=rule, relation="eats", support=2, label="learner_inferred", timestamp=time.time()))

            normalized_patterns: List[PatternRecord] = []
            for item in discovered:
                if isinstance(item, PatternRecord):
                    normalized_patterns.append(item)
                else:
                    normalized_patterns.append(PatternRecord(template=item.template, relation=item.relation, support=item.count, label=item.label, timestamp=time.time()))
            self.memory.update_patterns(normalized_patterns)
            self.concepts.ingest_triplets(triplets)

            hypotheses = self._update_hypotheses(self.memory.patterns, triplets)
            causal_hypotheses = self.hypothesis_engine.from_causal_rules(self.memory.get_causal_rules())
            for hypothesis in causal_hypotheses:
                self.logger.info("[hypothesis_generated] %s", hypothesis.rule)
            hypotheses = hypotheses + causal_hypotheses
            uncertainty = self.memory.hypothesis_uncertainty()
            self.curiosity.register_hypothesis_uncertainty(result.topic, uncertainty)

            ranked_hypotheses = self._curiosity_rank_hypotheses(result.topic, hypotheses[:12])
            if ranked_hypotheses:
                candidate, candidate_score = ranked_hypotheses[0]
                preferred_action = self._preferred_action_for_hypothesis(candidate, domain)
                target_concept = candidate.concepts[0] if candidate.concepts else None
                if domain in {"ecosystem", "astronomy", "chemistry", "technology"}:
                    transition = self._run_stateful_world_step(domain, preferred_action=preferred_action, target=target_concept)
                    self.curiosity.register_prediction_error(result.topic, transition["prediction_error"])

                predictions = self.predictor.predict([candidate], self.memory.all_entities(), self.type_system)
                prediction = predictions[0] if predictions else None
                if self.experimenter.should_schedule(candidate, self.curiosity.prediction_error_by_topic.get(result.topic, 0.0)):
                    experiment = self.experimenter.generate(candidate, self.memory.all_entities(), self.type_system, n=3)
                    outcome = self.experimenter.evaluate(experiment, triplets, self.type_system)
                    actual_success = outcome.supported >= outcome.contradicted
                    predicted_success = bool(prediction and prediction.valid)
                    evaluation = self.predictor.evaluate_outcome(predicted_success=predicted_success, actual_success=actual_success)

                    self.curiosity.register_prediction_error(result.topic, float(evaluation.prediction_error))
                    self.curiosity.register_experiment_signal(result.topic, candidate_score / 10.0)

                    self.memory.add_experiment(ExperimentRecord(rule=outcome.rule, name=experiment.name, supported=outcome.supported, contradicted=outcome.contradicted, timestamp=time.time()))
                    self.memory.update_hypothesis_evidence(outcome.rule, outcome.supported, outcome.contradicted)
                    if prediction:
                        self.memory.add_prediction(PredictionRecord(rule=prediction.rule, statement=prediction.statement, success=evaluation.success, timestamp=time.time()))

            if transition is None and domain in {"ecosystem", "astronomy", "chemistry", "technology"}:
                transition = self._run_stateful_world_step(domain)
                self.curiosity.register_prediction_error(result.topic, transition["prediction_error"])

            test_domain = None
            if hypotheses and len(hypotheses[0].rule.split()) >= 3:
                rel = hypotheses[0].rule.split()[1]
                rel_to_domain = {"hunts": "ecosystem", "orbits": "astronomy", "reacts_with": "chemistry"}
                test_domain = rel_to_domain.get(rel)

            maybe_generated = self._maybe_generate_topic(pages, selection_score, test_domain=test_domain if tick % 5 == 0 else None)
            if maybe_generated is not None and tick % 5 == 0 and hypotheses:
                self.logger.info("[tick=%s] hypothesis_test_domain=%s", tick, maybe_generated.get("domain"))

            novelty_score = self.curiosity.score_topics([selected_topic], weak_by_topic)[0].score
            curiosity_reward = self.curiosity.reward(novelty=novelty_score)

            self.memory.decay_confidence(decay_rate=0.005)
            self._sync_knowledge_graph()
            self.memory.save()
            self.knowledge_graph.save()

            self.logger.info(
                "[tick=%s] learned=%s new_facts=%s unknown=%s patterns=%s hypotheses=%s memory=%s unique_facts=%s reward=%.2f",
                tick,
                result.learned_facts,
                result.new_facts,
                sorted(result.unknown_concepts),
                len(self.memory.patterns),
                len(self.memory.world_model.get("rules", [])),
                len(self.memory.facts),
                len(self.memory.fact_index),
                curiosity_reward,
            )

            if max_ticks is not None and tick >= max_ticks:
                self.logger.info("Reached max_ticks=%s; stopping loop.", max_ticks)
                break

            time.sleep(self.tick_sleep_seconds)
