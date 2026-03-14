"""Main life-cycle loop for the digital baby agent."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
import time

from digital_baby.brain.concepts import ConceptHierarchy, ConceptTypeSystem
from digital_baby.brain.curiosity import CuriosityModel
from digital_baby.brain.experimenter import Experimenter
from digital_baby.brain.hypothesis import Hypothesis, HypothesisEngine
from digital_baby.brain.knowledge_expander import KnowledgeExpander
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
        self.generator = KnowledgeGenerator(self.world_path)
        self.tick_sleep_seconds = tick_sleep_seconds
        self.novelty_interval = max(1, novelty_interval)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stall_ticks = 0

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

    def _select_topic(self, pages: List[Dict], weak_by_topic: Dict[str, float]) -> Tuple[Dict, str, float]:
        topics = [page["topic"] for page in pages]
        goal = self.curiosity.pop_goal_concept()
        while goal is not None:
            matched = self._find_topic_for_concept(goal, pages)
            if matched:
                topic, reason = matched
                min_visits = min(self.curiosity.topic_visits[t] for t in topics) if topics else 0
                if self.curiosity.topic_visits[topic] <= min_visits + 1:
                    page = self._safe_topic_lookup(topic, pages)
                    score = self.curiosity.score_topics([page["topic"]], weak_by_topic)[0].score
                    return page, reason, score
            goal = self.curiosity.pop_goal_concept()

        topic_scores = self.curiosity.score_topics(topics, weak_by_topic)
        selected = topic_scores[0]
        page = self._safe_topic_lookup(selected.topic, pages)
        return page, f"curiosity:{selected.reason}", selected.score

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
                path = " -> ".join([edge[0] for edge in edges[:1]] + [edge[2] for edge in edges[:2]])
                self.logger.info("[world] expanded=%s", path)

    def _curiosity_rank_hypotheses(self, topic: str, hypotheses: List[Hypothesis]) -> List[Tuple[Hypothesis, float]]:
        ranked: List[Tuple[Hypothesis, float]] = []
        prediction_error = self.curiosity.prediction_error_by_topic.get(topic, 0.0)
        for hypothesis in hypotheses:
            rel = hypothesis.rule.split()[1] if len(hypothesis.rule.split()) >= 3 else "unknown"
            unexplored = [c for c in hypothesis.concepts if not self.type_system.is_known(c)]
            score = self.curiosity.curiosity_score(
                unknown_concepts=unexplored,
                prediction_error=prediction_error,
                unexplored_concepts=unexplored,
                new_relation=rel not in {"is", "eats", "hunts", "orbits", "reacts_with", "contains"},
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
            novelty_page = {
                "topic": "open_world_novelty",
                "domain": "open_world",
                "facts": novelty_facts,
                "generated": True,
            }
            result = self.learner.learn_from_page(novelty_page)
            self.curiosity.register_unknowns(result.topic, result.unknown_concepts)
            self.curiosity.register_unknowns(result.topic, result.unknown_relations)

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

            selected_page, selection_reason, selection_score = self._select_topic(pages, weak_by_topic)
            selected_topic = selected_page["topic"]
            self.logger.info("[tick=%s] selected_topic=%s reason=%s", tick, selected_topic, selection_reason)

            result = self.learner.learn_from_page(selected_page)
            self._expand_unknown_concepts(sorted(result.unknown_concepts))
            self.type_system.registry.refresh()

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
            self.type_system.infer_from_triplets(triplets)
            discovered = self.patterns.discover(triplets, min_support=2)
            learned_rules = self.learner.infer_general_rules(triplets)
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
            uncertainty = self.memory.hypothesis_uncertainty()
            self.curiosity.register_hypothesis_uncertainty(result.topic, uncertainty)

            ranked_hypotheses = self._curiosity_rank_hypotheses(result.topic, hypotheses[:10])
            if ranked_hypotheses:
                candidate, candidate_score = ranked_hypotheses[0]
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

                    self.memory.add_experiment(
                        ExperimentRecord(
                            rule=outcome.rule,
                            name=experiment.name,
                            supported=outcome.supported,
                            contradicted=outcome.contradicted,
                            timestamp=time.time(),
                        )
                    )
                    self.memory.update_hypothesis_evidence(outcome.rule, outcome.supported, outcome.contradicted)
                    if prediction:
                        self.memory.add_prediction(
                            PredictionRecord(
                                rule=prediction.rule,
                                statement=prediction.statement,
                                success=evaluation.success,
                                timestamp=time.time(),
                            )
                        )

            test_domain = None
            if hypotheses:
                rel = hypotheses[0].rule.split()[1]
                rel_to_domain = {"hunts": "ecosystem", "orbits": "astronomy", "reacts_with": "chemistry"}
                test_domain = rel_to_domain.get(rel)

            maybe_generated = self._maybe_generate_topic(pages, selection_score, test_domain=test_domain if tick % 5 == 0 else None)
            if maybe_generated is not None and tick % 5 == 0 and hypotheses:
                self.logger.info("[tick=%s] hypothesis_test_domain=%s", tick, maybe_generated.get("domain"))

            novelty_score = self.curiosity.score_topics([selected_topic], weak_by_topic)[0].score
            curiosity_reward = self.curiosity.reward(novelty=novelty_score)

            self.memory.decay_confidence(decay_rate=0.005)
            self.memory.save()

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
