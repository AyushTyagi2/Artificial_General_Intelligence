"""Main life-cycle loop for the digital baby agent."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
import time

from digital_baby.brain.concepts import ConceptHierarchy
from digital_baby.brain.curiosity import CuriosityModel
from digital_baby.brain.learner import Learner
from digital_baby.brain.memory import Memory, PatternRecord
from digital_baby.brain.patterns import PatternDiscoverer
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
    ) -> None:
        self.world_path = Path(world_path)
        self.memory = Memory(memory_path)
        self.reasoner = Reasoner()
        self.learner = Learner(self.memory, self.reasoner)
        self.curiosity = CuriosityModel()
        self.questions = QuestionGenerator()
        self.patterns = PatternDiscoverer()
        self.concepts = ConceptHierarchy()
        self.generator = KnowledgeGenerator(self.world_path)
        self.tick_sleep_seconds = tick_sleep_seconds
        self.logger = logging.getLogger(self.__class__.__name__)
        self.stall_ticks = 0

    def _load_pages(self) -> List[Dict]:
        """Auto-discover all JSON pages in world directory."""
        pages = []
        for file_path in sorted(self.world_path.glob("*.json")):
            page = json.loads(file_path.read_text(encoding="utf-8"))
            pages.append(page)
        return pages

    @staticmethod
    def _page_keywords(page: Dict) -> List[str]:
        words = {page.get("topic", "").lower()}
        for fact in page.get("facts", []):
            for token in fact.lower().split():
                words.add(token.strip())
        return sorted(words)

    def _find_topic_for_concept(self, concept: str, pages: List[Dict]) -> Optional[Tuple[str, str]]:
        """Find a topic tied to a concept by direct/fuzzy matching."""
        lowered = concept.lower().strip()
        topic_names = {page["topic"]: page for page in pages}

        if lowered in topic_names:
            return lowered, f"goal_direct_match:{lowered}"

        for page in pages:
            keywords = self._page_keywords(page)
            if lowered in keywords or any(lowered in keyword for keyword in keywords):
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
                min_visits = min(self.curiosity.topic_visits[t] for t in topics)
                if self.curiosity.topic_visits[topic] <= min_visits + 1:
                    page = next(p for p in pages if p["topic"] == topic)
                    goal_score = self.curiosity.score_topics([topic], weak_by_topic)[0].score
                    return page, reason, goal_score
            goal = self.curiosity.pop_goal_concept()

        topic_scores = self.curiosity.score_topics(topics, weak_by_topic)
        selected = topic_scores[0]
        page = next(p for p in pages if p["topic"] == selected.topic)
        return page, f"curiosity:{selected.reason}", selected.score

    def _maybe_generate_topic(self, pages: List[Dict], best_score: float) -> Optional[Dict]:
        """Generate a fresh topic when exploration stalls or novelty gets too low."""
        if self.stall_ticks >= 2 or best_score < 0.2:
            page = self.generator.generate_topic(persist=True)
            pages.append(page)
            self.logger.info("generated new topic=%s domain=%s", page["topic"], page.get("domain"))
            self.stall_ticks = 0
            return page
        return None

    def run(self, max_ticks: Optional[int] = None) -> None:
        """Run the agent indefinitely unless max_ticks is provided."""
        tick = 0
        while True:
            tick += 1
            pages = self._load_pages()
            if not pages:
                self.logger.warning("No knowledge pages found in %s; generating one.", self.world_path)
                pages = [self.generator.generate_topic(persist=True)]

            topics = [page["topic"] for page in pages]
            weak_by_topic = {
                topic: len(
                    [
                        record
                        for record in self.memory.facts.values()
                        if record.source_topic == topic and record.confidence < 0.5
                    ]
                )
                / max(1, len([r for r in self.memory.facts.values() if r.source_topic == topic]))
                for topic in topics
            }

            selected_page, selection_reason, selection_score = self._select_topic(pages, weak_by_topic)
            maybe_generated = self._maybe_generate_topic(pages, selection_score)
            if maybe_generated is not None:
                selected_page = maybe_generated
                selection_reason = "procedural_generation:stalled_or_low_novelty"

            selected_topic = selected_page["topic"]
            self.logger.info("[tick=%s] selected_topic=%s reason=%s", tick, selected_topic, selection_reason)

            prev_pattern_count = len(self.memory.patterns)
            result = self.learner.learn_from_page(selected_page)

            if result.new_facts == 0:
                self.stall_ticks += 1
            else:
                self.stall_ticks = 0

            # Structural novelty from entities, relation types, and fresh facts.
            structural_novelty = (
                (len(result.new_entities) * 0.12)
                + (len(result.new_relation_types) * 0.2)
                + (result.new_facts * 0.08)
            )
            self.curiosity.register_structural_novelty(result.topic, structural_novelty)

            self.curiosity.register_unknowns(result.topic, result.unknown_concepts)
            self.curiosity.register_unknowns(result.topic, result.unknown_relations)
            self.curiosity.mark_visited(result.topic)

            conflicts = self.memory.conflicting_relations_with_evidence()
            conflict_bonus = 0.0
            if conflicts:
                for entity, relation, ranked_values in conflicts:
                    self.logger.info("[tick=%s] conflict entity=%s relation=%s ranked_values=%s", tick, entity, relation, ranked_values)
                    concepts = [entity, relation] + [value for value, _evidence in ranked_values]
                    self.curiosity.register_unknowns(result.topic, concepts)
                    if len(ranked_values) > 1:
                        delta = abs(ranked_values[0][1] - ranked_values[1][1])
                        self.curiosity.register_prediction_error(result.topic, 1.0 / (1 + delta))
                conflict_bonus = len(conflicts) * 0.3

            generated_questions = self.questions.from_unknown_concepts(result.unknown_concepts)
            generated_questions.extend(self.questions.from_unknown_concepts(result.unknown_relations))
            generated_questions.extend(
                self.questions.from_conflicts([(e, r, [v for v, _ in vals]) for e, r, vals in conflicts])
            )
            self.curiosity.register_unknowns(result.topic, [q.target_concept for q in generated_questions])

            # Pattern discovery + hierarchy updates.
            triplets = self.memory.relation_triplets()
            discovered_rules = self.patterns.discover(triplets, min_support=2)
            pattern_delta = max(0, len(discovered_rules) - prev_pattern_count)
            if pattern_delta:
                self.curiosity.register_structural_novelty(result.topic, pattern_delta * 0.35)

            self.memory.update_patterns(
                [
                    PatternRecord(
                        template=rule.template,
                        relation=rule.relation,
                        support=rule.count,
                        label=rule.label,
                        timestamp=time.time(),
                    )
                    for rule in discovered_rules
                ]
            )
            self.concepts.ingest_triplets(triplets)

            # Memory compression for repeated relation structures.
            compressed = self.memory.compress_relation_facts("hunts", min_objects=2)

            novelty_score = self.curiosity.score_topics([selected_topic], weak_by_topic)[0].score
            curiosity_reward = self.curiosity.reward(novelty=novelty_score, conflict_bonus=conflict_bonus)

            self.memory.decay_confidence(decay_rate=0.005)
            self.memory.save()

            self.logger.info(
                "[tick=%s] learned=%s new_facts=%s unknown=%s new_entities=%s new_relations=%s patterns=%s compressed=%s memory=%s reward=%.2f",
                tick,
                result.learned_facts,
                result.new_facts,
                sorted(result.unknown_concepts),
                len(result.new_entities),
                len(result.new_relation_types),
                len(discovered_rules),
                len(compressed),
                len(self.memory.facts),
                curiosity_reward,
            )

            if max_ticks is not None and tick >= max_ticks:
                self.logger.info("Reached max_ticks=%s; stopping loop.", max_ticks)
                break

            time.sleep(self.tick_sleep_seconds)
