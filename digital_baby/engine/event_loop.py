"""Main life-cycle loop for the digital baby agent."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
import time

from digital_baby.brain.curiosity import CuriosityModel
from digital_baby.brain.learner import Learner
from digital_baby.brain.memory import Memory
from digital_baby.brain.questions import QuestionGenerator
from digital_baby.brain.reasoning import Reasoner


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
        self.tick_sleep_seconds = tick_sleep_seconds
        self.logger = logging.getLogger(self.__class__.__name__)

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

        # 1) Direct topic name match.
        if lowered in topic_names:
            return lowered, f"goal_direct_match:{lowered}"

        # 2) Fuzzy keyword match over page topic/facts.
        for page in pages:
            keywords = self._page_keywords(page)
            if lowered in keywords or any(lowered in keyword for keyword in keywords):
                return page["topic"], f"goal_keyword_match:{lowered}"

        # 3) Match against known memory topics mentioning concept.
        memory_topics = self.memory.find_topics_for_concept(lowered)
        if memory_topics:
            return memory_topics[0], f"goal_memory_match:{lowered}"

        return None

    def _select_topic(self, pages: List[Dict], weak_by_topic: Dict[str, float]) -> Tuple[Dict, str]:
        topics = [page["topic"] for page in pages]

        # Goal-directed: use unknown-concept queue first.
        goal = self.curiosity.pop_goal_concept()
        while goal is not None:
            matched = self._find_topic_for_concept(goal, pages)
            if matched:
                topic, reason = matched
                # Avoid getting stuck on one topic due repeated concept matches.
                min_visits = min(self.curiosity.topic_visits[t] for t in topics)
                if self.curiosity.topic_visits[topic] <= min_visits + 1:
                    page = next(p for p in pages if p["topic"] == topic)
                    return page, reason
            goal = self.curiosity.pop_goal_concept()

        # Fallback: curiosity-based scoring.
        topic_scores = self.curiosity.score_topics(topics, weak_by_topic)
        selected = topic_scores[0]
        page = next(p for p in pages if p["topic"] == selected.topic)
        return page, f"curiosity:{selected.reason}"

    def run(self, max_ticks: Optional[int] = None) -> None:
        """Run the agent indefinitely unless max_ticks is provided."""
        tick = 0
        while True:
            tick += 1
            pages = self._load_pages()
            if not pages:
                self.logger.warning("No knowledge pages found in %s", self.world_path)
                time.sleep(self.tick_sleep_seconds)
                continue

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

            selected_page, selection_reason = self._select_topic(pages, weak_by_topic)
            selected_topic = selected_page["topic"]
            self.logger.info(
                "[tick=%s] selected_topic=%s reason=%s",
                tick,
                selected_topic,
                selection_reason,
            )

            result = self.learner.learn_from_page(selected_page)
            self.curiosity.register_unknowns(result.topic, result.unknown_concepts)
            self.curiosity.register_unknowns(result.topic, result.unknown_relations)
            self.curiosity.mark_visited(result.topic)

            conflicts = self.memory.conflicting_relations()
            conflict_bonus = 0.0
            if conflicts:
                for entity, relation, values in conflicts:
                    self.logger.info(
                        "[tick=%s] conflict entity=%s relation=%s values=%s",
                        tick,
                        entity,
                        relation,
                        values,
                    )
                    self.curiosity.register_unknowns(result.topic, [entity, relation, *values])
                conflict_bonus = len(conflicts) * 0.2

            generated_questions = self.questions.from_unknown_concepts(result.unknown_concepts)
            generated_questions.extend(
                self.questions.from_unknown_concepts(result.unknown_relations)
            )
            generated_questions.extend(self.questions.from_conflicts(conflicts))
            # Feed question targets back into concept queue for goal-directed search.
            self.curiosity.register_unknowns(
                result.topic,
                [q.target_concept for q in generated_questions],
            )

            novelty_score = self.curiosity.score_topics([selected_topic], weak_by_topic)[0].score
            curiosity_reward = self.curiosity.reward(novelty=novelty_score, conflict_bonus=conflict_bonus)

            self.memory.decay_confidence(decay_rate=0.005)
            self.memory.save()

            self.logger.info(
                "[tick=%s] learned=%s unknown_concepts=%s unknown_relations=%s "
                "questions=%s memory_size=%s reward=%.2f",
                tick,
                result.learned_facts,
                sorted(result.unknown_concepts),
                sorted(result.unknown_relations),
                [q.text for q in generated_questions[:3]],
                len(self.memory.facts),
                curiosity_reward,
            )

            if max_ticks is not None and tick >= max_ticks:
                self.logger.info("Reached max_ticks=%s; stopping loop.", max_ticks)
                break

            time.sleep(self.tick_sleep_seconds)
