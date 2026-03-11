"""Main life-cycle loop for the digital baby agent."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
import time

from digital_baby.brain.curiosity import CuriosityModel
from digital_baby.brain.learner import Learner
from digital_baby.brain.memory import Memory
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
        self.tick_sleep_seconds = tick_sleep_seconds
        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_pages(self) -> List[Dict]:
        pages = []
        for file_path in sorted(self.world_path.glob("*.json")):
            page = json.loads(file_path.read_text(encoding="utf-8"))
            pages.append(page)
        return pages

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

            topic_scores = self.curiosity.score_topics(topics, weak_by_topic)
            selected = topic_scores[0]
            page = next(p for p in pages if p["topic"] == selected.topic)

            self.logger.info("tick=%s selected_topic=%s reason=%s", tick, selected.topic, selected.reason)

            result = self.learner.learn_from_page(page)
            self.curiosity.register_unknowns(result.topic, result.unknown_concepts)
            self.curiosity.mark_visited(result.topic)

            conflicts = self.reasoner.detect_contradictions(self.memory.relation_triplets())
            conflict_bonus = 0.0
            if conflicts:
                for conflict in conflicts:
                    self.logger.info(
                        "conflict detected entity=%s relation=%s values=%s",
                        conflict.entity,
                        conflict.relation,
                        conflict.values,
                    )
                    # Contradiction increases curiosity for this topic.
                    self.curiosity.register_unknowns(result.topic, {f"conflict:{conflict.entity}:{conflict.relation}"})
                conflict_bonus = len(conflicts) * 0.2

            curiosity_reward = self.curiosity.reward(novelty=selected.score, conflict_bonus=conflict_bonus)

            self.memory.decay_confidence(decay_rate=0.005)
            self.memory.save()

            self.logger.info(
                "tick=%s learned_facts=%s unknown_concepts=%s memory_size=%s reward=%.2f",
                tick,
                result.learned_facts,
                sorted(result.unknown_concepts),
                len(self.memory.facts),
                curiosity_reward,
            )

            if max_ticks is not None and tick >= max_ticks:
                self.logger.info("Reached max_ticks=%s; stopping loop.", max_ticks)
                break

            time.sleep(self.tick_sleep_seconds)
