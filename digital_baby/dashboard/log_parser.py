"""Parse digital_baby runtime logs into structured dashboard events."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List
import re


CONFLICT_RE = re.compile(
    r"\[tick=(?P<tick>\d+)\].*conflict entity=(?P<entity>\S+) relation=(?P<relation>\S+) "
    r"belief=(?P<belief>\S+) confidence=(?P<confidence>[0-9]*\.?[0-9]+)"
)
PRED_FAIL_RE = re.compile(r"\[tick=(?P<tick>\d+)\].*prediction_failed=(?P<prediction>.*?) rule=(?P<rule>.*)$")
PRED_INVALID_RE = re.compile(r"\[tick=(?P<tick>\d+)\].*prediction_invalid=(?P<prediction>.*?) rule=(?P<rule>.*)$")
EXPERIMENT_RE = re.compile(r"\[tick=(?P<tick>\d+)\].*experiment_(?P<kind>generated|result)=(?P<value>.*?)($| rule=(?P<rule>.*)$)")
SUMMARY_RE = re.compile(
    r"\[tick=(?P<tick>\d+)\].*learned=(?P<learned>\d+) new_facts=(?P<new_facts>\d+).*"
    r"patterns=(?P<patterns>\d+) hypotheses=(?P<hypotheses>\d+).*memory=(?P<memory>\d+).*reward=(?P<reward>[0-9]*\.?[0-9]+)"
)


@dataclass
class ConflictEvent:
    tick: int
    entity: str
    relation: str
    belief: str
    confidence: float


@dataclass
class PredictionFailureEvent:
    tick: int
    prediction: str
    rule: str
    kind: str


@dataclass
class TickSummary:
    tick: int
    learned: int
    new_facts: int
    patterns: int
    hypotheses: int
    memory: int
    reward: float


@dataclass
class ExperimentEvent:
    tick: int
    kind: str
    value: str
    rule: str


class LogParser:
    """Parses log text into events usable by the dashboard backend."""

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)

    def parse(self) -> Dict[str, List[Dict[str, Any]]]:
        conflicts: List[ConflictEvent] = []
        prediction_failures: List[PredictionFailureEvent] = []
        reward_history: List[Dict[str, Any]] = []
        summaries: List[TickSummary] = []
        experiments: List[ExperimentEvent] = []

        if not self.log_path.exists():
            return {
                "conflicts": [],
                "prediction_failures": [],
                "reward_history": [],
                "summaries": [],
                "experiments": [],
            }

        for line in self.log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            cm = CONFLICT_RE.search(line)
            if cm:
                conflicts.append(
                    ConflictEvent(
                        tick=int(cm.group("tick")),
                        entity=cm.group("entity"),
                        relation=cm.group("relation"),
                        belief=cm.group("belief"),
                        confidence=float(cm.group("confidence")),
                    )
                )

            pm = PRED_FAIL_RE.search(line)
            if pm:
                prediction_failures.append(
                    PredictionFailureEvent(
                        tick=int(pm.group("tick")),
                        prediction=pm.group("prediction").strip(),
                        rule=pm.group("rule").strip(),
                        kind="failed",
                    )
                )

            pim = PRED_INVALID_RE.search(line)
            if pim:
                prediction_failures.append(
                    PredictionFailureEvent(
                        tick=int(pim.group("tick")),
                        prediction=pim.group("prediction").strip(),
                        rule=pim.group("rule").strip(),
                        kind="invalid",
                    )
                )

            em = EXPERIMENT_RE.search(line)
            if em:
                experiments.append(
                    ExperimentEvent(
                        tick=int(em.group("tick")),
                        kind=em.group("kind"),
                        value=(em.group("value") or "").strip(),
                        rule=(em.group("rule") or "").strip(),
                    )
                )

            sm = SUMMARY_RE.search(line)
            if sm:
                summary = TickSummary(
                    tick=int(sm.group("tick")),
                    learned=int(sm.group("learned")),
                    new_facts=int(sm.group("new_facts")),
                    patterns=int(sm.group("patterns")),
                    hypotheses=int(sm.group("hypotheses")),
                    memory=int(sm.group("memory")),
                    reward=float(sm.group("reward")),
                )
                summaries.append(summary)
                reward_history.append({"tick": summary.tick, "reward": summary.reward})

        return {
            "conflicts": [asdict(c) for c in conflicts],
            "prediction_failures": [asdict(p) for p in prediction_failures],
            "reward_history": reward_history,
            "summaries": [asdict(s) for s in summaries],
            "experiments": [asdict(e) for e in experiments],
        }
