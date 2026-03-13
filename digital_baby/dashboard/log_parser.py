"""Parse digital_baby runtime logs for dashboard visualization."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List
import re


CONFLICT_RE = re.compile(
    r"\[tick=(?P<tick>\d+)\].*conflict entity=(?P<entity>\S+) relation=(?P<relation>\S+) "
    r"belief=(?P<belief>\S+) confidence=(?P<confidence>[0-9]*\.?[0-9]+)"
)
PRED_FAIL_RE = re.compile(
    r"\[tick=(?P<tick>\d+)\].*prediction_failed=(?P<prediction>.*?) rule=(?P<rule>.*)$"
)
REWARD_RE = re.compile(
    r"\[tick=(?P<tick>\d+)\].*reward=(?P<reward>[0-9]*\.?[0-9]+)"
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


@dataclass
class RewardPoint:
    tick: int
    reward: float


class LogParser:
    """Parse log file into dashboard-friendly structures."""

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)

    def parse(self) -> Dict[str, List[Dict[str, Any]]]:
        conflicts: List[ConflictEvent] = []
        prediction_failures: List[PredictionFailureEvent] = []
        reward_history: List[RewardPoint] = []

        if not self.log_path.exists():
            return {
                "conflicts": [],
                "prediction_failures": [],
                "reward_history": [],
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
                    )
                )

            rm = REWARD_RE.search(line)
            if rm:
                reward_history.append(
                    RewardPoint(tick=int(rm.group("tick")), reward=float(rm.group("reward")))
                )

        return {
            "conflicts": [asdict(c) for c in conflicts],
            "prediction_failures": [asdict(p) for p in prediction_failures],
            "reward_history": [asdict(r) for r in reward_history],
        }
