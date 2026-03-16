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
PRED_FAIL_RE = re.compile(
    r"\[tick=(?P<tick>\d+)\].*prediction_failed=(?P<prediction>.*?) rule=(?P<rule>.*)$"
)
PRED_INVALID_RE = re.compile(
    r"\[tick=(?P<tick>\d+)\].*prediction_invalid=(?P<prediction>.*?) rule=(?P<rule>.*)$"
)
EXPERIMENT_RE = re.compile(
    r"\[tick=(?P<tick>\d+)\].*experiment_(?P<kind>generated|result)=(?P<value>.*?)($| rule=(?P<rule>.*)$)"
)

# Structured tick summary:
#   [tick=N] topic=X domain=Y action=Z pred_err=W new_facts=V patterns=P hypotheses=H memory=M reward=R graph nodes=A edges=B new_edges=C ...
SUMMARY_RE = re.compile(
    r"\[tick=(?P<tick>\d+)\]"
    r"(?:.*?\btopic=(?P<topic>\S+))?"
    r"(?:.*?\bdomain=(?P<domain>\S+))?"
    r"(?:.*?\baction=(?P<action>\S+))?"
    r".*?\bnew_facts=(?P<new_facts>\d+)"
    r"(?:.*?\bpatterns=(?P<patterns>\d+))?"
    r".*?\bhypotheses=(?P<hypotheses>\d+)"
    r".*?\bmemory=(?P<memory>\d+)"
    r".*?\breward=(?P<reward>[0-9]*\.?[0-9]+)"
    r".*?\bgraph nodes=(?P<nodes>\d+)\s+edges=(?P<edges>\d+)\s+new(?:_edges)?=(?P<new_edges>\d+)"
)

# Causal rule discovery:
#   [causal_rule_discovered] cause direction_affects effect confidence=X
CAUSAL_RE = re.compile(
    # v3: "[tick=N] ... causal_rule_discovered ... rule confidence=X"
    # v4: "[causal_rule] rule conf=X"  (no tick in line — caller uses last_tick)
    r"(?:\[tick=(?P<tick>\d+)\].*?causal_rule_discovered|\[causal_rule\])"
    r".*?(?P<rule>\S+\s+\S+_affects\s+\S+)\s+conf(?:idence)?=(?P<confidence>[0-9]*\.?[0-9]+)"
)

# FIX: Validator events (new in v2 event loop):
#   [validator] tick=N rule='...' verdict=supported delta=+0.12 source=wikipedia
VALIDATOR_RE = re.compile(
    r"\[validator\].*?tick=(?P<tick>\d+)"
    r".*?rule=(?P<rule>['\"].*?['\"]|[^\s]+)"
    r".*?verdict=(?P<verdict>\S+)"
    r".*?delta=(?P<delta>[+-]?[0-9]*\.?[0-9]+)"
    r"(?:.*?source=(?P<source>\S+))?"
)

# FIX: Law discovery events — tick is NOT on this line, we track last seen tick separately
#   [law_discovered] reaction_rate = 2.3 × exp(0.08 × temperature)  r2=0.912  n=42  reason=...
LAW_RE = re.compile(
    r"\[law_discovered\]\s+(?P<equation>.+?)\s+r2=(?P<r2>[0-9]*\.?[0-9]+)"
    r"(?:\s+n=(?P<n>\d+))?"
)

# FIX: Memory consolidation events:
#   [consolidation] tick=N promoted=P forgotten=F evicted=E graph_size=G
CONSOLIDATION_RE = re.compile(
    r"\[consolidation\].*?tick=(?P<tick>\d+)"
    r".*?promoted=(?P<promoted>\d+)"
    r".*?forgotten=(?P<forgotten>\d+)"
    r".*?evicted=(?P<evicted>\d+)"
    r"(?:.*?spec_pruned=(?P<spec_pruned>\d+))?"
    r".*?graph(?:_size)?=(?P<graph_size>\d+)"
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
    topic: str
    domain: str
    action: str
    new_facts: int
    patterns: int
    hypotheses: int
    memory: int
    reward: float
    new_edges: int
    nodes: int
    edges: int


@dataclass
class ExperimentEvent:
    tick: int
    kind: str
    value: str
    rule: str


@dataclass
class CausalDiscoveryEvent:
    tick: int
    rule: str
    confidence: float


@dataclass
class ValidationEvent:
    """Hypothesis validation result from real-world grounding."""
    tick:             int
    hypothesis_rule:  str
    verdict:          str    # "supported" | "contradicted" | "inconclusive"
    confidence_delta: float
    source:           str


@dataclass
class LawDiscoveryEvent:
    """Mathematical law discovered by the law discovery module."""
    tick:     int
    equation: str
    r2:       float
    n:        int


@dataclass
class ConsolidationEvent:
    """Memory consolidation tick summary."""
    tick:       int
    promoted:   int
    forgotten:  int
    evicted:    int
    graph_size: int


class LogParser:
    """Parses log text into events usable by the dashboard backend."""

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)

    def parse(self) -> Dict[str, List[Dict[str, Any]]]:
        conflicts:           List[ConflictEvent]        = []
        prediction_failures: List[PredictionFailureEvent] = []
        reward_history:      List[Dict[str, Any]]       = []
        summaries:           List[TickSummary]           = []
        experiments:         List[ExperimentEvent]       = []
        causal_discoveries:  List[CausalDiscoveryEvent]  = []
        validation_events:   List[ValidationEvent]       = []
        law_discovery_events: List[LawDiscoveryEvent]    = []
        consolidation_events: List[ConsolidationEvent]   = []

        empty = {
            "conflicts":           [],
            "prediction_failures": [],
            "reward_history":      [],
            "summaries":           [],
            "experiments":         [],
            "causal_discoveries":  [],
            "validation_events":   [],
            "law_discovery_events":[],
            "consolidation_events":[],
        }

        if not self.log_path.exists():
            return empty

        last_tick: int = 0  # track current tick for events that don't carry it (e.g. law_discovered)

        for line in self.log_path.read_text(encoding="utf-8", errors="ignore").splitlines():

            cm = CONFLICT_RE.search(line)
            if cm:
                conflicts.append(ConflictEvent(
                    tick=int(cm.group("tick")),
                    entity=cm.group("entity"),
                    relation=cm.group("relation"),
                    belief=cm.group("belief"),
                    confidence=float(cm.group("confidence")),
                ))

            pm = PRED_FAIL_RE.search(line)
            if pm:
                prediction_failures.append(PredictionFailureEvent(
                    tick=int(pm.group("tick")),
                    prediction=pm.group("prediction").strip(),
                    rule=pm.group("rule").strip(),
                    kind="failed",
                ))

            pim = PRED_INVALID_RE.search(line)
            if pim:
                prediction_failures.append(PredictionFailureEvent(
                    tick=int(pim.group("tick")),
                    prediction=pim.group("prediction").strip(),
                    rule=pim.group("rule").strip(),
                    kind="invalid",
                ))

            em = EXPERIMENT_RE.search(line)
            if em:
                experiments.append(ExperimentEvent(
                    tick=int(em.group("tick")),
                    kind=em.group("kind"),
                    value=(em.group("value") or "").strip(),
                    rule=(em.group("rule") or "").strip(),
                ))

            sm = SUMMARY_RE.search(line)
            if sm:
                tick = int(sm.group("tick"))
                last_tick = tick
                summary = TickSummary(
                    tick=tick,
                    topic=(sm.group("topic") or "unknown"),
                    domain=(sm.group("domain") or "unknown"),
                    action=(sm.group("action") or "none"),
                    new_facts=int(sm.group("new_facts")),
                    patterns=int(sm.group("patterns")) if sm.group("patterns") else 0,
                    hypotheses=int(sm.group("hypotheses")),
                    memory=int(sm.group("memory")),
                    reward=float(sm.group("reward")),
                    new_edges=int(sm.group("new_edges") or 0),
                    nodes=int(sm.group("nodes") or 0),
                    edges=int(sm.group("edges") or 0),
                )
                summaries.append(summary)
                reward_history.append({"tick": summary.tick, "reward": summary.reward})

            causal_m = CAUSAL_RE.search(line)
            if causal_m:
                # v4 [causal_rule] lines have no tick= group — use last_tick
                tick_val = int(causal_m.group("tick")) if causal_m.group("tick") else last_tick
                causal_discoveries.append(CausalDiscoveryEvent(
                    tick=tick_val,
                    rule=causal_m.group("rule").strip(),
                    confidence=float(causal_m.group("confidence")),
                ))

            # FIX: parse validator events
            val_m = VALIDATOR_RE.search(line)
            if val_m:
                raw_rule = val_m.group("rule").strip().strip("'\"")
                validation_events.append(ValidationEvent(
                    tick=int(val_m.group("tick")),
                    hypothesis_rule=raw_rule,
                    verdict=val_m.group("verdict").strip(),
                    confidence_delta=float(val_m.group("delta")),
                    source=(val_m.group("source") or "unknown").strip(),
                ))

            # FIX: parse law discovery events — use last_tick since tick is not in this line
            law_m = LAW_RE.search(line)
            if law_m:
                law_discovery_events.append(LawDiscoveryEvent(
                    tick=last_tick,
                    equation=law_m.group("equation").strip(),
                    r2=float(law_m.group("r2")),
                    n=int(law_m.group("n") or 0),
                ))

            # FIX: parse memory consolidation events
            con_m = CONSOLIDATION_RE.search(line)
            if con_m:
                consolidation_events.append(ConsolidationEvent(
                    tick=int(con_m.group("tick")),
                    promoted=int(con_m.group("promoted")),
                    forgotten=int(con_m.group("forgotten")),
                    evicted=int(con_m.group("evicted")),
                    graph_size=int(con_m.group("graph_size")),
                ))

        return {
            "conflicts":            [asdict(c) for c in conflicts],
            "prediction_failures":  [asdict(p) for p in prediction_failures],
            "reward_history":       reward_history,
            "summaries":            [asdict(s) for s in summaries],
            "experiments":          [asdict(e) for e in experiments],
            "causal_discoveries":   [asdict(c) for c in causal_discoveries],
            "validation_events":    [asdict(v) for v in validation_events],
            "law_discovery_events": [asdict(l) for l in law_discovery_events],
            "consolidation_events": [asdict(c) for c in consolidation_events],
        }