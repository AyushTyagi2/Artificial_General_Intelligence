"""Research Agent — v2.

Closes explanation gaps in the knowledge graph using a two-stage pipeline:

Stage 1 — Graph path search (zero cost, instant):
    KnowledgeGraph.find_paths(cause, effect, max_depth=3)
    If a path exists → extract intermediate nodes as the mechanism
    → insert bridging edges at higher confidence (already implied by graph)

Stage 2 — External search (only when graph has no path):
    Call tools.wikipedia with loosened verb-pattern regex
    If Wikipedia yields nothing → try tools.search as fallback
    Results inserted into graph and cached for future reuse

Changes from v1
---------------
- Added Stage 1 graph path search (_find_graph_mechanisms)
- Added tools.search fallback (_call_search)
- Loosened verb regex: handles "can affect", "may reduce", "the X"
- Graph-path edges inserted at conf=0.50 (higher than external 0.35)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

SEARCH_CONFIDENCE_THRESHOLD:   float = 0.40
GRAPH_MECHANISM_CONFIDENCE:    float = 0.50
EXTERNAL_MECHANISM_CONFIDENCE: float = 0.35
COOLDOWN_TICKS:                int   = 100
MAX_GAPS_PER_CYCLE:            int   = 3
RESEARCH_INTERVAL:             int   = 50

CAUSAL_RELATIONS: Set[str] = {
    "positive_affects", "negative_affects", "affects",
    "mixed_affects", "conditional_affects",
}

_RELATION_VERBS = (
    "affects", "activates", "inhibits", "increases", "decreases",
    "regulates", "modulates", "controls", "triggers", "promotes",
    "reduces", "causes", "mediates", "enables", "prevents",
    "stimulates", "blocks", "enhances", "suppresses",
)

_VERB_RE = re.compile(
    r"([a-z][a-z_\s]{1,25}?)\s+(?:can\s+|may\s+|will\s+|directly\s+|indirectly\s+)?("
    + "|".join(_RELATION_VERBS)
    + r")\s+(?:the\s+|a\s+|an\s+)?([a-z][a-z_\s]{1,25})",
    re.IGNORECASE,
)


@dataclass
class InvestigationRecord:
    cause:        str
    effect:       str
    question:     str
    tick_started: int
    mechanisms:   List[Tuple[str, str, str]] = field(default_factory=list)
    source:       str = "none"
    notes_key:    str = ""


def _extract_triplets(text: str, limit: int = 5) -> List[Tuple[str, str, str]]:
    if not text:
        return []
    found: List[Tuple[str, str, str]] = []
    for m in _VERB_RE.finditer(text):
        subj = re.sub(r"[^a-z0-9_]", "", m.group(1).strip().lower().replace(" ", "_"))
        verb = m.group(2).lower()
        obj  = re.sub(r"[^a-z0-9_]", "", m.group(3).strip().lower().replace(" ", "_"))
        if not subj or not obj or subj == obj:
            continue
        if verb in ("activates", "increases", "promotes", "stimulates",
                    "enhances", "enables", "triggers"):
            rel = "positive_affects"
        elif verb in ("inhibits", "decreases", "reduces", "prevents",
                      "blocks", "suppresses"):
            rel = "negative_affects"
        else:
            rel = "affects"
        found.append((subj, rel, obj))
        if len(found) >= limit:
            break
    return found


class ResearchAgent:
    """Autonomously closes explanation gaps using graph reasoning + external search."""

    def __init__(self, knowledge_graph, tool_router) -> None:
        self._graph   = knowledge_graph
        self._tools   = tool_router
        self._investigated: Dict[Tuple[str, str], int] = {}
        self._records: List[InvestigationRecord]        = []
        self._priority_queue: List[str] = []   # high-priority named mechanism queries
        self._named_queue:    List[str] = []   # normal-priority named queries

    def queue_query(self, query: str, priority: str = "normal") -> None:
        """Queue a named mechanism query for investigation on the next run_cycle.

        High-priority queries are prepended so they are investigated first.
        Duplicates are silently dropped.

        Parameters
        ----------
        query : str
            Natural-language mechanism query, e.g.
            "mechanisms of trophic cascade".
        priority : str
            "high" prepends to the queue; anything else appends.
        """
        if query in self._priority_queue or query in self._named_queue:
            return
        if priority == "high":
            self._priority_queue.insert(0, query)
            logger.debug("[research_agent] queued_high %r (queue=%d)", query, len(self._priority_queue))
        else:
            self._named_queue.append(query)
            logger.debug("[research_agent] queued_normal %r (queue=%d)", query, len(self._named_queue))

    def run_cycle(self, current_tick: int = 0) -> int:
        # Drain named/priority queues first (mechanism queries from MechanismMatcher)
        queued = (self._priority_queue + self._named_queue)[:MAX_GAPS_PER_CYCLE]
        named_investigated = 0
        for query in queued:
            # Translate "mechanisms of trophic_cascade" -> investigate each word pair
            words = [w for w in query.lower().replace("mechanisms of ", "").split() if len(w) > 3]
            if len(words) >= 2:
                self._investigate(words[0], words[-1], current_tick)
                named_investigated += 1
            if query in self._priority_queue:
                self._priority_queue.remove(query)
            elif query in self._named_queue:
                self._named_queue.remove(query)

        gaps = self._find_gaps(current_tick)
        gap_slots = max(0, MAX_GAPS_PER_CYCLE - named_investigated)
        investigated = named_investigated
        for cause, effect in gaps[:gap_slots]:
            self._investigate(cause, effect, current_tick)
            investigated += 1
        if investigated:
            logger.info(
                "[research_agent] tick=%d investigated=%d gaps named=%d",
                current_tick, investigated, named_investigated,
            )
        return investigated

    def get_records(self) -> List[InvestigationRecord]:
        return list(self._records)

    def _find_gaps(self, current_tick: int) -> List[Tuple[str, str]]:
        causal_edges = [
            e for e in self._graph.edges
            if e.relation in CAUSAL_RELATIONS
            and e.confidence >= SEARCH_CONFIDENCE_THRESHOLD
        ]
        out_edges: Dict[str, Set[str]] = {}
        in_edges:  Dict[str, Set[str]] = {}
        for e in self._graph.edges:
            out_edges.setdefault(e.source, set()).add(e.target)
            in_edges.setdefault(e.target, set()).add(e.source)

        gaps: List[Tuple[str, str]] = []
        for edge in causal_edges:
            cause, effect = edge.source, edge.target
            pair = (cause, effect)
            last = self._investigated.get(pair, -COOLDOWN_TICKS - 1)
            if current_tick - last < COOLDOWN_TICKS:
                continue
            intermediates = out_edges.get(cause, set()) & in_edges.get(effect, set())
            if not intermediates:
                gaps.append(pair)

        gap_conf = {(e.source, e.target): e.confidence for e in causal_edges}
        gaps.sort(key=lambda p: -gap_conf.get(p, 0))
        return gaps

    def _investigate(self, cause: str, effect: str, tick: int) -> None:
        self._investigated[(cause, effect)] = tick
        question = (
            f"why does {cause.replace('_', ' ')} "
            f"affect {effect.replace('_', ' ')}?"
        )
        record = InvestigationRecord(
            cause=cause, effect=effect, question=question, tick_started=tick,
        )
        logger.info(
            "[research_agent] investigating cause=%r effect=%r question=%r",
            cause, effect, question,
        )

        # Stage 1: graph path search
        graph_mechs = self._find_graph_mechanisms(cause, effect)
        if graph_mechs:
            record.mechanisms = graph_mechs
            record.source = "graph"
            for subj, rel, obj in graph_mechs:
                self._graph.add_or_update_edge(
                    subj, obj, rel,
                    confidence=GRAPH_MECHANISM_CONFIDENCE,
                    evidence_increment=1,
                    provenance="research_agent_graph",
                    current_tick=tick,
                )
            logger.info(
                "[research_agent] graph_path cause=%r effect=%r mechanisms=%d",
                cause, effect, len(graph_mechs),
            )
        else:
            # Stage 2a: Wikipedia
            wiki_text = self._call_wikipedia(cause)
            triplets  = _extract_triplets(wiki_text)
            if not triplets:
                wiki_text2 = self._call_wikipedia(effect)
                triplets   = _extract_triplets(wiki_text2)

            relevant = [
                (s, r, o) for s, r, o in triplets
                if cause in s or cause in o or effect in s or effect in o
                or s in cause or o in effect
            ]
            if not relevant:
                relevant = triplets[:2]

            # Stage 2b: search fallback
            if not relevant:
                search_text = self._call_search(
                    f"mechanism {cause.replace('_', ' ')} affects "
                    f"{effect.replace('_', ' ')}"
                )
                relevant = _extract_triplets(search_text)
                if relevant:
                    record.source = "search"

            if relevant and record.source == "none":
                record.source = "wikipedia"

            record.mechanisms = relevant
            for subj, rel, obj in relevant:
                self._graph.add_or_update_edge(
                    subj, obj, rel,
                    confidence=EXTERNAL_MECHANISM_CONFIDENCE,
                    evidence_increment=1,
                    provenance="research_agent",
                    current_tick=tick,
                )

        note_content = (
            f"Q: {question}\nSource: {record.source}\n"
            f"Mechanisms: {len(record.mechanisms)}\n"
        )
        for s, r, o in record.mechanisms:
            note_content += f"  {s} --[{r}]--> {o}\n"
        note_key = f"research_{cause}_{effect}"
        self._call_notes(note_key, note_content)
        record.notes_key = note_key
        self._records.append(record)

        logger.info(
            "[research_agent] done cause=%r effect=%r mechanisms=%d source=%s",
            cause, effect, len(record.mechanisms), record.source,
        )

    def _find_graph_mechanisms(
        self, cause: str, effect: str
    ) -> List[Tuple[str, str, str]]:
        try:
            paths = self._graph.find_paths(cause, effect, max_depth=3)
        except Exception as exc:
            logger.debug("[research_agent] find_paths error: %s", exc)
            return []
        if not paths:
            return []
        best_path = min(paths, key=len)
        if len(best_path) < 3:
            return []
        triplets: List[Tuple[str, str, str]] = []
        for i in range(len(best_path) - 1):
            src, tgt = best_path[i], best_path[i + 1]
            rel = self._get_edge_relation(src, tgt)
            triplets.append((src, rel, tgt))
        logger.debug(
            "[research_agent] graph_path %s→%s via %s",
            cause, effect, " → ".join(best_path[1:-1]),
        )
        return triplets

    def _get_edge_relation(self, source: str, target: str) -> str:
        s = self._graph._normalize(source)
        t = self._graph._normalize(target)
        for edge in self._graph.edges:
            if edge.source == s and edge.target == t:
                return edge.relation
        return "affects"

    def _call_wikipedia(self, query: str) -> str:
        topic = query.replace("_", " ")
        try:
            resp = self._tools.dispatch({"tool": "wikipedia", "input": {"query": topic}})
            if resp.get("status") == "success":
                out = resp.get("output", {})
                return out.get("summary", "") or out.get("content", "") or str(out)
        except Exception as exc:
            logger.debug("[research_agent] wikipedia call failed: %s", exc)
        return ""

    def _call_search(self, query: str) -> str:
        try:
            resp = self._tools.dispatch({"tool": "search", "input": {"query": query}})
            if resp.get("status") == "success":
                out = resp.get("output", {})
                if isinstance(out, list):
                    return " ".join(
                        str(r.get("snippet", "") or r.get("text", "") or r)
                        for r in out[:3]
                    )
                return str(out)
        except Exception as exc:
            logger.debug("[research_agent] search call failed: %s", exc)
        return ""

    def _call_notes(self, key: str, content: str) -> None:
        try:
            self._tools.dispatch({
                "tool": "notes",
                "input": {"key": key, "content": content, "action": "set"},
            })
        except Exception as exc:
            logger.debug("[research_agent] notes call failed: %s", exc)