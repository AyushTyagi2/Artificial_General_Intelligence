"""
Redesigned Digital Brain Dashboard — server_new.py
====================================================
Key architecture changes vs. original:

1. Single /api/delta endpoint replaces 14+ polling endpoints.
   Clients send their last-seen `seq` number; only NEW data is returned.
   This avoids re-sending the same large arrays on every poll.

2. /api/graph uses aggressive sampling:
   - Returns at most MAX_GRAPH_NODES nodes and MAX_GRAPH_EDGES edges
   - Edges sorted by confidence; low-quality edges dropped first
   - Separate /api/graph/neighbors/<node> for on-demand neighbourhood drilling

3. /stream (SSE) pushes a tiny heartbeat + counters every 2 s.
   The heavy data only arrives when the seq changes.

4. State is frozen into a DeltaStore that keeps a rolling log of
   change-sets (max DELTA_HISTORY entries). Each change-set has a
   monotonically increasing seq number so the frontend can ask for
   "only what changed since seq N".

5. All JSON reads are guarded by _safe_json_load (from original code).
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import time
import threading
import logging
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

# ── Optional tools bridge for Larry Tool Activity panel ───────────────────────
try:
    import sys as _sys
    _HERE = Path(__file__).resolve()
    _PROJECT_ROOT_DB = _HERE.parent.parent.parent
    _TOOL_EVENTS_FILE = Path(
        os.environ.get("TOOL_EVENTS_FILE", str(_PROJECT_ROOT_DB / "tool_events.jsonl"))
    )
    _HAS_TOOLS = True
except Exception as _te:
    _HAS_TOOLS = False
    _TOOL_EVENTS_FILE = Path("/tmp/tool_events_missing.jsonl")


def _read_tool_events_file(since_line: int = 0) -> tuple:
    """
    Read tool events from the cross-process JSON-lines file.
    Returns (events_list, total_line_count).
    Events are returned oldest-first.
    """
    if not _TOOL_EVENTS_FILE.exists():
        return [], 0
    try:
        lines = _TOOL_EVENTS_FILE.read_text(encoding="utf-8").splitlines()
        total = len(lines)
        events = []
        for line in lines[since_line:]:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return events, total
    except Exception:
        return [], 0

# ── Optional flask-sock for WebSocket fallback ───────────────────────────────
try:
    from flask_sock import Sock
    _HAS_SOCK = True
except ImportError:
    _HAS_SOCK = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

log = logging.getLogger(__name__)

_NUMERIC_RE = re.compile(r"^-?[0-9]+\.?[0-9]*$")

def _is_noise_node(name: str) -> bool:
    return bool(_NUMERIC_RE.match(name)) or name.endswith("_delta")

def _is_noise_edge(edge: dict) -> bool:
    return _is_noise_node(edge.get("source", "")) or _is_noise_node(edge.get("target", ""))

def _safe_json_load(path, default=None):
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
        log.warning("Corrupt JSON %s → backed up to %s", p, backup)
        return default

def _entropy(p: float) -> float:
    p = max(1e-9, min(1 - 1e-9, p))
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution  (mirrors original server.py)
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parent
_REPO_ROOT = BASE_DIR.parent.parent

TEMPLATES_DIR = BASE_DIR / "templates"

def _resolve(env_var: str, default_relative: str) -> Path:
    raw = os.environ.get(env_var, "")
    return Path(raw) if raw else _REPO_ROOT / default_relative

# ── Screen capture state file ────────────────────────────────────────────────
# Both the dashboard and the agent process share a tiny JSON file:
#   { "screen_capture_enabled": true }
# Dashboard writes it; ScreenTracker polls it every capture cycle.
# This works even when they run as separate processes (the normal case).

_SCREEN_STATE_FILE = _resolve("DIGITAL_BABY_SCREEN_STATE",
                               "digital_baby/world/screen_capture_state.json")

def _read_screen_state() -> dict:
    """Read current screen capture state from shared file."""
    try:
        if _SCREEN_STATE_FILE.exists():
            raw = _SCREEN_STATE_FILE.read_text(encoding="utf-8").strip()
            if raw:
                return json.loads(raw)
    except Exception:
        pass
    return {"screen_capture_enabled": True}   # default ON if file missing

def _write_screen_state(enabled: bool) -> None:
    """Persist screen capture state to shared file."""
    try:
        _SCREEN_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SCREEN_STATE_FILE.write_text(
            json.dumps({"screen_capture_enabled": enabled}), encoding="utf-8"
        )
    except Exception as exc:
        log.warning("[screen_state] write_failed: %s", exc)

def is_screen_capture_enabled() -> bool:
    return _read_screen_state().get("screen_capture_enabled", True)

# In-process registry (used when dashboard and agent share the same process)
_AGENT_REGISTRY: dict = {}

def register_screen_tracker(tracker) -> None:
    """Called by BabyEventLoop.__init__ to expose the tracker to the dashboard."""
    _AGENT_REGISTRY["screen_tracker"] = tracker

def get_screen_tracker():
    return _AGENT_REGISTRY.get("screen_tracker")

# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_LOG_PATH    = _resolve("DIGITAL_BABY_LOG",    "agent.log")
DEFAULT_MEMORY_PATH = _resolve("DIGITAL_BABY_MEMORY", "digital_baby/world/memory_store.json")
DEFAULT_KG_PATH     = _resolve("DIGITAL_BABY_KG",     "digital_baby/world/knowledge_graph.json")
DEFAULT_PERCEPTION_LOG = _resolve("DIGITAL_BABY_PERCEPTION", "digital_baby/world/perception_log.jsonl")
DEFAULT_WIKI_LOG       = _resolve("DIGITAL_BABY_WIKI",       "digital_baby/world/wiki_log.jsonl")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_GRAPH_NODES   = 150    # max nodes sent to D3 canvas
MAX_GRAPH_EDGES   = 300    # max edges sent to D3 canvas
DELTA_HISTORY     = 50     # number of delta snapshots retained
POLL_INTERVAL_S   = 3      # how often the background thread refreshes state
SSE_HEARTBEAT_S   = 2      # SSE ping interval
CAUSAL_LIMIT      = 60
HYPO_LIMIT        = 50
EVENTS_LIMIT      = 40


# ─────────────────────────────────────────────────────────────────────────────
# Log parser  (inline, lightweight — avoids importing original module)
# ─────────────────────────────────────────────────────────────────────────────

SUMMARY_RE = re.compile(
    r"\[tick=(?P<tick>\d+)\]"
    r"(?:.*?\btopic=(?P<topic>\S+))?"
    r"(?:.*?\bdomain=(?P<domain>\S+))?"
    r"(?:.*?\baction=(?P<action>\S+))?"
    r".*?\bnew_facts=(?P<new_facts>\d+)"
    r"(?:.*?\breinf=(?P<reinf>\d+))?"                 # optional — added in v4.2
    r"(?:.*?\bpatterns=(?P<patterns>\d+))?"           # optional — v4 event loop omits it
    r".*?\bhypotheses=(?P<hypotheses>\d+)"
    r".*?\bmemory=(?P<memory>\d+)"
    r".*?\breward=(?P<reward>[0-9]*\.?[0-9]+)"
    r".*?\bgraph nodes=(?P<nodes>\d+)\s+edges=(?P<edges>\d+)\s+new(?:_edges)?=(?P<new_edges>\d+)"
)  # new(?:_edges)? matches both legacy "new_edges=" and v4 "new="
CAUSAL_RE = re.compile(
    # v3: "[tick=N] ... causal_rule_discovered ... rule confidence=X"
    # v4: "[causal_rule] rule conf=X"  (no tick in line — caller uses last_tick)
    r"(?:\[tick=(?P<tick>\d+)\].*?causal_rule_discovered|\[causal_rule\])"
    r".*?(?P<rule>\S+\s+\S+_affects\s+\S+)\s+conf(?:idence)?=(?P<confidence>[0-9]*\.?[0-9]+)"
)
LAW_RE = re.compile(
    r"\[law_discovered\]\s+(?P<equation>.+?)\s+r2=(?P<r2>[0-9]*\.?[0-9]+)"
    r"(?:\s+n=(?P<n>\d+))?"
)
CONSOLIDATION_RE = re.compile(
    r"\[consolidation\].*?tick=(?P<tick>\d+)"
    r".*?promoted=(?P<promoted>\d+)"
    r".*?forgotten=(?P<forgotten>\d+)"
    r".*?evicted=(?P<evicted>\d+)"
    r"(?:.*?spec_pruned=(?P<spec_pruned>\d+))?"   # v4 new field, optional
    r".*?graph(?:_size)?=(?P<graph_size>\d+)"     # v3: graph_size=N  v4: graph=N
)
VALIDATOR_RE = re.compile(
    r"\[validator\].*?tick=(?P<tick>\d+)"
    r".*?rule=(?P<rule>['\"].*?['\"]|\S+)"
    r".*?verdict=(?P<verdict>\S+)"
    r".*?delta=(?P<delta>[+-]?[0-9]*\.?[0-9]+)"
    r"(?:.*?source=(?P<source>\S+))?"
)


def _parse_log(log_path: Path, max_lines: int = 20_000) -> dict:
    """Parse the tail of the log file. Returns categorised event lists."""
    out: dict = {
        "summaries": [],
        "causal_discoveries": [],
        "law_discovery_events": [],
        "consolidation_events": [],
        "validation_events": [],
        "reward_history": [],
        "failure_series": [],
    }
    if not log_path.exists():
        return out

    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out

    lines = text.splitlines()[-max_lines:]
    last_tick = 0

    for line in lines:
        m = SUMMARY_RE.search(line)
        if m:
            d = m.groupdict()
            tick = int(d["tick"])
            last_tick = tick
            summary = {
                "tick":      tick,
                "topic":     d.get("topic") or "",
                "domain":    d.get("domain") or "",
                "action":    d.get("action") or "",
                "new_facts": int(d["new_facts"]),
                "reinf":     int(d["reinf"]) if d.get("reinf") else 0,
                "patterns":  int(d["patterns"]) if d.get("patterns") else 0,
                "hypotheses":int(d["hypotheses"]),
                "memory":    int(d["memory"]),
                "reward":    float(d["reward"]),
                "nodes":     int(d["nodes"]),
                "edges":     int(d["edges"]),
                "new_edges": int(d["new_edges"]),
            }
            out["summaries"].append(summary)
            out["reward_history"].append(float(d["reward"]))
            continue

        m = CAUSAL_RE.search(line)
        if m:
            d = m.groupdict()
            # v4 [causal_rule] lines have no tick= — fall back to last_tick
            tick_val = int(d["tick"]) if d.get("tick") else last_tick
            out["causal_discoveries"].append({
                "tick":       tick_val,
                "rule":       d["rule"],
                "confidence": float(d["confidence"]),
                "kind":       "causal_discovered",
            })
            continue

        m = LAW_RE.search(line)
        if m:
            d = m.groupdict()
            out["law_discovery_events"].append({
                "tick":     last_tick,
                "equation": d["equation"],
                "r2":       float(d["r2"]),
                "n":        int(d["n"]) if d.get("n") else None,
            })
            continue

        m = CONSOLIDATION_RE.search(line)
        if m:
            d = m.groupdict()
            out["consolidation_events"].append({
                "tick":       int(d["tick"]),
                "promoted":   int(d["promoted"]),
                "forgotten":  int(d["forgotten"]),
                "evicted":    int(d["evicted"]),
                "graph_size": int(d["graph_size"]),
            })
            continue

        m = VALIDATOR_RE.search(line)
        if m:
            d = m.groupdict()
            out["validation_events"].append({
                "tick":    int(d["tick"]),
                "rule":    d["rule"].strip("'\""),
                "verdict": d["verdict"],
                "delta":   float(d["delta"]),
                "source":  d.get("source") or "",
            })

    return out


# ─────────────────────────────────────────────────────────────────────────────
# DeltaStore — the core of the new architecture
# ─────────────────────────────────────────────────────────────────────────────

def _read_perception_log(max_events: int = 20) -> list:
    """Read recent entries from the perception JSONL log."""
    if not DEFAULT_PERCEPTION_LOG.exists():
        return []
    try:
        lines = DEFAULT_PERCEPTION_LOG.read_text(encoding="utf-8").splitlines()
        events = []
        for line in reversed(lines[-200:]):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
            if len(events) >= max_events:
                break
        return list(reversed(events))
    except Exception:
        return []


class DeltaStore:
    """
    Maintains a rolling window of dashboard snapshots.
    Each refresh produces a new snapshot with a monotonically increasing `seq`.
    Clients send their last-seen seq; the store returns only what changed.
    """

    def __init__(self) -> None:
        self._lock   = threading.RLock()
        self._seq    = 0
        self._full: Dict[str, Any] = {}          # latest full snapshot
        self._deltas: deque = deque(maxlen=DELTA_HISTORY)  # (seq, diff)
        self._subscribers: List = []             # SSE queues

    # ── Internal refresh ─────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Re-read source files and update the store if anything changed."""
        new_snap = self._build_snapshot()
        with self._lock:
            if self._is_changed(new_snap):
                self._seq += 1
                diff = self._compute_diff(self._full, new_snap)
                diff["seq"] = self._seq
                diff["ts"]  = time.time()
                self._deltas.append(diff)
                self._full = new_snap
                self._full["seq"] = self._seq
                # Notify SSE subscribers
                heartbeat = {
                    "seq":   self._seq,
                    "ts":    diff["ts"],
                    "counters": diff.get("counters", {}),
                }
                for q in list(self._subscribers):
                    try:
                        q.append(heartbeat)
                    except Exception:
                        pass

    def _is_changed(self, new_snap: dict) -> bool:
        old = self._full
        if not old:
            return True
        return (
            new_snap.get("counters") != old.get("counters")
            or new_snap.get("latest_tick") != old.get("latest_tick")
        )

    def _compute_diff(self, old: dict, new: dict) -> dict:
        """
        Returns only the keys that changed. Lists return only new tail items.
        """
        if not old:
            return dict(new)
        diff: dict = {}
        for key, val in new.items():
            old_val = old.get(key)
            if isinstance(val, list) and isinstance(old_val, list):
                if len(val) > len(old_val):
                    diff[key] = val[len(old_val):]   # only new items
                elif val != old_val:
                    diff[key] = val                  # full replace if changed
            elif val != old_val:
                diff[key] = val
        return diff

    # ── Snapshot builder ─────────────────────────────────────────────────────

    def _build_snapshot(self) -> dict:
        log_data = _parse_log(DEFAULT_LOG_PATH)
        mem      = _safe_json_load(DEFAULT_MEMORY_PATH) or {}

        # Perception summary (screen)
        _perc_events = _read_perception_log(max_events=20)
        _perc_summary = {
            "total_events": len(_perc_events),
            "total_edges":  sum(len(e.get("triples", [])) for e in _perc_events),
        }

        # Wikipedia perception summary
        _wiki_events = _read_wiki_log(max_events=20)
        _wiki_summary = {
            "total_events": len(_wiki_events),
            "total_edges":  sum(e.get("triples_count", len(e.get("triples", []))) for e in _wiki_events),
        }

        summaries = log_data["summaries"][-500:]
        latest    = summaries[-1] if summaries else {}

        # Memory counters
        wm = mem.get("world_model", {})
        counters = {
            "facts":             len(mem.get("facts",        [])),
            "facts_cap":         10_000,   # show cap so dashboard can render a fill bar
            "causal_rules":      len(mem.get("causal_rules", [])),
            "rules":             len(wm.get("rules",         [])),
            # Use cumulative totals if available; fall back to buffer length for old stores
            "predictions":       mem.get("total_predictions",  len(wm.get("predictions",  []))),
            "experiments":       mem.get("total_experiments",  len(wm.get("experiments",  []))),
            # Also expose current buffer size so the dashboard can show "last N" context
            "predictions_buf":   len(wm.get("predictions",   [])),
            "experiments_buf":   len(wm.get("experiments",   [])),
            "patterns":          len(mem.get("patterns",     [])),
            "nodes":             latest.get("nodes", 0),
            "edges":        latest.get("edges", 0),
            "perception_events": _perc_summary.get("total_events", 0),
            "perception_edges":  _perc_summary.get("total_edges", 0),
            "screen_paused":     not is_screen_capture_enabled(),
            "wiki_events":       _wiki_summary.get("total_events", 0),
            "wiki_edges":        _wiki_summary.get("total_edges", 0),
        }

        # Reward series (last 200 ticks for sparkline)
        rewards = [s["reward"] for s in summaries[-200:]]

        # Recent causal discoveries
        causal_discoveries = log_data["causal_discoveries"][-CAUSAL_LIMIT:]

        # Causal rules from memory (richer data)
        mem_causal = sorted(
            mem.get("causal_rules", []),
            key=lambda r: r.get("observations", 0),
            reverse=True,
        )[:CAUSAL_LIMIT]

        # Hypotheses / rules from world model
        wm_rules = sorted(
            wm.get("rules", []),
            key=lambda r: r.get("confidence", 0),
            reverse=True,
        )[:HYPO_LIMIT]

        # Live event stream (last N items of each type)
        live_events: List[dict] = []
        for ev in log_data["causal_discoveries"][-10:]:
            live_events.append({**ev, "kind": "causal"})
        for ev in log_data["law_discovery_events"][-6:]:
            live_events.append({**ev, "kind": "law"})
        for ev in log_data["validation_events"][-6:]:
            live_events.append({**ev, "kind": "validation"})
        # Inject recent perception triples into the live event stream
        for _pe in _perc_events[-6:]:
            for _pt in _pe.get("triples", [])[:3]:
                live_events.append({
                    "kind":       "perception",
                    "tick":       0,
                    "subject":    _pt.get("subject", ""),
                    "relation":   _pt.get("relation", ""),
                    "object":     _pt.get("object", ""),
                    "confidence": _pt.get("confidence", 0.0),
                    "source_text": _pe.get("text", "")[:80],
                })
        # Inject recent wikipedia triples into the live event stream
        for _we in _wiki_events[-4:]:
            for _wt in _we.get("triples", [])[:2]:
                live_events.append({
                    "kind":       "wiki",
                    "tick":       0,
                    "subject":    _wt.get("subject", ""),
                    "relation":   _wt.get("relation", ""),
                    "object":     _wt.get("object", ""),
                    "confidence": _wt.get("confidence", 0.0),
                    "source_text": _wt.get("text", "")[:80],
                    "topic":      _we.get("topic", ""),
                })
        live_events.sort(key=lambda e: e.get("tick", 0))
        live_events = live_events[-EVENTS_LIMIT:]

        # Learning status
        learning = self._compute_learning_status(summaries[-30:], log_data["law_discovery_events"])

        # Epistemic summary (from KG)
        epistemic = self._compute_epistemic_summary()

        return {
            "latest_tick":         latest.get("tick", 0),
            "counters":            counters,
            "rewards":             rewards,
            "causal_discoveries":  causal_discoveries,
            "mem_causal_rules":    mem_causal,
            "wm_rules":            wm_rules,
            "live_events":         live_events,
            "summaries_tail":      summaries[-30:],
            "law_events":          log_data["law_discovery_events"][-20:],
            "consolidation_events":log_data["consolidation_events"][-20:],
            "validation_events":   log_data["validation_events"][-20:],
            "learning":            learning,
            "epistemic":           epistemic,
            "perception_events":   _perc_events,
            "wiki_events":         _wiki_events,
        }

    @staticmethod
    def _compute_learning_status(recent: List[dict], laws: List[dict]) -> dict:
        if not recent:
            return {"status": "no_data", "score": 0.0, "indicators": {}}

        new_edges   = [int(s.get("new_edges", 0)) for s in recent]
        avg_edges   = sum(new_edges) / len(new_edges)
        hyp_counts  = [int(s.get("hypotheses", 0)) for s in recent]
        hyp_trend   = (hyp_counts[-1] - hyp_counts[0]) if len(hyp_counts) >= 2 else 0
        pat_counts  = [int(s.get("patterns", 0)) for s in recent]
        pat_trend   = (pat_counts[-1] - pat_counts[0]) if len(pat_counts) >= 2 else 0
        action_ctr  = Counter(s.get("action", "none") for s in recent)
        n_unique    = len([a for a in action_ctr if a != "none"])
        act_div     = n_unique / max(len(recent), 1)
        first_tick  = int(recent[0].get("tick", 0)) if recent else 0
        recent_laws = [l for l in laws if l.get("tick", 0) >= first_tick]
        law_rate    = len(recent_laws) / max(len(recent), 1)

        # Reinforcement: average re-observations per tick (weak novelty signal).
        # Contributes a small bonus (up to 5%) so the score doesn't flatline at
        # 16% when the knowledge base is saturated but the brain is still active.
        reinf_vals  = [int(s.get("reinf", 0)) for s in recent]
        avg_reinf   = sum(reinf_vals) / max(len(reinf_vals), 1)
        reinf_term  = min(1.0, avg_reinf / 20.0)   # 20 reinf/tick → full credit

        score = min(1.0, (
            min(avg_edges / 3.0,       1.0) * 0.28 +
            min(max(hyp_trend, 0) / 10.0, 1.0) * 0.20 +
            min(max(pat_trend, 0) / 5.0,  1.0) * 0.14 +
            act_div * 0.20 +
            min(law_rate * 10, 1.0) * 0.13 +
            reinf_term * 0.05
        ))
        status = "learning" if score >= 0.55 else "plateauing" if score >= 0.25 else "stagnant"
        return {
            "status": status,
            "score":  round(score, 3),
            "indicators": {
                "avg_new_edges": round(avg_edges, 2),
                "hyp_trend":     hyp_trend,
                "pat_trend":     pat_trend,
                "act_diversity": round(act_div, 3),
                "law_rate":      round(law_rate, 4),
                "avg_reinf":     round(avg_reinf, 1),
            },
        }

    @staticmethod
    def _compute_epistemic_summary() -> dict:
        if not DEFAULT_KG_PATH.exists():
            return {"mean_entropy": 0.0, "uncertain_count": 0}
        payload = _safe_json_load(DEFAULT_KG_PATH) or {}
        edges   = [e for e in payload.get("edges", []) if not _is_noise_edge(e)]
        if not edges:
            return {"mean_entropy": 0.0, "uncertain_count": 0}
        all_h   = [_entropy(e.get("confidence", 0.5)) for e in edges]
        mean_h  = sum(all_h) / len(all_h)
        unc     = sum(1 for h in all_h if h > 0.6)
        return {"mean_entropy": round(mean_h, 4), "uncertain_count": unc}

    # ── Public API ────────────────────────────────────────────────────────────

    def get_full(self) -> dict:
        with self._lock:
            return dict(self._full)

    def get_delta_since(self, since_seq: int) -> dict:
        """
        Return all changes since `since_seq`.
        If the client is too far behind (seq gap > DELTA_HISTORY), return full snapshot.
        """
        with self._lock:
            if not self._deltas or since_seq < (self._seq - DELTA_HISTORY):
                snap = dict(self._full)
                snap["full_refresh"] = True
                return snap
            # Merge all deltas newer than since_seq
            merged: dict = {"seq": self._seq, "ts": time.time(), "full_refresh": False}
            for delta in self._deltas:
                if delta.get("seq", 0) > since_seq:
                    for k, v in delta.items():
                        if k in ("seq", "ts"):
                            continue
                        if isinstance(v, list) and isinstance(merged.get(k), list):
                            merged[k] = merged[k] + v
                        else:
                            merged[k] = v
            return merged

    def subscribe_sse(self) -> deque:
        q: deque = deque(maxlen=20)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe_sse(self, q: deque) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    @property
    def current_seq(self) -> int:
        return self._seq


# ─────────────────────────────────────────────────────────────────────────────
# Background refresh thread
# ─────────────────────────────────────────────────────────────────────────────

_store = DeltaStore()

# ── Cross-process tool event tail ─────────────────────────────────────────────
# Poll the tool_events.jsonl file written by the brain process and push
# new events to all live SSE subscribers.
_tool_file_line = 0   # how many lines we've already processed


def _read_wiki_log(max_events: int = 20) -> list:
    """Read recent entries from the Wikipedia perception JSONL log."""
    path = DEFAULT_WIKI_LOG
    if not os.path.exists(path):
        return []
    events = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        return []
    return events[-max_events:]


def _tail_tool_events() -> None:
    global _tool_file_line
    while True:
        try:
            events, total = _read_tool_events_file(since_line=_tool_file_line)
            if events:
                _tool_file_line = total
                payload_list = [{"tool_event": ev} for ev in events]
                for q in list(_store._subscribers):
                    try:
                        for p in payload_list:
                            q.append(p)
                    except Exception:
                        pass
        except Exception:
            pass
        time.sleep(1.0)

_tool_tail_thread = threading.Thread(target=_tail_tool_events, daemon=True)
_tool_tail_thread.start()

def _refresh_loop() -> None:
    while True:
        try:
            _store.refresh()
        except Exception:
            log.exception("DeltaStore refresh error")
        time.sleep(POLL_INTERVAL_S)

_thread = threading.Thread(target=_refresh_loop, daemon=True)
_thread.start()


# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
if _HAS_SOCK:
    sock = Sock(app)


# ── Main dashboard ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index_new.html")


# ── Delta API — THE primary endpoint ─────────────────────────────────────────

@app.route("/api/delta")
def api_delta():
    """
    Returns only what changed since the client's last-seen `seq`.
    Query param: ?seq=N  (omit or 0 for initial full snapshot)
    """
    since = int(request.args.get("seq", 0) or 0)
    data  = _store.get_delta_since(since)
    resp  = jsonify(data)
    resp.headers["Cache-Control"] = "no-store"
    return resp


# ── SSE push stream ───────────────────────────────────────────────────────────

@app.route("/stream")
def sse_stream():
    """
    Server-Sent Events endpoint.
    Pushes a tiny heartbeat JSON whenever the seq increments.
    Clients use this to know when to call /api/delta.
    """
    q = _store.subscribe_sse()

    def generate():
        try:
            # Send current seq immediately
            yield f"data: {json.dumps({'seq': _store.current_seq, 'ts': time.time()})}\n\n"
            deadline = time.time()
            while True:
                now = time.time()
                if q:
                    msg = q.popleft()
                    yield f"data: {json.dumps(msg)}\n\n"
                    deadline = now + SSE_HEARTBEAT_S
                elif now >= deadline:
                    # keepalive comment
                    yield f": keepalive {int(now)}\n\n"
                    deadline = now + SSE_HEARTBEAT_S
                else:
                    time.sleep(0.1)
        except GeneratorExit:
            pass
        finally:
            _store.unsubscribe_sse(q)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── Graph endpoint — heavily sampled ─────────────────────────────────────────

@app.route("/api/graph")
def api_graph():
    """
    Returns a SAMPLED knowledge graph:
    - At most MAX_GRAPH_NODES nodes / MAX_GRAPH_EDGES edges
    - Edges sorted by confidence descending (highest-quality first)
    - Noise nodes filtered
    - Optional filters: relation, provenance, min_confidence, min_evidence
    """
    if not DEFAULT_KG_PATH.exists():
        return jsonify({"nodes": [], "edges": [], "stats": {
            "node_count": 0, "edge_count": 0, "total_edges": 0,
            "relations": [], "sampled": False
        }})

    payload = _safe_json_load(DEFAULT_KG_PATH) or {}
    raw_edges = payload.get("edges", [])
    raw_nodes = payload.get("nodes", {})

    relation_f   = request.args.get("relation",   "").strip()
    provenance_f = request.args.get("provenance", "").strip()
    min_conf     = float(request.args.get("min_confidence", 0) or 0)
    min_ev       = int(request.args.get("min_evidence", 1) or 1)

    # Filter
    filtered = []
    for e in raw_edges:
        if _is_noise_edge(e):
            continue
        if relation_f   and e.get("relation") != relation_f:
            continue
        if provenance_f and e.get("provenance", "internal") != provenance_f:
            continue
        if e.get("confidence", 0.0) < min_conf:
            continue
        if e.get("evidence", 1) < min_ev:
            continue
        filtered.append(e)

    total_filtered = len(filtered)

    # Sort by confidence desc and cap
    filtered.sort(key=lambda e: e.get("confidence", 0.0), reverse=True)
    sampled_edges = filtered[:MAX_GRAPH_EDGES]
    sampled = total_filtered > MAX_GRAPH_EDGES

    # Collect active nodes
    degree: dict      = {}
    active_nodes: set = set()
    for e in sampled_edges:
        src, tgt = e["source"], e["target"]
        degree[src] = degree.get(src, 0) + 1
        degree[tgt] = degree.get(tgt, 0) + 1
        active_nodes.add(src)
        active_nodes.add(tgt)

    # If still too many nodes, keep highest-degree ones
    if len(active_nodes) > MAX_GRAPH_NODES:
        top_nodes = set(sorted(degree, key=degree.__getitem__, reverse=True)[:MAX_GRAPH_NODES])
        sampled_edges = [e for e in sampled_edges
                         if e["source"] in top_nodes and e["target"] in top_nodes]
        active_nodes  = top_nodes
        sampled = True

    nodes_out = [
        {
            "id":         nid,
            "type":       raw_nodes.get(nid, {}).get("type", "concept"),
            "degree":     degree.get(nid, 1),
        }
        for nid in active_nodes
    ]

    return jsonify({
        "nodes": nodes_out,
        "edges": sampled_edges,
        "stats": {
            "node_count":   len(nodes_out),
            "edge_count":   len(sampled_edges),
            "total_edges":  total_filtered,
            "relations":    sorted({e["relation"] for e in sampled_edges}),
            "sampled":      sampled,
        },
    })


# ── Graph neighbourhood drill-down ───────────────────────────────────────────

@app.route("/api/graph/neighbors/<node_id>")
def api_graph_neighbors(node_id: str):
    """Return 1-hop neighbourhood of a node (max 80 edges, all directions)."""
    if not DEFAULT_KG_PATH.exists():
        return jsonify({"nodes": [], "edges": []})

    payload   = _safe_json_load(DEFAULT_KG_PATH) or {}
    raw_edges = payload.get("edges", [])
    raw_nodes = payload.get("nodes", {})

    neighbourhood = [
        e for e in raw_edges
        if not _is_noise_edge(e) and (e.get("source") == node_id or e.get("target") == node_id)
    ]
    neighbourhood.sort(key=lambda e: e.get("confidence", 0.0), reverse=True)
    neighbourhood = neighbourhood[:80]

    node_ids = {node_id}
    for e in neighbourhood:
        node_ids.add(e["source"])
        node_ids.add(e["target"])

    nodes_out = [
        {"id": nid, "type": raw_nodes.get(nid, {}).get("type", "concept"),
         "center": nid == node_id}
        for nid in node_ids
    ]
    return jsonify({"nodes": nodes_out, "edges": neighbourhood})


# ── Causal chains ─────────────────────────────────────────────────────────────

@app.route("/api/causal_chains")
def api_causal_chains():
    if not DEFAULT_KG_PATH.exists():
        return jsonify({"inferred": [], "direct": [], "stats": {}})

    payload   = _safe_json_load(DEFAULT_KG_PATH) or {}
    raw_edges = payload.get("edges", [])

    inferred, direct = [], []
    for e in raw_edges:
        if _is_noise_edge(e):
            continue
        rel = e.get("relation", "")
        if rel == "inferred_affects":
            inferred.append(e)
        elif rel in ("positive_affects", "negative_affects", "mixed_affects",
                     "bidirectional_affects", "affects"):
            direct.append(e)

    inferred.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    direct.sort(key=lambda x: x.get("evidence", 0), reverse=True)

    # Enrich with memory store confidence
    if DEFAULT_MEMORY_PATH.exists():
        try:
            mem_rules = {
                (r["cause"], r["effect"]): r
                for r in (_safe_json_load(DEFAULT_MEMORY_PATH) or {}).get("causal_rules", [])
            }
            for e in direct:
                key = (e.get("source", ""), e.get("target", ""))
                if key in mem_rules:
                    mr = mem_rules[key]
                    e["observations"] = mr["observations"]
                    e["confidence"]   = round(mr["confidence"], 4)
                    e["direction"]    = mr["direction"]
        except Exception:
            pass

    return jsonify({
        "inferred": inferred[:50],
        "direct":   direct[:50],
        "stats": {
            "inferred_count": len(inferred),
            "direct_count":   len(direct),
            "total_causal":   len(inferred) + len(direct),
        },
    })


# ── Abstraction nodes ─────────────────────────────────────────────────────────

@app.route("/api/abstraction_nodes")
def api_abstraction_nodes():
    if not DEFAULT_KG_PATH.exists():
        return jsonify({"concepts": []})
    payload   = _safe_json_load(DEFAULT_KG_PATH) or {}
    raw_nodes = payload.get("nodes", {})
    raw_edges = payload.get("edges", [])
    abstract  = {k for k, v in raw_nodes.items() if v.get("type") == "abstract_concept"}
    concepts  = []
    for node in abstract:
        members = [e["source"] for e in raw_edges
                   if e.get("target") == node and e.get("relation") == "is_instance_of"]
        confs   = [e.get("confidence", 0.0) for e in raw_edges
                   if e.get("target") == node and e.get("relation") == "is_instance_of"]
        avg_c   = round(sum(confs) / len(confs), 3) if confs else 0.0
        concepts.append({"name": node, "members": members,
                          "member_count": len(members), "avg_confidence": avg_c})
    concepts.sort(key=lambda x: x["member_count"], reverse=True)
    return jsonify({"concepts": concepts})


# ── Provenance breakdown ──────────────────────────────────────────────────────

@app.route("/api/graph_provenance")
def api_graph_provenance():
    if not DEFAULT_KG_PATH.exists():
        return jsonify({})
    payload = _safe_json_load(DEFAULT_KG_PATH) or {}
    counts: dict = {}
    for e in payload.get("edges", []):
        if _is_noise_edge(e):
            continue
        prov        = e.get("provenance", "internal")
        counts[prov] = counts.get(prov, 0) + 1
    return jsonify(counts)


# ── Snapshot charts (unchanged from original) ─────────────────────────────────

@app.route("/api/snapshot_charts")
def api_snapshot_charts():
    empty = {"causal_rules": [], "hypothesis_dist": [], "top_patterns": [], "evidence_dist": []}
    mem   = _safe_json_load(DEFAULT_MEMORY_PATH) or {}
    if not mem:
        return jsonify(empty)

    causal_out = [
        {"label": f"{r['cause']} → {r['effect']}", "cause": r["cause"],
         "effect": r["effect"], "observations": r["observations"],
         "confidence": round(r["confidence"], 3), "direction": r["direction"]}
        for r in sorted(mem.get("causal_rules", []),
                        key=lambda x: x["observations"], reverse=True)
    ]

    rules = mem.get("world_model", {}).get("rules", [])
    buckets = [("0.0–0.2", 0), ("0.2–0.4", 0), ("0.4–0.6", 0), ("0.6–0.8", 0), ("0.8–1.0", 0)]
    for r in rules:
        idx = min(4, int(float(r.get("confidence", 0)) * 5))
        buckets[idx] = (buckets[idx][0], buckets[idx][1] + 1)

    rel_support: dict = {}
    for p in mem.get("patterns", []):
        rel = p.get("relation", "unknown")
        rel_support[rel] = rel_support.get(rel, 0) + p.get("support", 0)
    top_patterns = [{"relation": rel, "support": sup}
                    for rel, sup in sorted(rel_support.items(), key=lambda x: x[1], reverse=True)[:12]]

    ev_buckets = [("1", 0), ("2–9", 0), ("10–49", 0), ("50–199", 0), ("200+", 0)]
    for f in mem.get("facts", []):
        ev = f.get("evidence", 1)
        idx = 0 if ev == 1 else 1 if ev < 10 else 2 if ev < 50 else 3 if ev < 200 else 4
        ev_buckets[idx] = (ev_buckets[idx][0], ev_buckets[idx][1] + 1)

    return jsonify({
        "causal_rules":    causal_out,
        "hypothesis_dist": [{"bucket": b, "count": n} for b, n in buckets],
        "top_patterns":    top_patterns,
        "evidence_dist":   [{"bucket": b, "count": n} for b, n in ev_buckets],
    })


# ── Larry Tool Activity endpoint ──────────────────────────────────────────────

@app.route("/api/tool-events")
def api_tool_events():
    """
    Returns recent Larry tool usage events read from tool_events.jsonl.
    Optional query param: since=<line_index>
    """
    try:
        since = int(request.args.get("since", 0))
    except (ValueError, TypeError):
        since = 0

    events, total = _read_tool_events_file(since_line=since)

    return jsonify({
        "events":          events,
        "total":           total,
        "tools_available": _HAS_TOOLS,
        "file":            str(_TOOL_EVENTS_FILE),
        "file_exists":     _TOOL_EVENTS_FILE.exists(),
    })


# ── Perception API ───────────────────────────────────────────────────────────

@app.route("/api/perception")
def api_perception():
    """Return recent perception events and summary stats."""
    events = _read_perception_log(max_events=int(request.args.get("n", 20)))
    total_triples = sum(len(e.get("triples", [])) for e in events)
    return jsonify({
        "events":        events,
        "total_events":  len(events),
        "total_triples": total_triples,
        "log_exists":    DEFAULT_PERCEPTION_LOG.exists(),
    })


@app.route("/api/perception/toggle", methods=["POST"])
def api_perception_toggle():
    """Pause or resume automatic screen capture.

    Works in two modes:
    - Separate processes (normal): writes screen_capture_state.json which the
      agent polls on every capture cycle.
    - Same process: also calls pause()/resume() on the live tracker directly
      for instant effect.

    POST body (JSON): {"enabled": true|false}
    Omit body to toggle current state.
    """
    data = request.get_json(silent=True) or {}

    # Determine new desired state
    if "enabled" in data:
        new_enabled = bool(data["enabled"])
    else:
        # Toggle from current file state
        new_enabled = not is_screen_capture_enabled()

    # 1. Write to state file (works across processes)
    _write_screen_state(new_enabled)

    # 2. Also apply immediately if in-process tracker exists
    tracker = get_screen_tracker()
    if tracker is not None:
        if new_enabled:
            tracker.resume()
        else:
            tracker.pause()

    paused = not new_enabled
    return jsonify({"ok": True, "paused": paused, "enabled": new_enabled})


@app.route("/api/perception/status")
def api_perception_status():
    """Return current screen capture status.

    Reads from the state file first (works standalone), then enriches with
    live tracker metrics if the agent is in-process.
    """
    enabled = is_screen_capture_enabled()
    tracker = get_screen_tracker()
    resp = {
        "registered": tracker is not None,
        "paused":     not enabled,
        "enabled":    enabled,
        "frames":     getattr(tracker, "total_frames", 0),
        "triples":    getattr(tracker, "total_triples", 0),
        "kg_edges":   getattr(tracker, "total_perception_edges", 0),
    }
    return jsonify(resp)


@app.route("/api/wiki")
def api_wiki():
    """Return recent Wikipedia perception events and summary stats."""
    events = _read_wiki_log(max_events=int(request.args.get("n", 20)))
    total_triples = sum(e.get("triples_count", len(e.get("triples", []))) for e in events)
    return jsonify({
        "events":        events,
        "total_events":  len(events),
        "total_triples": total_triples,
        "log_exists":    DEFAULT_WIKI_LOG.exists(),
    })


# ── Health check ──────────────────────────────────────────────────────────────

@app.route("/api/health")
def api_health():
    snap = _store.get_full()
    return jsonify({
        "ok":         True,
        "seq":        _store.current_seq,
        "latest_tick": snap.get("latest_tick", 0),
        "log_exists": DEFAULT_LOG_PATH.exists(),
        "mem_exists": DEFAULT_MEMORY_PATH.exists(),
        "kg_exists":  DEFAULT_KG_PATH.exists(),
    })


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=5055, debug=False, threaded=True)