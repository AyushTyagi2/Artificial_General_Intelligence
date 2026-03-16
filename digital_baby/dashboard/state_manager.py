"""Dashboard state manager with lightweight caching and aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json

def _safe_json_load(path, default=None):
    """Read a JSON file tolerantly — returns default on missing, empty, or corrupt file."""
    import json, shutil, time, logging
    from pathlib import Path
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
        logging.getLogger(__name__).warning(
            "Corrupt JSON file %s — backed up to %s, using default.", p, backup
        )
        return default

import time

from digital_baby.dashboard.log_parser import LogParser


_REPO_ROOT = Path(__file__).resolve().parents[2]   # dashboard/ → digital_baby/ → repo root


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return _REPO_ROOT / p


class DashboardStateManager:
    """Loads and caches dashboard state from logs + memory JSON."""

    def __init__(self, log_path: str | Path, memory_path: str | Path) -> None:
        self.log_path    = _resolve_path(log_path)
        self.memory_path = _resolve_path(memory_path)
        self._last_log_mtime = -1.0
        self._last_mem_mtime = -1.0
        self._cached: Dict[str, Any] = {
            "events": {
                "conflicts":            [],
                "prediction_failures":  [],
                "reward_history":       [],
                "summaries":            [],
                "experiments":          [],
                "causal_discoveries":   [],
                "validation_events":    [],
                "law_discovery_events": [],
                "consolidation_events": [],
            },
            "memory": {
                "facts": 0, "rules": 0, "predictions": 0,
                "experiments": 0, "causal_rules": 0, "patterns": 0,
            },
            "updated_at": time.time(),
        }

    def _read_memory_summary(self) -> Dict[str, int]:
        if not self.memory_path.exists():
            return {"facts": 0, "rules": 0, "predictions": 0,
                    "experiments": 0, "causal_rules": 0, "patterns": 0}
        try:
            payload = _safe_json_load(self.memory_path) or {}
            wm = payload.get("world_model", {})
            return {
                "facts":           len(payload.get("facts",        [])),
                "rules":           len(wm.get("rules",             [])),
                # Use cumulative totals; fall back to buffer len for old stores
                "predictions":     payload.get("total_predictions",  len(wm.get("predictions", []))),
                "experiments":     payload.get("total_experiments",  len(wm.get("experiments", []))),
                "predictions_buf": len(wm.get("predictions",        [])),
                "experiments_buf": len(wm.get("experiments",        [])),
                "causal_rules":    len(payload.get("causal_rules",  [])),
                "patterns":        len(payload.get("patterns",      [])),
            }
        except Exception:
            return {"facts": 0, "rules": 0, "predictions": 0,
                    "experiments": 0, "causal_rules": 0, "patterns": 0}

    def refresh(self) -> Dict[str, Any]:
        now = time.time()

        log_mtime = self.log_path.stat().st_mtime if self.log_path.exists() else -1.0
        if log_mtime != self._last_log_mtime:
            self._cached["events"] = LogParser(self.log_path).parse()
            self._last_log_mtime   = log_mtime

        mem_mtime = self.memory_path.stat().st_mtime if self.memory_path.exists() else -1.0
        if mem_mtime != self._last_mem_mtime:
            self._cached["memory"] = self._read_memory_summary()
            self._last_mem_mtime   = mem_mtime

        self._cached["updated_at"] = now
        return self._cached

    def snapshot(self) -> Dict[str, Any]:
        state  = self.refresh()
        events = state["events"]

        summaries = events.get("summaries", [])

        failures_by_tick: Dict[int, int] = {}
        for f in events.get("prediction_failures", []):
            failures_by_tick[f["tick"]] = failures_by_tick.get(f["tick"], 0) + 1

        return {
            "updated_at":           state["updated_at"],
            "memory":               state["memory"],
            "conflicts":            events.get("conflicts",            []),
            "prediction_failures":  events.get("prediction_failures",  []),
            "experiments":          events.get("experiments",          []),
            "reward_history":       events.get("reward_history",       []),
            "summaries":            summaries,
            "causal_discoveries":   events.get("causal_discoveries",   []),
            # FIX: new event types now passed through to snapshot
            "validation_events":    events.get("validation_events",    []),
            "law_discovery_events": events.get("law_discovery_events", []),
            "consolidation_events": events.get("consolidation_events", []),
            "failure_series": [
                {"tick": t, "count": c}
                for t, c in sorted(failures_by_tick.items())
            ],
        }