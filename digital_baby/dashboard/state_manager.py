"""Dashboard state manager with lightweight caching and aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json
import time

from digital_baby.dashboard.log_parser import LogParser


class DashboardStateManager:
    """Loads and caches dashboard state from logs + memory JSON."""

    def __init__(self, log_path: str | Path, memory_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.memory_path = Path(memory_path)
        self._last_log_mtime = -1.0
        self._cached: Dict[str, Any] = {
            "events": {
                "conflicts": [],
                "prediction_failures": [],
                "reward_history": [],
                "summaries": [],
                "experiments": [],
            },
            "memory": {"facts": 0, "rules": 0, "predictions": 0, "experiments": 0},
            "updated_at": time.time(),
        }

    def _read_memory_summary(self) -> Dict[str, int]:
        if not self.memory_path.exists():
            return {"facts": 0, "rules": 0, "predictions": 0, "experiments": 0}
        payload = json.loads(self.memory_path.read_text(encoding="utf-8"))
        wm = payload.get("world_model", {})
        return {
            "facts": len(payload.get("facts", [])),
            "rules": len(wm.get("rules", [])),
            "predictions": len(wm.get("predictions", [])),
            "experiments": len(wm.get("experiments", [])),
        }

    def refresh(self) -> Dict[str, Any]:
        log_mtime = self.log_path.stat().st_mtime if self.log_path.exists() else -1.0
        if log_mtime != self._last_log_mtime:
            self._cached["events"] = LogParser(self.log_path).parse()
            self._cached["memory"] = self._read_memory_summary()
            self._cached["updated_at"] = time.time()
            self._last_log_mtime = log_mtime
        else:
            self._cached["memory"] = self._read_memory_summary()
            self._cached["updated_at"] = time.time()
        return self._cached

    def snapshot(self) -> Dict[str, Any]:
        state = self.refresh()
        events = state["events"]
        summaries = events.get("summaries", [])

        failures_by_tick: Dict[int, int] = {}
        for f in events.get("prediction_failures", []):
            failures_by_tick[f["tick"]] = failures_by_tick.get(f["tick"], 0) + 1

        return {
            "updated_at": state["updated_at"],
            "memory": state["memory"],
            "conflicts": events.get("conflicts", []),
            "prediction_failures": events.get("prediction_failures", []),
            "experiments": events.get("experiments", []),
            "reward_history": events.get("reward_history", []),
            "summaries": summaries,
            "failure_series": [{"tick": t, "count": c} for t, c in sorted(failures_by_tick.items())],
        }
