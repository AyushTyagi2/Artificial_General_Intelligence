"""
ToolRouter
==========
Dynamically discovers and dispatches tool calls.

Uses file-based loading (importlib.util.spec_from_file_location) so it
works correctly on Windows regardless of working directory or how Python
was invoked (script, -m, IDE, etc.).

Usage
-----
    from tools import ToolRouter

    router = ToolRouter()
    result = router.dispatch({
        "tool": "wikipedia",
        "input": {"query": "photosynthesis"}
    })
    # -> {"status": "success", "output": {"title": "...", "summary": "..."}}

Tool request format
-------------------
    {"tool": "<tool_name>", "input": { ... }}

Response format (always)
------------------------
    {"status": "success" | "error", "output": { ... }}
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from types import ModuleType
from typing import Callable, Deque, Dict, List, Optional

from .base import err

logger = logging.getLogger(__name__)

_SKIP = {"__pycache__"}

# Ensure project root is on sys.path so tool modules can import from
# 'tools.base' regardless of OS or how Python was invoked.
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
_PROJECT_ROOT_STR = str(_PROJECT_ROOT)
if _PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_STR)

# ── Cross-process event file ──────────────────────────────────────────────────
# Written by the brain process, read by the dashboard process.
# Each line is a JSON object (JSON-lines format).
_TOOL_EVENTS_FILE = Path(
    os.environ.get("TOOL_EVENTS_FILE", str(_PROJECT_ROOT / "tool_events.jsonl"))
)
_MAX_FILE_BYTES = 2 * 1024 * 1024  # 2 MB — rotate when exceeded


class ToolRouter:
    """Load all tools from the tools/ directory and route requests to them."""

    def __init__(self, auto_discover: bool = True) -> None:
        self._registry: Dict[str, Callable[[dict], dict]] = {}
        if auto_discover:
            self._discover()

    # ── Discovery ─────────────────────────────────────────────────────────

    def _discover(self) -> None:
        tools_dir = Path(__file__).parent
        logger.info("[tools_bridge] scanning tools directory: %s", tools_dir)

        candidates = sorted(
            e for e in tools_dir.iterdir()
            if e.is_dir() and e.name not in _SKIP and not e.name.startswith("_")
        )

        if not candidates:
            logger.warning("[tools_bridge] no tool directories found in %s", tools_dir)
            return

        for entry in candidates:
            self._try_load(entry.name, entry)

    def _try_load(self, name: str, tool_dir: Path) -> None:
        # Search order:
        #   1. tool.py          (preferred — most tools use this)
        #   2. __init__.py      (fallback — package-style tool)
        candidates = [
            (f"tools.{name}.tool",  tool_dir / "tool.py"),
            (f"tools.{name}",       tool_dir / "__init__.py"),
        ]
        for module_name, file_path in candidates:
            if not file_path.exists():
                continue
            try:
                mod = self._load_from_file(module_name, file_path)
                if mod and hasattr(mod, "run"):
                    self.register(name, mod.run)
                    logger.info("[tools_bridge] discovered tool: %s", name)
                    return
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[tools_bridge] failed to load tool '%s' from %s: %s",
                    name, file_path, exc, exc_info=True,
                )
                return  # stop trying candidates for this tool on hard error

        logger.warning(
            "[tools_bridge] tool '%s' has no loadable 'run' — skipped "
            "(checked: tool.py, __init__.py)",
            name,
        )

    @staticmethod
    def _load_from_file(module_name: str, file_path: Path) -> Optional[ModuleType]:
        """
        Import a module directly from its absolute file path.
        Works on Windows regardless of sys.path / working directory.
        Re-uses cached modules from sys.modules.
        """
        if module_name in sys.modules:
            return sys.modules[module_name]
        try:
            spec = importlib.util.spec_from_file_location(
                module_name, str(file_path.resolve())
            )
            if spec is None or spec.loader is None:
                return None
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod       # register before exec
            spec.loader.exec_module(mod)         # type: ignore[attr-defined]
            return mod
        except Exception as exc:
            sys.modules.pop(module_name, None)   # clean up on failure
            logger.debug("[router] load failed '%s': %s", module_name, exc)
            return None

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, name: str, fn: Callable[[dict], dict]) -> None:
        if name in self._registry:
            logger.warning("[router] overwriting tool '%s'", name)
        self._registry[name] = fn
        logger.debug("[router] registered '%s'", name)

    # ── Dispatch ──────────────────────────────────────────────────────────

    def dispatch(self, request: dict) -> dict:
        """
        Route a tool request.
        request : {"tool": "<n>", "input": {<kwargs>}, "caller": "<module>"}
        returns : {"status": "success"|"error", "output": {...}}
        """
        if not isinstance(request, dict):
            return err("Request must be a dict.")

        tool_name = request.get("tool")
        if not tool_name:
            return err("Missing required field 'tool'.")

        tool_input = request.get("input", {})
        if not isinstance(tool_input, dict):
            return err(f"'input' must be a dict, got {type(tool_input).__name__}.")

        # Detect caller module automatically if not provided
        caller = request.get("caller") or _get_caller_module()

        fn = self._registry.get(tool_name)
        if fn is None:
            _emit_tool_event(tool_name, tool_input, caller, "error")
            return err(
                f"Unknown tool '{tool_name}'.",
                available_tools=sorted(self._registry),
            )

        status = "success"
        try:
            result = fn(tool_input)
        except Exception as exc:
            logger.exception("[router] tool '%s' raised: %s", tool_name, exc)
            status = "error"
            _emit_tool_event(tool_name, tool_input, caller, status)
            return err(f"Tool '{tool_name}' raised an exception: {exc}")

        if not isinstance(result, dict) or \
                "status" not in result or "output" not in result:
            result = {"status": "success", "output": result}

        status = result.get("status", "success")
        _emit_tool_event(tool_name, tool_input, caller, status)
        return result

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def available_tools(self) -> List[str]:
        return sorted(self._registry)

    def __repr__(self) -> str:
        return f"ToolRouter(tools={self.available_tools})"


# ── Tool Event Store ──────────────────────────────────────────────────────────
# A lightweight in-process ring buffer for tool usage events.
# The dashboard server reads from this to populate the "Larry Tool Activity"
# panel without any file I/O or external dependencies.

_MAX_TOOL_EVENTS = 200

# Each entry: {type, tool, timestamp, caller, input_summary, status}
_tool_events: Deque[dict] = deque(maxlen=_MAX_TOOL_EVENTS)
_tool_event_listeners: List[Callable[[dict], None]] = []


def _make_input_summary(tool_name: str, tool_input: dict) -> str:
    """Build a compact human-readable summary of the tool call input."""
    if not tool_input:
        return "(no input)"
    parts = []
    priority_keys = ["query", "message", "text", "topic", "key", "content", "name"]
    # Skip boolean flags and internal keys — only show meaningful string/numeric values
    filtered = {
        k: v for k, v in tool_input.items()
        if not isinstance(v, bool) and k not in ("full_summary", "evidence_increment")
    }
    keys = sorted(filtered.keys(), key=lambda k: (k not in priority_keys, k))
    for k in keys[:2]:
        v = str(filtered[k])
        if len(v) > 60:
            v = v[:57] + "..."
        parts.append(f'{k}="{v}"')
    return ", ".join(parts) if parts else "(no input)"


def _get_caller_module() -> str:
    """Walk the call stack to find the first non-tools frame."""
    try:
        for frame_info in inspect.stack()[2:]:
            mod = frame_info[0].f_globals.get("__name__", "")
            if mod and not mod.startswith("tools") and mod != "__main__":
                return mod.split(".")[-1]
        return "unknown"
    except Exception:
        return "unknown"


def _emit_tool_event(
    tool_name: str,
    tool_input: dict,
    caller: str,
    status: str,
) -> None:
    """Record a tool usage event in-process and persist to the cross-process file."""
    event = {
        "type":          "tool_usage",
        "tool":          tool_name,
        "timestamp":     time.strftime("%H:%M:%S"),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "caller":        caller,
        "input_summary": _make_input_summary(tool_name, tool_input),
        "status":        status,
    }
    _tool_events.append(event)

    # Persist to file so the dashboard (separate process) can read it
    try:
        # Rotate if file is getting large
        if _TOOL_EVENTS_FILE.exists() and _TOOL_EVENTS_FILE.stat().st_size > _MAX_FILE_BYTES:
            _TOOL_EVENTS_FILE.rename(
                _TOOL_EVENTS_FILE.with_suffix(".jsonl.bak")
            )
        with _TOOL_EVENTS_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as _fe:
        logger.debug("[router] tool_event file write failed: %s", _fe)

    logger.debug("[router] tool_event tool=%s caller=%s status=%s", tool_name, caller, status)
    for cb in list(_tool_event_listeners):
        try:
            cb(event)
        except Exception:
            pass


def get_tool_events(since_index: int = 0) -> List[dict]:
    """Return tool events newer than since_index (0 = all)."""
    events = list(_tool_events)
    if since_index <= 0:
        return events
    return events[since_index:]


def subscribe_tool_events(callback: Callable[[dict], None]) -> None:
    """Register a callback that receives each new tool event dict."""
    _tool_event_listeners.append(callback)


def unsubscribe_tool_events(callback: Callable[[dict], None]) -> None:
    try:
        _tool_event_listeners.remove(callback)
    except ValueError:
        pass