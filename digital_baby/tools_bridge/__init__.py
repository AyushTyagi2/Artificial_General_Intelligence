"""
digital_baby.tools_bridge
=========================
Thin bridge between the digital_baby brain and the tools/ package.

Provides a single shared ToolRouter instance plus purpose-built helpers
the brain modules call directly.  Degrades gracefully to a no-op router
if the tools/ directory is absent or misconfigured.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Resolve project root (works regardless of CWD or invocation method) ───
# __file__ = .../Digital_Brain/digital_baby/tools_bridge/__init__.py
_HERE         = Path(__file__).resolve()
_TOOLS_BRIDGE = _HERE.parent          # .../digital_baby/tools_bridge/
_DIGITAL_BABY = _TOOLS_BRIDGE.parent  # .../digital_baby/
_PROJECT_ROOT = _DIGITAL_BABY.parent  # .../Digital_Brain/
_TOOLS_DIR    = _PROJECT_ROOT / "tools"

# Ensure project root is on sys.path so absolute imports work everywhere
for _p in (str(_PROJECT_ROOT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_router: Optional[object] = None


def _bootstrap_tools_package() -> None:
    """
    Register tools/ as a proper Python package in sys.modules so that
    all tool sub-modules can do `from tools.base import ok, err` correctly.
    Only runs once (guarded by sys.modules check).
    """
    if "tools" in sys.modules:
        return
    if not _TOOLS_DIR.exists():
        raise FileNotFoundError(f"tools/ directory not found at {_TOOLS_DIR}")

    # 1. Register the package shell
    pkg = types.ModuleType("tools")
    pkg.__path__    = [str(_TOOLS_DIR)]
    pkg.__package__ = "tools"
    pkg.__file__    = str(_TOOLS_DIR / "__init__.py")
    sys.modules["tools"] = pkg

    # 2. Execute __init__.py so it populates the namespace (imports ToolRouter etc.)
    init_path = _TOOLS_DIR / "__init__.py"
    spec = importlib.util.spec_from_file_location("tools", str(init_path))
    if spec and spec.loader:
        spec.loader.exec_module(pkg)  # type: ignore[attr-defined]


def get_router():
    """Return the shared ToolRouter, creating it on first call."""
    global _router
    if _router is not None:
        return _router

    try:
        _bootstrap_tools_package()

        # Load router.py via file path (Windows-safe)
        router_path = _TOOLS_DIR / "router.py"
        spec = importlib.util.spec_from_file_location("tools.router", str(router_path))
        if spec is None or spec.loader is None:
            raise ImportError("Cannot create spec for tools.router")

        mod = importlib.util.module_from_spec(spec)
        sys.modules["tools.router"] = mod
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]

        _router = mod.ToolRouter()
        logger.info("[tools_bridge] router ready  tools=%s",
                    getattr(_router, "available_tools", []))

    except Exception as exc:
        logger.warning("[tools_bridge] router unavailable: %s", exc)
        _router = _NoopRouter()

    return _router


# ── Noop fallback ──────────────────────────────────────────────────────────

class _NoopRouter:
    """Returned when tools/ cannot be loaded — all dispatches return an error."""
    available_tools: list = []

    def dispatch(self, request: dict) -> dict:
        return {"status": "error",
                "output": {"error": "tools package not available"}}