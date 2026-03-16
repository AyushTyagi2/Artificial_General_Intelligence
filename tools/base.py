"""
Shared helpers for building standardized tool responses.

Every tool must return:
    {"status": "success" | "error", "output": {...}}

Use ok() and err() to build these responses consistently.
"""
from __future__ import annotations
from typing import Any


def ok(output: dict | None = None) -> dict:
    """Build a success response."""
    return {"status": "success", "output": output or {}}


def err(message: str, **extra: Any) -> dict:
    """Build an error response."""
    return {"status": "error", "output": {"error": message, **extra}}