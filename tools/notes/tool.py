"""
Tool: notes  (long-term persistent memory)
==========================================
JSON-backed note store.  The brain uses this for durable knowledge that
must survive process restarts.

Actions
-------
    create  – title (str), content (str), tags (list[str], optional)
    read    – id (int)
    update  – id (int), title / content / tags (any subset)
    delete  – id (int)
    list    – tag (str, optional filter), limit (int, default 20)
    search  – query (str), limit (int, default 10)

All inputs also accept an optional  store_path (str)  to override the
default notes file location (useful for testing).

Output (success) varies by action — always inside
    {"status": "success", "output": {...}}
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from tools.base import ok, err

_DEFAULT_STORE = Path(os.getenv("NOTES_FILE", "notes.json"))


# ── Storage helpers ────────────────────────────────────────────────────────

def _load(path: Path) -> List[dict]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save(notes: List[dict], path: Path) -> None:
    path.write_text(
        json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ── CRUD helpers ───────────────────────────────────────────────────────────

def _create(notes, title, content, tags=None):
    new_id = max((n["id"] for n in notes), default=0) + 1
    note = {
        "id": new_id, "title": title, "content": content,
        "tags": tags or [], "created_at": _now(), "updated_at": _now(),
    }
    notes.append(note)
    return note


def _find(notes, note_id):
    return next((n for n in notes if n["id"] == note_id), None)


def _update(notes, note_id, **fields):
    note = _find(notes, note_id)
    if note is None:
        return None
    for f in ("title", "content", "tags"):
        if f in fields and fields[f] is not None:
            note[f] = fields[f]
    note["updated_at"] = _now()
    return note


def _delete(notes, note_id):
    for i, n in enumerate(notes):
        if n["id"] == note_id:
            notes.pop(i)
            return True
    return False


# ── Entry point ───────────────────────────────────────────────────────────

def run(input: dict) -> dict:
    action = input.get("action", "list")
    path   = Path(input.get("store_path", str(_DEFAULT_STORE)))
    notes  = _load(path)

    try:
        # ── create ──────────────────────────────────────────────────────
        if action == "create":
            title   = (input.get("title")   or "").strip()
            content = (input.get("content") or "").strip()
            if not title or not content:
                return err("'title' and 'content' are required for create.")
            note = _create(notes, title, content, input.get("tags"))
            _save(notes, path)
            return ok({"note": note,
                        "message": f"Note '{title}' created (id={note['id']})."})

        # ── read ─────────────────────────────────────────────────────────
        elif action == "read":
            note_id = input.get("id")
            if note_id is None:
                return err("'id' is required for read.")
            note = _find(notes, int(note_id))
            if note is None:
                return err(f"Note id={note_id} not found.")
            return ok({"note": note})

        # ── update ───────────────────────────────────────────────────────
        elif action == "update":
            note_id = input.get("id")
            if note_id is None:
                return err("'id' is required for update.")
            note = _update(notes, int(note_id),
                           title=input.get("title"),
                           content=input.get("content"),
                           tags=input.get("tags"))
            if note is None:
                return err(f"Note id={note_id} not found.")
            _save(notes, path)
            return ok({"note": note, "message": "Note updated."})

        # ── delete ───────────────────────────────────────────────────────
        elif action == "delete":
            note_id = input.get("id")
            if note_id is None:
                return err("'id' is required for delete.")
            if not _delete(notes, int(note_id)):
                return err(f"Note id={note_id} not found.")
            _save(notes, path)
            return ok({"deleted_id": note_id,
                        "message": f"Note id={note_id} deleted."})

        # ── list ─────────────────────────────────────────────────────────
        elif action == "list":
            tag   = input.get("tag")
            limit = max(1, int(input.get("limit", 20)))
            result = [n for n in notes if not tag or tag in n.get("tags", [])]
            result = result[-limit:]
            return ok({"notes": result, "count": len(result)})

        # ── search ───────────────────────────────────────────────────────
        elif action == "search":
            query = (input.get("query") or "").strip()
            if not query:
                return err("'query' is required for search.")
            limit = max(1, int(input.get("limit", 10)))
            q = query.lower()
            matches = [
                n for n in notes
                if q in n.get("title", "").lower()
                or q in n.get("content", "").lower()
            ]
            return ok({"query": query, "notes": matches[:limit],
                        "count": len(matches[:limit])})

        else:
            return err(
                f"Unknown action '{action}'. "
                "Valid: create, read, update, delete, list, search."
            )

    except Exception as exc:
        return err(f"Notes error: {exc}")