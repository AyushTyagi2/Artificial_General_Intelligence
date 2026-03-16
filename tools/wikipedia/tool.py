"""
Tool: wikipedia
===============
Fetches article summaries from Wikipedia.

Uses the REST summary API (en.wikipedia.org/api/rest_v1/page/summary/)
which is more reliable and faster than the MediaWiki search API.
Responses are cached in-process (TTL=10 min) to avoid duplicate network
calls — knowledge_ingestion and hypothesis_validator both query the same
concepts, so caching eliminates most redundant requests.

Input
-----
    query        : str   (required)
    max_results  : int   (optional, kept for API compat)
    full_summary : bool  (optional, default False — truncates to 500 chars)

Output (success)
----------------
    {
        "query":      "...",
        "title":      "...",
        "summary":    "...",
        "url":        "https://en.wikipedia.org/wiki/...",
        "candidates": []
    }
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import urllib.error
from threading import Lock
from typing import Dict, Optional, Tuple

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from tools.base import ok, err

# ── REST API (reliable — same one knowledge_ingestion uses) ──────────────────
_REST = "https://en.wikipedia.org/api/rest_v1/page/summary/"
_UA   = "DigitalBrain/2.0 (educational-ai-agent)"

# ── In-process response cache ────────────────────────────────────────────────
_CACHE:     Dict[str, Tuple[float, dict]] = {}
_CACHE_TTL: float = 600.0   # 10 minutes
_CACHE_MAX: int   = 500
_cache_lock = Lock()


def _cache_get(key: str) -> Optional[dict]:
    with _cache_lock:
        entry = _CACHE.get(key)
        if entry and (time.time() - entry[0]) < _CACHE_TTL:
            return entry[1]
        if entry:
            del _CACHE[key]
        return None


def _cache_set(key: str, value: dict) -> None:
    with _cache_lock:
        if len(_CACHE) >= _CACHE_MAX:
            oldest = min(_CACHE, key=lambda k: _CACHE[k][0])
            del _CACHE[oldest]
        _CACHE[key] = (time.time(), value)


def _get_json(url: str, timeout: float = 10.0, retries: int = 2) -> Optional[dict]:
    """GET a URL returning parsed JSON, with retry on transient errors."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 400):
                return None   # definitive miss — no retry
            last_exc = exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(0.3 * (attempt + 1))
    raise last_exc or RuntimeError("unknown fetch error")


def _fetch(title: str, timeout: float = 10.0) -> Optional[dict]:
    """Fetch one article; checks cache first."""
    key = title.strip().lower().replace(" ", "_")
    cached = _cache_get(key)
    if cached is not None:
        return cached

    url  = _REST + urllib.parse.quote(title.replace(" ", "_"), safe="")
    data = _get_json(url, timeout=timeout)
    if not data:
        return None
    if "disambiguation" in data.get("type", "").lower():
        return None
    if not data.get("extract", "").strip():
        return None

    _cache_set(key, data)
    return data


def run(inp: dict) -> dict:
    query = inp.get("query", "").strip()
    if not query:
        return err("'query' is required and must not be empty.")

    full_summary = bool(inp.get("full_summary", False))

    try:
        data = _fetch(query)
    except Exception as exc:
        return err(f"Wikipedia fetch failed for '{query}': {exc}")

    if data is None:
        return err(f"No Wikipedia article found for '{query}'.")

    raw   = data.get("extract", "").strip()
    title = data.get("title", query)
    url   = (data.get("content_urls", {}).get("desktop", {}).get("page")
             or "https://en.wikipedia.org/wiki/"
             + urllib.parse.quote(title.replace(" ", "_"), safe=""))

    summary = raw if full_summary else (raw[:500] + ("…" if len(raw) > 500 else ""))

    return ok({
        "query":      query,
        "title":      title,
        "summary":    summary,
        "url":        url,
        "candidates": [],
    })