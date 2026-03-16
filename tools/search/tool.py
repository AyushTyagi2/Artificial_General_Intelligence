"""
Tool: search
============
Performs a web search using the DuckDuckGo Instant Answer API.

Input
-----
    query       : str   (required)
    max_results : int   (optional, default 5, max 20)

Output (success)
----------------
    {
        "query":   "...",
        "results": [
            {"title": "...", "url": "...", "snippet": "..."},
            ...
        ],
        "count": int
    }
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Dict, List

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
from tools.base import ok, err


def _ddg_search(query: str, max_results: int) -> List[Dict[str, str]]:
    encoded = urllib.parse.quote_plus(query)
    url = (
        f"https://api.duckduckgo.com/?q={encoded}"
        "&format=json&no_html=1&skip_disambig=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "DigitalBrain/1.0"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode())

    results: List[Dict[str, str]] = []

    # Top abstract result
    if data.get("Abstract"):
        results.append({
            "title":   data.get("Heading", query),
            "url":     data.get("AbstractURL", ""),
            "snippet": data["Abstract"],
        })

    # Related topics
    for topic in data.get("RelatedTopics", []):
        if len(results) >= max_results:
            break
        if "Text" in topic:
            results.append({
                "title":   topic.get("Text", "")[:80],
                "url":     topic.get("FirstURL", ""),
                "snippet": topic.get("Text", ""),
            })
        for sub in topic.get("Topics", []):
            if len(results) >= max_results:
                break
            if "Text" in sub:
                results.append({
                    "title":   sub.get("Text", "")[:80],
                    "url":     sub.get("FirstURL", ""),
                    "snippet": sub.get("Text", ""),
                })

    return results[:max_results]


def run(input: dict) -> dict:
    query = input.get("query", "").strip()
    if not query:
        return err("'query' is required and must not be empty.")

    max_results = min(max(1, int(input.get("max_results", 5))), 20)

    try:
        results = _ddg_search(query, max_results)
    except Exception as exc:
        return err(f"Search failed: {exc}")

    return ok({"query": query, "results": results, "count": len(results)})