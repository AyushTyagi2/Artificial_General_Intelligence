"""Knowledge graph builders for dashboard visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
from pyvis.network import Network


def build_graph_data(
    conflicts: List[Dict],
    relation_filter: Optional[str] = None,
    min_confidence: float = 0.0,
    max_edges: int = 5000,
) -> Dict:
    """Build graph JSON payload for front-end graph rendering.

    Designed to stay responsive with large logs (edge cap).
    """
    g = nx.DiGraph()

    sorted_conflicts = sorted(conflicts, key=lambda x: float(x.get("confidence", 0.0)), reverse=True)[:max_edges]
    for event in sorted_conflicts:
        relation = event["relation"]
        confidence = float(event.get("confidence", 0.0))
        if relation_filter and relation != relation_filter:
            continue
        if confidence < min_confidence:
            continue

        src = event["entity"]
        dst = event["belief"]
        tick = int(event.get("tick", 0))
        g.add_node(src)
        g.add_node(dst)
        g.add_edge(src, dst, relation=relation, confidence=confidence, tick=tick)

    nodes = [{"id": n, "label": n, "title": f"Entity: {n}"} for n in g.nodes]
    edges = [
        {
            "from": s,
            "to": d,
            "label": attrs.get("relation", "rel"),
            "confidence": float(attrs.get("confidence", 0.0)),
            "tick": int(attrs.get("tick", 0)),
            "value": max(1.0, float(attrs.get("confidence", 0.0)) * 10),
            "title": f"{attrs.get('relation','rel')} (conf={float(attrs.get('confidence',0.0)):.2f})",
        }
        for s, d, attrs in g.edges(data=True)
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "nodes": len(nodes),
            "edges": len(edges),
            "relations": sorted({e["label"] for e in edges}),
        },
    }


def build_pyvis_html(graph_data: Dict, output_html: str | Path) -> None:
    """Optional PyVis export for standalone graph embedding."""
    net = Network(height="780px", width="100%", directed=True, notebook=False)
    net.barnes_hut()

    for node in graph_data.get("nodes", []):
        net.add_node(node["id"], label=node["label"], title=node.get("title", node["label"]))

    for edge in graph_data.get("edges", []):
        net.add_edge(
            edge["from"],
            edge["to"],
            label=edge.get("label", "rel"),
            value=edge.get("value", 1.0),
            title=edge.get("title", "relation"),
        )

    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_path), notebook=False, open_browser=False)
