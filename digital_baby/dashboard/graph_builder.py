"""Build interactive knowledge graph views from parsed conflict events."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import networkx as nx
from pyvis.network import Network


def build_conflict_graph(conflicts: List[Dict], output_html: str | Path) -> None:
    """Build directed graph from conflict records and export via PyVis."""
    g = nx.DiGraph()

    for event in conflicts:
        src = event["entity"]
        dst = event["belief"]
        relation = event["relation"]
        confidence = float(event.get("confidence", 0.0))

        if g.has_edge(src, dst):
            g[src][dst]["weight"] = max(g[src][dst].get("weight", 0.0), confidence)
            g[src][dst]["label"] = relation
        else:
            g.add_edge(src, dst, weight=confidence, label=relation)

    net = Network(height="780px", width="100%", directed=True, notebook=False)
    net.barnes_hut()

    for node in g.nodes:
        net.add_node(node, label=node, title=f"Entity: {node}")

    for src, dst, attrs in g.edges(data=True):
        weight = float(attrs.get("weight", 0.0))
        label = attrs.get("label", "relation")
        net.add_edge(
            src,
            dst,
            value=max(1.0, weight * 10.0),
            title=f"{label} (confidence={weight:.2f})",
            label=label,
        )

    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(output_path), notebook=False, open_browser=False)
