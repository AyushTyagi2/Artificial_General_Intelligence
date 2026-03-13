"""Real-time dashboard server for monitoring digital_baby learning internals."""

from __future__ import annotations

from pathlib import Path
import json
import os
import time

from flask import Flask, jsonify, render_template, request
from flask_sock import Sock

from digital_baby.dashboard.graph_builder import build_graph_data, build_pyvis_html
from digital_baby.dashboard.state_manager import DashboardStateManager


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
DEFAULT_LOG_PATH = Path(os.environ.get("DIGITAL_BABY_LOG", "/tmp/db_exp.log"))
DEFAULT_MEMORY_PATH = Path(os.environ.get("DIGITAL_BABY_MEMORY", "digital_baby/world/memory_store.json"))
PYVIS_GRAPH_PATH = TEMPLATES_DIR / "knowledge_graph.html"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
sock = Sock(app)
state_manager = DashboardStateManager(DEFAULT_LOG_PATH, DEFAULT_MEMORY_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/graph")
def graph():
    relation = request.args.get("relation")
    min_conf = float(request.args.get("min_confidence", "0") or 0.0)
    snapshot = state_manager.snapshot()
    graph_data = build_graph_data(snapshot["conflicts"], relation_filter=relation, min_confidence=min_conf)
    build_pyvis_html(graph_data, PYVIS_GRAPH_PATH)
    return render_template("knowledge_graph.html")


@app.route("/api/state")
def api_state():
    return jsonify(state_manager.snapshot())


@app.route("/api/metrics")
def api_metrics():
    snapshot = state_manager.snapshot()
    return jsonify(
        {
            "memory": snapshot["memory"],
            "reward_history": snapshot["reward_history"][-500:],
            "failure_series": snapshot["failure_series"][-500:],
            "summaries": snapshot["summaries"][-500:],
        }
    )


@app.route("/api/conflicts")
def api_conflicts():
    snapshot = state_manager.snapshot()
    return jsonify(snapshot["conflicts"][-2000:])


@app.route("/api/graph")
def api_graph():
    relation = request.args.get("relation")
    min_conf = float(request.args.get("min_confidence", "0") or 0.0)
    snapshot = state_manager.snapshot()
    return jsonify(build_graph_data(snapshot["conflicts"], relation_filter=relation, min_confidence=min_conf, max_edges=10000))


@app.route("/api/failures")
def api_failures():
    snapshot = state_manager.snapshot()
    return jsonify(snapshot["prediction_failures"][-500:])


@app.route("/api/stream-snapshot")
def api_stream_snapshot():
    snapshot = state_manager.snapshot()
    events = (snapshot["prediction_failures"][-20:] + snapshot["experiments"][-20:])[-30:]
    return jsonify(events)


@sock.route("/ws/stream")
def ws_stream(ws):
    while True:
        payload = state_manager.snapshot()
        compact = {
            "updated_at": payload["updated_at"],
            "memory": payload["memory"],
            "latest_summary": payload["summaries"][-1] if payload["summaries"] else None,
            "latest_failure": payload["prediction_failures"][-1] if payload["prediction_failures"] else None,
            "latest_experiment": payload["experiments"][-1] if payload["experiments"] else None,
        }
        ws.send(json.dumps(compact))
        time.sleep(2)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055, debug=False)
