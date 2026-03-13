"""Flask server for digital_baby learning dashboard."""

from __future__ import annotations

from pathlib import Path
import json
import os

from flask import Flask, render_template

from digital_baby.dashboard.graph_builder import build_conflict_graph
from digital_baby.dashboard.log_parser import LogParser


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
DEFAULT_LOG_PATH = Path(os.environ.get("DIGITAL_BABY_LOG", "/tmp/db_exp.log"))
DEFAULT_MEMORY_PATH = Path(os.environ.get("DIGITAL_BABY_MEMORY", "digital_baby/world/memory_store.json"))
GRAPH_HTML_PATH = TEMPLATES_DIR / "knowledge_graph.html"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))


def _memory_summary(memory_path: Path) -> dict:
    if not memory_path.exists():
        return {
            "facts": 0,
            "rules": 0,
            "predictions": 0,
            "experiments": 0,
        }

    payload = json.loads(memory_path.read_text(encoding="utf-8"))
    world_model = payload.get("world_model", {})
    return {
        "facts": len(payload.get("facts", [])),
        "rules": len(world_model.get("rules", [])),
        "predictions": len(world_model.get("predictions", [])),
        "experiments": len(world_model.get("experiments", [])),
    }


@app.route("/")
def index():
    parsed = LogParser(DEFAULT_LOG_PATH).parse()
    conflicts = parsed["conflicts"]
    prediction_failures = parsed["prediction_failures"][-30:]
    reward_history = parsed["reward_history"][-300:]
    memory_summary = _memory_summary(DEFAULT_MEMORY_PATH)

    build_conflict_graph(conflicts, GRAPH_HTML_PATH)

    return render_template(
        "index.html",
        conflicts_count=len(conflicts),
        prediction_failures=prediction_failures,
        reward_history=reward_history,
        memory_summary=memory_summary,
    )


@app.route("/graph")
def graph():
    parsed = LogParser(DEFAULT_LOG_PATH).parse()
    build_conflict_graph(parsed["conflicts"], GRAPH_HTML_PATH)
    return render_template("knowledge_graph.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5055, debug=False)
