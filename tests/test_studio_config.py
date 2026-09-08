import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_langgraph_config_exposes_package_graph_for_python_312() -> None:
    config = json.loads((ROOT / "langgraph.json").read_text())

    assert config["dependencies"] == ["."]
    assert config["env"] == ".env"
    assert config["python_version"] == "3.12"
    assert config["graphs"] == {
        "quote_agent": "./src/quote_agent/graph.py:graph"
    }
