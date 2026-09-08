from pathlib import Path

import pytest

from quote_agent.reset_dev_state import reset_dev_state


def test_reset_dev_state_requires_confirmation(tmp_path: Path) -> None:
    state_dir = tmp_path / ".langgraph_api"
    state_dir.mkdir()
    (state_dir / "store.pckl").write_text("state")
    (tmp_path / "langgraph.json").write_text("{}")

    with pytest.raises(RuntimeError, match="--yes"):
        reset_dev_state(tmp_path, confirmed=False, state_in_use=False)

    assert state_dir.exists()


def test_reset_dev_state_removes_only_project_runtime_state(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / ".langgraph_api"
    state_dir.mkdir()
    (state_dir / "store.pckl").write_text("state")
    keep = tmp_path / "keep.txt"
    keep.write_text("keep")
    (tmp_path / "langgraph.json").write_text("{}")

    changed = reset_dev_state(
        tmp_path,
        confirmed=True,
        state_in_use=False,
    )

    assert changed is True
    assert not state_dir.exists()
    assert keep.read_text() == "keep"


def test_reset_dev_state_refuses_while_server_uses_state(
    tmp_path: Path,
) -> None:
    (tmp_path / ".langgraph_api").mkdir()
    (tmp_path / "langgraph.json").write_text("{}")

    with pytest.raises(RuntimeError, match="Stop the LangGraph server"):
        reset_dev_state(tmp_path, confirmed=True, state_in_use=True)

    assert (tmp_path / ".langgraph_api").exists()
