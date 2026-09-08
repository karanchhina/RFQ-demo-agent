"""Safely clear the local LangGraph development server's persisted state."""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
from pathlib import Path


STATE_DIRECTORY = ".langgraph_api"


def _state_is_in_use(state_dir: Path, server_port: int) -> bool:
    """Return whether a process currently has a file open below ``state_dir``."""
    if shutil.which("lsof") is not None:
        checks = []
        if state_dir.exists():
            checks.append(["lsof", "+D", str(state_dir)])
        checks.append(
            ["lsof", "-nP", f"-iTCP:{server_port}", "-sTCP:LISTEN"]
        )
        for command in checks:
            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                return True
        return False

    # The in-memory development server does not keep every persistence file
    # open, so also guard the port on which this project's server normally runs.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.2)
        return connection.connect_ex(("127.0.0.1", server_port)) == 0


def reset_dev_state(
    project_root: Path,
    *,
    confirmed: bool,
    state_in_use: bool | None = None,
    server_port: int = 2024,
) -> bool:
    """Remove only this project's local Agent Server persistence directory."""
    root = project_root.resolve()
    if not (root / "langgraph.json").is_file():
        raise RuntimeError("Run this command from the RFQ demo project root.")

    state_dir = (root / STATE_DIRECTORY).resolve()
    if state_dir.parent != root or state_dir.name != STATE_DIRECTORY:
        raise RuntimeError("Refusing to reset an unexpected path.")

    in_use = (
        _state_is_in_use(state_dir, server_port)
        if state_in_use is None
        else state_in_use
    )
    if in_use:
        raise RuntimeError(
            "Stop the LangGraph server before resetting development state."
        )

    if not state_dir.exists():
        return False

    if not confirmed:
        raise RuntimeError(
            "This clears all local threads, checkpoints, and memory. Pass --yes to confirm."
        )

    shutil.rmtree(state_dir)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clear local LangGraph threads, checkpoints, and long-term memory."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="confirm deletion of this project's .langgraph_api directory",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=2024,
        help="local LangGraph server port to check before clearing state",
    )
    args = parser.parse_args()

    try:
        changed = reset_dev_state(
            Path.cwd(), confirmed=args.yes, server_port=args.server_port
        )
    except RuntimeError as error:
        parser.error(str(error))

    if changed:
        print("Cleared local LangGraph state. The next server start will be clean.")
    else:
        print("No local LangGraph state exists; the project is already clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
