"""Pytest trajectory evaluations logged as individual LangSmith rows."""

import os

import pytest
from dotenv import load_dotenv
from langsmith import testing as t

from evals.cases import CASE_DEFINITIONS, run_case

load_dotenv()
HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))


def _contains_in_order(actual: list[str], expected: list[str]) -> bool:
    remaining = iter(actual)
    return all(any(item == wanted for item in remaining) for wanted in expected)


@pytest.mark.integration
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OPENAI_API_KEY is required")
@pytest.mark.langsmith(output_keys=["expected_tools"])
@pytest.mark.parametrize(
    "case",
    CASE_DEFINITIONS,
    ids=[case["metadata"]["case"] for case in CASE_DEFINITIONS],
)
def test_quote_agent_tool_trajectory(case) -> None:
    """The agent uses the expected tools in order and avoids forbidden tools."""
    t.log_inputs({
        "case": case["metadata"]["case"],
        "rfq": case["inputs"]["rfq"],
    })

    outputs = run_case(case["inputs"])
    actual = outputs["tool_sequence"]
    expected = case["expected_tools"]
    forbidden = case["forbidden_tools"]

    t.log_outputs({
        "expected_tools": expected,
        "actual_tools": actual,
        "forbidden_tools": forbidden,
    })

    assert _contains_in_order(actual, expected), (
        f"Expected ordered tools {expected}; got {actual}"
    )
    assert not set(actual).intersection(forbidden), (
        f"Forbidden tools {forbidden} appeared in {actual}"
    )
