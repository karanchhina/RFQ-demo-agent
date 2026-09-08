"""Pytest end-to-end evaluations logged as individual LangSmith rows."""

import json
import os

import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import testing as t
from pydantic import BaseModel, Field

from evals.cases import CASE_DEFINITIONS, evaluate_outputs, run_case

load_dotenv()
HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))


class EndToEndGrade(BaseModel):
    passes: bool = Field(
        description="Whether the observed run satisfies every success criterion"
    )
    explanation: str


END_TO_END_JUDGE = ChatOpenAI(
    model="gpt-5.4-mini", temperature=0, seed=7
).with_structured_output(EndToEndGrade)


@pytest.mark.integration
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OPENAI_API_KEY is required")
@pytest.mark.langsmith(output_keys=["reference_outputs"])
@pytest.mark.parametrize(
    "case",
    CASE_DEFINITIONS,
    ids=[case["metadata"]["case"] for case in CASE_DEFINITIONS],
)
def test_quote_workflow_end_to_end(case) -> None:
    """Structured workflow results match the literal behavioral contract."""
    t.log_inputs({
        "case": case["metadata"]["case"],
        "rfq": case["inputs"]["rfq"],
    })

    outputs = run_case(case["inputs"])
    evaluation = evaluate_outputs(outputs, case["reference_outputs"])

    t.log_outputs({
        "reference_outputs": case["reference_outputs"],
        "observed_outputs": outputs,
        "evaluation": evaluation,
    })

    assert evaluation["score"] == 1, evaluation["comment"]


@pytest.mark.integration
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OPENAI_API_KEY is required")
@pytest.mark.langsmith(output_keys=["success_criteria"])
@pytest.mark.parametrize(
    "case",
    CASE_DEFINITIONS,
    ids=[case["metadata"]["case"] for case in CASE_DEFINITIONS],
)
def test_quote_workflow_with_llm_judge(case) -> None:
    """An LLM judge checks the complete run against semantic criteria."""
    t.log_inputs({
        "case": case["metadata"]["case"],
        "rfq": case["inputs"]["rfq"],
    })

    outputs = run_case(case["inputs"])
    grade = END_TO_END_JUDGE.invoke([
        {
            "role": "system",
            "content": (
                "Judge whether the observed RFQ-to-quote run satisfies every part "
                "of the success criteria. Use only the supplied structured evidence."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({
                "success_criteria": case["success_criteria"],
                "observed_outputs": outputs,
            }),
        },
    ])

    t.log_outputs({
        "success_criteria": case["success_criteria"],
        "observed_outputs": outputs,
        "judge_explanation": grade.explanation,
    })

    assert grade.passes, grade.explanation
