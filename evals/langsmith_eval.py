"""Sync the shared dataset and run LangSmith experiments from the CLI."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import Client
from pydantic import BaseModel, Field

from .cases import CASE_DEFINITIONS, evaluate_outputs, run_case

DATASET_NAME = "Quote Agent - Core Behavior"
DATASET_DESCRIPTION = (
    "Seven quote-agent behaviors covering exact products, ambiguity, unresolved "
    "substitution judgment, learned account preferences, memory isolation, and "
    "authority-gated writes."
)


@dataclass(frozen=True)
class SyncSummary:
    dataset_name: str
    dataset_id: str
    created: int
    updated: int


def require_credentials(
    environ: Mapping[str, str], *, need_openai: bool
) -> None:
    missing = []
    if not environ.get("LANGSMITH_API_KEY"):
        missing.append("LANGSMITH_API_KEY")
    if need_openai and not environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"Missing required environment variable(s): {names}. Add them to .env."
        )


def _validate_cases(cases: Sequence[dict]) -> list[str]:
    names = [case["metadata"]["case"] for case in cases]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate case name(s): {', '.join(duplicates)}")
    return names


def sync_dataset(
    client: Client,
    *,
    cases: Sequence[dict] = CASE_DEFINITIONS,
) -> SyncSummary:
    """Create/update seven examples by stable metadata name without duplicates."""
    _validate_cases(cases)
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            DATASET_NAME,
            description=DATASET_DESCRIPTION,
            metadata={"agent": "quote-agent", "schema_version": 1},
        )

    existing_by_case = {}
    for example in client.list_examples(dataset_id=dataset.id):
        case_name = (example.metadata or {}).get("case")
        if not case_name:
            continue
        if case_name in existing_by_case:
            raise ValueError(
                f"Online dataset contains duplicate case metadata: {case_name}"
            )
        existing_by_case[case_name] = example

    created = 0
    updated = 0
    for case in cases:
        case_name = case["metadata"]["case"]
        existing = existing_by_case.get(case_name)
        if existing is None:
            client.create_example(
                dataset_id=dataset.id,
                inputs=case["inputs"],
                outputs=case["reference_outputs"],
                metadata=case["metadata"],
            )
            created += 1
        else:
            client.update_example(
                existing.id,
                dataset_id=dataset.id,
                inputs=case["inputs"],
                outputs=case["reference_outputs"],
                metadata=case["metadata"],
            )
            updated += 1

    return SyncSummary(
        dataset_name=dataset.name,
        dataset_id=str(dataset.id),
        created=created,
        updated=updated,
    )


def experiment_target(inputs: dict) -> dict[str, object]:
    return run_case(inputs)


def behavioral_evaluator(
    outputs: dict[str, object], reference_outputs: dict[str, object]
) -> dict[str, object]:
    return evaluate_outputs(outputs, reference_outputs)


class ClarityGrade(BaseModel):
    clear_and_actionable: bool = Field(
        description=(
            "True only when the question explains the decision and provides "
            "actionable options."
        )
    )
    explanation: str


def build_clarity_evaluator():
    model = ChatOpenAI(
        model="gpt-5.4-mini", temperature=0, seed=7
    ).with_structured_output(ClarityGrade)

    def interrupt_clarity(
        outputs: dict[str, object], reference_outputs: dict[str, object]
    ) -> dict[str, object]:
        question = outputs.get("human_question")
        if not question:
            return {
                "key": "interrupt_clarity",
                "score": 1,
                "comment": "No human interrupt in this case.",
            }
        grade = model.invoke([
            {
                "role": "system",
                "content": (
                    "Judge only whether a human-facing interrupt question clearly "
                    "explains the decision and offers actionable options. Ignore SKU "
                    "correctness, authorization, memory, and write behavior."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"Options: {outputs.get('human_options', [])}"
                ),
            },
        ])
        return {
            "key": "interrupt_clarity",
            "score": int(grade.clear_and_actionable),
            "comment": grade.explanation,
        }

    return interrupt_clarity


def run_experiment(client: Client, *, with_llm_judge: bool = False):
    evaluators = [behavioral_evaluator]
    if with_llm_judge:
        evaluators.append(build_clarity_evaluator())
    return client.evaluate(
        experiment_target,
        data=DATASET_NAME,
        evaluators=evaluators,
        experiment_prefix="quote-agent",
        max_concurrency=2,
        error_handling="log",
        metadata={
            "graph_revision": os.getenv("LANGGRAPH_REVISION", "local"),
            "quote_model": "gpt-5.4",
            "support_model": "gpt-5.4-mini",
            "evaluator_version": "2",
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage the quote-agent LangSmith evaluation dataset."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("sync", help="Create or update the seven-case dataset.")
    run_parser = subparsers.add_parser("run", help="Run an online experiment.")
    run_parser.add_argument(
        "--with-llm-judge",
        action="store_true",
        help="Also judge interrupt clarity with the configured OpenAI model.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")
    args = _parser().parse_args(argv)
    need_openai = args.command == "run"
    require_credentials(os.environ, need_openai=need_openai)
    client = Client()

    if args.command == "sync":
        summary = sync_dataset(client)
        print(
            f"Dataset: {summary.dataset_name} ({summary.dataset_id})\n"
            f"Created: {summary.created}; updated: {summary.updated}"
        )
        return 0

    sync_dataset(client)
    experiment = run_experiment(
        client, with_llm_judge=args.with_llm_judge
    )
    name = getattr(experiment, "experiment_name", None) or str(experiment)
    print(f"Experiment: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
