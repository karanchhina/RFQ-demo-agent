from types import SimpleNamespace

import pytest

from evals.cases import CASE_DEFINITIONS
from evals.langsmith_eval import (
    DATASET_NAME,
    require_credentials,
    run_experiment,
    sync_dataset,
)


class FakeLangSmithClient:
    def __init__(self):
        self.dataset = None
        self.examples = []
        self.created = 0
        self.updated = 0
        self.evaluate_call = None

    def has_dataset(self, *, dataset_name=None, dataset_id=None):
        return self.dataset is not None and self.dataset.name == dataset_name

    def create_dataset(self, dataset_name, **kwargs):
        self.dataset = SimpleNamespace(id="dataset-1", name=dataset_name)
        return self.dataset

    def read_dataset(self, *, dataset_name=None, dataset_id=None):
        assert self.dataset.name == dataset_name
        return self.dataset

    def list_examples(self, dataset_id=None, dataset_name=None, **kwargs):
        return iter(self.examples)

    def create_example(
        self, inputs=None, dataset_id=None, outputs=None, metadata=None, **kwargs
    ):
        example = SimpleNamespace(
            id=f"example-{len(self.examples) + 1}",
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        self.examples.append(example)
        self.created += 1
        return example

    def update_example(self, example_id, **kwargs):
        example = next(item for item in self.examples if item.id == example_id)
        example.inputs = kwargs["inputs"]
        example.outputs = kwargs["outputs"]
        example.metadata = kwargs["metadata"]
        self.updated += 1

    def evaluate(self, target, **kwargs):
        self.evaluate_call = {"target": target, **kwargs}
        return SimpleNamespace(experiment_name="quote-agent-test")


def test_sync_is_idempotent_by_stable_case_metadata() -> None:
    client = FakeLangSmithClient()

    first = sync_dataset(client)
    second = sync_dataset(client)

    assert first.dataset_name == DATASET_NAME
    assert first.created == 7
    assert first.updated == 0
    assert second.created == 0
    assert second.updated == 7
    assert len(client.examples) == 7
    assert {example.metadata["case"] for example in client.examples} == {
        case["metadata"]["case"] for case in CASE_DEFINITIONS
    }


def test_sync_rejects_duplicate_local_case_names_before_client_calls() -> None:
    client = FakeLangSmithClient()
    duplicated = [CASE_DEFINITIONS[0], CASE_DEFINITIONS[0]]

    with pytest.raises(ValueError, match="Duplicate case name"):
        sync_dataset(client, cases=duplicated)

    assert client.dataset is None


def test_credential_check_names_missing_variables_without_values() -> None:
    with pytest.raises(RuntimeError, match="LANGSMITH_API_KEY") as sync_error:
        require_credentials({}, need_openai=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY") as run_error:
        require_credentials({"LANGSMITH_API_KEY": "present"}, need_openai=True)

    assert "present" not in str(run_error.value)
    assert "LANGSMITH_API_KEY=" not in str(sync_error.value)


def test_run_experiment_uses_dataset_and_deterministic_evaluator() -> None:
    client = FakeLangSmithClient()
    sync_dataset(client)

    result = run_experiment(client, with_llm_judge=False)

    assert DATASET_NAME == "Quote Agent - Core Behavior"
    assert result.experiment_name == "quote-agent-test"
    assert client.evaluate_call["data"] == DATASET_NAME
    assert client.evaluate_call["experiment_prefix"] == "quote-agent"
    assert client.evaluate_call["max_concurrency"] == 2
    assert len(client.evaluate_call["evaluators"]) == 1
    assert client.evaluate_call["metadata"] == {
        "graph_revision": "local",
        "quote_model": "gpt-5.4",
        "support_model": "gpt-5.4-mini",
        "evaluator_version": "2",
    }
