from evals.cases import CASE_DEFINITIONS, evaluate_outputs

EXPECTED_CASES = [
    "happy_path_exact_product",
    "ambiguity_interrupts_without_write",
    "unremembered_backorder_requires_judgment",
    "judgment_correction_updates_account_memory",
    "fresh_thread_reuses_account_memory",
    "account_memory_does_not_leak",
    "authority_blocks_then_allows_one_write",
]


def test_case_definitions_have_seven_unique_stable_names() -> None:
    names = [case["metadata"]["case"] for case in CASE_DEFINITIONS]

    assert names == EXPECTED_CASES
    assert len(names) == len(set(names))


def test_eval_cases_accept_only_the_rfq_as_graph_input() -> None:
    for case in CASE_DEFINITIONS:
        assert "account_id" not in case["inputs"]
        assert "rfq" in case["inputs"]
        assert case["metadata"]["account_id"] in {"nova", "peak"}
        assert any(
            name in case["inputs"]["rfq"]
            for name in ("Nova Robotics", "Peak Manufacturing")
        )


def test_deterministic_evaluator_compares_only_reference_fields() -> None:
    reference = {
        "final_sku": "VAL-SS-100-NORTHSTAR",
        "writes_after_resume": 1,
    }

    passing = evaluate_outputs(
        {
            "final_sku": "VAL-SS-100-NORTHSTAR",
            "writes_after_resume": 1,
            "diagnostic": "ignored",
        },
        reference,
    )
    failing = evaluate_outputs(
        {"final_sku": "VAL-SS-100-BLUEPEAK", "writes_after_resume": 1},
        reference,
    )

    assert passing == {
        "key": "behavioral_contract",
        "score": 1,
        "comment": "All referenced fields matched.",
    }
    assert failing["score"] == 0
    assert "final_sku" in failing["comment"]
    assert "diagnostic" not in failing["comment"]
