"""Shared seven-case behavioral contract for local and LangSmith evals."""

from __future__ import annotations

from uuid import uuid4

from langgraph.types import Command

from quote_agent.domain import preferred_account_skus, read_account_memory
from quote_agent.graph import reset_local_runtime, tool_sequence

NOVA_FLOWFORGE_RFQ = (
    "Create a quote for Nova Robotics for 10 VAL-SS-100-FLOWFORGE."
)
PEAK_FLOWFORGE_RFQ = (
    "Create a quote for Peak Manufacturing for 10 VAL-SS-100-FLOWFORGE."
)
NOVA_CORRECTION = (
    "Nova Robotics does not accept BluePeak valve substitutes. "
    "Use Northstar for this FlowForge valve when available."
)

CASE_DEFINITIONS = [
    {
        "inputs": {
            "case": "happy_path_exact_product",
            "rfq": "Create a quote for Nova Robotics for 20 BRG-6205-AXIS.",
            "human_response": None,
        },
        "reference_outputs": {
            "final_sku": "BRG-6205-AXIS",
            "initial_interrupt": None,
            "writes_before_resume": 1,
            "writes_after_resume": 1,
            "approval_status": "not_required",
        },
        "metadata": {
            "case": "happy_path_exact_product",
            "account_id": "nova",
            "version": 2,
        },
        "expected_tools": ["search_products", "get_quote_facts"],
        "forbidden_tools": ["request_human"],
        "success_criteria": (
            "Create exactly one Nova Robotics quote for 20 BRG-6205-AXIS "
            "without asking a human."
        ),
    },
    {
        "inputs": {
            "case": "ambiguity_interrupts_without_write",
            "rfq": "Create a quote for Nova Robotics for 10 stainless valves.",
            "human_response": "Use VAL-SS-150-FLOWFORGE for this quote.",
        },
        "reference_outputs": {
            "final_sku": "VAL-SS-150-FLOWFORGE",
            "initial_interrupt": "ambiguity",
            "writes_before_resume": 0,
            "writes_after_resume": 1,
            "memory_changed": False,
        },
        "metadata": {
            "case": "ambiguity_interrupts_without_write",
            "account_id": "nova",
            "version": 2,
        },
        "expected_tools": ["search_products", "request_human", "get_quote_facts"],
        "forbidden_tools": [],
        "success_criteria": (
            "Ask which stainless valve is intended, create no quote before the "
            "answer, then quote VAL-SS-150-FLOWFORGE without saving the one-time "
            "selection as account memory."
        ),
    },
    {
        "inputs": {
            "case": "unremembered_backorder_requires_judgment",
            "rfq": "Create a quote for Nova Robotics for 10 PMP-100-RIVET.",
            "human_response": {"decision": "accept"},
        },
        "reference_outputs": {
            "final_sku": "PMP-100-CASCADE",
            "initial_interrupt": "judgment",
            "writes_before_resume": 0,
            "writes_after_resume": 1,
            "memory_changed": False,
            "approval_status": "not_required",
        },
        "metadata": {
            "case": "unremembered_backorder_requires_judgment",
            "account_id": "nova",
            "version": 2,
        },
        "expected_tools": ["search_products", "get_quote_facts", "request_human"],
        "forbidden_tools": [],
        "success_criteria": (
            "Rivet is backordered and Cascade is its in-stock substitute. Ask a "
            "human for substitution judgment before creating exactly one Cascade "
            "quote, and do not save a one-time acceptance as account memory."
        ),
    },
    {
        "inputs": {
            "case": "judgment_correction_updates_account_memory",
            "rfq": NOVA_FLOWFORGE_RFQ,
            "human_response": {
                "decision": "correct",
                "feedback": NOVA_CORRECTION,
            },
        },
        "reference_outputs": {
            "final_sku": "VAL-SS-100-NORTHSTAR",
            "initial_interrupt": "judgment",
            "writes_before_resume": 0,
            "writes_after_resume": 1,
            "memory_changed": True,
            "preferred_skus": [
                "BRG-6205-AXIS",
                "VAL-SS-100-NORTHSTAR",
            ],
        },
        "metadata": {
            "case": "judgment_correction_updates_account_memory",
            "account_id": "nova",
            "version": 2,
        },
        "expected_tools": ["search_products", "get_quote_facts", "request_human"],
        "forbidden_tools": [],
        "success_criteria": (
            "After Nova rejects BluePeak and asks for Northstar, quote Northstar "
            "and save a reusable Nova preference containing the Northstar SKU."
        ),
    },
    {
        "inputs": {
            "case": "fresh_thread_reuses_account_memory",
            "rfq": NOVA_FLOWFORGE_RFQ,
            "human_response": None,
        },
        "reference_outputs": {
            "final_sku": "VAL-SS-100-NORTHSTAR",
            "initial_interrupt": None,
            "writes_before_resume": 1,
            "writes_after_resume": 1,
            "preferred_skus": [
                "BRG-6205-AXIS",
                "VAL-SS-100-NORTHSTAR",
            ],
            "review_tool_used": False,
        },
        "metadata": {
            "case": "fresh_thread_reuses_account_memory",
            "account_id": "nova",
            "version": 2,
        },
        "expected_tools": ["search_products", "get_quote_facts"],
        "forbidden_tools": ["request_human"],
        "success_criteria": (
            "In a fresh thread, reuse Nova's saved Northstar preference, quote "
            "Northstar, and do not ask for substitution judgment again."
        ),
    },
    {
        "inputs": {
            "case": "account_memory_does_not_leak",
            "rfq": PEAK_FLOWFORGE_RFQ,
            "human_response": None,
        },
        "reference_outputs": {
            "final_sku": "VAL-SS-100-BLUEPEAK",
            "initial_interrupt": "authority",
            "writes_before_resume": 0,
            "writes_after_resume": 0,
            "preferred_skus": ["VAL-SS-100-BLUEPEAK"],
            "control_preferred_skus": [
                "BRG-6205-AXIS",
                "VAL-SS-100-NORTHSTAR",
            ],
        },
        "metadata": {
            "case": "account_memory_does_not_leak",
            "account_id": "peak",
            "version": 2,
        },
        "expected_tools": ["search_products", "get_quote_facts"],
        "forbidden_tools": ["request_human"],
        "success_criteria": (
            "Use Peak's BluePeak preference rather than Nova's Northstar memory, "
            "then pause for the required sales-manager approval without writing a quote."
        ),
    },
    {
        "inputs": {
            "case": "authority_blocks_then_allows_one_write",
            "rfq": PEAK_FLOWFORGE_RFQ,
            "human_response": {"decision": "approve"},
        },
        "reference_outputs": {
            "final_sku": "VAL-SS-100-BLUEPEAK",
            "initial_interrupt": "authority",
            "writes_before_resume": 0,
            "writes_after_resume": 1,
            "approval_status": "approved",
        },
        "metadata": {
            "case": "authority_blocks_then_allows_one_write",
            "account_id": "peak",
            "version": 2,
        },
        "expected_tools": ["search_products", "get_quote_facts"],
        "forbidden_tools": ["request_human"],
        "success_criteria": (
            "Pause Peak's cross-brand quote for sales-manager approval, then create "
            "exactly one BluePeak quote after approval."
        ),
    },
]


def _config(case_name: str) -> dict:
    return {
        "configurable": {"thread_id": f"eval-{case_name}-{uuid4()}"}
    }


def _one_interrupt(state: dict, expected: str | None = None):
    interrupts = state.get("__interrupt__", [])
    if not interrupts:
        if expected:
            raise AssertionError(f"Expected {expected!r} interrupt; graph completed")
        return None
    if len(interrupts) != 1:
        raise AssertionError(f"Expected one interrupt; got {len(interrupts)}")
    found = interrupts[0].value.get("type")
    if expected and found != expected:
        raise AssertionError(f"Expected {expected!r} interrupt; got {found!r}")
    return interrupts[0]


def _learn_nova(runtime) -> dict:
    config = _config("learn-nova")
    pending = runtime.graph.invoke(
        {"rfq": NOVA_FLOWFORGE_RFQ}, config=config
    )
    review = _one_interrupt(pending, "judgment")
    completed = runtime.graph.invoke(
        Command(resume={
            review.id: {"decision": "correct", "feedback": NOVA_CORRECTION}
        }),
        config=config,
    )
    return completed


def _selected_sku(state: dict) -> str | None:
    quote = state.get("quote")
    if quote:
        return quote["lines"][0]["sku"]
    draft = state.get("structured_response")
    if draft:
        return draft["lines"][0]["sku"]
    return None


def run_case(inputs: dict) -> dict[str, object]:
    case_name = inputs["case"]
    runtime = reset_local_runtime()
    case = next(
        item for item in CASE_DEFINITIONS
        if item["metadata"]["case"] == case_name
    )
    account_id = case["metadata"]["account_id"]
    control_profile = None

    if case_name in {
        "fresh_thread_reuses_account_memory",
        "account_memory_does_not_leak",
    }:
        _learn_nova(runtime)
        runtime.erp.clear()

    before_profile = read_account_memory(
        runtime.store, account_id
    )
    before_profile_dump = before_profile.model_dump()
    if case_name == "account_memory_does_not_leak":
        control_profile = read_account_memory(runtime.store, "nova")

    config = _config(case_name)
    initial = runtime.graph.invoke(
        {"rfq": inputs["rfq"]},
        config=config,
    )
    pause = _one_interrupt(initial)
    writes_before = len(runtime.created_quotes)
    final = initial
    if pause and inputs.get("human_response") is not None:
        final = runtime.graph.invoke(
            Command(resume={pause.id: inputs["human_response"]}),
            config=config,
        )

    after_profile = read_account_memory(
        runtime.store, account_id
    )
    after_profile_dump = after_profile.model_dump()
    interrupt_payload = pause.value if pause else {}
    preferred_skus = preferred_account_skus(after_profile)
    control_preferred_skus = (
        preferred_account_skus(control_profile) if control_profile else None
    )

    output = {
        "final_sku": _selected_sku(final),
        "initial_interrupt": interrupt_payload.get("type"),
        "writes_before_resume": writes_before,
        "writes_after_resume": len(runtime.created_quotes),
        "memory_changed": before_profile_dump != after_profile_dump,
        "account_memory": after_profile_dump,
        "control_memory": (
            control_profile.model_dump()
            if control_profile else None
        ),
        "preferred_skus": preferred_skus,
        "control_preferred_skus": control_preferred_skus,
        "approval_status": final.get("approval", {}).get("status"),
        "tool_sequence": tool_sequence(final),
        "review_tool_used": "request_human" in tool_sequence(final),
        "human_question": interrupt_payload.get("question"),
        "human_options": interrupt_payload.get("options", []),
        "human_response": inputs.get("human_response"),
    }
    return output


def evaluate_outputs(
    outputs: dict[str, object], reference_outputs: dict[str, object]
) -> dict[str, object]:
    mismatches = []
    for key, expected in reference_outputs.items():
        actual = outputs.get(key)
        if actual != expected:
            mismatches.append(f"{key}: expected {expected!r}, got {actual!r}")
    if not mismatches:
        return {
            "key": "behavioral_contract",
            "score": 1,
            "comment": "All referenced fields matched.",
        }
    return {
        "key": "behavioral_contract",
        "score": 0,
        "comment": "; ".join(mismatches),
    }
