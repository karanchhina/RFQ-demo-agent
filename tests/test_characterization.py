import json
import os
import asyncio
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage

load_dotenv()
HAS_OPENAI_KEY = bool(os.environ.get("OPENAI_API_KEY"))
os.environ.setdefault("OPENAI_API_KEY", "not-used-by-offline-tests")
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "quote_agent_demo.ipynb"

from quote_agent.domain import (  # noqa: E402
    MemoryDecision,
    account_memory,
    apply_memory_decision,
    read_account_memory,
)
import quote_agent.domain as domain  # noqa: E402
graph_module = import_module("quote_agent.graph")  # noqa: E402
from quote_agent.graph import (  # noqa: E402
    QUOTE_AGENT_PROMPT,
    TOOLS,
    build_async_quote_prompt,
    create_quote,
    reset_local_runtime,
)


@pytest.fixture
def runtime():
    return reset_local_runtime()


def test_reset_restores_only_seed_state(runtime) -> None:
    runtime.erp.created_quotes.append({"quote_id": "Q-stale"})
    runtime.store.put(account_memory("nova"), "stale", {"text": "stale"})

    fresh = reset_local_runtime()

    assert fresh.created_quotes == []
    assert fresh.store is not runtime.store
    assert fresh.graph is not runtime.graph
    nova = read_account_memory(fresh.store, "nova")
    peak = read_account_memory(fresh.store, "peak")
    assert nova.model_dump() == {
        "preferred_skus": ["BRG-6205-AXIS"],
        "preferences": [
            "Nova Robotics prefers Axis bearings in manufacturer-sealed packaging."
        ],
    }
    assert peak.model_dump() == {
        "preferred_skus": ["VAL-SS-100-BLUEPEAK"],
        "preferences": [
            "Peak Manufacturing accepts BluePeak for this FlowForge valve."
        ],
    }


def test_account_memory_is_stored_as_one_consolidated_profile(runtime) -> None:
    items = runtime.store.search(account_memory("nova"))

    assert [(item.key, item.value) for item in items] == [(
        "profile",
        {
            "preferred_skus": ["BRG-6205-AXIS"],
            "preferences": [
                "Nova Robotics prefers Axis bearings in "
                "manufacturer-sealed packaging."
            ],
        },
    )]


def test_seed_migrates_append_only_account_memories_into_profile(runtime) -> None:
    namespace = account_memory("nova")
    runtime.store.delete(namespace, "profile")
    runtime.store.put(namespace, "seed-1", {
        "preferred_skus": ["BRG-6205-AXIS"],
        "text": (
            "Nova Robotics prefers Axis bearings in manufacturer-sealed "
            "packaging."
        ),
    })
    runtime.store.put(namespace, "learned-northstar", {
        "preferred_skus": ["VAL-SS-100-NORTHSTAR"],
        "text": "Nova Robotics prefers Northstar for stainless valves.",
    })
    runtime.store.put(namespace, "obsolete", {"notes": ["ignore me"]})

    domain.seed_memories(runtime.store)

    assert [(item.key, item.value) for item in runtime.store.search(namespace)] == [(
        "profile",
        {
            "preferred_skus": [
                "BRG-6205-AXIS",
                "VAL-SS-100-NORTHSTAR",
            ],
            "preferences": [
                "Nova Robotics prefers Axis bearings in "
                "manufacturer-sealed packaging.",
                "Nova Robotics prefers Northstar for stainless valves.",
            ],
        },
    )]


def test_memory_decision_overwrites_one_revised_profile(runtime) -> None:
    decision = MemoryDecision.model_validate({
        "save": True,
        "preferred_skus": [
            "brg-6205-axis",
            "val-ss-100-northstar",
            "VAL-SS-100-NORTHSTAR",
        ],
        "preferences": [
            "Nova Robotics prefers Axis bearings in manufacturer-sealed packaging.",
            "Nova Robotics prefers Northstar for stainless valves.",
            "  nova robotics PREFERS northstar for stainless valves.  ",
        ],
    })

    changed = domain.apply_memory_decision(runtime.store, "nova", decision)

    assert changed is True
    assert domain.read_account_memory(runtime.store, "nova").model_dump() == {
        "preferred_skus": [
            "BRG-6205-AXIS",
            "VAL-SS-100-NORTHSTAR",
        ],
        "preferences": [
            "Nova Robotics prefers Axis bearings in manufacturer-sealed packaging.",
            "Nova Robotics prefers Northstar for stainless valves.",
        ],
    }
    assert [
        item.key for item in runtime.store.search(account_memory("nova"))
    ] == ["profile"]


def test_ignored_memory_decision_does_not_persist(runtime) -> None:
    before = read_account_memory(runtime.store, "nova").model_dump()

    changed = apply_memory_decision(
        runtime.store,
        "nova",
        MemoryDecision(save=False, preferred_skus=[], preferences=[]),
    )

    after = read_account_memory(runtime.store, "nova").model_dump()
    assert changed is False
    assert after == before


def test_account_update_is_normalized_and_isolated(runtime) -> None:
    decision = MemoryDecision(
        save=True,
        preferred_skus=["BRG-6205-AXIS", "val-ss-100-northstar"],
        preferences=[
            "Nova Robotics prefers Axis bearings in manufacturer-sealed packaging.",
            "Nova rejects BluePeak valve substitutes and prefers Northstar.",
        ],
    )

    apply_memory_decision(runtime.store, "nova", decision)

    nova = read_account_memory(runtime.store, "nova")
    peak = read_account_memory(runtime.store, "peak")
    assert nova.model_dump() == {
        "preferred_skus": ["BRG-6205-AXIS", "VAL-SS-100-NORTHSTAR"],
        "preferences": [
            "Nova Robotics prefers Axis bearings in manufacturer-sealed packaging.",
            "Nova rejects BluePeak valve substitutes and prefers Northstar.",
        ],
    }
    assert peak.model_dump() == {
        "preferred_skus": ["VAL-SS-100-BLUEPEAK"],
        "preferences": [
            "Peak Manufacturing accepts BluePeak for this FlowForge valve."
        ],
    }


def test_pump_backorder_fixture_requires_an_unremembered_substitute(runtime) -> None:
    search = graph_module.search_products.func("PMP-100-RIVET", limit=5)
    requested = domain.check_price_and_stock("nova", "PMP-100-RIVET", 10)
    substitute = domain.check_price_and_stock("nova", "PMP-100-CASCADE", 10)
    remembered = domain.preferred_account_skus(
        domain.read_account_memory(runtime.store, "nova")
    )

    assert {
        "matches": search.get("matches"),
        "requested_availability": requested.get("availability"),
        "substitute_availability": substitute.get("availability"),
        "substitute_is_remembered": "PMP-100-CASCADE" in remembered,
    } == {
        "matches": [{
            "sku": "PMP-100-RIVET",
            "name": "1-inch centrifugal pump",
            "brand": "Rivet",
            "category": "pump",
            "substitutes": [{
                "sku": "PMP-100-CASCADE",
                "name": "1-inch centrifugal pump",
                "brand": "Cascade",
            }],
            "score": 100,
        }],
        "requested_availability": "backorder",
        "substitute_availability": "in_stock",
        "substitute_is_remembered": False,
    }


def test_agent_surface_has_three_focused_tools() -> None:
    assert [tool.name for tool in TOOLS] == [
        "search_products",
        "get_quote_facts",
        "request_human",
    ]
    assert len(QUOTE_AGENT_PROMPT.split()) < 130
    assert "approval policy" not in QUOTE_AGENT_PROMPT.casefold()
    assert "create_quote" not in QUOTE_AGENT_PROMPT


def test_protected_write_is_idempotent_per_execution(runtime) -> None:
    state = {
        "account_id": "nova",
        "rfq": "Quote 20 BRG-6205-AXIS",
        "structured_response": {"lines": [{
            "requested_sku": "BRG-6205-AXIS",
            "sku": "BRG-6205-AXIS",
            "quantity": 20,
        }]},
        "messages": [ToolMessage(
            content=json.dumps({
                "status": "ok",
                "matches": [{"sku": "BRG-6205-AXIS", "substitutes": []}],
            }),
            name="search_products",
            tool_call_id="search-1",
        )],
        "approval": {"required": False, "status": "not_required"},
    }
    config = {"configurable": {"thread_id": "idempotency-test"}}

    first = create_quote(state, config, runtime.erp)
    second = create_quote(state, config, runtime.erp)

    assert first["status"] == "created"
    assert second["status"] == "already_created"
    assert first["quote"] == second["quote"]
    assert len(runtime.created_quotes) == 1


def test_write_boundary_trusts_the_agent_substitution_choice(runtime) -> None:
    state = {
        "account_id": "nova",
        "rfq": (
            "Create a quote for Nova Robotics for 10 "
            "VAL-SS-100-FLOWFORGE."
        ),
        "structured_response": {"lines": [{
            "requested_sku": "VAL-SS-100-FLOWFORGE",
            "sku": "VAL-SS-100-NORTHSTAR",
            "quantity": 10,
        }]},
        "messages": [],
        "approval": {"required": False, "status": "not_required"},
    }

    result = create_quote(
        state,
        {"configurable": {"thread_id": "agent-choice-test"}},
        runtime.erp,
    )

    assert result["quote"]["lines"][0]["sku"] == "VAL-SS-100-NORTHSTAR"
    assert len(runtime.created_quotes) == 1


def test_notebook_consumes_package_instead_of_redefining_runtime() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "from quote_agent" in code
    for duplicate_definition in (
        "StateGraph(",
        "@tool",
        "class AccountMemory(",
        "PRODUCTS = {",
        "def create_quote(",
    ):
        assert duplicate_definition not in code


@pytest.mark.integration
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OPENAI_API_KEY is required")
@pytest.mark.parametrize(
    ("rfq", "interrupt_type", "expected_sku"),
    [
        (
            "Create a quote for Nova Robotics for 20 of the 6205 bearings they normally buy.",
            None,
            "BRG-6205-AXIS",
        ),
        (
            "Create a quote for Nova Robotics for 10 stainless valves.",
            "ambiguity",
            None,
        ),
        (
            "Create a quote for Peak Manufacturing for 10 VAL-SS-100-FLOWFORGE.",
            "authority",
            None,
        ),
    ],
)
def test_graph_routes_clean_requests(
    runtime, rfq, interrupt_type, expected_sku
) -> None:
    result = runtime.graph.invoke(
        {"rfq": rfq},
        config={"configurable": {"thread_id": f"characterize-{uuid4()}"}},
    )

    interrupts = result.get("__interrupt__", [])
    if interrupt_type:
        assert len(interrupts) == 1
        assert interrupts[0].value["type"] == interrupt_type
        assert runtime.created_quotes == []
    else:
        assert not interrupts
        assert result["quote"]["lines"][0]["sku"] == expected_sku
        assert len(runtime.created_quotes) == 1


@pytest.mark.integration
@pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OPENAI_API_KEY is required")
def test_ambiguous_rfq_reuses_unique_preferred_sku(runtime) -> None:
    apply_memory_decision(
        runtime.store,
        "nova",
        MemoryDecision(
            save=True,
            preferred_skus=["BRG-6205-AXIS", "VAL-SS-100-NORTHSTAR"],
            preferences=[
                "Nova Robotics prefers Axis bearings in manufacturer-sealed "
                "packaging.",
                "Nova Robotics prefers VAL-SS-100-NORTHSTAR for stainless valves.",
            ],
        ),
    )

    result = runtime.graph.invoke(
        {"rfq": "Create a quote for Nova Robotics for 10 stainless valves."},
        config={
            "configurable": {
                "thread_id": f"remembered-ambiguity-{uuid4()}"
            }
        },
    )

    assert not result.get("__interrupt__")
    assert result["quote"]["lines"][0]["sku"] == "VAL-SS-100-NORTHSTAR"
    assert "request_human" not in graph_module.tool_sequence(result)


def test_rfq_text_is_interpreted_by_the_account_llm(monkeypatch) -> None:
    class FakeAccountModel:
        def invoke(self, messages):
            assert "Nova Robotics" in messages[-1]["content"]
            return {"account_id": "nova"}

    monkeypatch.setattr(graph_module, "ACCOUNT_MODEL", FakeAccountModel())

    state = graph_module.identify_account({
        "rfq": "Create a quote for Nova Robotics for 20 BRG-6205-AXIS."
    })

    assert state["account_id"] == "nova"
    assert state["messages"] == [{
        "role": "user",
        "content": "Create a quote for Nova Robotics for 20 BRG-6205-AXIS.",
    }]


def test_judgment_interrupt_accepts_plain_english_feedback(monkeypatch) -> None:
    monkeypatch.setattr(
        graph_module,
        "interrupt",
        lambda payload: "Use the Northstar variant for this quote.",
    )

    result = graph_module.request_human.func(
        kind="judgment",
        question="Should the quote use the proposed substitute?",
        options=["BluePeak", "Northstar"],
        runtime=SimpleNamespace(),
        requested_sku="VAL-SS-100-FLOWFORGE",
        proposed_sku="VAL-SS-100-BLUEPEAK",
    )

    assert result["answer"] == "Use the Northstar variant for this quote."


def test_human_approval_accepts_plain_text_approve(monkeypatch) -> None:
    monkeypatch.setattr(graph_module, "interrupt", lambda payload: "approve")
    state = {
        "approval": {
            "required": True,
            "event_id": "authority-test",
            "role": "sales_manager",
            "reason": "cross_brand_substitute",
            "lines": [],
            "status": "pending",
        },
        "structured_response": {
            "lines": [{
                "requested_sku": "VAL-SS-100-FLOWFORGE",
                "sku": "VAL-SS-100-BLUEPEAK",
                "quantity": 10,
            }]
        },
    }

    command = graph_module.human_approval(state)

    assert command.update["approval"]["status"] == "approved"
    assert command.update["status"] == "approved"
    assert command.goto == "update_memory"


def test_public_graph_has_no_memory_load_or_draft_retry_nodes(runtime) -> None:
    assert set(runtime.graph.get_graph().nodes) == {
        "__start__",
        "identify_account",
        "quote_agent",
        "update_memory",
        "check_approval",
        "human_approval",
        "create_quote",
        "__end__",
    }


def test_public_graph_requires_only_rfq_input(runtime) -> None:
    schema = runtime.graph.get_input_jsonschema()

    assert schema["required"] == ["rfq"]
    assert set(schema["properties"]) == {"rfq"}


def test_async_quote_prompt_uses_only_the_async_store_interface(runtime) -> None:
    class AsyncOnlyStore:
        def get(self, *args, **kwargs):
            raise AssertionError("sync get must not be used")

        def search(self, *args, **kwargs):
            raise AssertionError("sync search must not be used")

        def put(self, *args, **kwargs):
            raise AssertionError("sync put must not be used")

        async def aget(self, *args, **kwargs):
            return runtime.store.get(*args, **kwargs)

        async def asearch(self, *args, **kwargs):
            return runtime.store.search(*args, **kwargs)

        async def aput(self, *args, **kwargs):
            return runtime.store.put(*args, **kwargs)

    request = SimpleNamespace(
        runtime=SimpleNamespace(store=AsyncOnlyStore()),
        state={"account_id": "nova"},
    )

    prompt = asyncio.run(build_async_quote_prompt(request))

    assert "Nova Robotics" in prompt
    assert "BRG-6205-AXIS" in prompt
