"""LangGraph quote workflow shared by the notebook, tests, and Agent Server."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Literal, NotRequired, TypedDict
from uuid import uuid4

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import ToolRuntime
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime, get_runtime
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from quote_agent.domain import (
    ACCOUNTS,
    APPROVAL_POLICIES,
    DISTRIBUTOR_MEMORY,
    PRODUCTS,
    MemoryDecision,
    MockERP,
    aapply_memory_decision,
    aread_account_memory,
    aread_memories,
    aseed_memories,
    apply_memory_decision,
    check_price_and_stock,
    read_account_memory,
    read_memories,
    seed_memories,
)


class ReviewResponse(BaseModel):
    decision: Literal["accept", "correct"]
    feedback: str | None = None


class ApprovalResponse(BaseModel):
    decision: Literal["approve", "reject"]
    feedback: str | None = None


class AccountSelection(BaseModel):
    account_id: Literal["nova", "peak"] = Field(
        description="Account named or clearly referenced by the RFQ"
    )


class QuoteLineDraft(BaseModel):
    requested_sku: str = Field(description="Exact catalog SKU originally requested")
    sku: str = Field(description="Exact catalog SKU selected for the quote")
    quantity: int = Field(gt=0, description="Requested whole-unit quantity")


class QuoteDraft(BaseModel):
    lines: list[QuoteLineDraft] = Field(min_length=1)


class QuoteInput(TypedDict):
    rfq: str


class QuoteWorkflowState(AgentState[dict[str, object]]):
    rfq: str
    account_id: str
    processed_human_input_ids: NotRequired[list[str]]
    approval: NotRequired[dict[str, object]]
    quote: NotRequired[dict[str, object]]
    memory_updated: NotRequired[bool]
    status: NotRequired[str]


def _normalize_search_text(value: str) -> str:
    tokens = value.lower().replace("-", " ").split()
    singular_tokens = [
        token[:-1] if len(token) > 3 and token.endswith("s") else token
        for token in tokens
    ]
    return " ".join(singular_tokens)


@tool
def search_products(query: str, limit: int = 5) -> dict:
    """Search the catalog. Multiple matches mean the request is ambiguous."""
    normalized_query = _normalize_search_text(query)
    if not normalized_query or limit <= 0:
        return {
            "status": "invalid_request",
            "message": "query and positive limit are required",
        }

    query_tokens = normalized_query.split()
    matches = []
    for product in PRODUCTS.values():
        normalized_sku = _normalize_search_text(product["sku"])
        normalized_aliases = [
            _normalize_search_text(alias) for alias in product["aliases"]
        ]
        searchable = _normalize_search_text(
            " ".join(
                [
                    product["sku"],
                    product["name"],
                    product["brand"],
                    product["category"],
                    *product["aliases"],
                ]
            )
        )
        if normalized_query == normalized_sku:
            score = 100
        elif normalized_query in normalized_aliases:
            score = 90
        elif all(token in searchable for token in query_tokens):
            score = 70 + len(query_tokens)
        else:
            continue

        matches.append({
            "sku": product["sku"],
            "name": product["name"],
            "brand": product["brand"],
            "category": product["category"],
            "substitutes": [
                {
                    "sku": substitute_sku,
                    "name": PRODUCTS[substitute_sku]["name"],
                    "brand": PRODUCTS[substitute_sku]["brand"],
                }
                for substitute_sku in product["substitutes"]
            ],
            "score": score,
        })

    matches.sort(key=lambda match: (-match["score"], match["sku"]))
    return {"status": "ok", "query": query, "matches": matches[:limit]}


@tool
def get_quote_facts(sku: str, quantity: int, runtime: ToolRuntime) -> dict:
    """Get trusted price, margin, availability, and lead time for the account."""
    return check_price_and_stock(runtime.state["account_id"], sku, quantity)


def _prior_ambiguity_answer(runtime: ToolRuntime, line_index: int) -> dict | None:
    for message in reversed(runtime.state.get("messages", [])):
        if getattr(message, "name", None) != "request_human":
            continue
        content = message.content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                continue
        if (
            isinstance(content, dict)
            and content.get("type") == "ambiguity"
            and content.get("status") == "answered"
            and content.get("line_index", 0) == line_index
        ):
            return content
    return None


@tool
def request_human(
    kind: Literal["ambiguity", "judgment"],
    question: str,
    options: list[str],
    runtime: ToolRuntime,
    requested_sku: Annotated[
        str | None, "Required for judgment: original exact SKU"
    ] = None,
    proposed_sku: Annotated[
        str | None, "Required for judgment: proposed exact SKU"
    ] = None,
    line_index: int = 0,
) -> dict:
    """Ask a human to resolve ambiguity or judge a substitution.

    Call this before drafting any cross-SKU substitute unless exact account
    memory already names the proposed SKU as preferred.
    """
    if kind == "ambiguity":
        prior = _prior_ambiguity_answer(runtime, line_index)
        if prior:
            return {**prior, "status": "already_resolved"}
        answer = interrupt({
            "type": kind,
            "line_index": line_index,
            "question": question,
            "options": options,
        })
        return {
            "status": "answered",
            "type": kind,
            "line_index": line_index,
            "question": question,
            "options": options,
            "answer": answer,
        }

    if not requested_sku or not proposed_sku:
        return {
            "status": "invalid_request",
            "type": kind,
            "message": "Judgment requires requested_sku and proposed_sku.",
        }
    normalized_requested_sku = requested_sku.strip().upper()
    normalized_proposed_sku = proposed_sku.strip().upper()
    human_response = interrupt({
        "type": kind,
        "line_index": line_index,
        "requested_sku": normalized_requested_sku,
        "proposed_sku": normalized_proposed_sku,
        "question": question,
        "options": options,
    })
    if isinstance(human_response, str):
        response_fields = {"answer": human_response}
    else:
        response_fields = ReviewResponse.model_validate(
            human_response
        ).model_dump()
    return {
        "status": "answered",
        "requested_sku": normalized_requested_sku,
        "proposed_sku": normalized_proposed_sku,
        "type": kind,
        "line_index": line_index,
        "question": question,
        "options": options,
        **response_fields,
    }


TOOLS = [search_products, get_quote_facts, request_human]
QUOTE_MODEL = "gpt-5.4"
SUPPORT_MODEL = "gpt-5.4-mini"

QUOTE_AGENT_PROMPT = """
Draft quotes with tools and memory.
Use search_products, then inspect the requested item and each substitute with get_quote_facts.
After searching, compare the matches with account-memory preferred_skus first. If exactly one relevant match is preferred, select it without asking again.
Only when account memory does not uniquely resolve several matches, call request_human(kind="ambiguity") next. Do not draft until the human answers.
When the requested item is unavailable, use an in-stock listed substitute named in account memory without asking again. If memory names none, choose the best listed substitute using distributor guidance, then call request_human(kind="judgment") before drafting.
Never draft an unavailable requested SKU when an in-stock listed substitute exists.
For judgment, provide exact requested and proposed SKUs.
Never invent products or facts.
Return QuoteDraft only. The workflow handles approval and the write.
""".strip()


def build_quote_prompt(
    account_id: str,
    distributor_memory: list[str],
    account_memory: dict[str, object],
) -> str:
    distributor_lines = "\n".join(
        f"- {text}" for text in distributor_memory
    ) or "- None"
    return f"""{QUOTE_AGENT_PROMPT}

Relevant long-term memory:

Distributor:
{distributor_lines}

Trusted account: {account_id} ({ACCOUNTS[account_id]['name']})
Account memory:
{json.dumps(account_memory, indent=2)}
"""


def build_sync_quote_prompt(request: ModelRequest) -> str:
    seed_memories(request.runtime.store)
    account_memory = read_account_memory(
        request.runtime.store, request.state["account_id"]
    )
    return build_quote_prompt(
        account_id=request.state["account_id"],
        distributor_memory=read_memories(
            request.runtime.store, DISTRIBUTOR_MEMORY
        ),
        account_memory=account_memory.model_dump(),
    )


async def build_async_quote_prompt(request: ModelRequest) -> str:
    await aseed_memories(request.runtime.store)
    account_memory = await aread_account_memory(
        request.runtime.store, request.state["account_id"]
    )
    return build_quote_prompt(
        account_id=request.state["account_id"],
        distributor_memory=await aread_memories(
            request.runtime.store, DISTRIBUTOR_MEMORY
        ),
        account_memory=account_memory.model_dump(),
    )


class QuotePromptMiddleware(AgentMiddleware):
    """Use the Store interface appropriate to sync or async execution."""

    def wrap_model_call(self, request, handler):
        prompt = build_sync_quote_prompt(request)
        return handler(request.override(
            system_message=SystemMessage(content=prompt)
        ))

    async def awrap_model_call(self, request, handler):
        prompt = await build_async_quote_prompt(request)
        return await handler(request.override(
            system_message=SystemMessage(content=prompt)
        ))


quote_prompt = QuotePromptMiddleware()


quote_agent = create_agent(
    model=ChatOpenAI(model=QUOTE_MODEL, temperature=0, seed=7),
    tools=TOOLS,
    middleware=[quote_prompt],
    state_schema=QuoteWorkflowState,
    response_format=ToolStrategy(
        QuoteDraft.model_json_schema(),
        tool_message_content="Quote draft returned to the workflow.",
    ),
    name="quote_agent",
)

MEMORY_PROMPT = """
Revise account memory only when the human response states a reusable preference.
If reusable, set save to true and return the complete revised preferred_skus and preferences lists.
Preserve unrelated preferences, replace conflicting or obsolete ones, and remove duplicates.
Use short standalone preference sentences and only exact catalog SKUs the customer prefers.
Do not save one-time selections, quote approvals or rejections, or distributor-wide proposals.
If nothing is reusable, set save to false and return empty lists.
""".strip()

MEMORY_MODEL = ChatOpenAI(
    model=SUPPORT_MODEL, temperature=0, seed=7
).with_structured_output(MemoryDecision)


ACCOUNT_PROMPT = """
Identify the customer account referenced by the RFQ.
Choose only from the supplied account directory. Do not infer a new account.
""".strip()

ACCOUNT_MODEL = ChatOpenAI(
    model=SUPPORT_MODEL, temperature=0, seed=7
).with_structured_output(AccountSelection)


def identify_account(state: QuoteWorkflowState) -> dict:
    rfq = state["rfq"].strip()
    if not rfq:
        raise ValueError("RFQ cannot be empty")
    selection = ACCOUNT_MODEL.invoke([
        {"role": "system", "content": ACCOUNT_PROMPT},
        {
            "role": "user",
            "content": json.dumps({"rfq": rfq, "accounts": ACCOUNTS}),
        },
    ])
    account_id = AccountSelection.model_validate(selection).account_id
    return {
        "rfq": rfq,
        "account_id": account_id,
        "messages": [{"role": "user", "content": rfq}],
        "status": "received",
    }


def tool_sequence(state: QuoteWorkflowState) -> list[str]:
    agent_tools = {item.name for item in TOOLS}
    names = []
    for message in state.get("messages", []):
        if getattr(message, "name", None) not in agent_tools:
            continue
        content = message.content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                pass
        if (
            message.name == "request_human"
            and isinstance(content, dict)
            and content.get("status") == "already_resolved"
        ):
            continue
        names.append(message.name)
    return names


def human_input_events(state: QuoteWorkflowState) -> list[dict]:
    events = []
    for message in state["messages"]:
        if getattr(message, "name", None) != "request_human":
            continue
        content = message.content
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                continue
        if not isinstance(content, dict) or content.get("status") != "answered":
            continue
        events.append({
            "id": f"request_human:{message.tool_call_id}",
            "type": content["type"],
            "details": content,
        })

    approval = state.get("approval", {})
    if approval.get("human_response") and approval.get("event_id"):
        events.append({
            "id": approval["event_id"],
            "type": "authority",
            "details": {
                "role": approval["role"],
                "reason": approval["reason"],
                "quote_draft": QuoteDraft.model_validate(
                    state["structured_response"]
                ).model_dump(),
                "response": approval["human_response"],
            },
        })
    return events


def pending_human_input_events(state: QuoteWorkflowState) -> list[dict]:
    processed = set(state.get("processed_human_input_ids", []))
    return [
        event for event in human_input_events(state) if event["id"] not in processed
    ]


def route_after_agent(
    state: QuoteWorkflowState,
) -> Literal["update_memory", "check_approval"]:
    return "update_memory" if pending_human_input_events(state) else "check_approval"


def memory_request(
    event: dict,
    state: QuoteWorkflowState,
    memory,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": MEMORY_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "account": ACCOUNTS[state["account_id"]],
                    "rfq": state["rfq"],
                    "human_input": event,
                    "current_account_memory": memory.model_dump(),
                    "current_quote_draft": state.get("structured_response"),
                    "catalog": [
                        {
                            "sku": product["sku"],
                            "brand": product["brand"],
                            "substitutes": product["substitutes"],
                        }
                        for product in PRODUCTS.values()
                    ],
                },
                indent=2,
                default=str,
            ),
        },
    ]


def decide_memory(
    event: dict, state: QuoteWorkflowState, store
) -> MemoryDecision:
    memory = read_account_memory(store, state["account_id"])
    decision = MEMORY_MODEL.invoke(memory_request(event, state, memory))
    return MemoryDecision.model_validate(decision)


async def adecide_memory(
    event: dict, state: QuoteWorkflowState, store
) -> MemoryDecision:
    memory = await aread_account_memory(store, state["account_id"])
    decision = await MEMORY_MODEL.ainvoke(
        memory_request(event, state, memory)
    )
    return MemoryDecision.model_validate(decision)


def memory_update_command(
    state: QuoteWorkflowState,
    processed_ids: set[str],
    memory_updated: bool,
) -> Command[Literal["check_approval", "create_quote", "__end__"]]:
    update = {
        "processed_human_input_ids": sorted(processed_ids),
        "memory_updated": memory_updated,
    }
    approval_status = state.get("approval", {}).get("status")
    if approval_status == "approved":
        return Command(update=update, goto="create_quote")
    if approval_status == "rejected":
        return Command(update=update, goto=END)
    return Command(update=update, goto="check_approval")


def update_memory(
    state: QuoteWorkflowState,
    runtime: Runtime,
) -> Command[Literal["check_approval", "create_quote", "__end__"]]:
    processed_ids = set(state.get("processed_human_input_ids", []))
    memory_updated = False
    for event in human_input_events(state):
        if event["id"] in processed_ids:
            continue
        decision = decide_memory(event, state, runtime.store)
        memory_updated = apply_memory_decision(
            runtime.store, state["account_id"], decision
        ) or memory_updated
        processed_ids.add(event["id"])

    return memory_update_command(state, processed_ids, memory_updated)


async def aupdate_memory(
    state: QuoteWorkflowState,
    runtime: Runtime,
) -> Command[Literal["check_approval", "create_quote", "__end__"]]:
    processed_ids = set(state.get("processed_human_input_ids", []))
    memory_updated = False
    for event in human_input_events(state):
        if event["id"] in processed_ids:
            continue
        decision = await adecide_memory(event, state, runtime.store)
        memory_updated = await aapply_memory_decision(
            runtime.store, state["account_id"], decision
        ) or memory_updated
        processed_ids.add(event["id"])

    return memory_update_command(state, processed_ids, memory_updated)


def update_memory_node(state: QuoteWorkflowState):
    return update_memory(state, get_runtime())


async def aupdate_memory_node(state: QuoteWorkflowState):
    return await aupdate_memory(state, get_runtime())


def check_approval(
    state: QuoteWorkflowState,
) -> Command[Literal["human_approval", "create_quote"]]:
    draft = QuoteDraft.model_validate(state["structured_response"])
    rule = APPROVAL_POLICIES.get(state["account_id"], {}).get(
        "cross_brand_substitute"
    )
    cross_brand_lines = []
    for line in draft.lines:
        requested = PRODUCTS.get(line.requested_sku.strip().upper())
        quoted = PRODUCTS.get(line.sku.strip().upper())
        if requested and quoted and requested["brand"] != quoted["brand"]:
            cross_brand_lines.append({
                "requested_sku": requested["sku"],
                "requested_brand": requested["brand"],
                "quoted_sku": quoted["sku"],
                "quoted_brand": quoted["brand"],
            })

    if rule and cross_brand_lines:
        return Command(
            update={
                "approval": {
                    "required": True,
                    "event_id": f"authority:{uuid4()}",
                    "role": rule,
                    "reason": "cross_brand_substitute",
                    "lines": cross_brand_lines,
                    "status": "pending",
                },
                "status": "awaiting_approval",
            },
            goto="human_approval",
        )
    return Command(
        update={
            "approval": {"required": False, "status": "not_required"},
            "status": "ready_to_create",
        },
        goto="create_quote",
    )


def human_approval(
    state: QuoteWorkflowState,
) -> Command[Literal["update_memory"]]:
    human_response = interrupt({
        "type": "authority",
        "role": state["approval"]["role"],
        "reason": state["approval"]["reason"],
        "question": (
            "Approve or reject this cross-brand substitute quote before it is "
            "created."
        ),
        "options": ["approve", "reject"],
        "quote_draft": QuoteDraft.model_validate(
            state["structured_response"]
        ).model_dump(),
        "policy_details": state["approval"]["lines"],
    })
    response = ApprovalResponse.model_validate(
        {"decision": human_response.strip().lower()}
        if isinstance(human_response, str)
        else human_response
    )
    approval = {
        **state["approval"],
        "status": (
            "approved" if response.decision == "approve" else "rejected"
        ),
        "feedback": response.feedback,
        "human_response": response.model_dump(),
    }
    return Command(
        update={
            "approval": approval,
            "status": (
                "approved" if response.decision == "approve" else "rejected"
            ),
        },
        goto="update_memory",
    )


def create_quote(
    state: QuoteWorkflowState,
    config: RunnableConfig,
    erp: MockERP,
) -> dict:
    approval = state.get("approval", {})
    if approval.get("required") and approval.get("status") != "approved":
        raise PermissionError("Required quote approval is missing")
    execution_id = config.get("configurable", {}).get("thread_id")
    if not execution_id:
        raise ValueError("create_quote requires a workflow thread_id")
    if execution_id in erp.by_execution:
        return {
            "quote": erp.by_execution[execution_id],
            "status": "already_created",
        }

    draft = QuoteDraft.model_validate(state["structured_response"])
    quote_lines = []
    for line in draft.lines:
        requested_sku = line.requested_sku.strip().upper()
        sku = line.sku.strip().upper()
        facts = check_price_and_stock(state["account_id"], sku, line.quantity)
        quote_lines.append({
            "requested_sku": requested_sku,
            "sku": facts["sku"],
            "quantity": facts["quantity"],
            "unit_price": facts["unit_price"],
            "extended_price": facts["extended_price"],
            "availability": facts["availability"],
            "lead_time_days": facts["lead_time_days"],
        })

    quote = {
        "quote_id": f"Q-{len(erp.created_quotes) + 1:04d}",
        "account_id": state["account_id"],
        "lines": quote_lines,
        "total": round(
            sum(line["extended_price"] for line in quote_lines), 2
        ),
        "requested_by": "sales_rep",
    }
    erp.created_quotes.append(quote)
    erp.by_execution[execution_id] = quote
    return {"quote": quote, "status": "created"}


def build_graph(*, checkpointer=None, store=None, erp: MockERP | None = None):
    """Build the workflow; omit persistence for Agent Server injection."""
    active_erp = erp or MockERP()
    builder = StateGraph(QuoteWorkflowState, input_schema=QuoteInput)
    builder.add_node("identify_account", identify_account)
    builder.add_node("quote_agent", quote_agent)
    builder.add_node(
        "update_memory",
        RunnableLambda(update_memory_node, afunc=aupdate_memory_node),
    )
    builder.add_node("check_approval", check_approval)
    builder.add_node("human_approval", human_approval)
    builder.add_node("create_quote", partial(create_quote, erp=active_erp))

    builder.add_edge(START, "identify_account")
    builder.add_edge("identify_account", "quote_agent")
    builder.add_conditional_edges(
        "quote_agent",
        route_after_agent,
        {"update_memory": "update_memory", "check_approval": "check_approval"},
    )
    builder.add_edge("create_quote", END)
    if checkpointer is None and store is None:
        return builder.compile()
    return builder.compile(checkpointer=checkpointer, store=store)


@dataclass
class LocalRuntime:
    graph: object
    store: InMemoryStore
    checkpointer: InMemorySaver
    erp: MockERP

    @property
    def created_quotes(self) -> list[dict]:
        return self.erp.created_quotes


def build_local_graph() -> LocalRuntime:
    store = InMemoryStore()
    seed_memories(store)
    checkpointer = InMemorySaver()
    erp = MockERP()
    compiled = build_graph(checkpointer=checkpointer, store=store, erp=erp)
    return LocalRuntime(
        graph=compiled,
        store=store,
        checkpointer=checkpointer,
        erp=erp,
    )


def reset_local_runtime() -> LocalRuntime:
    return build_local_graph()


# Agent Server injects its own checkpointer and Store into this compiled graph.
graph = build_graph()
