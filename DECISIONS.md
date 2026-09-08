# Design decisions

This file records the current architecture. Git history preserves the phased experiments that led here.

## Let LLMs do the interpretive work

The workflow receives one natural-language RFQ. An LLM identifies the account, the Quote Agent resolves products and substitutions with tools, and a Memory LLM decides whether human feedback is reusable. Ambiguity and unresolved judgment go to a human; the response returns to the agent as natural language.

There is no parent-level draft validator, substitution checker, or retry loop that duplicates the agent's reasoning.

## Keep deterministic code at real boundaries

Deterministic code remains where trust must not depend on a prompt:

- catalog, price, inventory, and account data come from tool boundaries;
- a Peak cross-brand quote cannot bypass the explicit approval gate;
- the protected ERP write recomputes commercial facts and is idempotent per workflow thread;
- Pydantic keeps LLM and human payloads structurally usable.

These are application guarantees, not attempts to reproduce the LLM's product judgment.

## Keep memory simple and local to the account

Account memory is one profile containing `preferred_skus` and `preferences` lists. The Quote Agent reads the relevant account and distributor memory inside its dynamic prompt before every model call. Human input is sent to the Memory LLM, which either returns the complete revised profile—preserving unrelated facts and replacing obsolete ones—or chooses not to save it.

Thread checkpoints, long-term memory, and ERP quote records are separate concepts.

## Keep the agent surface focused

The Quote Agent receives only `search_products`, `get_quote_facts`, and `request_human`. Memory writes, authority approval, and `create_quote` stay outside the agent because they are side effects or enforcement boundaries.

## Share one installable implementation

The notebook, Pytest evaluations, LangSmith experiments, and Agent Server all import `quote_agent`. The notebook is a narrated client rather than a second implementation.

`graph` is compiled without an application-supplied checkpointer or Store so Agent Server can inject managed persistence. `reset_local_runtime()` supplies isolated in-memory persistence and a mock ERP for the notebook and tests.

## Run the demo and evaluations directly

```bash
uv run langgraph dev

# To start completely fresh, stop the server first, then run:
uv run quote-agent-reset --yes

LANGSMITH_TEST_SUITE='Quote Agent - Pytest' \
LANGSMITH_EXPERIMENT='quote-agent' \
uv run pytest tests/test_trajectory.py tests/test_end_to_end.py -q

uv run python -m evals.langsmith_eval sync
uv run python -m evals.langsmith_eval run --with-llm-judge
```

The LangSmith workspace supports traces, datasets, and experiments. Managed deployment requires an eligible LangSmith plan and Docker, but the same `langgraph.json` is ready for it.
