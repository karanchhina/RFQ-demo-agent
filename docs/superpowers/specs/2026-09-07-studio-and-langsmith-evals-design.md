# Local Studio and LangSmith evaluation design

## Objective

Make the quote agent selectable in LangSmith Studio through a local Agent Server, while storing its six evaluation cases as a persistent LangSmith dataset and running repeatable online experiments with `Client.evaluate(...)`.

The repository will also be structurally ready for `langgraph deploy` after the LangSmith workspace is upgraded to Plus. No cloud deployment will be attempted on the current Developer plan.

## Constraints

- Direct CLI is the chosen deployment path; no GitHub remote is required.
- LangSmith Developer supports traces, datasets, and offline experiments but not managed Deployment.
- `langgraph dev` provides the local Agent Server and connects it to the web-based Studio UI without Docker.
- The Docker daemon is currently stopped. Docker and Buildx are needed only for a later managed CLI deployment.
- `.env` remains local and untracked. The notebook and commands may print only credential availability, never values.
- The notebook, Studio, tests, and experiments must run the same graph implementation.

## Source-of-truth architecture

The notebook-only runtime will move into one small installable package:

```text
src/quote_agent/
├── __init__.py
├── domain.py     # fixtures, schemas, scoped memory, deterministic facts
└── graph.py      # tools, prompt, nodes, graph builder, exported graph
```

`domain.py` owns the tiny catalog, account master data, `AccountMemory`, memory-update validation, and mock quote facts. `graph.py` owns the three agent tools, short prompt, workflow state, human-input handling, deterministic draft/approval/write boundaries, and graph assembly.

`graph.py` exports:

- `graph`: a compiled graph without an application-supplied checkpointer or Store, allowing Agent Server to inject its managed persistence.
- `build_local_graph()`: a factory returning a graph with `InMemorySaver` and `InMemoryStore` for notebook scenarios and tests.
- `reset_local_runtime()`: a deterministic local reset for the notebook and evaluation target.

Seed memories are written idempotently when an empty Store is first used. This supports both a fresh local Store and the Agent Server Store without overwriting learned account memory.

The notebook imports this package rather than dynamically serving as executable source. It continues to display the prompt, state schema, memory schema, policy, responsibility map, and rendered graph, but does not duplicate implementation.

## Agent Server and Studio

The project becomes an installable `src` package and adds:

```text
langgraph.json
.env.example
```

`langgraph.json` declares `dependencies: ["."]`, loads `.env`, and exposes `./src/quote_agent/graph.py:graph` under the graph ID `quote_agent`. Python support is broadened to `>=3.12,<3.15`; the deployment config selects Python 3.12 for portability while the existing local 3.14 environment remains valid.

The development command is:

```bash
uv run langgraph dev
```

It starts the local Agent Server, prints the Studio URL, and exposes the graph, threads, interrupts, checkpoint state, and traces. The process must remain running while Studio is connected.

After a future Plus upgrade and Docker startup, the same project can be deployed with:

```bash
uv run langgraph deploy --name rfq-demo-agent --deployment-type dev
```

That later hosted deployment will receive its checkpointer and persistent Store from Agent Server rather than from application code.

## Dataset

The six existing cases become one LangSmith key-value dataset named `Quote Agent - Core Behavior`:

1. `happy_path_exact_product`
2. `ambiguity_interrupts_without_write`
3. `judgment_correction_updates_account_memory`
4. `fresh_thread_reuses_account_memory`
5. `account_memory_does_not_leak`
6. `authority_blocks_then_allows_one_write`

Each example contains:

```python
inputs = {
    "case": "...",
    "rfq": "...",
    "account_id": "...",
    "human_response": {...} | None,
}

reference_outputs = {
    "final_sku": "..." | None,
    "initial_interrupt": "ambiguity" | "judgment" | "authority" | None,
    "writes_before_resume": 0,
    "writes_after_resume": 1,
    # Additional expected memory, isolation, approval, and tool fields per case.
}
```

Dataset synchronization is idempotent. Examples use stable case names in metadata; the sync command updates existing examples instead of creating duplicates.

## Experiment target

The local experiment target accepts one dataset input and returns normalized, evaluator-friendly observations rather than a raw LangGraph state:

```python
{
    "final_sku": "VAL-SS-100-NORTHSTAR",
    "initial_interrupt": "judgment",
    "writes_before_resume": 0,
    "writes_after_resume": 1,
    "memory_scope": "account",
    "account_profile": {...},
    "control_profile": {...},
    "approval_status": "approved",
    "tool_sequence": [...],
    "human_question": "...",
}
```

Every target invocation creates and resets its own local runtime. Cases that need a learned Nova preference create it inside that case. Dataset order and prior experiment runs therefore cannot affect results.

`evals/run_evals.py` remains the fast local table runner but consumes the package directly. `evals/langsmith_eval.py` owns dataset synchronization and `Client.evaluate(...)` experiment execution.

## Evaluators

Deterministic evaluators remain primary because exact SKUs, interrupt types, namespaces, memory fields, approval state, tool use, and write counts have objective answers. A single evaluator compares the relevant reference-output fields and returns a score plus a concise mismatch explanation.

An optional secondary LLM judge evaluates only the clarity and actionability of a human-facing interrupt question. It does not judge authorization, catalog validity, memory isolation, or quote counts. The CLI flag `--with-llm-judge` enables it; default experiments remain deterministic and lower cost.

Experiment runs use a stable prefix such as `quote-agent` and attach model, graph revision, and evaluator-version metadata so LangSmith comparisons remain intelligible.

## Commands

```bash
# Start local Agent Server and open the graph in Studio
uv run langgraph dev

# Create or update the online dataset
uv run python -m evals.langsmith_eval sync

# Run a deterministic online experiment
uv run python -m evals.langsmith_eval run

# Add the secondary clarity judge
uv run python -m evals.langsmith_eval run --with-llm-judge
```

## Error handling

- Missing OpenAI or LangSmith credentials fail at command startup with a short actionable message.
- Dataset sync validates unique case names and reports created versus updated examples.
- One failed case appears as a failed LangSmith experiment row; it does not prevent other rows from completing.
- Human-interrupt targets assert the expected pause before resuming, preserving the zero-write evidence.
- Local runtime reset remains independent of the Agent Server-managed Store.

## Verification

Implementation is complete only when:

1. Unit and integration tests import the package rather than executing notebook cells.
2. The notebook runs top to bottom with clean ordered outputs and six passing local eval rows.
3. `langgraph dev` starts successfully and its API reports the `quote_agent` graph.
4. A Studio run can pause at an interrupt and expose the explicit `human_approval` node.
5. Dataset sync creates exactly six uniquely named examples online and is idempotent on a second run.
6. `Client.evaluate(...)` creates an online experiment containing six rows and deterministic feedback.
7. The local test suite and local evaluator both pass after the package extraction.
8. No secrets are committed or emitted.

## Deliberate non-goals

- Upgrading the LangSmith subscription
- Creating a managed cloud deployment before Plus access exists
- GitHub deployment integration
- A remote-dataset target that calls a hosted deployment
- Production ERP, authentication, database, or user interface work
