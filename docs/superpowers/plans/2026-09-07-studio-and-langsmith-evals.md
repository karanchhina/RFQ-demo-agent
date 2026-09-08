# Local Studio and LangSmith Evals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the quote agent a single installable package that runs unchanged in the notebook, LangGraph Studio, local tests, and persistent LangSmith dataset experiments.

**Architecture:** Extract the working notebook runtime into `src/quote_agent`, with domain data and memory rules separated from graph orchestration. Keep process-local graph factories for notebook/eval isolation, export an unconfigured compiled graph for Agent Server persistence injection, and normalize the six behavioral scenarios behind one shared evaluator contract.

**Tech Stack:** Python 3.12–3.14, uv, LangChain 1.4, LangGraph 1.2, LangSmith 0.12, Pydantic 2, pytest, JupyterLab.

**Spec:** `docs/superpowers/specs/2026-09-07-studio-and-langsmith-evals-design.md`

## Global Constraints

- Direct CLI is the deployment path; no GitHub remote is required.
- Do not attempt managed Deployment while the LangSmith workspace remains on Developer.
- `graph` must compile without an application-supplied checkpointer or Store.
- `build_local_graph()` must provide isolated `InMemorySaver` and `InMemoryStore` instances.
- Notebook, Studio, tests, and experiments must import the same package implementation.
- `.env` remains untracked and no credential values may be printed.
- Keep exactly six named behavioral cases in the online dataset and local evaluator.
- Deterministic checks remain authoritative; the interrupt-clarity LLM judge is optional.

---

### Task 1: Installable quote-agent package and local runtime

**Files:**
- Create: `src/quote_agent/__init__.py`
- Create: `src/quote_agent/domain.py`
- Create: `src/quote_agent/graph.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `tests/test_characterization.py`

**Interfaces:**
- Produces: `graph`, `build_local_graph() -> CompiledStateGraph`, `reset_local_runtime() -> LocalRuntime`, `LocalRuntime.graph`, `LocalRuntime.store`, `LocalRuntime.created_quotes`, `read_account_memory(store, account_id) -> AccountMemory`, and existing graph/domain types used by scenarios.
- Preserves: the exact three-tool agent surface, explicit `human_approval` node, deterministic validation, account-scoped memory, and idempotent quote writes.

- [ ] **Step 1: Replace notebook execution in characterization tests with package imports**

  Import `quote_agent.graph` and assert a local runtime resets seed memory, task-only corrections do not persist, account corrections validate and remain isolated, the agent exposes exactly three tools, review authorization is line-specific, normalization honors human input, and the write boundary is idempotent.

- [ ] **Step 2: Run the focused test and verify RED**

  Run: `uv run pytest tests/test_characterization.py -q`

  Expected: collection fails with `ModuleNotFoundError: No module named 'quote_agent'`.

- [ ] **Step 3: Extract domain behavior into `domain.py`**

  Move the distributor, account, product, pricing, inventory, discounts, approval policy, Pydantic memory models, namespace helpers, idempotent seed logic, memory validation, and mock quote-fact calculation out of the notebook. Make `seed_memories(store)` write only missing seed/profile records so Agent Server stores retain learned memory.

- [ ] **Step 4: Extract orchestration into `graph.py`**

  Move the three tools, short prompt, typed workflow state, correction classifier, human-event handling, deterministic draft normalization/validation, approval routing, protected write, and graph assembly. Replace module-global write lists with a runtime-scoped mock ERP object accessible through runtime context, while preserving process-local convenience access for the demo.

- [ ] **Step 5: Export server and local graph variants**

  Compile `graph = build_graph()` without persistence arguments. Implement `build_local_graph()` with fresh in-memory persistence and `reset_local_runtime()` returning all local state needed by notebook/tests. Export the public API from `__init__.py`.

- [ ] **Step 6: Make the project installable and portable**

  Set `requires-python = ">=3.12,<3.15"`, remove `package = false`, configure uv/setuptools for `src/quote_agent`, and add `langgraph-cli[inmem]>=0.4,<0.5` to the development group. Run `uv lock` and `uv sync --locked`.

- [ ] **Step 7: Run the focused test and verify GREEN**

  Run: `uv run pytest tests/test_characterization.py -q`

  Expected: all characterization tests pass; OpenAI integration tests skip only when the key is absent.

- [ ] **Step 8: Commit the package extraction**

  ```bash
  git add pyproject.toml uv.lock src/quote_agent tests/test_characterization.py
  git commit -m "extract reusable quote agent package"
  ```

### Task 2: Shared six-case local evaluation contract

**Files:**
- Create: `evals/cases.py`
- Modify: `evals/run_evals.py`
- Modify: `tests/test_evals.py`

**Interfaces:**
- Consumes: `reset_local_runtime()` and package graph/domain APIs from Task 1.
- Produces: `CASE_DEFINITIONS`, `run_case(inputs) -> dict[str, object]`, `evaluate_outputs(outputs, reference_outputs) -> dict[str, object]`, `run_evals() -> list[dict[str, str]]`.

- [ ] **Step 1: Write package-based evaluator tests**

  Add literal expectations for the six stable case names, normalized happy-path output, zero writes at each pause, durable Nova correction, fresh-thread reuse, Peak isolation, and exactly one authorized write.

- [ ] **Step 2: Run the evaluator tests and verify RED**

  Run: `uv run pytest tests/test_evals.py -q`

  Expected: import failure for `evals.cases` or missing `run_case`.

- [ ] **Step 3: Define six dataset-ready cases**

  Store each case as literal `inputs`, `reference_outputs`, and stable metadata name. Include human responses only where a resume is part of the case. Keep learned memory setup inside the relevant case so execution order cannot affect results.

- [ ] **Step 4: Implement normalized case execution**

  Each `run_case` call creates a fresh local runtime, invokes the graph with a unique thread ID, records the first interrupt and pre-resume write count, resumes when configured, and returns final SKU, write counts, account/control profiles, approval status, tool sequence, and human question.

- [ ] **Step 5: Implement deterministic comparison and local table output**

  Compare only keys present in `reference_outputs`; return `key = "behavioral_contract"`, score `1` or `0`, and a concise comment listing literal mismatches. Rebuild `run_evals.py` as a package consumer with no notebook parsing or dynamic `exec`.

- [ ] **Step 6: Run evaluator and regression tests and verify GREEN**

  Run: `uv run pytest tests/test_evals.py tests/test_characterization.py -q`

  Run: `uv run python -m evals.run_evals`

  Expected: six PASS rows and zero test failures.

- [ ] **Step 7: Commit shared evaluation behavior**

  ```bash
  git add evals/cases.py evals/run_evals.py tests/test_evals.py
  git commit -m "share six quote agent evaluation cases"
  ```

### Task 3: Persistent LangSmith dataset and online experiments

**Files:**
- Create: `evals/langsmith_eval.py`
- Create: `tests/test_langsmith_eval.py`
- Modify: `.env.example`

**Interfaces:**
- Consumes: `CASE_DEFINITIONS`, `run_case`, and `evaluate_outputs` from Task 2.
- Produces: `sync_dataset(client) -> SyncSummary`, `run_experiment(client, with_llm_judge=False)`, and CLI subcommands `sync` and `run`.

- [ ] **Step 1: Write client-boundary tests for idempotent sync**

  Use a small in-memory fake implementing the LangSmith methods called by this module. Assert the first sync creates six examples, the second creates zero duplicates and updates six by stable metadata name, duplicate local names fail before network calls, and absent credentials produce an actionable startup error.

- [ ] **Step 2: Run focused tests and verify RED**

  Run: `uv run pytest tests/test_langsmith_eval.py -q`

  Expected: import failure for `evals.langsmith_eval`.

- [ ] **Step 3: Implement dataset synchronization**

  Find or create the key-value dataset named `Quote Agent - Core Behavior`; list its examples; index them by `metadata["case"]`; create missing examples and update existing ones with current inputs, reference outputs, and metadata. Print only dataset name/ID and created/updated counts.

- [ ] **Step 4: Implement experiment target and deterministic evaluator**

  Wrap `run_case` as the `Client.evaluate` target. Pass `evaluate_outputs` as the primary evaluator, use experiment prefix `quote-agent`, set `max_concurrency=2`, and attach graph revision, model, and evaluator-version metadata.

- [ ] **Step 5: Add the optional interrupt-clarity judge**

  When `--with-llm-judge` is present, add one structured-output OpenAI evaluator that scores only whether a non-empty human question explains the decision and gives actionable options. Do not let it judge SKU correctness, authorization, memory, or writes.

- [ ] **Step 6: Run focused and complete offline tests and verify GREEN**

  Run: `uv run pytest tests/test_langsmith_eval.py -q`

  Run: `uv run pytest -q`

  Expected: all offline tests pass; credential-gated integration tests may skip only when their documented key is absent.

- [ ] **Step 7: Commit LangSmith integration**

  ```bash
  git add .env.example evals/langsmith_eval.py tests/test_langsmith_eval.py
  git commit -m "add LangSmith dataset and experiment commands"
  ```

### Task 4: Notebook as a clean package-backed demo

**Files:**
- Modify: `notebooks/quote_agent_demo.ipynb`
- Modify: `tests/test_characterization.py`

**Interfaces:**
- Consumes: public package API and shared evaluator APIs from Tasks 1–3.
- Produces: a top-to-bottom notebook with one environment cell, concise architecture/schema/tool displays, graph rendering, four narrated workflows covering all six eval behaviors, and a six-row PASS table.

- [ ] **Step 1: Add notebook contract checks**

  Assert no runtime cell dynamically defines `StateGraph`, `@tool`, product fixtures, or memory models; assert the notebook imports `quote_agent`; retain checks for ordered execution, no stored errors, graph display, human approval, memory reuse, and eval table.

- [ ] **Step 2: Run notebook contract tests and verify RED**

  Run: `uv run pytest tests/test_characterization.py -q`

  Expected: the notebook-duplication assertion fails against the current embedded implementation.

- [ ] **Step 3: Replace embedded runtime cells with focused imports/displays**

  Keep the conceptual order: setup, business environment, state/memory/policy responsibilities, tools/prompt, graph, happy path, ambiguity, judgment correction and fresh-thread reuse, account isolation and authority approval, local evals, decisions/limitations. Use `reset_local_runtime()` once per independent story and package helpers for readable JSON/tool output.

- [ ] **Step 4: Execute a clean notebook artifact**

  Run nbconvert in place with the project kernel and a bounded timeout. Confirm cells execute monotonically, no error outputs remain, the graph image renders, all interrupt/resume stories complete, and the eval table has six PASS rows.

- [ ] **Step 5: Run notebook and evaluator regression tests**

  Run: `uv run pytest -q`

  Run: `uv run python -m evals.run_evals`

  Expected: all tests pass and six local cases pass.

- [ ] **Step 6: Commit notebook migration**

  ```bash
  git add notebooks/quote_agent_demo.ipynb tests/test_characterization.py
  git commit -m "make notebook consume the quote agent package"
  ```

### Task 5: LangGraph Studio configuration and end-to-end verification

**Files:**
- Create: `langgraph.json`
- Modify: `.env.example`
- Modify: `DECISIONS.md`
- Create: `tests/test_studio_config.py`

**Interfaces:**
- Consumes: exported `quote_agent.graph:graph` and installable project.
- Produces: graph ID `quote_agent`, `uv run langgraph dev`, documented online dataset commands, and future managed CLI deployment instructions.

- [ ] **Step 1: Write configuration contract test**

  Load `langgraph.json` and assert `dependencies == ["."]`, `.env` is selected, Python is `3.12`, and graph `quote_agent` resolves to `./src/quote_agent/graph.py:graph`.

- [ ] **Step 2: Run focused test and verify RED**

  Run: `uv run pytest tests/test_studio_config.py -q`

  Expected: failure because `langgraph.json` does not exist.

- [ ] **Step 3: Add Studio configuration and operating notes**

  Create the exact server config, document `uv run langgraph dev`, dataset sync/run commands, optional judge, Developer-plan limitation, and later Plus command `uv run langgraph deploy --name rfq-demo-agent --deployment-type dev`. Keep credential names—not values—in `.env.example`.

- [ ] **Step 4: Run configuration tests and verify GREEN**

  Run: `uv run pytest tests/test_studio_config.py -q`

  Expected: pass.

- [ ] **Step 5: Start the local Agent Server and probe its API**

  Run: `uv run langgraph dev --no-browser`

  From a second process, call its documented assistants/graphs endpoint and confirm `quote_agent` is registered. Invoke one ambiguity request through the SDK/API and confirm it pauses, then inspect the graph representation to confirm `human_approval` exists as an explicit node.

- [ ] **Step 6: Sync the real online dataset twice**

  Run: `uv run python -m evals.langsmith_eval sync` twice.

  Expected: the dataset contains exactly six unique cases after both runs; the second run creates zero examples and updates six.

- [ ] **Step 7: Run the real deterministic online experiment**

  Run: `uv run python -m evals.langsmith_eval run`

  Expected: one LangSmith experiment with six completed rows and passing deterministic feedback.

- [ ] **Step 8: Run final verification**

  Run: `uv lock --check`

  Run: `uv sync --locked --dry-run`

  Run: `uv run pytest -q`

  Run: `uv run python -m evals.run_evals`

  Run: `git diff --check && git status --short`

  Inspect the executed notebook for error outputs and scan tracked files for credential-value patterns.

- [ ] **Step 9: Commit the verified Studio integration**

  ```bash
  git add langgraph.json .env.example DECISIONS.md tests/test_studio_config.py notebooks/quote_agent_demo.ipynb
  git commit -m "configure local LangGraph Studio workflow"
  ```

