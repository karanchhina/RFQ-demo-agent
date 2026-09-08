# RFQ Agent Demo

A working RFQ-to-quote agent for an industrial distributor. A natural-language
RFQ enters a LangGraph workflow, an LLM identifies the account, and a
tool-calling Quote Agent researches products, checks trusted pricing and
availability, handles substitutions, and drafts the quote. Human feedback can
update account memory, while an explicit approval gate protects the final ERP
write.

The project uses Python, LangGraph, LangChain's `create_agent`, OpenAI models,
Pydantic, Pytest, and LangSmith.

## What is real and what is mocked

The agent execution, model calls, tool loop, interrupts, long-term memory,
LangSmith traces, and evaluations are real.

For a focused demo, the following are mocked in memory:

- customer accounts and product catalog;
- pricing and inventory;
- ERP quote creation.

The demo accepts RFQ text directly. Email and document ingestion, OCR,
authentication, customer notifications, order conversion, fulfillment, and
production database integrations are outside its scope.

## How it works

- The public graph accepts one field: `rfq`.
- An Account LLM identifies the customer from the RFQ.
- A `create_agent` subgraph uses three tools: `search_products`,
  `get_quote_facts`, and `request_human`.
- Thread state holds the current RFQ. Long-term memory holds distributor rules
  and account-specific preferences across threads.
- LangGraph interrupts pause for ambiguity, substitution judgment, or required
  approval and resume with the human response.
- A Memory LLM saves durable account preferences and ignores one-time answers
  and approvals.
- A deterministic account-policy check protects the idempotent mock ERP write.

## Prerequisites

- Python 3.12 through 3.14
- [`uv`](https://docs.astral.sh/uv/)
- An OpenAI API key
- A LangSmith API key for traces, datasets, and experiments

On macOS, install `uv` with:

```bash
brew install uv
```

## Setup

Clone the repository, enter it, and install the project and development
dependencies:

```bash
git clone https://github.com/karanchhina/RFQ-demo-agent.git
cd RFQ-demo-agent
uv sync --dev
```

Create the local environment file:

```bash
cp .env.example .env
```

Add your values to `.env`:

```dotenv
OPENAI_API_KEY=your-openai-api-key

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=rfq-demo-agent

# Only needed if the LangSmith key belongs to more than one workspace.
LANGSMITH_WORKSPACE_ID=
```

The `.env` file is ignored by Git and must not be committed.

## Run the Jupyter notebook

```bash
uv run jupyter lab notebooks/quote_agent_demo.ipynb
```

Open the URL printed by JupyterLab if the browser does not open automatically.
Run the notebook from top to bottom. It imports the same package used by
LangGraph Studio and the evaluations.

## Run LangGraph Studio

Start the local Agent Server:

```bash
uv run langgraph dev
```

The command opens LangGraph Studio and registers the graph as `quote_agent`.
Studio inputs are JSON objects:

```json
{
  "rfq": "Quote 20 of the 6205 bearings for Nova Robotics."
}
```

Use a new Studio thread for each independent RFQ. Resume an interrupt in the
same thread with the JSON value requested by Studio.

### Reset Studio state

Studio keeps local threads, checkpoints, and long-term memory. To return to the
seed state, stop the Agent Server and run:

```bash
uv run quote-agent-reset --yes
```

Restart `uv run langgraph dev` afterward. Do not reset between a memory-learning
scenario and the fresh-thread scenario intended to reuse that memory.

## Studio demo scenarios

### 1. Exact product

```json
{
  "rfq": "Quote 20 of the 6205 bearings for Nova Robotics."
}
```

The agent resolves the natural-language product, checks its commercial facts,
and creates a quote without human input.

### 2. Ambiguity, memory, and reuse

```json
{
  "rfq": "Quote 10 stainless valves for Nova Robotics."
}
```

Resume the interrupt with:

```json
"Use the Northstar variant, and remember that preference for the next time Nova asks for stainless valves."
```

Start a new thread and submit the same RFQ again. Nova's saved Northstar
preference should resolve it without another interrupt.

### 3. Out-of-stock substitution

```json
{
  "rfq": "Quote 10 Rivet pumps for Nova Robotics."
}
```

Resume with:

```json
"Use the Cascade substitute, and remember it for the next time the Rivet pump is unavailable."
```

In a new thread, repeat the RFQ to demonstrate memory reuse.

### 4. Account isolation and approval

```json
{
  "rfq": "Quote 10 one-inch FlowForge stainless valves for Peak Manufacturing."
}
```

Peak uses its own BluePeak preference and requires sales-manager approval for a
cross-brand substitution. Resume the authority interrupt with:

```json
"approve"
```

The protected write creates the quote only after approval.

## Run the tests and evaluations

Run the complete test suite:

```bash
uv run pytest -q
```

Run only the model-backed end-to-end evaluations:

```bash
uv run pytest tests/test_end_to_end.py -q -s
```

Run only the tool-trajectory evaluations:

```bash
uv run pytest tests/test_trajectory.py -q -s
```

The end-to-end file includes exact behavioral checks and LLM-as-a-judge tests.
The trajectory tests verify required tool order and forbidden tool use. With
LangSmith configured, the decorated Pytest cases also log their inputs and
outputs to LangSmith.

To group those Pytest runs under explicit LangSmith names:

```bash
LANGSMITH_TEST_SUITE='Quote Agent - Pytest' \
LANGSMITH_EXPERIMENT='quote-agent' \
uv run pytest tests/test_trajectory.py tests/test_end_to_end.py -q
```

## Run a LangSmith dataset experiment

The shared online dataset is named `Quote Agent - Core Behavior` and
contains seven behavioral cases.

Create or update the dataset without running an experiment:

```bash
uv run python -m evals.langsmith_eval sync
```

Run all seven examples with the deterministic behavioral evaluator:

```bash
uv run python -m evals.langsmith_eval run
```

Add the LLM judge for the clarity of human-facing interrupt questions:

```bash
uv run python -m evals.langsmith_eval run --with-llm-judge
```

The experiment appears in LangSmith with evaluator scores and a trace for each
dataset example.

## Project layout

```text
src/quote_agent/
  domain.py            Mock business data, pricing, inventory, and memory
  graph.py             Tools, prompts, agent subgraph, and LangGraph workflow
  reset_dev_state.py   Safe local Studio-state reset command
evals/
  cases.py             Shared seven-scenario behavioral contract
  langsmith_eval.py    Dataset sync and LangSmith experiment CLI
tests/
  test_end_to_end.py   End-to-end and LLM-as-a-judge evaluations
  test_trajectory.py   Agent tool-trajectory evaluations
notebooks/
  quote_agent_demo.ipynb  Narrated interactive demo
langgraph.json         Local Agent Server and Studio configuration
```

Architecture decisions and demo tradeoffs are recorded in
[`DECISIONS.md`](DECISIONS.md).
