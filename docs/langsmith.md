# LangSmith tracing and evaluation

Status: live tracing and both 18-question experiments were verified on October 31,
2026. All 36 stored experiment runs completed without application errors and had
evaluator feedback. These runs used local extractive answers, not LLM generation.
All 36 local tests passed after the guided evidence-check fixes. Initial failed setup experiments remain in LangSmith
for troubleshooting; use the successful experiments linked in the evidence file.
No AWS service needs to run for this workflow.

## Account and local configuration

Create an account at https://smith.langchain.com/ and choose the Developer plan.
Check current pricing at https://www.langchain.com/pricing before sending traces.
As of October 31, 2026, Developer lists 5,000 included base traces per month,
then paid usage. Keep usage within the included allowance. No paid LLM evaluator
is used by this project. Do not enable paid services as part of this setup.

In your workspace settings, create an API key. Copy `.env.example` to `.env`,
which Git ignores, and enter the key there using a local editor. Match the
endpoint to your workspace's region. Do not commit the key or paste it into chat.
The template targets the agent-based workspace: `LANGSMITH_AGENT_ID=hybrid-search-rag`
and `LANGSMITH_AGENT_ENVIRONMENT=local`. It requires LangSmith SDK 0.14.12 or newer.
Do not combine these settings with `LANGSMITH_PROJECT`, `LANGCHAIN_PROJECT` or
`LANGCHAIN_SESSION`; unset legacy variables if they were previously exported.

Install the declared dependencies and export the local configuration:

```sh
source .venv/bin/activate
pip install -r requirements.txt
set -a
source .env
set +a
```

`.env` is not loaded automatically. Use shell-safe values if sourcing it.
Tracing is disabled in the template. Set `LANGSMITH_TRACING=true` only when you
intend to upload inputs, retrieved document text, prompts, model answers and
execution details to LangSmith. Use only the bundled public corpus initially.
Never enable tracing for sensitive uploads without reviewing what is sent.
The AWS task definition does not contain the key or enable tracing.

## Trace one run

With the above environment loaded, set `LANGSMITH_TRACING=true` and start:

```sh
uvicorn api:app --app-dir src --port 8000
```

Send a request to `/ask/guided` and inspect its trace in LangSmith. The custom
root spans record the answer and sources without serializing the search engine,
model object or embedding array. Retrieval has a child span; the LangGraph nodes
and LangChain Ollama calls use their existing native tracing integrations.
Local Ollama is optional: without it the app returns cited excerpts or abstains.
Do not report local excerpts as LLM-generated answers. Token reporting depends
on the model integration; local inference does not imply a paid API token cost.

## Evaluate the 18-question dataset

Validate the dataset without network calls or model runs:

```sh
python evaluation/langsmith_eval.py --dry-run
```

Then explicitly upload the bundled questions, labels and results:

```sh
python evaluation/langsmith_eval.py --offline-excerpts
```

This creates the versioned `hybrid-search-demo-v1` dataset (12 answerable and six
unanswerable cases) and separate original/guided experiments. Repeat runs reuse
examples without duplicating them. Changed labels require a new dataset name,
for example `--dataset hybrid-search-demo-v2`.

For actual local-model evaluation, start Ollama with the configured model and run:

```sh
python evaluation/langsmith_eval.py
```

Concurrency is one. Evaluators use existing code-based checks: source hit,
expected-source citation, citation validity, abstention matching the label, and
whether an offline excerpt was returned. LangSmith also records run latency.
These are proxies on a small demo, not factual-correctness or production-quality
scores. An offline-excerpt flag of 1 means no model answer was produced.
Review retrieved text and answers manually; no LLM-as-judge is billed.

Source references:
- https://docs.langchain.com/langsmith/annotate-code
- https://docs.langchain.com/langsmith/evaluate-llm-application

For version 2 before/after results and evaluation limitations, see
[evidence-check improvements](evidence/guided-rag-improvements.md).
