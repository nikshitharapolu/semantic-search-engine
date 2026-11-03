# Semantic Search Engine

Search documents using BM25 keyword retrieval and MiniLM semantic embeddings, then generate cited answers with a local Ollama model.

**[Live Demo](https://semantic-search-engine-nr.streamlit.app/)**

> The public interface discovers papers across subjects through OpenAlex and reranks up to 40 candidates with BM25 and MiniLM. It shows available authors, year, venue, abstracts and paper links. Guided evidence selects a retrieved abstract or abstains; it does not verify scientific claims. Original local API routes still support sample/upload search and Ollama-generated answers.

## Features

- **Hybrid search:** adjustable fusion of keyword and semantic relevance.
- **Document uploads:** PDF, TXT, and Markdown ingestion with persistent embedding caches.
- **Guided RAG:** LangGraph evidence checks, one bounded query rewrite, explicit abstention, and visible execution traces.
- **Local generation:** LangChain–Ollama integration with an extractive fallback.
- **Deployment:** Streamlit interface, FastAPI endpoints, Docker, and Kubernetes configurations.

## Results

Evaluated on **BEIR SciFact**, containing **5,183 documents** and **300 test queries**. Hybrid weight was selected using **809 separate training queries**.

| Retrieval setup | Recall@10 | NDCG@10 |
|---|---:|---:|
| BM25 | 0.7757 | 0.6523 |
| Semantic | 0.7833 | 0.6451 |
| Hybrid — alpha 0.5 | 0.8242 | 0.7107 |
| Tuned hybrid — alpha 0.4 | **0.8366** | **0.7157** |

Compared with this project's BM25 baseline, tuned hybrid improved:

- **Recall@10 by 6.09 percentage points.**
- **NDCG@10 by 9.7% relative.**

Paired bootstrap analysis with 10,000 resamples produced 95% intervals of **[0.0317, 0.0911]** for the Recall@10 gain and **[0.0411, 0.0865]** for the NDCG@10 gain.

These results measure retrieval on SciFact, not generated-answer correctness or performance across other datasets. The application's default alpha remains 0.5. RRF and cross-encoder reranking were tested on training data but did not outperform weighted hybrid.

[Benchmark report](reports/scifact-benchmark.json) · [Per-query results](reports/scifact-benchmark-per-query.csv) · [Bootstrap results](reports/scifact-bootstrap.json)

## Quick Start

Requires **Python 3.10+**. The first run downloads the embedding model.

```bash
git clone https://github.com/nikshitharapolu/semantic-search-engine.git
cd semantic-search-engine

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run src/app.py
```

For generated answers, install Ollama and run:

```bash
ollama pull llama3.2:3b
ollama serve
```

Without Ollama, the application falls back to retrieved evidence.

## API

```bash
uvicorn api:app --app-dir src --reload --port 8000
```

Open the API documentation at `http://localhost:8000/docs`.

Endpoints include `/health`, `/search`, `/ask`, `/ask/guided`, and `/documents`.

## AWS deployment demonstration

The FastAPI backend has been deployed and verified on **AWS ECS Fargate** using
Terraform, with an S3-backed index snapshot, CloudWatch request logs and alarms,
and SNS email notifications. Compute is stopped after testing; this is an
on-demand demonstration, separate from the Streamlit demo.

[Deployment evidence](docs/evidence/aws-deployment.md) · [Runbook](infra/aws-demo/README.md)

## React + TypeScript interface

The evidence explorer provides typed search requests, ranked passage cards,
retrieval controls and a guided RAG workflow trace. The public Streamlit app
hosts it through a Python component bridge; standalone local development uses
FastAPI. Both interfaces use OpenAlex discovery through the same research
request handler. The original sample/upload routes remain available locally.

### Research workflow

1. Enter a research question in **Hybrid search**; optionally select a subject.
2. Inspect ranked titles, available publication metadata, and full abstracts.
3. Open the DOI/publisher link (or catalog record when a source link is missing).
   **Copy reference** copies available metadata and a source link; verify the
   original publication before using the reference in an assignment.
4. Optionally use **Guided evidence** for heuristic relevance checks and a
   selected abstract. This is not claim verification or generated scientific advice.
5. Insufficient-evidence responses retain closest matches for exploration.
   Title-only records cannot serve as guided evidence.

Search terms are sent to OpenAlex; do not submit personal or confidential text.
The provider first selects up to 40 candidates; local hybrid scores only rank
that pool, not the entire catalog. Missing abstracts limit evidence checks. A
subject filter uses the primary OpenAlex topic classification and may omit
interdisciplinary papers. Known retracted works are excluded by a provider filter;
metadata is not independently verified. No full-text crawling is performed.

The app deliberately uses **keyless access**. It does not configure API billing,
read an OpenAlex API key, or use prepaid credit. Provider access/rate limits can
interrupt discovery; the UI reports failures without silently changing collections.
A shared in-memory cache holds up to 32 query/subject candidate sets for 30 minutes
per entry. Queries and responses are not written to disk by this integration.

The standalone research routes are `/research/search` and `/research/ask/guided`.
They accept `subject`: `all`, `computer-science`, `engineering`, `life-sciences`,
`health-sciences`, `social-sciences`, or `physical-sciences`. The public component
uses `/search` and `/ask/guided` internally with the same handler.

**Evaluation scope:** SciFact remains an internal reproducible benchmark; its
reported gains do not measure OpenAlex candidate discovery/ranking. The live
candidate ranker uses positive BM25 IDF for small pools. The original 18-question
sample-corpus evaluation also does not establish scientific-answer quality.
[Packaged SciFact attribution](data/scientific/README.md) ·
[Live cross-subject integration checks](docs/evidence/openalex-live-smoke.json) ·
[OpenAlex API documentation](https://help.openalex.org/api/)

## Evaluation

Download and extract SciFact once:

```bash
mkdir -p evaluation/datasets

curl -L --fail \
  https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip \
  -o evaluation/datasets/scifact.zip

unzip -q evaluation/datasets/scifact.zip -d evaluation/datasets
```

Run the retrieval benchmark and bootstrap analysis:

```bash
python evaluation/benchmark_scifact.py
python evaluation/bootstrap_scifact.py
```

The benchmark saves aggregate and per-query results under `reports/`. Downloaded datasets and embedding caches are excluded from Git.

Compare original and guided RAG on the separate 18-question demo set, with Ollama running:

```bash
python evaluation/compare_rag_paths.py
```

## Optional LangSmith tracing

Opt-in traces cover the original and guided RAG paths and hybrid retrieval.
A versioned 18-question dataset runner compares the paths using code-based
citation, source-hit and abstention checks. Live tracing and both 18-question experiments were verified in LangSmith using
extractive answers and code-based evaluators; model-generated answer evaluation remains separate.

[LangSmith setup and evaluation](docs/langsmith.md)

## Tests

```bash
python -m unittest discover -s tests -v
```

## Stack

Python · BM25 · Sentence Transformers · LangChain · LangGraph · Ollama · Streamlit · FastAPI · Docker · Kubernetes

## Author

**Nikshitha Rapolu**

[GitHub](https://github.com/nikshitharapolu) · [LinkedIn](https://www.linkedin.com/in/nikshitha-rapolu-18960321a/) . [Portfolio] (https://nikshitharapolu.github.io)