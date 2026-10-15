# Semantic Search Engine

Search documents using BM25 keyword retrieval and MiniLM semantic embeddings, then generate cited answers with a local Ollama model.

**[Live Demo](https://semantic-search-engine-nr.streamlit.app/)**

> The public demo returns retrieved evidence without Ollama. Run locally for generated answers.

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

## Tests

```bash
python -m unittest discover -s tests -v
```

## Stack

Python · BM25 · Sentence Transformers · LangChain · LangGraph · Ollama · Streamlit · FastAPI · Docker · Kubernetes

## Author

**Nikshitha Rapolu**

[GitHub](https://github.com/nikshitharapolu) · [LinkedIn](https://www.linkedin.com/in/nikshitha-rapolu-18960321a/) . [Portfolio] (https://nikshitharapolu.github.io)