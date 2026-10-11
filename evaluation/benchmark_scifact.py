"""Benchmark the existing search engine on BEIR SciFact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import HybridSearchEngine, MODEL_NAME
from export_scifact_results import export_report


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_qrels(path):
    frame = pd.read_csv(path, sep="\t", dtype=str)
    judgments = {}
    for row in frame.to_dict("records"):
        score = float(row["score"])
        if score > 0:
            judgments.setdefault(str(row["query-id"]), {})[
                str(row["corpus-id"])
            ] = score
    return judgments


def metrics(ranked, relevant, k):
    top = ranked[:k]
    recall = len(set(top) & set(relevant)) / len(relevant)

    # Linear relevance gain matches BEIR's trec-style NDCG.
    # SciFact judgments are binary.
    dcg = sum(
        relevant.get(doc_id, 0) / math.log2(position + 2)
        for position, doc_id in enumerate(top)
    )
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(
        score / math.log2(position + 2)
        for position, score in enumerate(ideal)
    )
    return recall, dcg / idcg


def cache_scores(engine, queries, qrels, label):
    """Get the actual engine's component scores once per query."""
    cached = {}
    doc_ids = engine.df["doc_id"].astype(str).tolist()

    for index, query_id in enumerate(sorted(qrels), 1):
        results = engine.search(
            queries[query_id], alpha=0.5, top_k=len(doc_ids)
        )
        results["doc_id"] = results["doc_id"].astype(str)
        results = results.set_index("doc_id").reindex(doc_ids)

        lexical = engine.minmax(results["bm25_score"].to_numpy())
        semantic = engine.minmax(results["semantic_score"].to_numpy())
        cached[query_id] = (lexical, semantic)

        if index == 1 or index % 25 == 0 or index == len(qrels):
            print(f"{label}: {index}/{len(qrels)} queries", flush=True)

    return cached


def evaluate(engine, cached, qrels, alpha, k):
    recalls, ndcgs, details = [], [], []
    doc_ids = engine.df["doc_id"].astype(str).tolist()

    for query_id, (lexical, semantic) in cached.items():
        scores = alpha * lexical + (1 - alpha) * semantic

        # Match the engine's pandas sorting behavior.
        ranking = pd.DataFrame({
            "doc_id": doc_ids,
            "hybrid_score": scores,
        }).sort_values("hybrid_score", ascending=False)

        retrieved = ranking.head(k)["doc_id"].tolist()
        recall, ndcg = metrics(retrieved, qrels[query_id], k)
        recalls.append(recall)
        ndcgs.append(ndcg)
        details.append({
            "query_id": query_id,
            "retrieved_doc_ids": retrieved,
            "relevant_doc_ids": qrels[query_id],
            f"Recall@{k}": recall,
            f"NDCG@{k}": ndcg,
        })

    return {
        "alpha": alpha,
        "query_count": len(details),
        f"Recall@{k}": float(np.mean(recalls)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
        "per_query": details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Check only 10 queries per split")
    args = parser.parse_args()

    dataset = ROOT / "evaluation/datasets/scifact"
    corpus_path = dataset / "corpus.jsonl"
    query_path = dataset / "queries.jsonl"
    train_path = dataset / "qrels/train.tsv"
    test_path = dataset / "qrels/test.tsv"

    for path in (corpus_path, query_path, train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")

    corpus = read_jsonl(corpus_path)
    queries = {
        str(row["_id"]): row["text"]
        for row in read_jsonl(query_path)
    }
    train = read_qrels(train_path)
    test = read_qrels(test_path)

    if set(train) & set(test):
        raise ValueError("Training and test query IDs overlap.")

    known_docs = {str(row["_id"]) for row in corpus}
    for split in (train, test):
        for query_id, relevant in split.items():
            if query_id not in queries:
                raise ValueError(f"Missing query: {query_id}")
            if not set(relevant).issubset(known_docs):
                raise ValueError(f"Missing judged document for {query_id}")

    if args.smoke:
        train = dict(sorted(train.items())[:10])
        test = dict(sorted(test.items())[:10])

    # Create a separate corpus compatible with your existing engine.
    prepared = ROOT / "evaluation/datasets/scifact-engine.csv"
    pd.DataFrame([
        {
            "doc_id": str(row["_id"]),
            "title": row.get("title", ""),
            "text": row.get("text", ""),
            "category": "SciFact",
        }
        for row in corpus
    ]).to_csv(prepared, index=False)

    print(f"Corpus: {len(corpus)} documents", flush=True)
    print(f"Train: {len(train)} queries; test: {len(test)} queries",
          flush=True)
    print("Loading engine and building/reusing embeddings...", flush=True)

    engine = HybridSearchEngine(
        str(prepared),
        index_dir=str(ROOT / "evaluation/datasets/scifact-index"),
    )
    k = 10

    train_scores = cache_scores(engine, queries, train, "Train")
    tuning = []
    for alpha in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        result = evaluate(engine, train_scores, train, alpha, k)
        tuning.append({
            key: value for key, value in result.items()
            if key != "per_query"
        })
        print(
            f"Train alpha={alpha:.1f}: "
            f"NDCG@10={result['NDCG@10']:.4f}, "
            f"Recall@10={result['Recall@10']:.4f}"
        )

    # Select using training scores only. Prefer 0.5 if tied.
    best = max(
        tuning,
        key=lambda result: (
            result["NDCG@10"],
            result["Recall@10"],
            -abs(result["alpha"] - 0.5),
        ),
    )
    selected_alpha = best["alpha"]
    del train_scores

    print(f"\nSelected hybrid alpha: {selected_alpha}", flush=True)
    test_scores = cache_scores(engine, queries, test, "Test")

    results = {}
    for name, alpha in (
        ("BM25", 1.0),
        ("Semantic", 0.0),
        ("Hybrid (fixed)", 0.5),
        ("Hybrid (train-tuned)", selected_alpha),
    ):
        results[name] = evaluate(engine, test_scores, test, alpha, k)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "BEIR SciFact",
        "smoke_test": args.smoke,
        "document_count": len(corpus),
        "train_query_count": len(train),
        "test_query_count": len(test),
        "embedding_model": MODEL_NAME,
        "top_k": k,
        "selected_alpha": selected_alpha,
        "selection_metric": "Training NDCG@10",
        "dataset_sha256": {
            path.name if path.suffix != ".tsv" else
            f"qrels/{path.name}":
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (corpus_path, query_path, train_path, test_path)
        },
        "training_results": tuning,
        "test_results": results,
        "limitations": [
            "Retrieval only; generation and abstention are not evaluated.",
            "Uses the existing engine's MiniLM truncation behavior.",
            "Unjudged documents receive zero relevance for scoring.",
            "These scores describe this implementation, not all BM25 "
            "or dense retrieval systems.",
        ],
    }

    output = ROOT / "reports" / (
        "scifact-smoke.json" if args.smoke else "scifact-benchmark.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    csv_output = export_report(output, query_path)

    print("\nTEST RESULTS")
    for name, result in results.items():
        print(
            f"{name}: alpha={result['alpha']:.1f}, "
            f"Recall@10={result['Recall@10']:.4f}, "
            f"NDCG@10={result['NDCG@10']:.4f}"
        )
    print(f"\nSaved report: {output}")
    print(f"Saved per-query CSV: {csv_output}")
    if args.smoke:
        print("Smoke check only: do not use these numbers as benchmark results.")


if __name__ == "__main__":
    main()
