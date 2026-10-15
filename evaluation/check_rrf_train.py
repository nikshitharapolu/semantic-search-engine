"""Compare reciprocal rank fusion with weighted hybrid on training data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_scifact import (
    ROOT,
    HybridSearchEngine,
    cache_scores,
    evaluate,
    metrics,
    read_jsonl,
    read_qrels,
)

dataset = ROOT / "evaluation/datasets/scifact"
queries = {
    str(row["_id"]): row["text"]
    for row in read_jsonl(dataset / "queries.jsonl")
}
qrels = read_qrels(dataset / "qrels/train.tsv")

engine = HybridSearchEngine(
    str(ROOT / "evaluation/datasets/scifact-engine.csv"),
    index_dir=str(ROOT / "evaluation/datasets/scifact-index"),
)
cached = cache_scores(engine, queries, qrels, "Train")
doc_ids = engine.df["doc_id"].astype(str).to_numpy()

baseline = evaluate(engine, cached, qrels, alpha=0.4, k=10)
print(
    "\nWeighted hybrid alpha=0.4: "
    f"Recall@10={baseline['Recall@10']:.4f}, "
    f"NDCG@10={baseline['NDCG@10']:.4f}"
)

# Cache each retriever's top-100 positions.
candidates = {}
for query_id, (lexical, semantic) in cached.items():
    candidates[query_id] = (
        pd.Series(lexical).sort_values(ascending=False).index[:100].to_numpy(),
        pd.Series(semantic).sort_values(ascending=False).index[:100].to_numpy(),
    )

results = {}
for constant in (10, 30, 60, 100):
    details = []
    for query_id, (lexical_indices, semantic_indices) in candidates.items():
        scores = np.zeros(len(doc_ids))
        for indices in (lexical_indices, semantic_indices):
            ranks = np.arange(1, len(indices) + 1)
            scores[indices] += 1.0 / (constant + ranks)

        top = pd.Series(scores).sort_values(ascending=False).index[:10]
        retrieved = doc_ids[top].tolist()
        recall, ndcg = metrics(retrieved, qrels[query_id], 10)
        details.append({
            "query_id": query_id,
            "retrieved_doc_ids": retrieved,
            "Recall@10": recall,
            "NDCG@10": ndcg,
        })

    result = {
        "rank_constant": constant,
        "Recall@10": float(np.mean([x["Recall@10"] for x in details])),
        "NDCG@10": float(np.mean([x["NDCG@10"] for x in details])),
        "per_query": details,
    }
    results[str(constant)] = result
    print(
        f"RRF constant={constant}: "
        f"Recall@10={result['Recall@10']:.4f}, "
        f"NDCG@10={result['NDCG@10']:.4f}"
    )

output = ROOT / "reports/scifact-rrf-train.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({
    "split": "train",
    "query_count": len(qrels),
    "candidate_depth_per_retriever": 100,
    "top_k": 10,
    "weighted_hybrid": baseline,
    "rrf": results,
}, indent=2) + "\n", encoding="utf-8")
print(f"\nSaved training results: {output}")