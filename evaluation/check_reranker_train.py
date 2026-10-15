"""Compare weighted hybrid with candidate-union reranking on training data."""

import argparse
import json
import random
import statistics
import time
from datetime import datetime, timezone

import numpy as np
from sentence_transformers import CrossEncoder

from benchmark_scifact import (
    ROOT,
    HybridSearchEngine,
    metrics,
    read_jsonl,
    read_qrels,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of training queries; 0 means all")
    args = parser.parse_args()

    dataset = ROOT / "evaluation/datasets/scifact"
    queries = {
        str(row["_id"]): row["text"]
        for row in read_jsonl(dataset / "queries.jsonl")
    }
    qrels = read_qrels(dataset / "qrels/train.tsv")

    if not 0 <= args.limit <= len(qrels):
        parser.error("--limit must be between 0 and the training query count")

    query_ids = sorted(qrels)
    if args.limit:
        query_ids = random.Random(42).sample(query_ids, args.limit)

    print(f"Training queries: {len(query_ids)}", flush=True)
    engine = HybridSearchEngine(
        str(ROOT / "evaluation/datasets/scifact-engine.csv"),
        index_dir=str(ROOT / "evaluation/datasets/scifact-index"),
    )
    model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker = CrossEncoder(model_name, max_length=512)
    print(f"Reranker device: {reranker.device}", flush=True)

    doc_ids = engine.df["doc_id"].astype(str).tolist()
    rows = []

    for index, query_id in enumerate(query_ids, 1):
        query = queries[query_id]
        relevant = qrels[query_id]

        # Existing weighted hybrid ranking.
        ranking = engine.search(query, alpha=0.4, top_k=len(doc_ids))
        ranking["doc_id"] = ranking["doc_id"].astype(str)
        hybrid_ids = ranking.head(10)["doc_id"].tolist()

        # Restore corpus order before sorting the component scores.
        original = ranking.set_index("doc_id").loc[doc_ids]
        lexical_ids = original.sort_values(
            "bm25_score", ascending=False
        ).head(100).index.tolist()
        semantic_ids = original.sort_values(
            "semantic_score", ascending=False
        ).head(100).index.tolist()

        # Deduplicate without adding any ground-truth documents.
        candidate_ids = list(dict.fromkeys(lexical_ids + semantic_ids))
        candidates = original.loc[candidate_ids]
        passages = candidates["combined_text"].tolist()

        started = time.perf_counter()
        scores = np.asarray(reranker.predict(
            [(query, passage) for passage in passages],
            batch_size=16,
            show_progress_bar=False,
        )).reshape(-1)
        rerank_seconds = time.perf_counter() - started

        order = np.argsort(-scores, kind="stable")[:10]
        reranked_ids = [candidate_ids[position] for position in order]

        hybrid_recall, hybrid_ndcg = metrics(hybrid_ids, relevant, 10)
        rerank_recall, rerank_ndcg = metrics(reranked_ids, relevant, 10)

        rows.append({
            "query_id": query_id,
            "query": query,
            "relevant_doc_ids": relevant,
            "candidate_count": len(candidate_ids),
            "candidate_recall": len(set(candidate_ids) & set(relevant))
                                / len(relevant),
            "hybrid": {
                "Recall@10": hybrid_recall,
                "NDCG@10": hybrid_ndcg,
                "retrieved_doc_ids": hybrid_ids,
            },
            "reranked": {
                "Recall@10": rerank_recall,
                "NDCG@10": rerank_ndcg,
                "retrieved_doc_ids": reranked_ids,
            },
            "rerank_seconds": rerank_seconds,
        })

        print(
            f"{index}/{len(query_ids)} | query {query_id} | "
            f"{len(candidate_ids)} candidates | "
            f"NDCG: {hybrid_ndcg:.4f} -> {rerank_ndcg:.4f} | "
            f"reranking: {rerank_seconds:.1f}s",
            flush=True,
        )

    summary = {}
    for setup in ("hybrid", "reranked"):
        summary[setup] = {
            metric: statistics.mean(row[setup][metric] for row in rows)
            for metric in ("Recall@10", "NDCG@10")
        }

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "split": "train",
        "query_count": len(rows),
        "sample_seed": 42 if args.limit else None,
        "smoke_check": len(rows) < len(qrels),
        "hybrid_alpha": 0.4,
        "candidate_depth_per_retriever": 100,
        "reranker": model_name,
        "max_length": 512,
        "batch_size": 16,
        "device": str(reranker.device),
        "summary": summary,
        "mean_candidate_recall": statistics.mean(
            row["candidate_recall"] for row in rows
        ),
        "median_rerank_seconds": statistics.median(
            row["rerank_seconds"] for row in rows
        ),
        "per_query": rows,
    }

    suffix = "smoke" if report["smoke_check"] else "full"
    output = ROOT / f"reports/scifact-rerank-train-{suffix}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\nTRAINING RESULTS")
    print(json.dumps(summary, indent=2))
    print(f"Median reranking time: {report['median_rerank_seconds']:.2f}s")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()