from __future__ import annotations
import os
import numpy as np
import pandas as pd
from utils import HybridSearchEngine

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))


def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    if k == 0:
        return 0.0
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / k


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / len(relevant)


def average_precision(retrieved, relevant):
    if not relevant:
        return 0.0
    precisions = []
    hits = 0
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hits += 1
            precisions.append(hits / i)
    return sum(precisions) / len(relevant) if precisions else 0.0


def reciprocal_rank(retrieved, relevant):
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1 / i
    return 0.0


def evaluate(alpha=0.5, top_k=5):
    engine = HybridSearchEngine(os.path.join(DATA_DIR, "corpus.csv"))
    queries = pd.read_csv(os.path.join(DATA_DIR, "queries.csv"))
    qrels = pd.read_csv(os.path.join(DATA_DIR, "qrels.csv"))

    p_at_k, r_at_k, ap_scores, rr_scores = [], [], [], []

    for _, row in queries.iterrows():
        qid = row["query_id"]
        query = row["query"]
        relevant = set(qrels[qrels["query_id"] == qid]["doc_id"].tolist())
        results = engine.search(query, alpha=alpha, top_k=top_k)
        retrieved = results["doc_id"].tolist()

        p_at_k.append(precision_at_k(retrieved, relevant, top_k))
        r_at_k.append(recall_at_k(retrieved, relevant, top_k))
        ap_scores.append(average_precision(retrieved, relevant))
        rr_scores.append(reciprocal_rank(retrieved, relevant))

    metrics = {
        f"Precision@{top_k}": np.mean(p_at_k),
        f"Recall@{top_k}": np.mean(r_at_k),
        "MAP": np.mean(ap_scores),
        "MRR": np.mean(rr_scores),
    }
    return metrics


if __name__ == "__main__":
    metrics = evaluate(alpha=0.5, top_k=5)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")