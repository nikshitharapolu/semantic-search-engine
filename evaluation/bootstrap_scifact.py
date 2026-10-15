"""Paired bootstrap intervals for tuned hybrid versus BM25."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
report = json.loads(
    (ROOT / "reports/scifact-benchmark.json").read_text()
)
results = report["test_results"]

baseline = {
    row["query_id"]: row
    for row in results["BM25"]["per_query"]
}
hybrid = {
    row["query_id"]: row
    for row in results["Hybrid (train-tuned)"]["per_query"]
}

if baseline.keys() != hybrid.keys():
    raise ValueError("The setups must contain the same query IDs.")

query_ids = sorted(baseline)
output = {
    "comparison": "Hybrid (train-tuned) minus BM25",
    "query_count": len(query_ids),
    "resamples": 10000,
    "seed": 42,
    "method": "Paired bootstrap, 95% percentile intervals",
    "metrics": {},
}

for metric in ("Recall@10", "NDCG@10"):
    differences = np.array([
        hybrid[qid][metric] - baseline[qid][metric]
        for qid in query_ids
    ])
    rng = np.random.default_rng(42)
    means = rng.choice(
        differences,
        size=(10000, len(differences)),
        replace=True,
    ).mean(axis=1)

    lower, upper = np.percentile(means, [2.5, 97.5])
    output["metrics"][metric] = {
        "mean_gain": float(differences.mean()),
        "lower": float(lower),
        "upper": float(upper),
    }

destination = ROOT / "reports/scifact-bootstrap.json"
destination.write_text(
    json.dumps(output, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(output, indent=2))
print(f"\nSaved: {destination}")