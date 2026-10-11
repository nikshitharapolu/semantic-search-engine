"""Export per-query benchmark results without rerunning retrieval."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export_report(report_path: Path, queries_path: Path | None = None) -> Path:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    queries = {}
    if queries_path is not None:
        with queries_path.open(encoding="utf-8") as handle:
            queries = {str(row["_id"]): row["text"] for row in
                       (json.loads(line) for line in handle if line.strip())}
    k = report["top_k"]
    output = report_path.with_name(report_path.stem + "-per-query.csv")
    fields = ["setup", "alpha", "query_id", "query", f"Recall@{k}",
              f"NDCG@{k}", "retrieved_doc_ids", "relevant_doc_ids",
              "missed_relevant_doc_ids"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for setup, result in report["test_results"].items():
            for row in result["per_query"]:
                relevant = row["relevant_doc_ids"]
                retrieved = row["retrieved_doc_ids"]
                writer.writerow({
                    "setup": setup,
                    "alpha": result["alpha"],
                    "query_id": row["query_id"],
                    "query": queries.get(str(row["query_id"]), ""),
                    f"Recall@{k}": row[f"Recall@{k}"],
                    f"NDCG@{k}": row[f"NDCG@{k}"],
                    "retrieved_doc_ids": json.dumps(retrieved),
                    "relevant_doc_ids": json.dumps(relevant, sort_keys=True),
                    "missed_relevant_doc_ids": json.dumps(
                        sorted(set(relevant) - set(retrieved))
                    ),
                })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path,
                        default=ROOT / "reports/scifact-benchmark.json")
    parser.add_argument("--queries", type=Path,
                        default=ROOT / "evaluation/datasets/scifact/queries.jsonl")
    args = parser.parse_args()
    print(f"Saved per-query CSV: {export_report(args.report, args.queries)}")


if __name__ == "__main__":
    main()
