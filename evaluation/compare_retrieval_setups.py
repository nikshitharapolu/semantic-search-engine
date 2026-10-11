"""Compare BM25, semantic, and hybrid retrieval on the RAG cases."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compare_rag_paths import load_cases, measure
from generation import (
    ExtractiveFallbackGenerator,
    OllamaGenerator,
    ResilientLocalGenerator,
)
from guided_rag import GuidedRAGPipeline
from service import BASE_DATA_DIR, STATE_DATA_DIR
from utils import HybridSearchEngine


class FixedAlphaEngine:
    """Keep each setup's weight fixed, including guided retries."""

    def __init__(self, engine, alpha):
        self.engine = engine
        self.alpha = alpha

    def search(self, query, alpha=0.5, top_k=5):
        return self.engine.search(
            query, alpha=self.alpha, top_k=top_k
        )

    def __getattr__(self, name):
        return getattr(self.engine, name)


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return None
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def ndcg_at_k(retrieved, relevant, k):
    """NDCG with binary relevance and logarithmic discount."""
    if not relevant:
        return None

    dcg = sum(
        1 / math.log2(rank + 2)
        for rank, doc_id in enumerate(retrieved[:k])
        if doc_id in relevant
    )
    ideal_dcg = sum(
        1 / math.log2(rank + 2)
        for rank in range(min(k, len(relevant)))
    )
    return dcg / ideal_dcg


def mean(values):
    return sum(values) / len(values) if values else None


def format_metric(value, percentage=False):
    if value is None:
        return "N/A"
    return f"{value:.1%}" if percentage else f"{value:.4f}"


def make_readme_section(report):
    k = report["top_k"]
    lines = [
        "## Retrieval Setup Comparison",
        "",
        f"Compared three retrieval setups on the same "
        f"{report['case_count']} hand-labeled questions "
        f"({report['answerable_count']} answerable and "
        f"{report['unanswerable_count']} unanswerable).",
        "",
        f"Recall@{k} and binary-relevance NDCG@{k} measure "
        "initial retrieval on answerable questions. Explicit abstention "
        "is measured using the same guided RAG workflow for all setups; "
        "each setup's retrieval weight stays fixed during retries.",
        "",
        f"| Setup | Recall@{k} | NDCG@{k} | "
        "Unanswerable abstention | Answerable abstention |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in report["results"].items():
        lines.append(
            f"| {name} | "
            f"{format_metric(result['recall_at_k'])} | "
            f"{format_metric(result['ndcg_at_k'])} | "
            f"{format_metric(result['unanswerable_abstention_rate'], True)} | "
            f"{format_metric(result['answerable_abstention_rate'], True)} |"
        )
    lines.extend([
        "",
        f"Generation model: `{report['model']}`. "
        "No offline fallback responses were used.",
        "",
        "These are small bundled-demo results, not a production benchmark. "
        "Unanswerable questions have no relevant documents, so they are "
        "excluded from Recall and NDCG averages. Abstention counts explicit "
        "application responses; model-written refusals and answer correctness "
        "require manual review. Inspect the saved answers before making claims.",
    ])
    return "\n".join(lines)


def update_readme(section):
    path = ROOT / "README.md"
    start = "<!-- retrieval-comparison:start -->"
    end = "<!-- retrieval-comparison:end -->"
    block = f"{start}\n{section}\n{end}"
    content = path.read_text(encoding="utf-8")

    if start in content and end in content:
        before, rest = content.split(start, 1)
        _, after = rest.split(end, 1)
        content = before + block + after
    elif start in content or end in content:
        raise ValueError("README has an incomplete comparison marker.")
    else:
        content = content.rstrip() + "\n\n" + block + "\n"

    path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20")

    cases = load_cases(ROOT / "evaluation/rag_comparison_cases.json")
    engine = HybridSearchEngine(
        str(BASE_DATA_DIR / "corpus.csv"),
        index_dir=str(STATE_DATA_DIR / "index" / "retrieval-comparison"),
    )
    model = OllamaGenerator()
    generator = ResilientLocalGenerator(
        model, ExtractiveFallbackGenerator()
    )

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases),
        "answerable_count": sum(c["answerable"] for c in cases),
        "unanswerable_count": sum(not c["answerable"] for c in cases),
        "top_k": args.top_k,
        "model": model.model,
        "results": {},
        "cases": [],
    }

    for name, alpha in [
        ("BM25", 1.0),
        ("Semantic", 0.0),
        ("Hybrid", 0.5),
    ]:
        fixed_engine = FixedAlphaEngine(engine, alpha)
        pipeline = GuidedRAGPipeline(
            fixed_engine, generator, rewriter=model
        )
        recalls, ndcgs, positive, negative = [], [], [], []

        for index, case in enumerate(cases, 1):
            print(f"{name}: question {index}/{len(cases)}", flush=True)
            relevant = {str(x) for x in case["expected_doc_ids"]}
            initial = fixed_engine.search(
                case["question"], top_k=args.top_k
            )
            retrieved = [str(x) for x in initial["doc_id"].tolist()]

            recall = recall_at_k(retrieved, relevant, args.top_k)
            ndcg = ndcg_at_k(retrieved, relevant, args.top_k)
            answer = measure(
                case, pipeline, alpha=alpha, top_k=args.top_k
            )

            if answer["offline_excerpt"]:
                raise RuntimeError(
                    "Offline fallback detected. Check Ollama and rerun. "
                    "The README has not been updated."
                )

            if case["answerable"]:
                recalls.append(recall)
                ndcgs.append(ndcg)
                positive.append(answer["abstained"])
            else:
                negative.append(answer["abstained"])

            report["cases"].append({
                "setup": name,
                **case,
                "initial_source_ids": retrieved,
                "recall_at_k": recall,
                "ndcg_at_k": ndcg,
                "guided_response": answer,
            })

        report["results"][name] = {
            "alpha": alpha,
            "recall_at_k": mean(recalls),
            "ndcg_at_k": mean(ndcgs),
            "unanswerable_abstention_rate": mean(negative),
            "answerable_abstention_rate": mean(positive),
        }

    output = ROOT / "reports" / "retrieval-setup-comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    section = make_readme_section(report)
    print("\n" + section)
    print(f"\nDetailed report: {output}")

    if args.update_readme:
        update_readme(section)
        print("Updated README.md")


if __name__ == "__main__":
    main()