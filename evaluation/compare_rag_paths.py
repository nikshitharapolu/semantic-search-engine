"""Compare both RAG paths on the same labeled questions and document collection."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag import CITATION_PATTERN, LocalRAGPipeline  # noqa: E402


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation cases must be a nonempty list.")

    seen_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError(f"Invalid evaluation case: {case!r}")

        case_id = case.get("id")
        answerable = case.get("answerable")
        expected_ids = case.get("expected_doc_ids")
        if (
            not isinstance(case_id, str)
            or not isinstance(case.get("question"), str)
            or not isinstance(answerable, bool)
            or not isinstance(expected_ids, list)
            or answerable != bool(expected_ids)
        ):
            raise ValueError(f"Invalid evaluation case: {case!r}")
        if case_id in seen_ids:
            raise ValueError(f"Duplicate evaluation case ID: {case_id}")
        seen_ids.add(case_id)
    return cases


def measure(case: dict[str, Any], pipeline: Any, *, alpha: float, top_k: int) -> dict[str, Any]:
    started = time.perf_counter()
    response = pipeline.answer(case["question"], alpha=alpha, top_k=top_k)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    sources = response.sources.reset_index(drop=True)
    source_ids = [str(doc_id) for doc_id in sources.get("doc_id", [])]
    cited_ids = []
    for number in CITATION_PATTERN.findall(response.answer):
        index = int(number) - 1
        if 0 <= index < len(source_ids) and source_ids[index] not in cited_ids:
            cited_ids.append(source_ids[index])

    expected_ids = set(case["expected_doc_ids"])
    trace = getattr(response, "trace", [])
    outcome = getattr(response, "outcome", "original RAG")
    valid_citations = LocalRAGPipeline.has_valid_citation(response.answer, len(sources))
    # This counts explicit application abstentions. Model-written refusals need review.
    abstained = (
        outcome in ("insufficient evidence", "blocked")
        or response.answer.startswith("The available documents do not contain enough")
        or response.answer == "The indexed documents do not contain enough evidence to answer this question."
    )
    offline_excerpt = (
        any(event.get("mode") == "offline excerpt" for event in trace)
        or response.answer.startswith("The strongest retrieved evidence says:")
    )
    if case["answerable"]:
        source_hit = bool(expected_ids.intersection(source_ids))
        expected_source_cited = valid_citations and bool(expected_ids.intersection(cited_ids))
    else:
        source_hit = None
        expected_source_cited = None

    return {
        "answer": response.answer,
        "outcome": outcome,
        "source_ids": source_ids,
        "cited_source_ids": cited_ids,
        "valid_citations": valid_citations,
        "source_hit": source_hit,
        "expected_source_cited": expected_source_cited,
        "abstained": abstained,
        "offline_excerpt": offline_excerpt,
        "search_attempts": sum(event.get("step") == "retrieve" for event in trace) if trace else 1,
        "latency_ms": latency_ms,
        "trace": trace,
    }


def summarize(rows: list[dict[str, Any]], path: str) -> dict[str, Any]:
    measurements = [row[path] for row in rows]
    positive = [result for row, result in zip(rows, measurements) if row["answerable"]]
    negative = [result for row, result in zip(rows, measurements) if not row["answerable"]]
    answered = [result for result in measurements if not result["abstained"]]

    def rate(values: list[bool]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    return {
        "answerable_source_hit_rate": rate([item["source_hit"] for item in positive]),
        "answerable_expected_source_citation_rate": rate(
            [item["expected_source_cited"] for item in positive]
        ),
        "answerable_abstention_rate": rate([item["abstained"] for item in positive]),
        "unanswerable_explicit_abstention_rate": rate([item["abstained"] for item in negative]),
        "answered_valid_citation_rate": rate([item["valid_citations"] for item in answered]),
        "median_latency_ms": round(
            statistics.median(item["latency_ms"] for item in measurements), 1
        ),
        "mean_search_attempts": round(
            statistics.mean(item["search_attempts"] for item in measurements), 2
        ),
        "offline_excerpt_count": sum(item["offline_excerpt"] for item in measurements),
    }


def compare(
    cases: list[dict[str, Any]],
    original: Any,
    guided: Any,
    *,
    alpha: float = 0.5,
    top_k: int = 5,
) -> dict[str, Any]:
    rows = []
    for case in cases:
        rows.append(
            {
                **case,
                "original": measure(case, original, alpha=alpha, top_k=top_k),
                "guided": measure(case, guided, alpha=alpha, top_k=top_k),
            }
        )
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(rows),
        "answerable_count": sum(row["answerable"] for row in rows),
        "unanswerable_count": sum(not row["answerable"] for row in rows),
        "alpha": alpha,
        "top_k": top_k,
        "summary": {name: summarize(rows, name) for name in ("original", "guided")},
        "limitations": [
            "Citation/source metrics are proxies; review each answer against the retrieved passage.",
            "Only explicit app abstentions are counted; review model-written refusals manually.",
            "The 12-document bundled corpus and hand-labeled questions are a small demo, not a benchmark.",
            "Offline excerpts are flagged; rerun with Ollama active before comparing generation quality.",
        ],
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "evaluation/rag_comparison_cases.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/rag-path-comparison.json")
    parser.add_argument(
        "--limit", type=int, default=None, help="Run the first N cases for a quick check"
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not 0 <= args.alpha <= 1 or not 1 <= args.top_k <= 20:
        parser.error("--alpha must be in [0, 1] and --top-k in [1, 20]")

    # The fixed bundled corpus makes reruns comparable even after users upload files.
    from generation import ExtractiveFallbackGenerator, OllamaGenerator, ResilientLocalGenerator
    from guided_rag import GuidedRAGPipeline
    from service import BASE_DATA_DIR, STATE_DATA_DIR
    from utils import HybridSearchEngine

    engine = HybridSearchEngine(
        str(BASE_DATA_DIR / "corpus.csv"),
        index_dir=str(STATE_DATA_DIR / "index" / "comparison"),
    )
    model = OllamaGenerator()
    generator = ResilientLocalGenerator(model, ExtractiveFallbackGenerator())
    cases = load_cases(args.cases)
    report = compare(
        cases[:args.limit],
        LocalRAGPipeline(engine, generator),
        GuidedRAGPipeline(engine, generator, rewriter=model),
        alpha=args.alpha,
        top_k=args.top_k,
    )
    report["model"] = model.model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"case_count": report["case_count"], "summary": report["summary"]}, indent=2))
    print(f"Saved answer-by-answer report to {args.output}")


if __name__ == "__main__":
    main()
