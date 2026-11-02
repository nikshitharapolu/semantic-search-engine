"""Publish the fixed demo set and compare RAG versions with deterministic evaluators."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from uuid import uuid5, NAMESPACE_URL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from evaluation.compare_rag_paths import load_cases, measure


def reference(case):
    return {key: case[key] for key in ("id", "answerable", "expected_doc_ids")}


def sync_dataset(client, name, cases):
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
    else:
        dataset = client.create_dataset(dataset_name=name, description="18 labeled questions over the bundled 12-document demo corpus. Metrics are proxies, not answer correctness.")
    existing = {str(e.id): e for e in client.list_examples(dataset_id=dataset.id)}
    pending = []
    expected = set()
    for case in cases:
        example_id = str(uuid5(NAMESPACE_URL, f"hybrid-search/{dataset.id}/{case['id']}"))
        expected.add(example_id)
        inputs = {"question": case["question"]}
        outputs = reference(case)
        if example_id in existing:
            old = existing[example_id]
            if old.inputs != inputs or old.outputs != outputs:
                raise ValueError("Dataset labels changed. Use a new versioned dataset name.")
        else:
            pending.append({"id": example_id, "inputs": inputs, "outputs": outputs,
                            "metadata": {"case_id": case["id"]}})
    if set(existing) - expected:
        raise ValueError("Dataset contains unrelated examples. Use a new dataset name.")
    if pending:
        client.create_examples(dataset_id=dataset.id, examples=pending)
    return dataset


def metric_scores(run, example):
    output = run.outputs or {}
    truth = example.outputs or {}
    scores = [{"key": "abstention_matches_label", "score": int(bool(output.get("abstained")) == (not truth["answerable"]))},
              {"key": "offline_excerpt", "score": int(bool(output.get("offline_excerpt")))}]
    if not output.get("abstained"):
        scores.append({"key": "answered_valid_citations", "score": int(bool(output.get("valid_citations")))})
    if truth["answerable"]:
        scores.extend({"key": key, "score": int(bool(output.get(key)))}
                      for key in ("source_hit", "expected_source_cited"))
    return {"results": scores}


def evaluation_target(pipeline, cases):
    by_question = {case["question"]: case for case in cases}
    def target(inputs):
        return measure(by_question[inputs["question"]], pipeline, alpha=0.5, top_k=5)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "evaluation/rag_comparison_cases.json")
    parser.add_argument("--label", default="")
    parser.add_argument("--dataset", default="hybrid-search-demo-v1")
    parser.add_argument("--path", choices=("original", "guided", "both"), default="both")
    parser.add_argument("--dry-run", action="store_true", help="Validate cases without any network calls or model runs")
    parser.add_argument("--offline-excerpts", action="store_true", help="Use excerpts only; no Ollama model calls")
    args = parser.parse_args()
    cases = load_cases(args.cases)
    if args.dry_run:
        print(json.dumps({"case_count": len(cases), "answerable": sum(c["answerable"] for c in cases), "paths": args.path, "uploads": False}))
        return
    if not os.environ.get("LANGSMITH_API_KEY"):
        parser.error("Set LANGSMITH_API_KEY locally; never commit it or paste it into chat.")
    # Explicit CLI invocation opts in to uploading the bundled public questions,
    # retrieved passages, prompts, answers and experiment results.
    os.environ["LANGSMITH_TRACING"] = "true"
    from langsmith import Client
    from generation import ExtractiveFallbackGenerator, OllamaGenerator, ResilientLocalGenerator
    from guided_rag import GuidedRAGPipeline
    from rag import LocalRAGPipeline
    from utils import HybridSearchEngine
    engine = HybridSearchEngine(str(ROOT / "data/corpus.csv"), index_dir=str(ROOT / "data/index/comparison"))
    model = None if args.offline_excerpts else OllamaGenerator()
    generator = ExtractiveFallbackGenerator() if model is None else ResilientLocalGenerator(model, ExtractiveFallbackGenerator())
    pipelines = {"original": LocalRAGPipeline(engine, generator),
                 "guided": GuidedRAGPipeline(engine, generator, rewriter=model)}
    client = Client()
    dataset = sync_dataset(client, args.dataset, cases)
    names = pipelines if args.path == "both" else [args.path]
    for name in names:
        results = client.evaluate(evaluation_target(pipelines[name], cases), data=dataset.id,
                        evaluators=[metric_scores], experiment_prefix=(f"hybrid-{name}-{args.label}" if args.label else f"hybrid-{name}"),
                        max_concurrency=1, blocking=True,
                        metadata={"path": name, "alpha": 0.5, "top_k": 5,
                                  "generation": "offline-excerpts" if model is None else model.model,
                                  "limitations": "Small demo; citation and source metrics do not establish factual correctness."})
        failed = [row for row in results if row["run"].error]
        client.flush()
        if failed:
            raise RuntimeError(f"{name}: {len(failed)} evaluation runs failed; inspect the experiment before reporting results.")
        print(f"{name}: {len(cases)} cases completed without application errors.")


if __name__ == "__main__":
    main()
