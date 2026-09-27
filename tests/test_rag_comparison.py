from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluation.compare_rag_paths import compare, load_cases


class FixedPipeline:
    def __init__(self, answer: str, source_id: str):
        self.answer_text = answer
        self.source_id = source_id

    def answer(self, question: str, alpha: float, top_k: int):
        return SimpleNamespace(
            answer=self.answer_text,
            sources=pd.DataFrame([{"doc_id": self.source_id}]),
        )


class ComparisonTests(unittest.TestCase):
    def test_labeled_cases_cover_answerable_and_unanswerable_questions(self):
        cases = load_cases(ROOT / "evaluation/rag_comparison_cases.json")
        self.assertEqual(len(cases), 18)
        self.assertEqual(sum(case["answerable"] for case in cases), 12)
        self.assertEqual(sum(not case["answerable"] for case in cases), 6)

    def test_expected_source_citation_differs_from_merely_valid_citation(self):
        cases = [{"id": "p1", "question": "What is BM25?", "answerable": True,
                  "expected_doc_ids": ["D2"]}]
        report = compare(
            cases,
            FixedPipeline("Wrong source but valid marker [1]", "D9"),
            FixedPipeline("Document-supported answer [1]", "D2"),
        )

        self.assertTrue(report["cases"][0]["original"]["valid_citations"])
        self.assertFalse(report["cases"][0]["original"]["expected_source_cited"])
        metric = "answerable_expected_source_citation_rate"
        self.assertEqual(report["summary"]["original"][metric], 0)
        self.assertEqual(report["summary"]["guided"][metric], 1)


if __name__ == "__main__":
    unittest.main()
