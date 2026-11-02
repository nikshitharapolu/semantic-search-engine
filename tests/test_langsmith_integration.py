import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from uuid import uuid4

import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from tracing import inputs_without_self, response_view, retrieval_view
from evaluation.langsmith_eval import sync_dataset, metric_scores

CASE = {"id": "p1", "question": "What is search?", "answerable": True, "expected_doc_ids": ["D1"]}


class LangSmithIntegrationTests(unittest.TestCase):
    def test_trace_views_exclude_engine_and_serialize_documents(self):
        self.assertEqual(inputs_without_self({"self": object(), "query": "hello"}), {"query": "hello"})
        frame = pd.DataFrame([{"doc_id": "D1", "text": "Evidence"}])
        self.assertEqual(retrieval_view(frame)["documents"][0]["page_content"], "Evidence")
        self.assertEqual(response_view(SimpleNamespace(answer="Answer [1]", sources=frame))["sources"][0]["doc_id"], "D1")
        self.assertEqual(response_view(None), {})

    def test_direct_extractive_generator_is_labeled_offline(self):
        from guided_rag import GuidedRAGPipeline
        from generation import ExtractiveFallbackGenerator
        engine = Mock()
        engine.search.return_value = pd.DataFrame([{
            "doc_id": "D1", "title": "Hybrid search", "text": "Hybrid search combines retrieval methods.",
            "semantic_score": 0.9,
        }])
        response = GuidedRAGPipeline(engine, ExtractiveFallbackGenerator()).answer("What is hybrid search?")
        self.assertEqual(response.outcome, "cited excerpt")
        self.assertEqual(response.trace[-1]["mode"], "offline excerpt")

    def client(self):
        client = Mock()
        client.has_dataset.return_value = True
        client.read_dataset.return_value = SimpleNamespace(id=uuid4())
        client.list_examples.return_value = []
        return client

    def test_repeating_sync_does_not_duplicate_examples(self):
        client = self.client()
        sync_dataset(client, "demo", [CASE])
        example = client.create_examples.call_args.kwargs["examples"][0]
        client.list_examples.return_value = [SimpleNamespace(**example)]
        client.create_examples.reset_mock()
        sync_dataset(client, "demo", [CASE])
        client.create_examples.assert_not_called()

    def test_changed_labels_are_rejected_without_upload(self):
        client = self.client()
        sync_dataset(client, "demo", [CASE])
        example = client.create_examples.call_args.kwargs["examples"][0]
        client.list_examples.return_value = [SimpleNamespace(**example)]
        client.create_examples.reset_mock()
        changed = {**CASE, "expected_doc_ids": ["D2"]}
        with self.assertRaisesRegex(ValueError, "labels changed"):
            sync_dataset(client, "demo", [changed])
        client.create_examples.assert_not_called()

    def test_unrelated_dataset_examples_are_rejected(self):
        client = self.client()
        client.list_examples.return_value = [SimpleNamespace(id=uuid4())]
        with self.assertRaisesRegex(ValueError, "unrelated"):
            sync_dataset(client, "demo", [CASE])
        client.create_examples.assert_not_called()

    def test_unanswerable_cases_do_not_get_source_hit_score(self):
        run = SimpleNamespace(outputs={"abstained": True, "valid_citations": False, "offline_excerpt": False})
        example = SimpleNamespace(outputs={"answerable": False})
        scores = {r["key"]: r["score"] for r in metric_scores(run, example)["results"]}
        self.assertEqual(scores["abstention_matches_label"], 1)
        self.assertNotIn("source_hit", scores)
