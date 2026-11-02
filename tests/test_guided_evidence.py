import sys
from pathlib import Path
import unittest
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from guided_rag import GuidedRAGPipeline

class RequestedFactTests(unittest.TestCase):
    def check(self, question, text):
        return GuidedRAGPipeline.missing_requested_fact(question, pd.DataFrame([{'text':text}]))

    def test_topic_match_is_not_an_answer_to_quantity(self):
        self.assertIsNotNone(self.check('How much does the service cost?', 'The service hosts web applications.'))
        self.assertIsNone(self.check('How much does the service cost?', 'The price is 12 dollars monthly.'))

    def test_date_requires_date_evidence(self):
        self.assertIsNotNone(self.check('In what year was the library introduced?', 'The library supports retrieval.'))
        self.assertIsNone(self.check('In what year was the library introduced?', 'It was introduced in 2021.'))

    def test_attribution_requires_attribution_evidence(self):
        self.assertIsNotNone(self.check('Who wrote the paper?', 'The paper describes lexical scoring.'))
        self.assertIsNone(self.check('Who wrote the paper?', 'It was written by Ada Lovelace.'))

    def test_normal_explanation_is_not_rejected(self):
        self.assertIsNone(self.check('How does the index map terms?', 'The index maps terms to documents.'))

class RetrievalGuardTests(unittest.TestCase):
    def test_promotes_only_consistently_stronger_evidence(self):
        from unittest.mock import Mock
        from generation import ExtractiveFallbackGenerator
        engine = Mock()
        engine.search.return_value = pd.DataFrame([
            {"doc_id":"weak", "title":"Search", "text":"Relevant documents exist.", "semantic_score":0.5},
            {"doc_id":"better", "title":"Feedback", "text":"Judged relevant documents refine searches.", "semantic_score":0.6},
        ])
        result = GuidedRAGPipeline(engine, ExtractiveFallbackGenerator()).answer("How do judged relevant documents refine searches?")
        self.assertEqual(result.sources.iloc[0]["doc_id"], "better")

    def test_retry_cannot_discard_more_relevant_original_results(self):
        from unittest.mock import Mock
        from generation import ExtractiveFallbackGenerator
        engine = Mock()
        first = pd.DataFrame([{"doc_id":"first", "text":"Planet atmosphere composition", "semantic_score":0.1}])
        engine.search.side_effect = [first, pd.DataFrame([{"doc_id":"drift", "text":"Football match", "semantic_score":0.9}])]
        pipeline = GuidedRAGPipeline(engine, ExtractiveFallbackGenerator())
        state = {"question":"Planet atmosphere composition?", "query":"football", "alpha":0.5,"top_k":5,"attempts":1,"sources":first}
        self.assertEqual(pipeline._retrieve(state)["sources"].iloc[0]["doc_id"], "first")
