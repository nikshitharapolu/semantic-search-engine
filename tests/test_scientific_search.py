import gzip
import csv
import hashlib
import json
from pathlib import Path
import sys
from unittest import TestCase
from unittest.mock import Mock
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from generation import ExtractiveFallbackGenerator
from scientific_search import ScientificEvidencePipeline, spelling_suggestion


class ScientificCollectionTests(TestCase):
    def test_packaged_corpus_matches_embedding_index(self):
        directory = ROOT / 'data/scientific'
        manifest = json.loads((directory / 'manifest.json').read_text())
        for name, expected in manifest['artifacts'].items():
            self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), expected)
        with gzip.open(directory / 'corpus.csv.gz', 'rt') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 5183)
        self.assertEqual(len({r['doc_id'] for r in rows}), 5183)
        digest = hashlib.sha256(manifest['model'].encode())
        for row in rows:
            digest.update(row['doc_id'].encode() + b'\0')
            digest.update((row['title'] + ' ' + row['text']).encode() + b'\0')
        self.assertEqual(digest.hexdigest(), manifest['fingerprint'])
        embeddings = np.load(directory / 'embeddings.npy', allow_pickle=False)
        self.assertEqual(embeddings.shape, (5183, 384))
        self.assertTrue(np.isfinite(embeddings).all())

    def pipeline(self, text, title='Exercise study'):
        engine = Mock()
        engine.search.return_value = pd.DataFrame([{
            'doc_id':'scientific-example', 'title':title, 'text':text,
            'category':'SciFact', 'semantic_score':0.8, 'hybrid_score':1.0,
        }])
        return ScientificEvidencePipeline(engine, ExtractiveFallbackGenerator())

    def test_high_similarity_cannot_override_a_missing_outcome(self):
        pipeline = self.pipeline('Exercise physiology in children and adults.')
        result = pipeline.answer('Exercise improves memory')
        self.assertEqual(result.outcome, 'insufficient evidence')
        self.assertTrue(result.sources.empty)
        self.assertEqual(sum(e['step'] == 'retrieve' for e in result.trace), 2)

    def test_selected_abstract_preserves_study_limitations(self):
        text = 'Exercise and memory were investigated in mice. No human participants were studied.'
        response = self.pipeline(text).answer('Exercise improves memory')
        self.assertEqual(response.outcome, 'cited excerpt')
        self.assertEqual(response.answer, text + ' [1]')

    def test_population_mismatch_is_rejected(self):
        result = self.pipeline('Exercise and memory were studied in mice.').answer('Exercise improves memory in children')
        self.assertEqual(result.outcome, 'insufficient evidence')

    def test_spelling_suggestion_does_not_change_the_query(self):
        engine = Mock(tokenized_corpus=[['exercise', 'improves', 'memory']])
        query = 'Exercise improves memor'
        self.assertEqual(spelling_suggestion(query, engine), 'Exercise improves memory')
        self.assertEqual(query, 'Exercise improves memor')
        self.assertIsNone(spelling_suggestion('Exercise improves memory', engine))
