import io
import json
from pathlib import Path
import sys
from unittest import TestCase
from unittest.mock import Mock, patch
from urllib.error import HTTPError
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from openalex_search import (OpenAlexSearchEngine, CatalogUnavailableError,
                             abstract_text, normalize_work, fetch_works)
from public_search import handle_request


def work(identifier='W1', title='Attention in transformers', abstract=True):
    return {'id':'https://openalex.org/' + identifier, 'display_name':title,
            'doi':'https://doi.org/10.1234/example', 'publication_year':2024,
            'authorships':[{'author':{'display_name':'Example Author'}}],
            'primary_location':{'source':{'display_name':'Example Journal'}},
            'primary_topic':{'field':{'display_name':'Computer Science'}},
            'abstract_inverted_index':{'Transformers':[0], 'use':[1], 'attention.':[2]} if abstract else None}


class FakeModel:
    def encode(self, texts, **kwargs):
        return np.array([[1., 0.] if 'attention' in text.lower() else [0., 1.] for text in texts])


class OpenAlexTests(TestCase):
    def test_reconstructs_abstract_positions_and_keeps_only_supplied_metadata(self):
        self.assertEqual(abstract_text({'second':[1], 'first':[0], 'bad':[-1,10000,True]}), 'first second')
        record = normalize_work(work())
        self.assertEqual(record['text'], 'Transformers use attention.')
        self.assertEqual(record['authors'], ['Example Author'])
        self.assertEqual(record['source_url'], 'https://doi.org/10.1234/example')
        sparse = normalize_work({'id':'https://openalex.org/W2', 'display_name':'Sparse title'})
        self.assertEqual(sparse['text'], '')
        self.assertEqual(sparse['authors'], [])
        self.assertIsNone(sparse['publication_year'])

    def test_bad_urls_cannot_become_source_links(self):
        item = work()
        item['doi'] = 'javascript:alert(1)'
        item['primary_location']['landing_page_url'] = 'https://user:secret@example.com'
        self.assertEqual(normalize_work(item)['source_url'], item['id'])
        self.assertIsNone(normalize_work({'id':'https://example.com/W1', 'display_name':'Bad ID'}))

    def test_hybrid_rank_cache_and_subject_isolation(self):
        rows = [normalize_work(work()), normalize_work(work('W2','Cooking potatoes'))]
        fetcher = Mock(return_value=rows)
        engine = OpenAlexSearchEngine(FakeModel(), fetcher)
        engine._last_fetch = -1000
        self.assertEqual(engine.search('attention', alpha=0, subject='computer-science').iloc[0]['doc_id'], 'W1')
        engine.search('attention', alpha=1, top_k=1, subject='computer-science')
        self.assertEqual(fetcher.call_count, 1)
        with patch('openalex_search.time.sleep'):
            engine.for_subject('engineering').search('attention')
        self.assertEqual(fetcher.call_args.args, ('attention','engineering'))
        self.assertEqual(fetcher.call_count, 2)

    def test_fetch_is_bounded_keyless_and_excludes_retracted_records(self):
        body = json.dumps({'results':[work(),work()]}).encode()
        with patch('openalex_search.urlopen') as opening:
            opening.return_value.__enter__.return_value = io.BytesIO(body)
            rows = fetch_works('attention & memory','computer-science')
        request = opening.call_args.args[0]
        from urllib.parse import urlparse, parse_qs
        params = parse_qs(urlparse(request.full_url).query)
        self.assertEqual(params['per_page'], ['40'])
        self.assertEqual(params['filter'], ['is_retracted:false,primary_topic.field.id:17'])
        self.assertEqual(params['search'], ['attention & memory'])
        self.assertNotIn('api_key', params)
        self.assertFalse(request.has_header('Authorization'))
        self.assertEqual(len(rows), 1)

    def test_provider_limit_is_reported_without_silent_corpus_fallback(self):
        error = HTTPError('https://api.openalex.org/works',429,'limit',{},None)
        with patch('openalex_search.urlopen', side_effect=error):
            with self.assertRaisesRegex(CatalogUnavailableError, 'free request limit'):
                fetch_works('attention','all')
        engine = OpenAlexSearchEngine(FakeModel(), Mock(side_effect=CatalogUnavailableError('Catalog unavailable.')))
        response = handle_request({'id':'x','path':'/search','payload':{'query':'attention'}},lambda:engine)
        self.assertEqual(response['error'], 'Catalog unavailable.')

    def test_guided_does_not_use_title_only_record_as_an_abstract(self):
        engine = OpenAlexSearchEngine(FakeModel(), Mock(return_value=[normalize_work(work(abstract=False))]))
        with patch('openalex_search.time.sleep'):
            result = handle_request({'id':'x','path':'/ask/guided', 'payload':{'query':'Attention in transformers','subject':'computer-science'}},lambda:engine)['body']
        self.assertEqual(result['outcome'], 'insufficient evidence')
        self.assertEqual(result['sources'], [])
        self.assertEqual(result['related_results'][0]['authors'], ['Example Author'])
        self.assertEqual(result['collection']['name'], 'OpenAlex')

    def test_empty_catalog_result_stays_empty_without_loading_a_model(self):
        engine = OpenAlexSearchEngine(fetcher=lambda *args:[])
        response = handle_request({'id':'x','path':'/search','payload':{'query':'unknown'}},lambda:engine)
        self.assertEqual(response['body']['results'], [])
        self.assertIsNone(engine.model)

    def test_invalid_subject_never_initializes_engine(self):
        factory = Mock()
        response = handle_request({'id':'x','path':'/search','payload':{'query':'attention','subject':'bogus'}},factory)
        self.assertIn('error',response)
        factory.assert_not_called()
