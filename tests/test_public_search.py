import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from public_search import handle_request, records

class PublicSearchTests(TestCase):
    def setUp(self):
        self.engine = Mock()
        self.engine.search.return_value = pd.DataFrame([{'doc_id':'D1','title':'Hybrid retrieval','text':'Hybrid retrieval combines lexical and semantic evidence.','semantic_score':0.9,'bm25_score':2.0,'hybrid_score':1.0}])
        self.factory = Mock(return_value=self.engine)
        self.request = {'id':'request-1','path':'/search','payload':{'query':'hybrid retrieval','alpha':0.5,'top_k':3}}

    def test_search_keeps_the_frontend_response_contract(self):
        result = handle_request(self.request, self.factory)
        self.assertEqual(result['id'], 'request-1')
        self.assertEqual(result['body']['results'][0]['doc_id'], 'D1')
        self.engine.search.assert_called_once_with('hybrid retrieval',alpha=0.5,top_k=3)

    def test_invalid_inputs_never_initialize_the_model(self):
        for key,value in [('alpha',True),('alpha',float('nan')),('top_k',True),('top_k',21),('query',' '),('query','x'*2001)]:
            with self.subTest(key=key,value=str(value)[:20]):
                request = {**self.request, 'payload':{**self.request['payload'], key:value}}
                self.assertIn('error', handle_request(request,self.factory))
        self.factory.assert_not_called()

    def test_unsupported_endpoint_is_rejected(self):
        self.assertIn('error',handle_request({**self.request,'path':'/documents'},self.factory))
        self.factory.assert_not_called()

    def test_guided_uses_an_extractive_answer(self):
        result = handle_request({**self.request,'path':'/ask/guided'},self.factory)
        self.assertEqual(result['body']['outcome'],'cited excerpt')
        self.assertIn('trace',result['body'])
        self.assertTrue(result['body']['answer'].startswith('The strongest retrieved evidence says:'))

    def test_guided_abstains_for_missing_attribution(self):
        request = {**self.request,'path':'/ask/guided','payload':{**self.request['payload'],'query':'Who wrote the paper?'}}
        result = handle_request(request,self.factory)
        self.assertEqual(result['body']['outcome'],'insufficient evidence')
        self.assertEqual(result['body']['sources'],[])

    def test_exception_internals_are_not_returned(self):
        self.factory.side_effect = RuntimeError('private internal detail')
        with self.assertLogs('public_search',level='ERROR'):
            result = handle_request(self.request,self.factory)
        self.assertNotIn('private internal detail',result['error'])

    def test_nonfinite_scores_become_json_null(self):
        self.assertIsNone(records(pd.DataFrame([{'doc_id':'D1','text':'x','hybrid_score':float('nan')}]))[0]['hybrid_score'])
