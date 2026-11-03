from pathlib import Path
import sys
from unittest import TestCase
from unittest.mock import Mock, patch
import pandas as pd
from fastapi.testclient import TestClient
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from api import app


class ResearchAPITests(TestCase):
    def test_research_routes_use_public_dispatch_and_validate_inputs(self):
        engine = Mock()
        engine.search.return_value = pd.DataFrame([{
            'doc_id':'scientific-1', 'category':'SciFact', 'title':'Memory study',
            'text':'Exercise and memory were studied in mice.', 'semantic_score':0.8,
        }])
        with patch('api.scientific_engine', return_value=engine):
            client = TestClient(app)
            result = client.post('/research/search', json={'query':'Exercise improves memory'})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()['collection']['document_count'], 5183)
            self.assertEqual(result.json()['results'][0]['doc_id'], 'scientific-1')
            guided = client.post('/research/ask/guided', json={'query':'Exercise improves memory'})
            self.assertEqual(guided.status_code, 200)
            self.assertEqual(guided.json()['outcome'], 'cited excerpt')
            self.assertEqual(client.post('/research/search', json={'query':'x', 'top_k':21}).status_code, 422)

    def test_research_failure_has_a_safe_http_error(self):
        with patch('api.scientific_engine', side_effect=RuntimeError('private server detail')):
            with self.assertLogs('public_search', level='ERROR'):
                response = TestClient(app).post('/research/search', json={'query':'memory'})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('private server detail', response.text)
