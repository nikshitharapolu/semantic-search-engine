"""Read-only search dispatch for the public Streamlit component.

No AWS, LLM service, upload directory, or external tracing is required.
"""
from typing import Callable, Any
import math
from openalex_search import CatalogUnavailableError


def create_engine():
    from openalex_search import OpenAlexSearchEngine
    return OpenAlexSearchEngine()


def records(frame):
    import numpy as np
    result = []
    for row in frame.to_dict(orient='records'):
        cleaned = {}
        for key in ('doc_id', 'title', 'category', 'source', 'text', 'bm25_score', 'semantic_score', 'hybrid_score', 'authors', 'publication_year', 'venue', 'doi', 'source_url', 'catalog_url'):
            value = row.get(key)
            if isinstance(value, np.generic):
                value = value.item()
            cleaned[key] = None if isinstance(value, float) and not math.isfinite(value) else value
        result.append(cleaned)
    return result


def handle_request(request: dict, engine_factory: Callable[[], Any]) -> dict:
    request_id = request.get('id')
    try:
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 100:
            raise ValueError('Invalid request identifier.')
        if request.get('path') not in ('/search', '/ask/guided'):
            raise ValueError('This demo supports search and guided answers only.')
        payload = request.get('payload')
        if not isinstance(payload, dict):
            raise ValueError('Invalid search request.')
        from openalex_search import SUBJECTS, OpenAlexSearchEngine
        subject = payload.get('subject', 'all')
        if not isinstance(subject, str) or subject not in SUBJECTS:
            raise ValueError('Choose a supported subject.')
        question = payload.get('query')
        alpha = payload.get('alpha', 0.5)
        top_k = payload.get('top_k', 5)
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
            raise ValueError('Enter a question between 1 and 2,000 characters.')
        if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('Keyword weight must be between 0 and 1.')
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError('Request between 1 and 20 documents.')
        from langsmith import tracing_context
        with tracing_context(enabled=False):
            engine = engine_factory()
            if isinstance(engine, OpenAlexSearchEngine):
                engine = engine.for_subject(subject)
            if request['path'] == '/search':
                frame = engine.search(question.strip(), alpha=alpha, top_k=top_k)
                body = {'query': question.strip(), 'results': records(frame)}
            else:
                from generation import ExtractiveFallbackGenerator
                from scientific_search import ScientificEvidencePipeline
                response = ScientificEvidencePipeline(engine, ExtractiveFallbackGenerator()).answer(question.strip(), alpha=alpha, top_k=top_k)
                body = {'query': question.strip(), 'answer': response.answer,
                        'sources': records(response.sources), 'outcome': response.outcome, 'trace': response.trace}
                if response.outcome == 'insufficient evidence':
                    body['answer'] = 'The retrieved abstracts did not provide enough evidence to answer your question. This does not mean the claim is false.'
                    body['related_results'] = records(engine.search(question.strip(), alpha=alpha, top_k=top_k))
        from scientific_search import COLLECTION, spelling_suggestion
        collection = getattr(engine, 'collection', None)
        body['collection'] = collection if isinstance(collection, dict) else COLLECTION
        body['query_suggestion'] = spelling_suggestion(question.strip(), engine)
        return {'id': request_id, 'body': body}
    except CatalogUnavailableError as error:
        return {'id': request_id, 'error': str(error)}
    except ValueError as error:
        return {'id': request_id, 'error': str(error)}
    except Exception:
        # Do not expose credentials, server paths or exception internals publicly.
        import logging
        logging.getLogger(__name__).exception('Public search failed')
        return {'id': request_id, 'error': 'Search could not complete. Please retry while the app warms up.'}
