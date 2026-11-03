"""Bounded, keyless OpenAlex discovery with local hybrid candidate ranking.

Catalog search chooses candidates; our BM25/MiniLM ranking only covers those
candidates. No API key, paid plan, full-text scraping or query persistence.
"""
from collections import Counter, OrderedDict
from dataclasses import dataclass
import json
import math
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from text_utils import preprocess

SUBJECTS = {
    'all': None,
    'computer-science': 'primary_topic.field.id:17',
    'engineering': 'primary_topic.field.id:22',
    'life-sciences': 'primary_topic.domain.id:1',
    'health-sciences': 'primary_topic.domain.id:4',
    'social-sciences': 'primary_topic.domain.id:2',
    'physical-sciences': 'primary_topic.domain.id:3',
}
COLLECTION = {
    'name': 'OpenAlex',
    'description': 'Cross-subject scholarly catalog; hybrid ranking applies to up to 40 fetched candidates, not the entire catalog.',
    'attribution_url': 'https://openalex.org',
}
CANDIDATE_LIMIT = 40
CACHE_SECONDS = 1800
CACHE_ENTRIES = 32
MAX_RESPONSE_BYTES = 8_000_000


class CatalogUnavailableError(RuntimeError):
    """A safe user-facing catalog error, without provider internals."""


def safe_url(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = urlparse(value)
        return value if parsed.scheme == 'https' and parsed.hostname and not parsed.username and not parsed.password else None
    except ValueError:
        return None


def abstract_text(index):
    if not isinstance(index, dict):
        return ''
    words = {}
    for token, positions in index.items():
        if not isinstance(token, str) or not isinstance(positions, list):
            continue
        for position in positions:
            if type(position) is int and 0 <= position < 10000:
                words[position] = token
    return ' '.join(words[position] for position in sorted(words))


def normalize_work(work):
    if not isinstance(work, dict):
        return None
    identifier = safe_url(work.get('id'))
    title = work.get('display_name')
    if not identifier or urlparse(identifier).hostname != 'openalex.org' or not isinstance(title, str) or not title.strip():
        return None
    location = work.get('primary_location') or {}
    topic = work.get('primary_topic') or {}
    field = topic.get('field') or {}
    authors = []
    for item in work.get('authorships') or []:
        name = (item.get('author') or {}).get('display_name') if isinstance(item, dict) else None
        if isinstance(name, str) and name.strip():
            authors.append(name)
    year = work.get('publication_year')
    return {
        'doc_id': identifier.rsplit('/', 1)[-1], 'title': title.strip(),
        'text': abstract_text(work.get('abstract_inverted_index')),
        'category': field.get('display_name') or 'Research', 'source': 'OpenAlex',
        'authors': authors, 'publication_year': year if type(year) is int and 1500 <= year <= 2200 else None,
        'venue': (location.get('source') or {}).get('display_name') or None,
        'doi': safe_url(work.get('doi')), 'catalog_url': identifier,
        'source_url': safe_url(work.get('doi')) or safe_url(location.get('landing_page_url')) or identifier,
    }


def fetch_works(query, subject):
    params = {'search':query, 'per_page':CANDIDATE_LIMIT,
              'select':'id,display_name,doi,publication_year,authorships,primary_location,abstract_inverted_index,primary_topic',
              'filter':'is_retracted:false' + (',' + SUBJECTS[subject] if SUBJECTS[subject] else '')}
    # Deliberately keyless: this application cannot spend prepaid/account credit.
    request = Request('https://api.openalex.org/works?' + urlencode(params),
                      headers={'User-Agent':'HybridSearchResearch/1.0', 'Accept':'application/json'})
    try:
        with urlopen(request, timeout=12) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise CatalogUnavailableError('The paper catalog returned too much data. Try a more specific question.')
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get('results'), list):
            raise ValueError('Invalid catalog response')
        rows, seen = [], set()
        for work in data['results'][:CANDIDATE_LIMIT]:
            row = normalize_work(work)
            if row and row['doc_id'] not in seen:
                rows.append(row)
                seen.add(row['doc_id'])
        return rows
    except HTTPError as error:
        if error.code in (401, 403, 429):
            raise CatalogUnavailableError('OpenAlex access or its free request limit is temporarily unavailable. Please try again later.') from None
        raise CatalogUnavailableError('The paper catalog is temporarily unavailable. Please try again later.') from None
    except (URLError, TimeoutError, OSError, ValueError, TypeError, AttributeError):
        raise CatalogUnavailableError('The paper catalog could not be reached or returned an invalid response. Please try again later.') from None


@dataclass
class Candidates:
    created: float
    frame: pd.DataFrame
    tokens: list
    embeddings: np.ndarray


class OpenAlexSearchEngine:
    collection = COLLECTION

    def __init__(self, model=None, fetcher=fetch_works):
        self.model = model
        self.fetcher = fetcher
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._last_fetch = 0.0

    @staticmethod
    def minmax(scores):
        scores = np.asarray(scores, dtype=float)
        return np.ones_like(scores) if np.allclose(scores.max(), scores.min()) else (scores - scores.min()) / (scores.max() - scores.min())

    def for_subject(self, subject):
        return SubjectSearch(self, subject)

    def search(self, query, alpha=0.5, top_k=5, subject='all'):
        if subject not in SUBJECTS:
            raise ValueError('Choose a supported subject.')
        # Serialize the model and upstream access within this app instance.
        with self._lock:
            key = (query.strip().casefold(), subject)
            cached = self._cache.get(key)
            if cached is None or time.monotonic() - cached.created > CACHE_SECONDS:
                pause = 1.0 - (time.monotonic() - self._last_fetch)
                if pause > 0:
                    time.sleep(pause)
                self._last_fetch = time.monotonic()
                rows = self.fetcher(query, subject)
                frame = pd.DataFrame(rows)
                tokens = [preprocess(row['title'] + ' ' + row['text']) or ['__untokenized__'] for row in rows]
                if rows:
                    if self.model is None:
                        from sentence_transformers import SentenceTransformer
                        from utils import MODEL_NAME
                        self.model = SentenceTransformer(MODEL_NAME)
                    embeddings = self.model.encode([row['title'] + ' ' + row['text'] for row in rows],
                                                   convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                else:
                    frame = pd.DataFrame(columns=['doc_id','title','text','category','source'])
                    embeddings = np.empty((0, 0))
                cached = Candidates(time.monotonic(), frame, tokens, embeddings)
                self._cache[key] = cached
                self._cache.move_to_end(key)
                while len(self._cache) > CACHE_ENTRIES:
                    self._cache.popitem(last=False)
            else:
                self._cache.move_to_end(key)
            if cached.frame.empty:
                return cached.frame.copy()
            # Positive IDF avoids inverted keyword ranking in small pools
            # where an important term occurs in most fetched papers.
            bm25 = BM25Okapi(cached.tokens)
            frequencies = Counter(term for tokens in cached.tokens for term in set(tokens))
            n = len(cached.tokens)
            bm25.idf = {term:math.log(1 + (n-count+0.5)/(count+0.5)) for term, count in frequencies.items()}
            lexical = np.asarray(bm25.get_scores(preprocess(query)))
            vector = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
            semantic = cached.embeddings @ vector
            frame = cached.frame.copy()
            frame['bm25_score'] = lexical
            frame['semantic_score'] = semantic
            frame['hybrid_score'] = alpha * self.minmax(lexical) + (1-alpha) * self.minmax(semantic)
            return frame.sort_values('hybrid_score', ascending=False, kind='stable').head(top_k).reset_index(drop=True)


class SubjectSearch:
    """Per-request subject binding; a guided retry cannot change other sessions."""
    collection = COLLECTION

    def __init__(self, engine, subject):
        self.engine, self.subject = engine, subject
        self.tokenized_corpus = []

    def search(self, query, alpha=0.5, top_k=5):
        frame = self.engine.search(query, alpha=alpha, top_k=top_k, subject=self.subject)
        self.tokenized_corpus = [preprocess(str(row['title']) + ' ' + str(row['text'])) for row in frame.to_dict('records')]
        return frame
