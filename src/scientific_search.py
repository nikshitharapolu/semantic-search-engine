"""Public scientific search: a fixed, attributed corpus and conservative excerpts.

These relevance heuristics do not establish scientific support or contradiction.
The original guided pipeline and its sample-corpus evaluation remain separate.
"""
from pathlib import Path
import difflib
import hashlib
import json

from guided_rag import GuidedRAGPipeline, QUESTION_WORDS
from text_utils import preprocess

COLLECTION = {
    'name': 'SciFact', 'document_count': 5183,
    'description': 'A fixed collection of scientific abstracts, primarily biomedical research; not a live literature search.',
    'attribution_url': 'https://github.com/allenai/scifact',
}
# Remove grammatical and generic task words, while retaining populations,
# negation, interventions and outcomes. No claim labels or document IDs used.
GENERIC = QUESTION_WORDS | {
    'research', 'paper', 'papers', 'study', 'studies', 'evidence', 'find', 'show',
    'shows', 'tell', 'me', 'about', 'between', 'relate', 'related', 'relationship',
    'improve', 'improves', 'improved', 'affect', 'affects', 'effect', 'effects',
    'increase', 'increases', 'decrease', 'decreases', 'there', 'that', 'it', 'as',
    'by', 'from', 'have', 'has', 'was', 'were', 'be', 'been', 'an', 'a', 'the',
}


def concepts(text):
    return {term for term in preprocess(text) if term not in GENERIC and len(term) > 2}


def create_scientific_engine():
    from utils import HybridSearchEngine
    directory = Path(__file__).resolve().parents[1] / 'data/scientific'
    manifest = json.loads((directory / 'manifest.json').read_text())
    for name in ('corpus.csv.gz', 'embeddings.npy'):
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest['artifacts'][name]:
            raise ValueError('Scientific collection integrity check failed. Repackage the collection.')
    engine = HybridSearchEngine(str(directory / 'corpus.csv.gz'), index_dir=str(directory))
    if len(engine.df) != COLLECTION['document_count']:
        raise ValueError('Scientific collection document count mismatch.')
    return engine


def spelling_suggestion(query, engine):
    """Offer a suggestion; never silently rewrite the student's question."""
    if not hasattr(engine, 'tokenized_corpus') or not isinstance(engine.tokenized_corpus, list):
        return None
    vocabulary = {token for document in engine.tokenized_corpus for token in document}
    replacements = {}
    for token in preprocess(query)[:20]:
        if len(token) < 5 or token in vocabulary:
            continue
        matches = difflib.get_close_matches(token, vocabulary, n=2, cutoff=0.88)
        # Only suggest an unambiguous close match. Not a general spell checker.
        if len(matches) == 1:
            replacements[token] = matches[0]
    if not replacements:
        return None
    import re
    return re.sub(r'\b\w+\b', lambda m: replacements.get(m.group().lower(), m.group()), query)


class ScientificEvidencePipeline(GuidedRAGPipeline):
    def _assess(self, state):
        assessment = super()._assess(state)
        terms = concepts(state['question'])
        sources = state['sources']
        matched = set()
        if not sources.empty:
            top = sources.iloc[0]
            matched = terms & concepts(f"{top.get('title', '')} {top.get('text', '')}")
        # A short claim must cover all its concepts in the selected abstract.
        # Longer questions require at least 70% coverage. Still only relevance.
        covered = not sources.empty and bool(str(sources.iloc[0].get('text', '')).strip()) and bool(terms) and len(matched) / len(terms) >= (1.0 if len(terms) <= 3 else 0.7)
        if not covered:
            assessment['supported'] = False
            assessment['trace'][-1].update(result='weak evidence', reason='selected abstract is missing key query terms')
        return assessment

    def _generate(self, state):
        # Preserve the complete abstract, including population and study limits;
        # truncating its first characters can omit the actual findings.
        top = state['sources'].iloc[0]
        return {
            'answer': str(top.get('text', '')) + ' [1]',
            'outcome': 'cited excerpt',
            'trace': self._record(state, 'generate', result='related abstract', mode='offline excerpt'),
        }
