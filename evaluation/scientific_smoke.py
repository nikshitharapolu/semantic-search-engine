"""Development smoke checks, not held-out scientific claim verification.

Inspected examples include the user's exercise/memory question. Expected document
IDs are assertions in this diagnostic only; neither retrieval nor evidence checks
read this file. No prompts, weights or benchmark labels are optimized here.
"""
import json
import os
from pathlib import Path
import resource
import sys
import time

os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from public_search import handle_request
from scientific_search import create_scientific_engine as create_engine

CASES = [
    ('Exercise improves memory', 'cited excerpt', '57574395'),
    ('Does aerobic fitness relate to memory in children?', 'cited excerpt', '2424794'),
    ('Exercise improves memor', 'insufficient evidence', None),
    ('Who won the 2026 World Cup?', 'insufficient evidence', None),
    ('Who authored the exercise and memory paper?', 'insufficient evidence', None),
]


def main():
    started = time.perf_counter()
    engine = create_engine()
    load_seconds = time.perf_counter() - started
    results = []
    for query, outcome, expected_top in CASES:
        responses = {}
        for path in ('/search', '/ask/guided'):
            start = time.perf_counter()
            response = handle_request({'id':'smoke', 'path':path,
                                       'payload':{'query':query, 'top_k':5}}, lambda: engine)
            if 'error' in response:
                raise RuntimeError(response['error'])
            responses[path] = {**response['body'], 'seconds': round(time.perf_counter() - start, 4)}
        search, guided = responses['/search'], responses['/ask/guided']
        ids = [str(row['doc_id']) for row in search['results']]
        checks = {'guided_outcome': guided['outcome'] == outcome}
        if expected_top:
            checks['top_document'] = ids[0] == expected_top
        if outcome == 'insufficient evidence':
            checks['no_supporting_sources'] = not guided['sources']
            checks['closest_matches_retained'] = len(guided['related_results']) == 5
        if query.endswith('memor'):
            checks['visible_spelling_suggestion'] = guided['query_suggestion'] == 'Exercise improves memory'
        results.append({'query':query, 'checks':checks, 'retrieved_ids':ids,
                        'top_title':search['results'][0]['title'], 'guided_outcome':guided['outcome'],
                        'search_seconds': search['seconds'], 'guided_seconds': guided['seconds']})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    output = {'scope':'Development smoke checks; not independent evaluation or claim-verification accuracy.',
              'document_count':len(engine.df), 'engine_load_seconds':round(load_seconds, 3),
              'peak_process_rss_mib':round(rss / (1024**2 if sys.platform == 'darwin' else 1024), 1),
              'performance_note':'Local process with model already cached. Hosting startup, total Streamlit memory and performance may differ.',
              'cases':results}
    path = ROOT / 'docs/evidence/scientific-search-smoke.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    passed = all(all(r['checks'].values()) for r in results)
    print(json.dumps({'passed':passed, 'cases':len(results), 'engine_load_seconds':output['engine_load_seconds'],
                      'peak_process_rss_mib':output['peak_process_rss_mib']}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
