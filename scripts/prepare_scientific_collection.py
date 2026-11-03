"""Package the public SciFact collection without queries, labels or private uploads.

Run after the retrieval benchmark has prepared its CSV and embedding cache.
The benchmark inputs and evaluation metrics are not modified.
"""
from pathlib import Path
import gzip
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def prepare():
    import csv
    import numpy as np
    source = ROOT / 'evaluation/datasets/scifact-engine.csv'
    cache = ROOT / 'evaluation/datasets/scifact-index'
    destination = ROOT / 'data/scientific'
    rows = list(csv.DictReader(source.open(encoding='utf-8', newline='')))
    manifest = json.loads((cache / 'manifest.json').read_text())
    digest = hashlib.sha256(manifest['model'].encode())
    for row in rows:
        digest.update(row['doc_id'].encode())
        digest.update(b'\0')
        digest.update((row['title'] + ' ' + row['text']).encode())
        digest.update(b'\0')
    embeddings = np.load(cache / 'embeddings.npy', allow_pickle=False)
    if len(rows) != 5183 or digest.hexdigest() != manifest['fingerprint']:
        raise ValueError('The benchmark corpus and index must match the 5,183-paper collection.')
    if embeddings.shape != (5183, 384) or not np.isfinite(embeddings).all():
        raise ValueError('Invalid embedding cache.')
    destination.mkdir(parents=True, exist_ok=True)
    # Fixed gzip timestamp makes repeated packaging reproducible.
    with (destination / 'corpus.csv.gz').open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
            zipped.write(source.read_bytes())
    shutil.copyfile(cache / 'embeddings.npy', destination / 'embeddings.npy')
    manifest['artifacts'] = {name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
                             for name in ('corpus.csv.gz', 'embeddings.npy')}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Packaged {len(rows):,} abstracts and matching embeddings; no queries or labels included.')


if __name__ == '__main__':
    prepare()
