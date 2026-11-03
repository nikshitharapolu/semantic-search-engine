# Bundled SciFact regression collection

5,183 titles and abstracts from SciFact, packaged from the BEIR SciFact corpus.
The public app now uses OpenAlex; this package remains for offline regression
checks and reproducible diagnostics, without adding another interface mode.
This is a fixed research collection, primarily biomedical, not a current or
comprehensive literature index. Some records may be other publication types.

## Attribution and source

SciFact: David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen,
Arman Cohan, and Hannaneh Hajishirzi, *Fact or Fiction: Verifying Scientific Claims*
(2020): https://arxiv.org/abs/2004.14974

- Original dataset and documentation: https://github.com/allenai/scifact
- Original license notice: https://github.com/allenai/scifact/blob/master/LICENSE.md
- Corpus source: Semantic Scholar S2ORC, https://github.com/allenai/s2orc
- Abstract corpus license: ODC Attribution License 1.0,
  https://opendatacommons.org/licenses/by/1-0/
- BEIR distribution used:
  https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip

The upstream SciFact notice identifies the abstract corpus as ODC-By 1.0.
This package contains no claim/evidence annotations. Document IDs are upstream
S2ORC identifiers, not DOIs. The BEIR export has empty metadata: authors,
publication dates and DOI links are not supplied here. The UI offers labeled
Google Scholar title searches, not verified direct paper links or complete
bibliographic citations.

## Reproduce

Download the benchmark as described in the repository README. Run the existing
benchmark to prepare `scifact-engine.csv` and `scifact-index`, then run:

```sh
python scripts/prepare_scientific_collection.py
```

Transformations: upstream IDs, titles and abstract texts are converted to CSV,
a `SciFact` category is added, and the CSV is gzip-compressed with a fixed
timestamp. MiniLM embeddings of title + abstract are packaged without changing
the corpus order. `manifest.json` records the corpus fingerprint and SHA-256
hashes of both artifacts. The packaged index avoids recomputing 5,183 document
embeddings on each hosting restart. The small embedding model must still load.

The sample corpus, private uploads and benchmark queries remain separate.
