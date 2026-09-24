from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def preprocess(text: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


class HybridSearchEngine:
    def __init__(self, corpus_path: str, extra_corpus_path: str | None = None, index_dir: str | None = None):
        frames = [pd.read_csv(corpus_path)]
        if extra_corpus_path:
            extra_path = Path(extra_corpus_path)
            if extra_path.exists() and extra_path.stat().st_size > 0:
                frames.append(pd.read_csv(extra_path))

        self.df = pd.concat(frames, ignore_index=True, sort=False)
        self.df = self.df.drop_duplicates(subset=["doc_id"], keep="last").reset_index(drop=True)
        for column in ("title", "text", "category"):
            if column not in self.df.columns:
                self.df[column] = ""
        self.df["combined_text"] = self.df["title"].fillna("") + " " + self.df["text"].fillna("")
        self.tokenized_corpus = self.df["combined_text"].apply(preprocess).tolist()
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.model = SentenceTransformer(MODEL_NAME)
        self.doc_embeddings = self._load_or_create_embeddings(index_dir)

    def _corpus_fingerprint(self) -> str:
        digest = hashlib.sha256(MODEL_NAME.encode("utf-8"))
        for row in self.df[["doc_id", "combined_text"]].itertuples(index=False):
            digest.update(str(row.doc_id).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(row.combined_text).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _encode_corpus(self) -> np.ndarray:
        return self.model.encode(
            self.df["combined_text"].tolist(),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def _load_or_create_embeddings(self, index_dir: str | None) -> np.ndarray:
        if not index_dir:
            return self._encode_corpus()

        directory = Path(index_dir)
        embeddings_path = directory / "embeddings.npy"
        manifest_path = directory / "manifest.json"
        fingerprint = self._corpus_fingerprint()

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            embeddings = np.load(embeddings_path, allow_pickle=False)
            if manifest.get("fingerprint") == fingerprint and embeddings.shape[0] == len(self.df):
                return embeddings
        except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError):
            pass

        embeddings = self._encode_corpus()
        directory.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, embeddings, allow_pickle=False)
        manifest_path.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "model": MODEL_NAME,
                    "document_count": len(self.df),
                    "dimensions": int(embeddings.shape[1]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return embeddings

    @staticmethod
    def minmax(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if np.allclose(scores.max(), scores.min()):
            return np.ones_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())

    def search(self, query: str, alpha: float = 0.5, top_k: int = 5):
        query_tokens = preprocess(query)
        bm25_scores = np.array(self.bm25.get_scores(query_tokens), dtype=float)

        query_embedding = self.model.encode(
            [query], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
        )
        semantic_scores = cosine_similarity(query_embedding, self.doc_embeddings)[0]

        bm25_norm = self.minmax(bm25_scores)
        semantic_norm = self.minmax(semantic_scores)
        hybrid_scores = alpha * bm25_norm + (1 - alpha) * semantic_norm

        results = self.df.copy()
        results["bm25_score"] = bm25_scores
        results["semantic_score"] = semantic_scores
        results["hybrid_score"] = hybrid_scores
        results = results.sort_values("hybrid_score", ascending=False).head(top_k).reset_index(drop=True)
        return results
