from __future__ import annotations
import re
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
    def __init__(self, corpus_path: str):
        self.df = pd.read_csv(corpus_path)
        self.df["combined_text"] = self.df["title"].fillna("") + " " + self.df["text"].fillna("")
        self.tokenized_corpus = self.df["combined_text"].apply(preprocess).tolist()
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.model = SentenceTransformer(MODEL_NAME)
        self.doc_embeddings = self.model.encode(
            self.df["combined_text"].tolist(),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

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