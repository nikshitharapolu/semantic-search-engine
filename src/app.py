from __future__ import annotations
import os
import sys
import streamlit as st
import pandas as pd

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from utils import HybridSearchEngine

st.set_page_config(page_title="Hybrid Semantic Search Engine", page_icon="🔎", layout="wide")

@st.cache_resource
def load_engine():
    corpus_path = os.path.join(PROJECT_ROOT, "data", "corpus.csv")
    return HybridSearchEngine(corpus_path)

engine = load_engine()

st.title("🔎 Semantic Search Engine with Hybrid Ranking")
st.markdown(
    "Search documents using BM25, semantic embeddings, and a weighted hybrid score."
)

with st.sidebar:
    st.header("Search controls")
    alpha = st.slider("BM25 weight (alpha)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    top_k = st.slider("Top K results", min_value=3, max_value=10, value=5, step=1)
    st.caption("Hybrid score = alpha × BM25 + (1-alpha) × semantic score")

query = st.text_input("Enter your search query", value="hybrid retrieval with bm25 and embeddings")

if query:
    results = engine.search(query=query, alpha=alpha, top_k=top_k)

    st.subheader("Top results")
    for i, row in results.iterrows():
        with st.container(border=True):
            st.markdown(f"### {i+1}. {row['title']}")
            st.write(row["text"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Doc ID", row["doc_id"])
            c2.metric("BM25", f"{row['bm25_score']:.4f}")
            c3.metric("Semantic", f"{row['semantic_score']:.4f}")
            c4.metric("Hybrid", f"{row['hybrid_score']:.4f}")
            st.caption(f"Category: {row['category']}")

    st.subheader("Results table")
    st.dataframe(
        results[["doc_id", "title", "category", "bm25_score", "semantic_score", "hybrid_score"]],
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.subheader("Corpus preview")
st.dataframe(engine.df[["doc_id", "title", "category"]], use_container_width=True, hide_index=True)