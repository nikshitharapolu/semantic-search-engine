from __future__ import annotations
import os
import sys
import streamlit as st

CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from object_storage import ObjectStorageError
from service import get_engine, get_rag_pipeline, ingest_document

st.set_page_config(page_title="Hybrid Semantic Search Engine", page_icon="🔎", layout="wide")

@st.cache_resource
def load_engine():
    return get_engine()

engine = load_engine()

@st.cache_resource
def load_rag_pipeline():
    return get_rag_pipeline()

rag_pipeline = load_rag_pipeline()

st.title("🔎 Semantic Search Engine with Hybrid Ranking")
st.markdown(
    "Search documents using BM25, semantic embeddings, and a weighted hybrid score."
)

with st.sidebar:
    st.header("Search controls")
    experience = st.radio("Experience", ["Ask with RAG", "Search documents"])
    alpha = st.slider("BM25 weight (alpha)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    top_k = st.slider("Top K results", min_value=3, max_value=10, value=5, step=1)
    st.caption("Hybrid score = alpha × BM25 + (1-alpha) × semantic score")

    st.divider()
    st.header("Add documents")
    uploaded_file = st.file_uploader("Upload PDF, TXT, or Markdown", type=["pdf", "txt", "md"])
    if uploaded_file is not None and st.button("Index document", width="stretch"):
        try:
            ingestion, stored = ingest_document(uploaded_file.name, uploaded_file.getvalue())
            if ingestion.duplicate:
                st.info(f"{ingestion.filename} is already indexed.")
            else:
                st.session_state["ingestion_message"] = (
                    f"Stored and indexed {ingestion.filename} as {ingestion.chunks_added} searchable chunks."
                )
                st.cache_resource.clear()
                st.rerun()
        except (ValueError, ObjectStorageError) as error:
            st.error(str(error))

    if message := st.session_state.pop("ingestion_message", None):
        st.success(message)
    uploaded_count = int((engine.df["category"] == "uploaded").sum())
    st.caption(f"{len(engine.df)} total chunks indexed · {uploaded_count} uploaded chunks")

query = st.text_input(
    "Ask a question" if experience == "Ask with RAG" else "Enter your search query",
    value="How does hybrid retrieval combine lexical and semantic search?",
)

if query:
    if experience == "Ask with RAG":
        response = rag_pipeline.answer(question=query, alpha=alpha, top_k=top_k)
        results = response.sources
        st.subheader("Grounded answer")
        st.info(response.answer)
        st.caption(
            "Answers are generated locally with Ollama when available. "
            "If Ollama is offline, the app returns the strongest retrieved evidence."
        )
    else:
        results = engine.search(query=query, alpha=alpha, top_k=top_k)

    st.subheader("Sources" if experience == "Ask with RAG" else "Top results")
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
        width="stretch",
        hide_index=True,
    )

st.divider()
st.subheader("Corpus preview")
st.dataframe(engine.df[["doc_id", "title", "category"]], width="stretch", hide_index=True)
