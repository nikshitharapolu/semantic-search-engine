from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from generation import ExtractiveFallbackGenerator, OllamaGenerator, ResilientLocalGenerator
from ingestion import UploadedCorpusStore
from object_storage import LocalObjectStore, ObjectStore, S3ObjectStore
from rag import LocalRAGPipeline
from utils import HybridSearchEngine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DATA_DIR = Path(
    os.getenv("RAG_BASE_DATA_DIR", os.getenv("RAG_DATA_DIR", str(PROJECT_ROOT / "data")))
)
STATE_DATA_DIR = Path(
    os.getenv("RAG_STATE_DATA_DIR", os.getenv("RAG_DATA_DIR", str(PROJECT_ROOT / "data")))
)


@lru_cache(maxsize=1)
def get_engine() -> HybridSearchEngine:
    return HybridSearchEngine(
        str(BASE_DATA_DIR / "corpus.csv"),
        str(STATE_DATA_DIR / "uploaded_corpus.csv"),
        str(STATE_DATA_DIR / "index"),
    )


@lru_cache(maxsize=1)
def get_rag_pipeline() -> LocalRAGPipeline:
    generator = ResilientLocalGenerator(
        primary=OllamaGenerator(),
        fallback=ExtractiveFallbackGenerator(),
    )
    return LocalRAGPipeline(get_engine(), generator)


def get_uploaded_corpus_store() -> UploadedCorpusStore:
    return UploadedCorpusStore(STATE_DATA_DIR / "uploaded_corpus.csv")


@lru_cache(maxsize=1)
def get_object_store() -> ObjectStore:
    backend = os.getenv("OBJECT_STORAGE_BACKEND", "local").lower()
    if backend == "s3":
        return S3ObjectStore(
            bucket=os.getenv("S3_BUCKET", "rag-documents"),
            endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
            region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )
    if backend == "local":
        return LocalObjectStore(STATE_DATA_DIR / "objects")
    raise ValueError(f"Unsupported object storage backend: {backend}")


def ingest_document(filename: str, content: bytes):
    stored = get_object_store().put(filename, content)
    ingestion = get_uploaded_corpus_store().add(filename, content)
    if not ingestion.duplicate:
        refresh_indexes()
    return ingestion, stored


def refresh_indexes() -> None:
    """Force the next request to load uploaded chunks and rebuild stale embeddings."""

    get_rag_pipeline.cache_clear()
    get_engine.cache_clear()
