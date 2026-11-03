from __future__ import annotations

import json
import logging
import os
import time
from functools import lru_cache
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from pydantic import BaseModel, Field

from ingestion import MAX_FILE_BYTES
from object_storage import ObjectStorageError
from service import get_engine, get_guided_rag_pipeline, get_rag_pipeline, ingest_document


app = FastAPI(
    title="Hybrid Search and RAG API",
    description="Hybrid retrieval and citation-grounded RAG API.",
    version="0.3.0",
)


request_log = logging.getLogger("hybrid_search.requests")
if not request_log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    request_log.addHandler(handler)
request_log.setLevel(logging.INFO)
request_log.propagate = False


@app.middleware("http")
async def log_request(request: Request, call_next):
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        request_log.info(json.dumps({
            "event": "request", "method": request.method,
            "route": getattr(route, "path", "unmatched"), "status": status,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        }))


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=20)


class ResearchRequest(RetrievalRequest):
    subject: Literal['all', 'computer-science', 'engineering', 'life-sciences', 'health-sciences', 'social-sciences', 'physical-sciences'] = 'all'


def records_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        column
        for column in (
            "doc_id",
            "title",
            "category",
            "source",
            "chunk_index",
            "text",
            "bm25_score",
            "semantic_score",
            "hybrid_score",
        )
        if column in frame.columns
    ]
    records = []
    for record in frame[columns].to_dict(orient="records"):
        cleaned = {}
        for key, value in record.items():
            if isinstance(value, np.generic):
                value = value.item()
            if pd.isna(value):
                value = None
            cleaned[key] = value
        records.append(cleaned)
    return records


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "service": "hybrid-search-rag"}


@app.post("/search")
def search(request: RetrievalRequest) -> dict[str, Any]:
    results = get_engine().search(request.query, alpha=request.alpha, top_k=request.top_k)
    return {"query": request.query, "results": records_from_frame(results)}


@app.post("/ask")
def ask(request: RetrievalRequest) -> dict[str, Any]:
    response = get_rag_pipeline().answer(request.query, alpha=request.alpha, top_k=request.top_k)
    return {
        "query": request.query,
        "answer": response.answer,
        "sources": records_from_frame(response.sources),
    }


@app.post("/ask/guided")
def ask_guided(request: RetrievalRequest) -> dict[str, Any]:
    response = get_guided_rag_pipeline().answer(request.query, alpha=request.alpha, top_k=request.top_k)
    return {
        "query": request.query,
        "answer": response.answer,
        "sources": records_from_frame(response.sources),
        "outcome": response.outcome,
        "trace": response.trace,
    }


@app.post("/research/search")
def research_search(request: ResearchRequest) -> dict[str, Any]:
    return research_request('/search', request)


@app.post("/research/ask/guided")
def research_guided(request: ResearchRequest) -> dict[str, Any]:
    return research_request('/ask/guided', request)


def research_request(path: str, request: ResearchRequest) -> dict[str, Any]:
    # Same cross-subject catalog as the Streamlit app; existing API routes retain
    # their local sample/upload corpus and Ollama behavior.
    from public_search import handle_request
    result = handle_request({'id': 'api-research', 'path': path,
                             'payload': request.model_dump()}, scientific_engine)
    if 'error' in result:
        raise HTTPException(status_code=503, detail=result['error'])
    return result['body']


@lru_cache(maxsize=1)
def scientific_engine():
    from public_search import create_engine
    return create_engine()


@app.post("/documents", status_code=201)
async def add_document(file: UploadFile = File(...)) -> dict[str, Any]:
    if os.getenv("RAG_READ_ONLY") == "1":
        raise HTTPException(status_code=403, detail="Document uploads are disabled for this demo replica.")
    content = await file.read(MAX_FILE_BYTES + 1)
    try:
        result, stored = ingest_document(file.filename or "uploaded-document", content)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ObjectStorageError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "filename": result.filename,
        "document_id": result.document_id,
        "chunks_added": result.chunks_added,
        "duplicate": result.duplicate,
        "object_uri": stored.uri,
        "checksum_sha256": stored.checksum_sha256,
    }
