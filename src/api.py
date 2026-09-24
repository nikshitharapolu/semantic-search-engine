from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ingestion import MAX_FILE_BYTES
from object_storage import ObjectStorageError
from service import get_engine, get_rag_pipeline, ingest_document


app = FastAPI(
    title="Hybrid Search and RAG API",
    description="Cost-free local API for hybrid retrieval and citation-grounded RAG.",
    version="0.3.0",
)


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=20)


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


@app.post("/documents", status_code=201)
async def add_document(file: UploadFile = File(...)) -> dict[str, Any]:
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
