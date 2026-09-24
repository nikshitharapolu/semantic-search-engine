from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


MAX_FILE_BYTES = 10 * 1024 * 1024
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}
CORPUS_COLUMNS = ["doc_id", "title", "category", "text", "source", "chunk_index", "document_id"]


def safe_filename(filename: str) -> str:
    """Discard directory components and control characters from an uploaded name."""

    name = Path(str(filename)).name
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    return name or "uploaded-document"


def extract_text(filename: str, content: bytes) -> str:
    """Extract text without writing user-controlled filenames to the filesystem."""

    if not content:
        raise ValueError("The uploaded file is empty.")
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("The uploaded file exceeds the 10 MB limit.")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Only PDF, TXT, and Markdown files are supported.")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(content))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise ValueError("The PDF could not be read or contains no extractable text.") from exc
    else:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Text files must use UTF-8 encoding.") from exc

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise ValueError("No extractable text was found in the uploaded file.")
    return text


def chunk_text(text: str, chunk_size: int = 180, overlap: int = 30) -> list[str]:
    """Create overlapping word chunks so evidence near boundaries is retained."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")

    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size])
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


@dataclass(frozen=True)
class IngestionResult:
    filename: str
    document_id: str
    chunks_added: int
    duplicate: bool


class UploadedCorpusStore:
    def __init__(self, corpus_path: str | Path):
        self.corpus_path = Path(corpus_path)

    def _read(self) -> pd.DataFrame:
        if not self.corpus_path.exists() or self.corpus_path.stat().st_size == 0:
            return pd.DataFrame(columns=CORPUS_COLUMNS)
        return pd.read_csv(self.corpus_path)

    def add(self, filename: str, content: bytes) -> IngestionResult:
        filename = safe_filename(filename)
        text = extract_text(filename, content)
        document_id = hashlib.sha256(content).hexdigest()[:16]
        existing = self._read()

        if "document_id" in existing.columns and document_id in set(existing["document_id"].astype(str)):
            return IngestionResult(
                filename=filename,
                document_id=document_id,
                chunks_added=0,
                duplicate=True,
            )

        chunks = chunk_text(text)
        rows = []
        title = Path(filename).stem.replace("_", " ").replace("-", " ").strip() or filename
        for index, chunk in enumerate(chunks):
            rows.append(
                {
                    "doc_id": f"upload-{document_id}-{index:04d}",
                    "title": title,
                    "category": "uploaded",
                    "text": chunk,
                    "source": filename,
                    "chunk_index": index,
                    "document_id": document_id,
                }
            )

        updated = pd.concat([existing, pd.DataFrame(rows, columns=CORPUS_COLUMNS)], ignore_index=True)
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        updated.to_csv(self.corpus_path, index=False)
        return IngestionResult(
            filename=filename,
            document_id=document_id,
            chunks_added=len(rows),
            duplicate=False,
        )
