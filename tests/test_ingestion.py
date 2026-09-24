from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingestion import UploadedCorpusStore, chunk_text, extract_text, safe_filename


class IngestionTests(unittest.TestCase):
    def test_chunks_overlap(self):
        words = [f"word-{index}" for index in range(12)]
        chunks = chunk_text(" ".join(words), chunk_size=5, overlap=2)

        self.assertEqual(chunks[0].split()[-2:], chunks[1].split()[:2])
        self.assertEqual(chunks[1].split()[-2:], chunks[2].split()[:2])

    def test_filename_is_reduced_to_safe_basename(self):
        self.assertEqual(safe_filename("../../private/notes.txt"), "notes.txt")

    def test_unsupported_and_empty_files_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            extract_text("notes.txt", b"")
        with self.assertRaisesRegex(ValueError, "Only PDF"):
            extract_text("notes.exe", b"content")

    def test_store_persists_chunks_and_detects_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            store = UploadedCorpusStore(Path(directory) / "uploaded_corpus.csv")
            content = ("retrieval augmented generation uses relevant context " * 50).encode("utf-8")

            first = store.add("rag-notes.txt", content)
            second = store.add("rag-notes.txt", content)

            self.assertGreater(first.chunks_added, 1)
            self.assertFalse(first.duplicate)
            self.assertTrue(second.duplicate)
            self.assertEqual(second.chunks_added, 0)


if __name__ == "__main__":
    unittest.main()
