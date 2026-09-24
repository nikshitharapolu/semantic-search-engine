from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from object_storage import LocalObjectStore, object_identity


class ObjectStorageTests(unittest.TestCase):
    def test_identity_is_deterministic_and_sanitized(self):
        content = b"retrieval evidence"
        name, checksum, key = object_identity("../../private/notes.txt", content)

        self.assertEqual(name, "notes.txt")
        self.assertEqual(checksum, hashlib.sha256(content).hexdigest())
        self.assertEqual(key, f"documents/{checksum[:16]}/notes.txt")

    def test_local_store_persists_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalObjectStore(directory)
            content = b"original document bytes"

            stored = store.put("paper.txt", content)

            path = Path(stored.uri.removeprefix("file://"))
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(stored.checksum_sha256, hashlib.sha256(content).hexdigest())


if __name__ == "__main__":
    unittest.main()
