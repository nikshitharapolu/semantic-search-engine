import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cloud_bootstrap import restore_snapshot, SNAPSHOT_FILES


class SnapshotTests(unittest.TestCase):
    def test_restores_only_known_files(self):
        client = Mock()
        def download(bucket, key, target):
            Path(target).write_text(key)
        client.download_file.side_effect = download
        with tempfile.TemporaryDirectory() as directory:
            restore_snapshot(client, "index", "snapshots/v1/", Path(directory))
            for name in SNAPSHOT_FILES:
                self.assertEqual((Path(directory) / name).read_text(), f"snapshots/v1/{name}")
            self.assertEqual(client.download_file.call_count, 3)

    def test_download_failure_does_not_replace_file(self):
        client = Mock()
        client.download_file.side_effect = RuntimeError("S3 unavailable")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "corpus.csv"
            target.write_text("previous snapshot")
            with self.assertRaises(RuntimeError):
                restore_snapshot(client, "index", "snapshots/v1", Path(directory))
            self.assertEqual(target.read_text(), "previous snapshot")


class CloudAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        import api
        cls.api = api
        cls.client = TestClient(api.app)

    def test_cloud_upload_disabled_before_ingestion(self):
        with patch.dict("os.environ", {"RAG_READ_ONLY": "1"}), patch.object(self.api, "ingest_document") as ingest:
            response = self.client.post("/documents", files={"file": ("a.txt", b"hello")})
            self.assertEqual(response.status_code, 403)
            ingest.assert_not_called()

    def test_logs_metrics_without_query_contents(self):
        with self.assertLogs(self.api.request_log, level="INFO") as logs:
            response = self.client.post("/search", json={"query": "private query", "top_k": 0})
        self.assertEqual(response.status_code, 422)
        record = json.loads(logs.records[-1].getMessage())
        self.assertEqual(record["status"], 422)
        self.assertIn("duration_ms", record)
        self.assertNotIn("private query", logs.records[-1].getMessage())

    def test_unhandled_error_logged_as_500(self):
        from fastapi.testclient import TestClient
        client = TestClient(self.api.app, raise_server_exceptions=False)
        with patch.object(self.api, "get_engine", side_effect=RuntimeError("failure")):
            with self.assertLogs(self.api.request_log, level="INFO") as logs:
                response = client.post("/search", json={"query": "test"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(json.loads(logs.records[-1].getMessage())["status"], 500)
