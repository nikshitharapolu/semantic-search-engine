"""Start a read-only search replica from a versioned S3 snapshot."""
from __future__ import annotations

import os
from pathlib import Path

SNAPSHOT_FILES = ("corpus.csv", "index/manifest.json", "index/embeddings.npy")


def restore_snapshot(client, bucket: str, prefix: str, destination: Path) -> None:
    # Only fixed filenames are accepted; remote keys cannot select local paths.
    for name in SNAPSHOT_FILES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".download")
        client.download_file(bucket, f"{prefix.rstrip('/')}/{name}", str(temporary))
        temporary.replace(target)


def main() -> None:
    import boto3
    import uvicorn

    destination = Path(os.environ.get("RAG_DATA_DIR", "/tmp/search-data"))
    restore_snapshot(boto3.client("s3"), os.environ["INDEX_BUCKET"],
                     os.environ["INDEX_PREFIX"], destination)
    os.environ["RAG_DATA_DIR"] = str(destination)
    os.environ["RAG_READ_ONLY"] = "1"
    # Load and validate the snapshot before accepting traffic.
    from service import get_engine
    get_engine()
    uvicorn.run("api:app", host="0.0.0.0", port=8000, workers=1, access_log=False)


if __name__ == "__main__":
    main()
