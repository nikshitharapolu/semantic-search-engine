from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ingestion import safe_filename


class ObjectStorageError(RuntimeError):
    """Raised when an original document cannot be persisted."""


@dataclass(frozen=True)
class StoredObject:
    uri: str
    checksum_sha256: str


class ObjectStore(Protocol):
    def put(self, filename: str, content: bytes) -> StoredObject:
        ...


def object_identity(filename: str, content: bytes) -> tuple[str, str, str]:
    checksum = hashlib.sha256(content).hexdigest()
    clean_name = safe_filename(filename)
    key = f"documents/{checksum[:16]}/{clean_name}"
    return clean_name, checksum, key


class LocalObjectStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def put(self, filename: str, content: bytes) -> StoredObject:
        _, checksum, key = object_identity(filename, content)
        destination = self.root / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return StoredObject(uri=destination.resolve().as_uri(), checksum_sha256=checksum)


class S3ObjectStore:
    def __init__(
        self,
        bucket: str,
        endpoint_url: str | None = None,
        region: str = "us-east-1",
    ):
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise ObjectStorageError("boto3 is required for S3 object storage.") from exc

        self.bucket = bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            config=Config(s3={"addressing_style": "path"}),
        )

    def put(self, filename: str, content: bytes) -> StoredObject:
        clean_name, checksum, key = object_identity(filename, content)
        content_type = mimetypes.guess_type(clean_name)[0] or "application/octet-stream"
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=content,
                ContentType=content_type,
                ServerSideEncryption="AES256",
                Metadata={"sha256": checksum},
            )
        except Exception as exc:
            raise ObjectStorageError(f"Could not store {clean_name} in S3-compatible storage.") from exc
        return StoredObject(uri=f"s3://{self.bucket}/{key}", checksum_sha256=checksum)
