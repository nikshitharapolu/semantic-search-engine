from __future__ import annotations

import json
import os
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://aws-mock:5000")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
BUCKET = "rag-documents"
POLICY_NAME = "RagDocumentAccess"

client_config = Config(s3={"addressing_style": "path"})
s3 = boto3.client("s3", endpoint_url=ENDPOINT, region_name=REGION, config=client_config)
iam = boto3.client("iam", endpoint_url=ENDPOINT, region_name=REGION)

try:
    s3.head_bucket(Bucket=BUCKET)
except ClientError:
    s3.create_bucket(Bucket=BUCKET)

s3.put_bucket_versioning(Bucket=BUCKET, VersioningConfiguration={"Status": "Enabled"})
s3.put_bucket_encryption(
    Bucket=BUCKET,
    ServerSideEncryptionConfiguration={
        "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
    },
)

policy_path = Path("infra/moto/document-access-policy.json")
policy_document = policy_path.read_text(encoding="utf-8")
try:
    iam.create_policy(PolicyName=POLICY_NAME, PolicyDocument=policy_document)
except ClientError as error:
    if error.response.get("Error", {}).get("Code") != "EntityAlreadyExists":
        raise

print(json.dumps({"bucket": BUCKET, "versioning": "Enabled", "policy": POLICY_NAME}))
