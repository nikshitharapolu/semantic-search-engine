from __future__ import annotations

import json
import os

import boto3
from botocore.config import Config


endpoint = os.getenv("AWS_ENDPOINT_URL", "http://aws-mock:5000")
region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
config = Config(s3={"addressing_style": "path"})
s3 = boto3.client("s3", endpoint_url=endpoint, region_name=region, config=config)
iam = boto3.client("iam", endpoint_url=endpoint, region_name=region)

bucket = "rag-documents"
objects = s3.list_objects_v2(Bucket=bucket).get("Contents", [])
result = {
    "buckets": [item["Name"] for item in s3.list_buckets().get("Buckets", [])],
    "versioning": s3.get_bucket_versioning(Bucket=bucket).get("Status"),
    "objects": [item["Key"] for item in objects],
    "local_policies": [item["PolicyName"] for item in iam.list_policies(Scope="Local").get("Policies", [])],
}
print(json.dumps(result, indent=2))
