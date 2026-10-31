# AWS deployment verification — October 30, 2026

The FastAPI backend was deployed to Amazon ECS Fargate in Ohio (`us-east-2`)
using `infra/aws-demo` Terraform. The demonstration used one Linux task with
1 vCPU and 2 GB memory and was stopped after verification. It is not an
always-on public endpoint. The existing Streamlit demo remains separate.

## Verified behavior

- The container restored the base corpus and embedding index from a private,
  encrypted, versioned S3 bucket before accepting requests.
- `GET /health` returned HTTP 200.
- `POST /search` returned three ranked results for `hybrid retrieval`.
- `POST /ask` returned cited extractive evidence. No Ollama server or paid LLM
  was deployed; this verifies the existing fallback, not LLM generation.
- `POST /documents` returned HTTP 403 for the read-only cloud replica.
- Real request status and duration events appeared in CloudWatch Logs.
- An SNS email subscription was confirmed. A deliberately forced alarm state
  successfully executed its SNS action. This tests notification wiring, not
  detection of a real production incident. A separate metric-filter test matched
  a synthetic 500 event and excluded a 200 event.
- Terraform validation and all 24 unit/API tests passed. The built Linux
  container also passed local smoke tests under a 2 GB memory limit.

## Configuration

- ECR repository: `hybrid-search-api`; immutable image tag and digest.
- ECS cluster: `hybrid-search-demo`; service: `hybrid-search-api`.
- CloudWatch log group: `/ecs/hybrid-search-api`; retention: seven days.
- Alarms: `hybrid-search-api-errors` and `hybrid-search-memory`.
- Network access restricted to the operator's IPv4 address; no load balancer,
  NAT gateway, autoscaling, hosted LLM, or public production endpoint.
- Task role can read the snapshot prefix only.

The account remained on its active AWS Free plan. The billing API reported
$120 in remaining credits during verification; billing/credit updates are
asynchronous. No upgrade was performed. Retained S3 objects, ECR images,
CloudWatch resources, and logs may consume credits even while compute is stopped.

Image digest: `sha256:3bbd8f2d30fc508115d5f7f4f4632935b093f5968bf77b6432b319694b1d3814`.

Raw API responses and infrastructure run logs are saved locally in the ignored
`.cloud-demo` directory. This document records deployment evidence rather than
availability or performance guarantees. See `infra/aws-demo/README.md` for
launch, monitoring, and cleanup instructions.
