# AWS ECS demonstration

This environment extends the local-emulator Terraform in `../terraform` with a
separate AWS state. Never apply the emulator configuration against a real account.
Use the project's assigned region: the connected account permits Ohio
(`us-east-2`), while its organization policy denies regional services in Virginia.
ECS/ECR read-only access was verified on the Free plan; no plan upgrade is required.

## Before provisioning

Keep the Free plan. Verify remaining credits and expiration in AWS Settings.
The existing `hybrid-search-spend-alert` budget is an alert, not a spending cap.
Do not activate the Paid plan as part of this procedure. Keep copies of the code
and snapshots locally: Free plan access ends when the credits or free period end.

Install a working Docker Desktop, then from the repository root:

```sh
python scripts/prepare_cloud_snapshot.py
docker build --platform linux/amd64 -f Dockerfile.api -t hybrid-search-api:demo-v1 .
```

This image uses CPU PyTorch, bundles the embedding model, and downloads a fixed
three-file S3 snapshot before accepting requests. It does not run Ollama or call
a paid LLM. `/ask` and `/ask/guided` use the existing extractive fallback when
Ollama is unavailable. Uploads are disabled on the cloud replica. S3 is the durable
snapshot store; local task storage is disposable. Local UI and upload behavior
are preserved.

Create an ignored `infra/aws-demo/terraform.tfvars` containing:

```hcl
region = "us-east-2"
allowed_cidr = "YOUR_PUBLIC_IPV4/32"
alert_email = "YOUR_EMAIL"
snapshot_dir = "../../.cloud-demo/snapshot"
snapshot_version = "v1"
run_service = false
```

Never use `0.0.0.0/0`. This is an IP-restricted HTTP demonstration, not a public
production endpoint. Do not send sensitive data. A public service needs TLS,
authentication and request limits before opening access.

## Provision and run

Use short-lived CLI credentials. If the provider cannot use the CLI login cache,
export credentials only inside the invoking process; never save or print them.
Initialize, validate and review a saved plan before applying it:

```sh
AWS_PROFILE=hybrid-search terraform -chdir=infra/aws-demo init
AWS_PROFILE=hybrid-search terraform -chdir=infra/aws-demo plan -out=tfplan
AWS_PROFILE=hybrid-search terraform -chdir=infra/aws-demo apply tfplan
```

The initial apply creates storage, an ECR repository, IAM roles, monitoring and
an ECS cluster; it launches no task. Confirm the SNS email subscription.
Get the ECR repository URL from `terraform output -raw repository_url`, authenticate
Docker with `aws ecr get-login-password --region us-east-2 --profile hybrid-search`,
then tag and push `demo-v1`. Pass the password to `docker login --password-stdin`.
Use the image's ECR sha256 digest in `image_digest` (not a mutable tag).
Set `run_service = true`, review a fresh plan, and apply it to start one task.
The service cannot autoscale and deployment settings permit only one task.

Find the task's ENI public IP using ECS `describe-tasks` and EC2
`describe-network-interfaces`. Test `http://IP:8000/health` and POST `/search` with
`{"query":"hybrid retrieval","top_k":3}`. Save the actual response, deployment
revision and CloudWatch evidence before making deployment claims.

## Monitoring and evidence

CloudWatch log group: `/ecs/hybrid-search-api`; retention: 7 days.
Structured request logs contain method, route, status and duration, with no
request bodies or query contents. CloudWatch alarms cover HTTP 5xx errors and
sustained memory utilization over 85%. Notification delivery requires confirming
the SNS email. ECS health checks monitor `/health`, and failed deployments use a
circuit breaker. No continuous external uptime probe is provisioned.

To verify the error alarm, use `aws cloudwatch set-alarm-state` on
`hybrid-search-api-errors` with `--state-value ALARM` and an explicit test reason.
Label this as a notification test; it does not prove the metric filter works.
Verify the filter separately with `aws logs test-metric-filter` and a synthetic
JSON log containing `{"event":"request","status":500}`. Do not introduce an
intentional failure endpoint into the deployed app.

## Stop and clean up

Set `run_service = false`, review/apply the plan, then wait for the service to
reach zero running AND pending tasks. Stopping a task directly is insufficient:
the service will replace it while desired count is one. No task = no ongoing
Fargate compute, but stored images, snapshots, alarms and logs may still cost.

For full cleanup, deliberately remove this demo's ECR images and all S3 object
versions/delete markers, then review and apply `terraform destroy`. Both stores
refuse destructive deletion while nonempty. Do not delete the separate billing
alert. Verify there are no remaining tasks, repository, bucket, alarms, or log
group; check billing after its delayed update. Keep local state until cleanup
has succeeded. Cloud resources have not been deployed merely by adding these files.
