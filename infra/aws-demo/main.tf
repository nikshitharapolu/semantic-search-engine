terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 6.66" }
  }
}
provider "aws" {
  region = var.region
  default_tags { tags = { Project = "hybrid-search", Environment = "demo", ManagedBy = "terraform" } }
}
variable "region" {
  type    = string
  default = "us-east-2"
}
variable "allowed_cidr" {
  description = "Your current public IPv4 address with /32. Never expose this HTTP demo publicly."
  type        = string
  validation {
    condition     = can(cidrhost(var.allowed_cidr, 0)) && endswith(var.allowed_cidr, "/32")
    error_message = "Provide a single IPv4 /32 address."
  }
}
variable "image_digest" {
  description = "ECR sha256 digest. Empty during the storage-only bootstrap."
  type        = string
  default     = ""
}
variable "snapshot_dir" {
  description = "Local directory containing corpus.csv and index/{manifest.json,embeddings.npy}."
  type        = string
}
variable "snapshot_version" {
  type    = string
  default = "v1"
  validation {
    condition     = can(regex("^[a-zA-Z0-9_-]+$", var.snapshot_version))
    error_message = "Use letters, numbers, underscores or hyphens."
  }
}
variable "run_service" {
  type    = bool
  default = false
}
variable "alert_email" { type = string }
data "aws_caller_identity" "current" {}
data "aws_vpc" "default" { default = true }
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}
resource "aws_s3_bucket" "index" {
  bucket = "hybrid-search-index-${data.aws_caller_identity.current.account_id}-${var.region}"
}
resource "aws_s3_bucket_public_access_block" "index" {
  bucket                  = aws_s3_bucket.index.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
resource "aws_s3_bucket_server_side_encryption_configuration" "index" {
  bucket = aws_s3_bucket.index.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}
resource "aws_s3_bucket_versioning" "index" {
  bucket = aws_s3_bucket.index.id
  versioning_configuration { status = "Enabled" }
}
resource "aws_s3_object" "snapshot" {
  for_each               = toset(["corpus.csv", "index/manifest.json", "index/embeddings.npy"])
  bucket                 = aws_s3_bucket.index.id
  key                    = "snapshots/${var.snapshot_version}/${each.value}"
  source                 = "${var.snapshot_dir}/${each.value}"
  source_hash            = filemd5("${var.snapshot_dir}/${each.value}")
  server_side_encryption = "AES256"
}
resource "aws_ecr_repository" "api" {
  name                 = "hybrid-search-api"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration { scan_on_push = true }
}
resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/hybrid-search-api"
  retention_in_days = 7
}
resource "aws_sns_topic" "alerts" { name = "hybrid-search-alerts" }
resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}
resource "aws_iam_role" "execution" {
  name               = "hybrid-search-execution"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Principal = { Service = "ecs-tasks.amazonaws.com" }, Action = "sts:AssumeRole" }] })
}
resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}
resource "aws_iam_role" "task" {
  name               = "hybrid-search-task"
  assume_role_policy = aws_iam_role.execution.assume_role_policy
}
resource "aws_iam_role_policy" "index_read" {
  role   = aws_iam_role.task.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Action = ["s3:GetObject"], Resource = "${aws_s3_bucket.index.arn}/snapshots/${var.snapshot_version}/*" }] })
}
resource "aws_security_group" "api" {
  name   = "hybrid-search-demo"
  vpc_id = data.aws_vpc.default.id
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = [var.allowed_cidr]
  }
  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
resource "aws_ecs_cluster" "api" { name = "hybrid-search-demo" }
resource "aws_ecs_task_definition" "api" {
  count                    = var.image_digest != "" ? 1 : 0
  family                   = "hybrid-search-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }
  container_definitions = jsonencode([{
    name         = "api", essential = true,
    image        = "${aws_ecr_repository.api.repository_url}@${var.image_digest}",
    portMappings = [{ containerPort = 8000, protocol = "tcp" }],
    environment = [
      { name = "INDEX_BUCKET", value = aws_s3_bucket.index.id },
      { name = "INDEX_PREFIX", value = "snapshots/${var.snapshot_version}" },
      { name = "AWS_DEFAULT_REGION", value = var.region }
    ],
    healthCheck = {
      command  = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)\""],
      interval = 30, timeout = 5, retries = 3, startPeriod = 120
    },
    logConfiguration = {
      logDriver = "awslogs",
      options   = { awslogs-group = aws_cloudwatch_log_group.api.name, awslogs-region = var.region, awslogs-stream-prefix = "api" }
    }
  }])
}
resource "aws_ecs_service" "api" {
  count                              = var.image_digest != "" ? 1 : 0
  name                               = "hybrid-search-api"
  cluster                            = aws_ecs_cluster.api.id
  task_definition                    = aws_ecs_task_definition.api[0].arn
  desired_count                      = var.run_service ? 1 : 0
  launch_type                        = "FARGATE"
  deployment_maximum_percent         = 100
  deployment_minimum_healthy_percent = 0
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  network_configuration {
    subnets          = [sort(data.aws_subnets.default.ids)[0]]
    security_groups  = [aws_security_group.api.id]
    assign_public_ip = true
  }
  depends_on = [aws_iam_role_policy_attachment.execution, aws_iam_role_policy.index_read, aws_s3_object.snapshot]
}
resource "aws_cloudwatch_log_metric_filter" "errors" {
  name           = "hybrid-search-5xx"
  log_group_name = aws_cloudwatch_log_group.api.name
  pattern        = "{ $.status >= 500 }"
  metric_transformation {
    name          = "ServerErrors"
    namespace     = "HybridSearch"
    value         = "1"
    default_value = 0
  }
}
resource "aws_cloudwatch_metric_alarm" "errors" {
  alarm_name          = "hybrid-search-api-errors"
  namespace           = "HybridSearch"
  metric_name         = "ServerErrors"
  statistic           = "Sum"
  period              = 60
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}
resource "aws_cloudwatch_metric_alarm" "memory" {
  alarm_name          = "hybrid-search-memory"
  namespace           = "AWS/ECS"
  metric_name         = "MemoryUtilization"
  statistic           = "Average"
  period              = 60
  evaluation_periods  = 3
  threshold           = 85
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  dimensions          = { ClusterName = aws_ecs_cluster.api.name, ServiceName = "hybrid-search-api" }
  alarm_actions       = [aws_sns_topic.alerts.arn]
}
output "repository_url" { value = aws_ecr_repository.api.repository_url }
output "index_bucket" { value = aws_s3_bucket.index.id }
output "cluster" { value = aws_ecs_cluster.api.name }
