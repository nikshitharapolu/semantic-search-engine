provider "aws" {
  region                      = var.aws_region
  access_key                  = "test"
  secret_key                  = "test"
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true
  s3_use_path_style           = true

  endpoints {
    s3  = var.aws_mock_endpoint
    iam = var.aws_mock_endpoint
  }
}

resource "aws_s3_bucket" "documents" {
  bucket        = var.document_bucket
  force_destroy = true

  tags = {
    Project     = "hybrid-search-rag"
    Environment = "local"
    ManagedBy   = "terraform"
  }
}

resource "aws_s3_bucket_versioning" "documents" {
  bucket = aws_s3_bucket.documents.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "documents" {
  bucket = aws_s3_bucket.documents.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "documents" {
  bucket                  = aws_s3_bucket.documents.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

data "aws_iam_policy_document" "document_access" {
  statement {
    sid       = "ListDocumentBucket"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.documents.arn]
  }

  statement {
    sid    = "ReadWriteDocumentObjects"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject"
    ]
    resources = ["${aws_s3_bucket.documents.arn}/documents/*"]
  }
}

resource "aws_iam_policy" "document_access" {
  name        = "RagDocumentAccessTerraform"
  description = "Least-privilege access to RAG source documents."
  policy      = data.aws_iam_policy_document.document_access.json
}
