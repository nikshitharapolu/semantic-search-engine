output "document_bucket" {
  description = "S3 bucket containing original documents."
  value       = aws_s3_bucket.documents.id
}

output "document_access_policy_arn" {
  description = "Least-privilege IAM policy for the application workload."
  value       = aws_iam_policy.document_access.arn
}
