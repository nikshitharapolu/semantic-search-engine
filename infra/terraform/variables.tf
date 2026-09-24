variable "aws_region" {
  description = "AWS region used by the local provider."
  type        = string
  default     = "us-east-1"
}

variable "aws_mock_endpoint" {
  description = "Local Moto AWS emulator endpoint."
  type        = string
  default     = "http://localhost:4566"
}

variable "document_bucket" {
  description = "Demonstration S3 bucket managed independently by Terraform."
  type        = string
  default     = "rag-documents-iac"
}
