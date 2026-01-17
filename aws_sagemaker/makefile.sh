#!/bin/bash

set -euo pipefail

required_vars=(AWS_REGION AWS_ACCOUNT_ID AWS_ECR_REPOSITORY AWS_ECR_IMAGE_URI)
for v in "${required_vars[@]}"; do
  if [[ -z "${!v:-}" ]]; then
    echo "Missing required env var: ${v}" >&2
    exit 1
  fi
done

# Login to ECR base image
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

# Login to ECR Image to push to
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# If above works build image
docker build -t ${AWS_ECR_REPOSITORY} -f aws_sagemaker/Dockerfile . --platform linux/amd64

# If Above works tag and push image
docker tag ${AWS_ECR_REPOSITORY}:latest ${AWS_ECR_IMAGE_URI}:latest
docker push ${AWS_ECR_IMAGE_URI}:latest
