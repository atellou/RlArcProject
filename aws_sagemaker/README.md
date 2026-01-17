# AWS SageMaker Integration

This directory contains the glue code that lets you launch **ArcD4PG** training jobs on Amazon SageMaker.

## Contents

- `launch.py` – CLI wrapper that creates and submits a `ModelTrainer` job using the SageMaker **v3** Python SDK.  Supports a `--dry-run` flag to print the request body without incurring costs.
- `train.py`  – Container entry-point executed on the training instances.  It unpacks the ARC dataset tarball, rewrites config paths, and starts the distributed `torchrun` job.

## Prerequisites

1. **AWS credentials** with permission to create SageMaker training jobs and read/write to the chosen S3 bucket.
2. An **execution role** (e.g. `arn:aws:iam::<account>:role/SageMakerExecRole`) with at least `AmazonSageMakerFullAccess` and S3 access.
3. ARC dataset packed as a single tarball and uploaded to S3, e.g. `s3://<bucket>/datasets/arc_train.tar.gz`.
4. Python environment with `pip install "rlarcproject[aws-sagemaker]"` to pull the SageMaker SDK extras.

## Quick start

```bash
python aws_sagemaker/launch.py \
  --role-arn arn:aws:iam::<acct>:role/SageMakerExecRole \
  --s3-train-tar s3://<bucket>/datasets/arc_train.tar.gz \
  --experiment arc_lr_td \
  --instance-type ml.g5.12xlarge \
  --epochs 10
```

### Dry-run

Add `--dry-run` to see the exact JSON payload instead of submitting the job:

```bash
python aws_sagemaker/launch.py ... --dry-run
```

### Custom container image (Python 3.12)

Build, tag and push a container that already contains **rlarcworld** and all Python dependencies:

```bash
# Authenticate to the DLC and your private ECR registries
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin 763104351884.dkr.ecr.$AWS_REGION.amazonaws.com
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.$AWS_REGION.amazonaws.com

# Build x86-64 image (works on Apple Silicon too)
docker build -t ${AWS_ECR_REPOSITORY} -f aws_sagemaker/Dockerfile . --platform linux/amd64

# Tag and push
docker tag ${AWS_ECR_REPOSITORY}:latest ${AWS_ECR_IMAGE_URI}:latest
docker push ${AWS_ECR_IMAGE_URI}:latest
```

Launch a training job with the custom image:

```bash
python aws_sagemaker/launch.py \
  --training-image ${AWS_ECR_IMAGE_URI}:latest \
  --role-arn $AWS_SAGEMAKER_ROLE_ARN \
  --s3-train-tar s3://$AWS_SAGEMAKER_BUCKET/datasets/arc_train.tar.gz \
  --experiment lr_1e3_nstep_1 \
  --instance-type ml.g5.xlarge \
  --epochs 1
```

`launch.py` uploads only the `aws_sagemaker/` package as the source archive, keeping the transfer minimal.

## Output

- Checkpoints and final `model.tar.gz` are stored in `s3://<default-bucket>/<training-job-name>/output/`.
- Logs stream to Amazon CloudWatch under the `/aws/sagemaker/TrainingJobs` log group.

---

## TODOs

- [x] Upload dataset tarball to S3.
- [x] Build/push ECR image.
- [x] Configure IAM role and AWS credentials.
- [x] Add `--dry-run` option to `aws_sagemaker/launch.py` that prints the `ModelTrainer` request JSON and skips job submission.
