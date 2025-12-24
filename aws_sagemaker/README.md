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

### Custom container image

If you need additional system libraries, build and push an image to ECR and supply the URI via `--training-image`.

## Distributed training details

`launch.py` configures `Torchrun(process_count_per_node=None)`, letting SageMaker automatically match the number of GPUs on the instance.  FSDP is enabled through the `cloud_base.yaml` configuration so that memory is sharded across GPUs.

## Output

- Checkpoints and final `model.tar.gz` are stored in `s3://<default-bucket>/<training-job-name>/output/`.
- Logs stream to Amazon CloudWatch under the `/aws/sagemaker/TrainingJobs` log group.

---

## TODOs

- [ ] Upload dataset tarball to S3.
- [ ] Build/push ECR image.
- [ ] Configure IAM role and AWS credentials.
- [x] Add `--dry-run` option to `aws_sagemaker/launch.py` that prints the `ModelTrainer` request JSON and skips job submission.
