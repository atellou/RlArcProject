"""Launch a SageMaker training job for RlArcProject.

This module lives under `aws_sagemaker/` to avoid shadowing the official AWS
`sagemaker` Python SDK.

It submits a GPU training job using the SageMaker PyTorch Estimator and runs
`aws_sagemaker/train.py` as the entrypoint.
"""

from __future__ import annotations

import argparse
import logging

from sagemaker.core.helper.session_helper import Session as SmSession
from sagemaker.core.training.configs import (
    Compute,
    InputData,
    SourceCode,
    StoppingCondition,
)
from sagemaker.train import ModelTrainer
from sagemaker.train.distributed import Torchrun

logger = logging.getLogger(__name__)


def main() -> None:
    """Submit a SageMaker training job using the PyTorch Estimator."""
    parser = argparse.ArgumentParser(description="Launch SageMaker training")
    parser.add_argument("--role-arn", required=True, type=str)
    parser.add_argument("--s3-train-tar", required=True, type=str)
    parser.add_argument("--experiment", required=True, type=str)
    parser.add_argument(
        "--experiments-config",
        default="experiments/lr_td_analysis/experiments.yaml",
        type=str,
    )
    parser.add_argument(
        "--base-config",
        default="experiments/configs/cloud_base.yaml",
        type=str,
    )
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--instance-type", default="ml.g5.12xlarge", type=str)
    parser.add_argument("--instance-count", default=1, type=int)
    parser.add_argument("--volume-size", default=200, type=int)
    parser.add_argument("--max-run", default=24 * 60 * 60, type=int)
    parser.add_argument("--job-name", default=None, type=str)
    parser.add_argument("--training-image", default=None, type=str)
    parser.add_argument("--pytorch-version", default="2.1.0", type=str)
    parser.add_argument("--py-version", default="py310", type=str)

    args = parser.parse_args()

    sess = SmSession()
    role = args.role_arn
    logger.info("SageMaker Session:")
    logger.info("  - role arn: %s", role)
    logger.info("  - bucket: %s", sess.default_bucket())
    logger.info("  - region: %s", sess.boto_region_name)

    training_image = args.training_image
    if training_image is None:
        try:
            from sagemaker.core.image_retriever import ImageRetriever
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Unable to import 'sagemaker.core.image_retriever'. "
                "Either install the full SageMaker Python SDK or pass --training-image "
                "with the full ECR image URI to use for training."
            ) from e

        training_image = ImageRetriever.retrieve(
            framework="pytorch",
            region=sess.boto_region_name,
            version=args.pytorch_version,
            py_version=args.py_version,
            image_scope="training",
            instance_type=args.instance_type,
        )

    source_code = SourceCode(
        source_dir=".",
        entry_script="aws_sagemaker/train.py",
    )

    # Let SageMaker infer GPU count per node; override with --processes-per-node
    distributed = Torchrun(process_count_per_node=None)

    compute = Compute(
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        volume_size_in_gb=args.volume_size,
    )

    stopping_condition = StoppingCondition(max_runtime_in_seconds=args.max_run)

    model_trainer = ModelTrainer(
        training_image=training_image,
        training_mode="SAGEMAKER_TRAINING_JOB",
        sagemaker_session=sess,
        role=role,
        base_job_name=args.experiment,
        source_code=source_code,
        distributed=distributed,
        compute=compute,
        stopping_condition=stopping_condition,
        hyperparameters={
            k: v
            for k, v in {
                "experiment": args.experiment,
                "experiments-config": args.experiments_config,
                "base-config": args.base_config,
                "epochs": str(args.epochs),
                "seed": None if args.seed is None else str(args.seed),
                "samples-path-list": "arc_sample.txt",
            }.items()
            if v is not None
        },
        environment={
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            "NCCL_DEBUG": "WARN",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    )

    if args.job_name is not None:
        logger.warning(
            "--job-name is not supported by ModelTrainer.train(); it will be ignored."
        )

    train_data = InputData(channel_name="train", data_source=args.s3_train_tar)
    model_trainer.train(input_data_config=[train_data], wait=True, logs=True)


if __name__ == "__main__":
    main()
