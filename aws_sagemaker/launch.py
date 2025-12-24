"""Launch a SageMaker training job for RlArcProject.

This module lives under `aws_sagemaker/` to avoid shadowing the official AWS
`sagemaker` Python SDK.

It submits a GPU training job using the SageMaker PyTorch Estimator and runs
`aws_sagemaker/train.py` as the entrypoint.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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


@dataclass
class SagemakerExperiment:
    """Configuration for a SageMaker training job.

    Attributes:
        role_arn: AWS IAM role ARN for SageMaker execution.
        s3_train_tar: S3 path to training data tarball.
        experiment: Name of the experiment configuration to run.
        experiments_config: Path to experiments YAML configuration file.
        base_config: Path to base YAML configuration file.
        epochs: Number of training epochs to run.
        seed: Random seed for reproducibility.
        instance_type: SageMaker instance type for training.
        instance_count: Number of instances to use for training.
        volume_size: EBS volume size in GB for training.
        max_run: Maximum job runtime in seconds.
        job_name: Custom name for the training job.
        training_image: Custom training image URI.
        pytorch_version: PyTorch version to use.
        py_version: Python version to use.
        dry_run: If True, only display the job configuration without submitting.
    """

    role_arn: str
    s3_train_tar: str
    experiment: str
    experiments_config: str = "experiments/lr_td_analysis/experiments.yaml"
    base_config: str = "experiments/configs/cloud_base.yaml"
    epochs: int = 10
    seed: int | None = None
    instance_type: str = "ml.g5.12xlarge"
    instance_count: int = 1
    volume_size: int = 200
    max_run: int = 24 * 60 * 60
    job_name: str | None = None
    training_image: str | None = None
    pytorch_version: str = "2.1.0"
    py_version: str = "py310"
    dry_run: bool = False


def run_experiment(args: SagemakerExperiment) -> None:
    """Submit a SageMaker training job using the PyTorch Estimator."""
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
    if args.dry_run:
        # pretty-print the training configuration and stop
        import pprint

        pprint.pp(model_trainer.to_request())  # full JSON request body
        logger.info("Dry-run: job not submitted.")
        return
    model_trainer.train(input_data_config=[train_data], wait=True, logs=True)
