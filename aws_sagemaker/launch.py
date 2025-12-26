"""Launch a SageMaker training job for RlArcProject.

This module lives under `aws_sagemaker/` to avoid shadowing the official AWS
`sagemaker` Python SDK.

It submits a GPU training job using the SageMaker PyTorch Estimator and runs
`aws_sagemaker/train.py` as the entrypoint.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sagemaker.core.helper.session_helper import Session as SmSession
from sagemaker.core.training.configs import (
    Compute,
    InputData,
    OutputDataConfig,
    SourceCode,
    StoppingCondition,
)
from sagemaker.train import ModelTrainer
from sagemaker.train.distributed import Torchrun

logger = logging.getLogger(__name__)


class SagemakerExperiment:
    """Configuration for a SageMaker training job."""

    def __init__(
        self,
        role_arn: str,
        output_data_config: OutputDataConfig,
        experiment: str,
        experiments_config: str,
        base_config: str = "experiments/configs/cloud_base.yaml",
        epochs: int = 10,
        seed: int | None = None,
        instance_type: str = "ml.g5.12xlarge",
        instance_count: int = 1,
        volume_size: int = 200,
        max_run: int = 1 * 60 * 60,
        job_name: str | None = None,
        training_image: str | None = None,
        pytorch_version: str = "2.1.0",
        py_version: str = "py310",
    ):
        """Configuration for a SageMaker training job.

        Attributes:
            role_arn: AWS IAM role ARN for SageMaker execution.
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
        """
        self.__role_arn = role_arn
        self.experiment = experiment
        self.experiments_config = experiments_config
        self.base_config = base_config
        self.epochs = epochs
        self.seed = seed
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.volume_size = volume_size
        self.max_run = max_run
        self.job_name = job_name
        self.training_image = training_image
        self.pytorch_version = pytorch_version
        self.py_version = py_version
        self.output_data_config = output_data_config

        self.session = SmSession()
        logger.info("SageMaker Session:")
        logger.info("  - role arn: %s", self.__role_arn)
        logger.info("  - bucket: %s", self.session.default_bucket())
        logger.info("  - region: %s", self.session.boto_region_name)

        self.training_image = self.__retreive_image()
        logger.info("Training image: %s", self.training_image)

        self.source_code = self.__source_code()
        logger.info("Source code: %s", self.source_code)

        self.compute = Compute(
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            volume_size_in_gb=self.volume_size,
        )
        logger.info("Compute: %s", self.compute)

        self.create_model_trainer()

    def __retreive_image(self):
        """Submit a SageMaker training job using the PyTorch Estimator."""
        training_image = self.training_image
        if training_image is None:
            try:
                from sagemaker.core import image_uris
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "Unable to import 'sagemaker.core.image_uris'. "
                    "Either install full SageMaker Python SDK or pass training-image"
                    "with the full ECR image URI to use for training."
                ) from e

            training_image = image_uris.retrieve(
                framework="pytorch",
                region=self.session.boto_region_name,
                version=self.pytorch_version,
                py_version=self.py_version,
                image_scope="training",
                instance_type=self.instance_type,
            )
        return training_image

    def __source_code(self) -> SourceCode:
        """Return the source code for the training job."""
        project_root = Path(__file__).resolve().parent.parent  # repo root

        source_code = SourceCode(
            source_dir=str(project_root),
            entry_script="aws_sagemaker/train.py",
            ignore_patterns=[
                "*.ipynb",
                ".git",
                ".github",
                ".venv",
                "**/__pycache__",
                "tests",
            ],
        )
        return source_code

    def create_model_trainer(self) -> None:
        """Create the model trainer for the training job."""
        stopping_condition = StoppingCondition(max_runtime_in_seconds=self.max_run)

        self.model_trainer = ModelTrainer(
            training_image=self.training_image,
            training_mode="SAGEMAKER_TRAINING_JOB",
            sagemaker_session=self.session,
            role=self.__role_arn,
            base_job_name=self.experiment,
            source_code=self.source_code,
            distributed=Torchrun(process_count_per_node=None),
            output_data_config=self.output_data_config,
            compute=self.compute,
            stopping_condition=stopping_condition,
            hyperparameters={
                k: v
                for k, v in {
                    "experiment": self.experiment,
                    "experiments-config": self.experiments_config,
                    "base-config": self.base_config,
                    "epochs": str(self.epochs),
                    "seed": None if self.seed is None else str(self.seed),
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
        logger.info(f"Model trainer created: {self.model_trainer}")

    def train(self, s3_train_data_tar: str) -> None:
        """Train the model using the SageMaker training job.

        Args:
            s3_train_data_tar: S3 path to training data tarball.
        """
        train_data = InputData(channel_name="train", data_source=s3_train_data_tar)
        self.model_trainer.train(input_data_config=[train_data], wait=True, logs=True)
