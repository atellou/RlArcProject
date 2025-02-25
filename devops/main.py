import os
import argparse
from typing import Optional
import google.auth
import vertexai
from google.cloud import aiplatform

import logging

logger = logging.getLogger(__name__)


def init_sample(
    project: Optional[str] = None,
    location: Optional[str] = None,
    experiment: Optional[str] = None,
    staging_bucket: Optional[str] = None,
    credentials: Optional[google.auth.credentials.Credentials] = None,
    encryption_spec_key_name: Optional[str] = None,
    service_account: Optional[str] = None,
):
    if project is None:
        project = os.environ.get("PROJECT_ID")
    if location is None:
        location = os.environ.get("REGION", "us-central1")
    if staging_bucket is None:
        staging_bucket = os.environ.get("STORAGE_URI")
    vertexai.init(
        project=project,
        location=location,
        experiment=experiment,
        staging_bucket=staging_bucket,
        credentials=credentials,
        encryption_spec_key_name=encryption_spec_key_name,
        service_account=service_account,
    )


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config_file", type=str, required=True)
        parser.add_argument("--config_key", type=str, required=True)
        parser.add_argument("--log_level", type=str, default="INFO")
        args = parser.parse_args()

        continer_uri = os.environ["GCP_TRAIN_IMAGE"]
        service_account = os.environ["SERVICE_ACCOUNT"]
        init_sample(experiment=f"custom-train-job-{args.config_key}")

        job = aiplatform.CustomContainerTrainingJob(
            display_name=f"custom-train-job-{args.config_key}",
            container_uri=continer_uri,
            command=["poetry", "run", "python3", "rlarcworld/jobs/train.py"],
            labels={"job-config": args.config_key},
        )
        job.run(
            sync=True,
            scheduling_strategy=aiplatform.compat.types.custom_job.Scheduling.Strategy.SPOT,
            timeout=600,
            machine_type="n1-standard-4",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            replica_count=1,
            enable_web_access=True,
            args=[
                "--config_file",
                args.config_file,
                "--config_key",
                args.config_key,
                "--log_level",
                args.log_level,
            ],
            service_account=service_account,
            environment_variables={
                "PROJECT_ID": os.environ["PROJECT_ID"],
                "REGION": os.environ["REGION"],
                "STORAGE_URI": os.environ["STORAGE_URI"],
                "AIP_TRAINING_DATA_URI": os.environ["AIP_TRAINING_DATA_URI"],
                "AIP_VALIDATION_DATA_URI": os.environ["AIP_VALIDATION_DATA_URI"],
                "AIP_TEST_DATA_URI": os.environ["AIP_TEST_DATA_URI"],
                "IMAGE_TAG": os.environ["IMAGE_TAG"],
                "IMAGE_NAME": os.environ["IMAGE_NAME"],
                "GCP_TRAIN_IMAGE": os.environ["GCP_TRAIN_IMAGE"],
                "SERVICE_ACCOUNT": os.environ["SERVICE_ACCOUNT"],
            },
            base_output_dir=os.environ["STORAGE_URI"],
        )
    except Exception as e:
        logger.error(e)
        exit(1)
