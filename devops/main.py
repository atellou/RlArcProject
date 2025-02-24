import os
from typing import Optional
import google.auth
import vertexai
from google.cloud import aiplatform, aiplatform_v1


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
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if location is None:
        location = os.environ.get("GCP_REGION", "us-central1")
    if staging_bucket is None:
        staging_bucket = os.environ.get("GCP_STORAGE_BUCKET")
    vertexai.init(
        project=project,
        location=location,
        experiment=experiment,
        staging_bucket=staging_bucket,
        credentials=credentials,
        encryption_spec_key_name=encryption_spec_key_name,
        service_account=service_account,
    )


def create_job(config_file: str, config_key: str):
    init_sample(experiment=f"custom-train-job-{config_key}")
    job = aiplatform.CustomContainerTrainingJob(
        display_name="test-train",
        container_uri=os.environ["GCP_TRAIN_IMAGE"],
        command=[
            "poetry",
            "run",
            "python3",
            "rlarcworld/jobs/train.py",
            "--config_file",
            config_file,
            "--config_key",
            config_key,
        ],
        labels={"job-config": config_key},
    )
    return job
