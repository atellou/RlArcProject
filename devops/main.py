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


def create_job(config_key: str, continer_uri: str):
    job = aiplatform.CustomContainerTrainingJob(
        display_name=f"custom-train-job-{config_key}",
        container_uri=continer_uri,
        command=["poetry", "run", "python3", "rlarcworld/jobs/train.py"],
        labels={"job-config": config_key},
    )
    return job


if __name__ == "__main__":
    try:
        config_file = "rlarcworld/jobs/config.yaml"
        config_key = "test"
        continer_uri = os.environ["GCP_TRAIN_IMAGE"]
        service_account = os.environ["SERVICE_ACCOUNT"]
        init_sample(experiment=f"custom-train-job-{config_key}")
        job = create_job(config_key=config_key, continer_uri=continer_uri)
        job.run(
            scheduling_strategy=aiplatform.compat.types.custom_job.Scheduling.Strategy.SPOT,
            timeout=600,
            machine_type="n1-standard-4",  # Define machine type here
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            replica_count=1,  # Set number of replicas here
            enable_web_access=True,
            args=[
                "--config_file",
                config_file,
                "--config_key",
                config_key,
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
        )
    except Exception as e:
        exit(1)
