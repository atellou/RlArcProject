"""Utilities to interact with AWS SageMaker.

This module provides helpers for dataset preparation and S3 interactions used by
`aws_sagemaker/launch.py` and `aws_sagemaker/train.py`.
"""

from __future__ import annotations

import logging
import tarfile
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

from botocore.exceptions import ClientError
from sagemaker.core.helper.session_helper import Session as SmSession


def load_arc_sample(file_path: str | Path) -> list[str]:
    """Load ARC sample JSON paths from a plain-text list.

    Blank lines are ignored. Returns the paths exactly as stored in the file
    (no path validation is performed here).
    """
    file_path = Path(file_path)
    with file_path.open("r") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """Return (bucket, key) from an *s3://bucket/key* URI."""
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    key = parsed.path.lstrip("/")
    return parsed.netloc, key


def create_and_upload_arc_tar(
    session: SmSession,
    sample_list_path: str | Path,
    bucket: str,
    key: str,
    *,
    overwrite: bool = False,
) -> str:
    """Package the JSON files listed in *sample_list_path* and upload the archive.

    Args:
        session : SmSession
            Active SageMaker session to obtain the S3 client.
        sample_list_path : str | Path
            Plain-text file; each non-blank line is a path to an ARC JSON file.
        bucket : str
            Destination S3 bucket.
        key : str
            Destination S3 key. Should end with ``.tar.gz``.
        overwrite : bool, default False
            If *False* and the object already exists, raises :class:`FileExistsError`.

    Returns:
        The full S3 URI of the uploaded tarball.
    """
    sample_paths = load_arc_sample(sample_list_path)
    if not sample_paths:
        raise ValueError("Sample list is empty; nothing to package.")

    resolved: list[Path] = []
    for p in sample_paths:
        path_obj = Path(p).expanduser().resolve()
        if not path_obj.exists():
            raise FileNotFoundError(path_obj)
        resolved.append(path_obj)

    # Build temp archive
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    with tarfile.open(tmp_path, "w:gz") as tf:
        used_names: set[str] = set()
        for src in resolved:
            base_name = src.name
            if base_name in used_names:
                stem, suffix = src.stem, src.suffix
                base_name = f"{stem}_{uuid.uuid4().hex}{suffix}"
            used_names.add(base_name)
            tf.add(src, arcname=base_name)

    s3_client = session.client("s3")
    try:
        if not overwrite:
            s3_client.head_object(Bucket=bucket, Key=key)
            logging.warning(
                "Object %s/%s already exists; skipping upload.", bucket, key
            )
            return f"s3://{bucket}/{key}"
    except ClientError as err:
        if err.response["Error"]["Code"] != "404":
            raise

    logging.info("Uploading %s to s3://%s/%s", tmp_path, bucket, key)
    s3_client.upload_file(str(tmp_path), bucket, key)
    tmp_path.unlink(missing_ok=True)
    return f"s3://{bucket}/{key}"
