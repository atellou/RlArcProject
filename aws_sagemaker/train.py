#!/usr/bin/env python
"""SageMaker training entrypoint for RlArcProject.

This module lives under `aws_sagemaker/` to avoid shadowing the official AWS
`sagemaker` Python SDK.

It extracts a dataset tarball from the SageMaker train channel, then runs the
existing experiment runner via the ArcD4PG trainer.
"""

import argparse
import logging
import os
import tarfile
from pathlib import Path

from rlarcworld.trainers.arc_d4pg import ArcD4PG
from rlarcworld.utils.config_loader import load_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _maybe_init_distributed() -> tuple[int, int]:
    world_size_env = os.environ.get("WORLD_SIZE")
    if world_size_env is None:
        return 0, 1

    try:
        world_size = int(world_size_env)
    except ValueError:
        return 0, 1

    if world_size <= 1:
        return 0, 1

    import torch
    import torch.distributed as dist

    if dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size


def _maybe_destroy_distributed() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        return


def _resolve_sagemaker_dir(env_key: str, default_path: str) -> Path:
    value = os.environ.get(env_key)
    if not value:
        return Path(default_path)
    return Path(value)


def _find_dataset_tarball(train_channel_dir: Path, tarball_name: str | None) -> Path:
    if tarball_name:
        p = train_channel_dir / tarball_name
        if not p.exists():
            raise FileNotFoundError(f"Dataset tarball not found: {p}")
        return p

    candidates: list[Path] = []
    for suffix in ("*.tar", "*.tar.gz", "*.tgz"):
        candidates.extend(sorted(train_channel_dir.glob(suffix)))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No dataset tarball found in {train_channel_dir}. "
            "Expected *.tar, *.tar.gz, or *.tgz"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple dataset tarballs found in train channel. "
            "Pass --dataset-tarball to disambiguate. Found: "
            + ", ".join(str(c) for c in candidates)
        )
    return candidates[0]


def _extract_tarball(tar_path: Path, extract_dir: Path) -> Path:
    """Extract a tarball safely into an extraction directory."""
    extract_dir.mkdir(parents=True, exist_ok=True)

    marker = extract_dir / ".extracted"
    if marker.exists():
        return extract_dir

    logger.info("Extracting dataset tarball: %s -> %s", tar_path, extract_dir)
    with tarfile.open(tar_path, "r:*") as tf:
        extract_root = extract_dir.resolve()
        for member in tf.getmembers():
            member_path = (extract_dir / member.name).resolve()
            if not str(member_path).startswith(str(extract_root)):
                raise RuntimeError(f"Unsafe path in tarball member: {member.name}")
        try:
            tf.extractall(path=extract_dir, filter="data")
        except TypeError:
            tf.extractall(path=extract_dir)

    marker.write_text(str(tar_path) + "\n")
    return extract_dir


def _load_samples_path_list(extracted_dir: Path, rel_path: str) -> list[str]:
    list_path = extracted_dir / rel_path
    if not list_path.exists():
        raise FileNotFoundError(f"samples path list not found: {list_path}")

    sample_paths: list[str] = []
    with list_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if p.is_absolute():
                sample_paths.append(str(p))
            else:
                sample_paths.append(str((extracted_dir / p).resolve()))
    return sample_paths


def main() -> None:
    """Main SageMaker entrypoint."""
    rank, world_size = _maybe_init_distributed()

    parser = argparse.ArgumentParser(
        description="SageMaker entrypoint for RlArcProject"
    )
    parser.add_argument("--experiment", required=True, type=str)
    parser.add_argument(
        "--experiments-config",
        required=True,
        type=str,
    )
    parser.add_argument("--base-config", default="experiments/configs/cloud_base.yaml")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--dataset-tarball", default=None, type=str)
    parser.add_argument(
        "--samples-path-list",
        default=None,
        type=str,
        help="Relative path to a text file listing JSON sample paths. \
        If omitted, all *.json files under the extracted tarball are used.",
    )

    args = parser.parse_args()

    if world_size > 1 and rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    train_channel_dir = _resolve_sagemaker_dir(
        "SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"
    )
    model_dir = _resolve_sagemaker_dir("SM_MODEL_DIR", "/opt/ml/model")
    output_dir = _resolve_sagemaker_dir("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    tar_path = _find_dataset_tarball(train_channel_dir, args.dataset_tarball)
    extracted_dir = _extract_tarball(tar_path, train_channel_dir / "extracted")

    config = load_experiment(args.experiment, args.experiments_config, args.base_config)

    if args.samples_path_list:
        config.dataset = _load_samples_path_list(extracted_dir, args.samples_path_list)
    else:
        # Use every JSON file in the extracted directory
        config.dataset = [str(p) for p in extracted_dir.rglob("*.json")]
        logger.info("Loaded %d samples from extracted dataset", len(config.dataset))

    # honour checkpoint directory from YAML; fall back to SageMaker default
    if not config.checkpoint.checkpoint_dir:
        config.checkpoint.checkpoint_dir = "/opt/ml/checkpoints"

    if args.seed is not None:
        config.reproducibility.global_seed = args.seed

    logger.info("SageMaker job paths:")
    logger.info("  - train_channel_dir=%s", train_channel_dir)
    logger.info("  - extracted_dir=%s", extracted_dir)
    logger.info("  - model_dir=%s", model_dir)
    logger.info("  - output_dir=%s", output_dir)

    logger.info(
        "Starting experiment=%s epochs=%s world_size=%s",
        args.experiment,
        args.epochs,
        world_size,
    )
    experiment = ArcD4PG(config)
    experiment.train(epochs=args.epochs)

    _maybe_destroy_distributed()


if __name__ == "__main__":
    main()
