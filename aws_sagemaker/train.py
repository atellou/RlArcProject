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
import re
import signal
import tarfile
from pathlib import Path

from rlarcworld.trainers.arc_d4pg import ArcD4PG
from rlarcworld.utils.config_loader import load_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _find_latest_checkpoint(checkpoint_dir: str | Path) -> str | None:
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None

    candidates: list[Path] = []
    for p in ckpt_dir.rglob("*.pth"):
        if p.is_file():
            candidates.append(p)
    if not candidates:
        return None

    spot_candidates: list[Path] = []
    periodic_candidates: list[Path] = []
    best_candidates: list[Path] = []
    for cand in candidates:
        cand_str = str(cand)
        if "checkpoints/spot" in cand_str or "spot" in cand_str:
            spot_candidates.append(cand)
        elif "checkpoints/periodic" in cand_str or "periodic" in cand_str:
            periodic_candidates.append(cand)
        elif "checkpoints/best" in cand_str or "best" in cand_str:
            best_candidates.append(cand)

    iter_re = re.compile(r"(?:^|[_-])i_(\d{6})(?:$|\D)")
    ts_re = re.compile(r"(\d{8}-\d{6})")

    def score(candidate: Path) -> tuple[int, int]:
        # Prefer timestamp in the name (spot / success / failure).
        m_ts = ts_re.search(candidate.name)
        if m_ts:
            try:
                ts_val = int(m_ts.group(1).replace("-", ""))
            except ValueError:
                ts_val = 0
            return (1, ts_val)

        # Otherwise try iteration if present (periodic / best).
        m_iter = iter_re.search(candidate.name)
        if m_iter:
            return (2, int(m_iter.group(1)))

        # Final fallback: filesystem mtime.
        try:
            mtime_val = int(candidate.stat().st_mtime)
        except OSError:
            mtime_val = 0
        return (0, mtime_val)

    if spot_candidates:
        return str(max(spot_candidates, key=score))
    if periodic_candidates:
        return str(max(periodic_candidates, key=score))
    if best_candidates:
        return str(max(best_candidates, key=score))

    return None


def _flush_logs() -> None:
    root = logging.getLogger()
    for h in root.handlers:
        try:
            h.flush()
        except Exception:
            pass


def _stderr_unbuffered(msg: str) -> None:
    try:
        os.write(2, (msg + "\n").encode("utf-8", errors="replace"))
    except Exception:
        return


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

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        # torch>=2.2 supports device_id to silence barrier() warnings and avoid hangs
        try:
            dist.init_process_group(
                backend=backend, init_method="env://", device_id=local_rank
            )
        except TypeError:
            dist.init_process_group(backend=backend, init_method="env://")

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

    # Auto-resume if SageMaker has restored checkpoints from S3.
    latest_ckpt = _find_latest_checkpoint(config.checkpoint.checkpoint_dir)
    if latest_ckpt is not None:
        logger.info("Resuming from checkpoint: %s", latest_ckpt)
        experiment.load_checkpoint(latest_ckpt)

    # Best-effort spot interruption handling.
    # SageMaker typically sends SIGTERM before stopping the container.
    def _sigterm_handler(signum: int, _frame) -> None:
        try:
            _stderr_unbuffered(
                f"[train.py] received signal {signum} (rank={rank}); attempting final checkpoint"  # noqa: E501
            )
            logger.warning(
                "Received signal %s; saving final checkpoint (best-effort)...",
                signum,
            )
            _flush_logs()

            # Only rank0 should attempt writing checkpoints to avoid corruption.
            if rank == 0:
                experiment.save_final_checkpoint(failed=False, reason="spot")
                logger.warning("Final checkpoint saved after SIGTERM")
            else:
                logger.warning("Non-rank0 process: skipping SIGTERM checkpoint")
            _flush_logs()
        except Exception as e:
            logger.exception("Failed to save SIGTERM checkpoint: %s", e)
            _flush_logs()
        finally:
            _maybe_destroy_distributed()
            raise SystemExit(0)

    logger.info("Registering SIGTERM handler (rank=%s)", rank)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    experiment.train(epochs=args.epochs)

    _maybe_destroy_distributed()


if __name__ == "__main__":
    main()
