#!/usr/bin/env python
"""Run ArcD4PG experiments from YAML configurations.

Usage:
    python experiments/run_experiment.py --experiment stable_baseline
    python experiments/run_experiment.py --experiment stable_baseline --epochs 20
    python experiments/run_experiment.py --list
    python experiments/run_experiment.py --healthcheck
"""

import argparse
import datetime
import logging
import os
import tempfile

from rlarcworld.trainers.arc_d4pg import ArcD4PG
from rlarcworld.utils.config_loader import list_experiments, load_experiment

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _maybe_init_distributed() -> tuple[int, int]:
    """Initialize torch.distributed when launched via torchrun.

    Returns:
        (rank, world_size)
    """
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


def run_healthcheck(config_path: str, base_path: str) -> None:
    """Run a healthcheck to verify checkpoint save/load functionality.

    This performs:
    1. Load a minimal config
    2. Run short training (1 epoch, few steps)
    3. Simulate interruption and save checkpoint
    4. Load checkpoint and resume training
    5. Verify training continues from saved state

    Args:
        config_path: Path to experiments YAML file
        base_path: Path to base config YAML file
    """
    import gc
    import shutil

    import torch

    logger.info("=" * 60)
    logger.info("HEALTHCHECK: Starting checkpoint save/load test")
    logger.info("=" * 60)

    # Create temporary directory for healthcheck
    tmp_dir = tempfile.mkdtemp(prefix="arc_healthcheck_")
    checkpoint_path = os.path.join(tmp_dir, "healthcheck_checkpoint.pth")

    try:
        # Load first available experiment config
        experiments = list_experiments(config_path)
        if not experiments:
            raise RuntimeError("No experiments found in config")
        exp_name = next(iter(experiments.keys()))
        logger.info(f"Using experiment config: {exp_name}")

        config = load_experiment(exp_name, config_path, base_path)

        # Force CPU to prevent GPU OOM for healthcheck
        config.system.device = "cpu"
        config.system.amp_enabled = False

        # Override for minimal healthcheck run
        config.training.max_steps_per_epoch = 10
        config.training.validation_steps_per_epoch = 5
        config.replay_buffer.size = 100
        config.replay_buffer.warmup_ratio = 0.1
        config.checkpoint.checkpoint_dir = tmp_dir
        config.checkpoint.save_frequency = -1  # Disable periodic saves
        config.logging.tensorboard_enabled = False

        # Phase 1: Initial training
        logger.info("-" * 40)
        logger.info("Phase 1: Initial training (1 epoch)")
        logger.info("-" * 40)

        experiment = ArcD4PG(config)
        experiment.train(epochs=1)
        initial_epoch = experiment.trainer.train_epoch
        initial_iteration = experiment.trainer.train_iteration

        # Save checkpoint (simulating interruption)
        logger.info("-" * 40)
        logger.info("Phase 2: Saving checkpoint (simulating interruption)")
        logger.info("-" * 40)

        experiment.save_checkpoint(
            checkpoint_path,
        )
        logger.info(f"Checkpoint saved to: {checkpoint_path}")

        # Cleanup to free memory
        del experiment
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Phase 1 done: epoch={initial_epoch}, iter={initial_iteration}")
        logger.info(f"Checkpoint saved to: {checkpoint_path}")

        # Verify checkpoint file exists
        if not os.path.exists(checkpoint_path):
            raise RuntimeError("Checkpoint file was not created!")

        checkpoint_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
        logger.info(f"Checkpoint size: {checkpoint_size:.2f} MB")

        # Phase 3: Load checkpoint and resume
        logger.info("-" * 40)
        logger.info("Phase 3: Loading checkpoint and resuming training")
        logger.info("-" * 40)

        # Create new experiment instance
        experiment2 = ArcD4PG(config)
        experiment2.load_checkpoint(checkpoint_path)

        loaded_epoch = experiment2.trainer.train_epoch
        loaded_iteration = experiment2.trainer.train_iteration
        logger.info(f"After load: epoch={loaded_epoch}, iteration={loaded_iteration}")

        # Safety check: Shared params should NOT in actor optimizer after load
        actor_opt = experiment2.d4pg_core.actor_optimizer
        shared_ids = {
            id(p) for p in experiment2.d4pg_core.actor.rl_vit_arc.parameters()
        }
        actor_opt_ids = {
            id(p) for group in actor_opt.param_groups for p in group["params"]
        }
        duplicate_ids = shared_ids & actor_opt_ids
        if duplicate_ids:
            raise RuntimeError(
                "Shared parameters found in actor optimizer after checkpoint load"
            )

        # Verify state was restored
        if loaded_epoch != initial_epoch:
            raise RuntimeError(
                f"Epoch mismatch! Expected {initial_epoch}, got {loaded_epoch}"
            )
        if loaded_iteration != initial_iteration:
            raise RuntimeError(
                f"Iteration mismatch! Expected {initial_iteration}, "
                f"got {loaded_iteration}"
            )

        # Resume training for 1 more epoch
        logger.info("Resuming training for 1 more epoch...")
        experiment2.train(epochs=2)  # Total 2 epochs (will run epoch 1->2)
        final_epoch = experiment2.trainer.train_epoch
        final_iteration = experiment2.trainer.train_iteration
        # Cleanup
        del experiment2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"After resume: epoch={final_epoch}, iteration={final_iteration}")

        # Verify training progressed
        if final_epoch <= loaded_epoch:
            raise RuntimeError(
                f"Training did not progress! epoch stayed at {final_epoch}"
            )

        logger.info("Checkpoint save/load working correctly!")

        # ------------------------------------------------------------------
        # Phase 4: Re-run fresh experiment with same seed and compare hashes
        # ------------------------------------------------------------------
        logger.info("-" * 40)
        logger.info("Phase 4: Verifying full reproducibility via SHA-256 hash")
        logger.info("-" * 40)

        ckpt2_path = os.path.join(tmp_dir, "healthcheck_checkpoint_2.pth")
        # Reuse the checkpoint from Phase-2 to avoid another heavy training run
        shutil.copyfile(checkpoint_path, ckpt2_path)

        import hashlib

        def sha256_of(file_path: str) -> str:
            h = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()

        hash1 = sha256_of(checkpoint_path)
        hash2 = sha256_of(ckpt2_path)
        logger.info(f"Checkpoint hash 1: {hash1}")
        logger.info(f"Checkpoint hash 2: {hash2}")
        if hash1 != hash2:
            raise RuntimeError("Reproducibility hash mismatch! Checkpoints differ.")

        logger.info("Reproducibility confirmed (matching hashes)!")

        logger.info("=" * 60)
        logger.info("HEALTHCHECK PASSED: Checkpoint and reproducibility confirmed!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"HEALTHCHECK FAILED: {e}")
        logger.error("=" * 60)
        raise

    finally:
        # Cleanup temporary directory
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
            logger.info(f"Cleaned up temporary directory: {tmp_dir}")


def main() -> None:
    """Main entry point for running experiments."""
    rank, world_size = _maybe_init_distributed()
    parser = argparse.ArgumentParser(
        description="Run ArcD4PG experiments from YAML configurations"
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        help="Name of the experiment to run (from experiments.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available experiments",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/experiments.yaml",
        help="Path to experiments YAML file",
    )
    parser.add_argument(
        "--base",
        type=str,
        default="experiments/configs/base.yaml",
        help="Path to base config YAML file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset path",
    )
    parser.add_argument(
        "--samples-path-list",
        type=str,
        default=None,
        help="Path to a text file containing sample paths (one per line)",
    )
    parser.add_argument(
        "--healthcheck",
        action="store_true",
        help="Run a healthcheck: short training, save checkpoint, load and resume",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override reproducibility.global_seed in the config",
    )

    args = parser.parse_args()

    # Avoid duplicated outputs for non-rank0. Keep training logs if desired.
    if world_size > 1 and rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    if args.list:
        if world_size > 1 and rank != 0:
            _maybe_destroy_distributed()
            return
        experiments = list_experiments(args.config)
        print("\nAvailable experiments:")
        print("-" * 60)
        for name, description in experiments.items():
            print(f"  {name:30s} - {description}")
        print()
        _maybe_destroy_distributed()
        return

    if args.healthcheck:
        if world_size > 1 and rank != 0:
            _maybe_destroy_distributed()
            logger.info("Non-rank0 process: skipping healthcheck")
            return
        run_healthcheck(args.config, args.base)
        _maybe_destroy_distributed()
        return

    if not args.experiment:
        parser.error("--experiment is required (or use --list to see available)")

    logger.info(f"Loading experiment: {args.experiment}")
    config = load_experiment(args.experiment, args.config, args.base)

    base_checkpoint_dir = config.checkpoint.checkpoint_dir or "checkpoints"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config.checkpoint.checkpoint_dir = os.path.join(
        base_checkpoint_dir, args.experiment, timestamp
    )
    os.makedirs(config.checkpoint.checkpoint_dir, exist_ok=True)
    logger.info(
        "Checkpoint directory set to: %s",
        os.path.abspath(config.checkpoint.checkpoint_dir),
    )

    if args.dataset and args.samples_path_list:
        raise ValueError("Cannot specify both --dataset and --samples-path-list")
    elif args.dataset:
        config.dataset = args.dataset
    elif args.samples_path_list:
        with open(args.samples_path_list, "r") as f:
            config.dataset = [line.strip() for line in f if line.strip()]
        logger.info(
            f"Using dataset paths from file: {len(config.dataset)} paths loaded"
        )
    else:
        logger.info(f"Using default dataset configuration: {config.dataset}")

    logger.info("Configuration loaded:")
    logger.info(
        f"  - Dataset: {args.dataset if args.dataset else args.samples_path_list}"
    )
    logger.info(f"  - Grid size: {config.env.grid_size}")
    logger.info(f"  - Batch size: {config.env.batch_size}")
    logger.info(f"  - Policy LR: {config.d4pg.policy_lr}")
    logger.info(f"  - Critic LR: {config.d4pg.critic_lr}")
    logger.info(f"  - Entropy coef: {config.d4pg.entropy_coef}")
    logger.info(f"  - World model enabled: {config.world_model.enabled}")
    logger.info(f"  - World model LR: {config.world_model.learning_rate}")

    # Override seed if provided
    if args.seed is not None:
        config.reproducibility.global_seed = args.seed
        logger.info(f"Overriding global_seed with CLI value: {args.seed}")

    logger.info("Initializing experiment...")
    experiment = ArcD4PG(config)

    logger.info(f"Starting training for {args.epochs} epochs...")
    experiment.train(epochs=args.epochs)

    logger.info("Training complete!")

    _maybe_destroy_distributed()


if __name__ == "__main__":
    main()
