#!/usr/bin/env python
"""Run ArcD4PG experiments from YAML configurations.

Usage:
    python experiments/run_experiment.py --experiment stable_baseline
    python experiments/run_experiment.py --experiment stable_baseline --epochs 20
    python experiments/run_experiment.py --list
    python experiments/run_experiment.py --healthcheck
"""

import argparse
import logging
import os
import tempfile

from rlarcworld.trainers.arc_d4pg import ArcD4PG
from rlarcworld.utils.config_loader import list_experiments, load_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
    import shutil

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
        logger.info(f"Phase 1 done: epoch={initial_epoch}, iter={initial_iteration}")

        # Save checkpoint (simulating interruption)
        logger.info("-" * 40)
        logger.info("Phase 2: Saving checkpoint (simulating interruption)")
        logger.info("-" * 40)

        experiment.save_checkpoint(
            checkpoint_path,
            include_core=True,
            include_trainer=True,
            include_env=True,
            include_stepper=True,
            include_rng=True,
            include_config=True,
            save_replay_buffer=True,
        )
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
        logger.info(f"After resume: epoch={final_epoch}, iteration={final_iteration}")

        # Verify training progressed
        if final_epoch <= loaded_epoch:
            raise RuntimeError(
                f"Training did not progress! epoch stayed at {final_epoch}"
            )

        logger.info("=" * 60)
        logger.info("HEALTHCHECK PASSED: Checkpoint save/load working correctly!")
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

    args = parser.parse_args()

    if args.list:
        experiments = list_experiments(args.config)
        print("\nAvailable experiments:")
        print("-" * 60)
        for name, description in experiments.items():
            print(f"  {name:30s} - {description}")
        print()
        return

    if args.healthcheck:
        run_healthcheck(args.config, args.base)
        return

    if not args.experiment:
        parser.error("--experiment is required (or use --list to see available)")

    logger.info(f"Loading experiment: {args.experiment}")
    config = load_experiment(args.experiment, args.config, args.base)

    if args.dataset and args.samples_path_list:
        raise ValueError("Cannot specify both --dataset and --samples-path-list")
    elif args.dataset:
        config.dataset = args.dataset
    elif args.samples_path_list:
        with open(args.samples_path_list, "r") as f:
            paths = [line.strip() for line in f if line.strip()]
        config.dataset = paths
        logger.info(f"Using dataset paths from file: {len(paths)} paths loaded")
    else:
        logger.info(f"Using default dataset configuration: {config.dataset}")

    logger.info("Configuration loaded:")
    logger.info(f"  - Dataset: {config.dataset}")
    logger.info(f"  - Grid size: {config.env.grid_size}")
    logger.info(f"  - Batch size: {config.env.batch_size}")
    logger.info(f"  - Policy LR: {config.d4pg.policy_lr}")
    logger.info(f"  - Critic LR: {config.d4pg.critic_lr}")
    logger.info(f"  - Entropy coef: {config.d4pg.entropy_coef}")
    logger.info(f"  - World model enabled: {config.world_model.enabled}")
    logger.info(f"  - World model LR: {config.world_model.learning_rate}")

    logger.info("Initializing experiment...")
    experiment = ArcD4PG(config)

    logger.info(f"Starting training for {args.epochs} epochs...")
    experiment.train(epochs=args.epochs)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
