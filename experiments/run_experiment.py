#!/usr/bin/env python
"""Run ArcD4PG experiments from YAML configurations.

Usage:
    python experiments/run_experiment.py --experiment stable_baseline
    python experiments/run_experiment.py --experiment stable_baseline --epochs 20
    python experiments/run_experiment.py --list
"""

import argparse
import logging

from rlarcworld.experiments.arc_d4pg import ArcD4PG
from rlarcworld.utils.config_loader import list_experiments, load_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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

    args = parser.parse_args()

    if args.list:
        experiments = list_experiments(args.config)
        print("\nAvailable experiments:")
        print("-" * 60)
        for name, description in experiments.items():
            print(f"  {name:30s} - {description}")
        print()
        return

    if not args.experiment:
        parser.error("--experiment is required (or use --list to see available)")

    logger.info(f"Loading experiment: {args.experiment}")
    config = load_experiment(args.experiment, args.config, args.base)

    if args.dataset:
        config.dataset_path = args.dataset

    logger.info("Configuration loaded:")
    logger.info(f"  - Dataset: {config.dataset_path}")
    logger.info(f"  - Grid size: {config.env.grid_size}")
    logger.info(f"  - Batch size: {config.env.batch_size}")
    logger.info(f"  - Policy LR: {config.d4pg.policy_lr}")
    logger.info(f"  - Critic LR: {config.d4pg.critic_lr}")
    logger.info(f"  - Entropy coef: {config.d4pg.entropy_coef}")
    logger.info(f"  - World model enabled: {config.world_model.enabled}")

    logger.info("Initializing experiment...")
    experiment = ArcD4PG(config)

    logger.info(f"Starting training for {args.epochs} epochs...")
    experiment.train(epochs=args.epochs)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
