#!/usr/bin/env python
"""Run ArcD4PG experiments from YAML configurations.

Usage:
    python experiments/run_experiment.py --experiment stable_baseline
    python experiments/run_experiment.py --experiment stable_baseline --epochs 20
    python experiments/run_experiment.py --list
"""

import argparse
import logging

from rlarcworld.trainers.arc_d4pg import ArcD4PG
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
    parser.add_argument(
        "--samples-path-list",
        type=str,
        default=None,
        help="Path to a text file containing sample paths (one per line)",
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
