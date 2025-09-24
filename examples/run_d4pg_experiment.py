#!/usr/bin/env python3
"""Example script demonstrating how to run a D4PG experiment on ARC tasks."""

import argparse
import logging
from pathlib import Path

from experiments.arc_d4pg import ArcD4PG


def main():
    """Run a D4PG experiment with configurable parameters."""
    parser = argparse.ArgumentParser(description="Run D4PG experiment on ARC tasks")

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--min-train-batch-size",
        type=int,
        default=2,
        help="Minimum effective batch size to perform a training step (default: 2)",
    )
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=100,
        help="Maximum steps per epoch (default: 100)",
    )

    # Environment parameters
    parser.add_argument(
        "--grid-size", type=int, default=30, help="ARC grid size (default: 30)"
    )
    parser.add_argument(
        "--color-values",
        type=int,
        default=10,
        help="Number of color values (default: 10)",
    )

    # Algorithm parameters
    parser.add_argument(
        "--policy-lr",
        type=float,
        default=3e-4,
        help="Actor learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=3e-3,
        help="Critic learning rate (default: 3e-3)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)"
    )

    # System parameters
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (auto-detect if None)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--no-amp", action="store_true", help="Disable automatic mixed precision"
    )

    # Data parameters
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to ARC dataset (use synthetic data if None)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints (default: ./checkpoints)",
    )
    parser.add_argument(
        "--load-checkpoint", type=str, default=None, help="Path to checkpoint to load"
    )
    parser.add_argument(
        "--save-checkpoint", action="store_true", help="Save checkpoint after training"
    )
    parser.add_argument(
        "--save-replay-buffer",
        action="store_true",
        help="Also save replay buffer alongside checkpoint (if available)",
    )
    parser.add_argument(
        "--load-replay-buffer",
        action="store_true",
        help="Also load replay buffer alongside checkpoint (if available)",
    )

    # Evaluation
    parser.add_argument(
        "--evaluate", action="store_true", help="Run evaluation after training"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation (default: 10)",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting D4PG experiment with configuration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    try:
        # Resolve dataset path: if None, use repo test data for a quick demo
        default_ds = Path(__file__).resolve().parents[1] / "tests" / "test_data"
        dataset_path = args.dataset_path or str(default_ds.resolve())

        # Initialize experiment
        experiment = ArcD4PG(
            grid_size=args.grid_size,
            color_values=args.color_values,
            batch_size=args.batch_size,
            min_train_batch_size=args.min_train_batch_size,
            policy_lr=args.policy_lr,
            critic_lr=args.critic_lr,
            gamma=args.gamma,
            max_steps_per_epoch=args.max_steps_per_epoch,
            amp_enabled=not args.no_amp,
            device=args.device,
            seed=args.seed,
            dataset_path=dataset_path,
        )

        # Load checkpoint if specified
        if args.load_checkpoint:
            logger.info(f"Loading checkpoint from {args.load_checkpoint}")
            experiment.load_checkpoint(
                args.load_checkpoint, load_replay_buffer=args.load_replay_buffer
            )

        # Run training
        logger.info("Starting training...")
        experiment.train(epochs=args.epochs)

        # Save checkpoint if requested
        if args.save_checkpoint:
            checkpoint_path = Path(args.checkpoint_dir) / "d4pg_final.pt"
            logger.info(f"Saving checkpoint to {checkpoint_path}")
            experiment.save_checkpoint(
                str(checkpoint_path), save_replay_buffer=args.save_replay_buffer
            )

        # Run evaluation if requested
        if args.evaluate:
            logger.info("Running evaluation...")
            metrics = experiment.evaluate(num_episodes=args.eval_episodes)
            logger.info("Final evaluation metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")

        logger.info("Experiment completed successfully!")

    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
