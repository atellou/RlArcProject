import argparse
import json
import logging
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchrl.data.replay_buffers import (
    LazyMemmapStorage,
    PrioritizedSampler,
    TensorDictReplayBuffer,
)

from rlarcworld.agent.actor import ArcActorNetwork
from rlarcworld.agent.critic import ArcCriticNetwork
from rlarcworld.algorithms.d4pg import D4PG
from rlarcworld.arc_dataset import ArcDataset, ArcSampleTransformer
from rlarcworld.environments.arc_batch_grid_env import ArcBatchGridEnv
from rlarcworld.environments.wrappers.rewards import PixelAwareRewardWrapper
from rlarcworld.utils.beta_scheduler import BetaScheduler
from rlarcworld.utils.configure_logger import configure_logger
from rlarcworld.utils.device import enable_cuda

logger = logging.getLogger(__name__)


def load_args_from_yaml(file_path: str, key: str) -> dict:
    with open(file_path) as f:
        args_dict = yaml.safe_load(f)
    return args_dict[key]


def check_args(args: dict) -> list[str]:
    """Validate the arguments and return a list of error messages for any issues found.

    Args:
        args: Dictionary of arguments to validate

    Returns:
        List of error messages, empty if all validations pass
    """

    should_have = {
        "grid_size",
        "color_values",
        "n_steps",
        "num_atoms",
        "v_min",
        "v_max",
    }
    errors = []

    args_keys = set(args.keys())
    missing_keys = should_have - args_keys
    if missing_keys:
        errors.append(f"Missing required fields: {missing_keys}")

    if args.get("n_steps") is not None:
        if args["n_steps"] > 1:
            if (
                args["num_atoms"].get("n_reward") is None
                or args["v_min"].get("n_reward") is None
                or args["v_max"].get("n_reward") is None
            ):
                errors.append(
                    "num_atoms.n_reward, v_min.n_reward, and v_max.n_reward are required when n_steps > 1"
                )
        else:
            if (
                args["num_atoms"].get("n_reward") is not None
                or args["v_min"].get("n_reward") is not None
                or args["v_max"].get("n_reward") is not None
            ):
                errors.append(
                    "num_atoms.n_reward, v_min.n_reward, and v_max.n_reward should not be specified when n_steps is None"
                )
    if errors:
        raise ValueError("Invalid arguments: \n" + "\n".join(errors))
    if (
        args["num_atoms"].get("binary") is not None
        or args["v_min"].get("binary") is not None
        or args["v_max"].get("binary") is not None
    ):
        errors.append(
            "num_atoms.binary, v_min.binary, and v_max.binary should not be specified when n_steps is None"
        )
    if (
        args["num_atoms"].get("pixel_wise") is not None
        or args["v_min"].get("pixel_wise") is not None
        or args["v_max"].get("pixel_wise") is not None
    ):
        errors.append(
            "num_atoms.pixel_wise, v_min.pixel_wise, and v_max.pixel_wise should not be specified when n_steps is None"
        )
    if errors:
        raise ValueError("Invalid arguments: \n" + "\n".join(errors))


def train_d4pg(
    config_key: str,
    training_data_uri: str,
    validation_data_uri: str,
    models_dir: str,
    checkpoint_dir: str,
    tensorboard_log_dir: str,
    args: dict,
):
    """
    Train a D4PG agent on the ARC environment.

    Args:
        config_key (str): Key for the configuration in the YAML file
        training_data_uri (str): Path to the training data directory
        validation_data_uri (str): Path to the validation data directory
        models_dir (str): Directory to save trained models
        checkpoint_dir (str): Directory to save training checkpoints
        tensorboard_log_dir (str): Directory for TensorBoard logs
        args (dict): Training arguments

    """
    logger.info(f"D4PG train configuration: \n{json.dumps(args, indent=4)}")
    device_configs = enable_cuda()
    logger.info(f"Device configurations: \n{json.dumps(device_configs, indent=4)}")
    check_args(args)

    # Saved models
    models_dir = os.path.join(models_dir, config_key)
    logger.info(f"Saved models directory: {models_dir}")

    # Checkpoints
    checkpoint_dir = os.path.join(checkpoint_dir, config_key)
    checkpoint_frequency = args["checkpoint_frequency"]
    logger.info(f"Checkpoints directory: {checkpoint_dir}")
    logger.info(f"Checkpoint frequency: {checkpoint_frequency}")

    # Tensorboard
    tensorboard_log_dir = os.path.join(tensorboard_log_dir, config_key)
    tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)
    logger.info(f"TensorBoard log directory: {tensorboard_log_dir}")

    # Set the seed
    seed = np.random.randint(0, 10000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device_configs["device"] == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    tb_writer.add_text(f"Seed for experiment {config_key}:", str(seed))
    logger.info(f"Seed for experiment {config_key}: {seed}")

    # Load Datasets
    logger.info(f"Load Datasets from {training_data_uri}")
    dataset = ArcDataset(
        arc_dataset_dir=training_data_uri,
        keep_in_memory=False,
        transform=ArcSampleTransformer(
            (args["grid_size"], args["grid_size"]), examples_stack_dim=10
        ),
    )
    train_samples = DataLoader(
        dataset=dataset,
        batch_size=args["loader_batch_size"],
        worker_init_fn=lambda worker_id: torch.manual_seed(seed + worker_id),
    )

    logger.info(f"Load Datasets from {validation_data_uri}")
    dataset = ArcDataset(
        arc_dataset_dir=validation_data_uri,
        keep_in_memory=False,
        transform=ArcSampleTransformer(
            (args["grid_size"], args["grid_size"]), examples_stack_dim=10
        ),
    )
    validation_samples = DataLoader(
        dataset=dataset,
        batch_size=args["loader_batch_size"],
        worker_init_fn=lambda worker_id: torch.manual_seed(seed + worker_id),
    )

    # Set up environment
    env = ArcBatchGridEnv(
        grid_size=args["grid_size"],
        color_values=args["color_values"],
        n_steps=args["n_steps"],
        gamma=args["gamma"],
        device=device_configs["device"],
    )
    env = PixelAwareRewardWrapper(
        env,
        max_penalty=args["max_penalty"],
        n_steps=args["n_steps"],
        gamma=args["gamma"],
        apply_clamp=True,
        v_min=args["v_min"]["pixel_wise"],
        v_max=args["v_max"]["pixel_wise"],
    )

    # Set up actor and critic networks
    actor = ArcActorNetwork(
        grid_size=args["grid_size"],
        color_values=args["color_values"],
        config=device_configs,
    )
    critic = ArcCriticNetwork(
        grid_size=args["grid_size"],
        color_values=args["color_values"],
        num_atoms=args["num_atoms"],
        v_min=args["v_min"],
        v_max=args["v_max"],
        config=device_configs,
    )

    # Set up replay buffer
    beta_scheduler = BetaScheduler(start=args["beta"], end=1, steps=args["max_steps"])
    sampler = PrioritizedSampler(
        args["replay_buffer_size"],
        alpha=args["alpha"],
        beta=args["beta"],
    )
    storage = LazyMemmapStorage(args["replay_buffer_size"])
    rb = TensorDictReplayBuffer(storage=storage, sampler=sampler)

    # Set up D4PG algorithm
    d4pg = D4PG(
        env=env,
        train_samples=train_samples,
        validation_samples=validation_samples,
        actor=actor,
        critic=critic,
        batch_size=args["batch_size"],
        replay_buffer=rb,
        warmup_buffer_ratio=args["warmup_buffer_ratio"],
        target_update_frequency=args["target_update_frequency"],
        beta_scheduler=beta_scheduler,
        n_steps=args["n_steps"],
        gamma=args["gamma"],
        policy_lr=args["policy_lr"],
        critic_lr=args["critic_lr"],
        lr_scheduler_kwargs=args["lr_scheduler"],
        tau=args["tau"],
        tb_writer=tb_writer,
        save_path=models_dir,
        config=device_configs,
    )

    # Set up training loop
    d4pg.fit(
        epochs=args["max_epochs"],
        max_steps=args["max_steps"],
        validation_steps_frequency=args["validation_steps_frequency"],
        validation_steps_per_train_step=args["validation_steps_per_train_step"],
        validation_steps_per_episode=args["validation_steps_per_episode"],
        logger_frequency=args["logger_frequency"],
        checkpoint_frequency=checkpoint_frequency,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train D4PG algorithm")
    parser.add_argument(
        "--config_file",
        type=str,
        default="rlarcworld/jobs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--config_key",
        type=str,
        default="test",
        help="Key to use in the YAML configuration file",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="WARNING",
        help="Logging level",
    )
    parser.add_argument(
        "--log_convention",
        type=str,
        default="json",
        help="Logging convention",
    )
    parser.add_argument(
        "--storage_uri",
        type=str,
        default=os.environ.get("STORAGE_URI"),
        help="Storage URI",
    )
    parser.add_argument(
        "--training_data_uri",
        type=str,
        default=os.environ.get("AIP_TRAINING_DATA_URI"),
        help="Training data URI",
    )
    parser.add_argument(
        "--validation_data_uri",
        type=str,
        default=os.environ.get("AIP_VALIDATION_DATA_URI"),
        help="Validation data URI",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default=os.environ.get("AIP_MODEL_DIR"),
        help="Directory to save models",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.environ.get("AIP_CHECKPOINT_DIR"),
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--tensorboard_log_dir",
        type=str,
        default=os.environ.get("AIP_TENSORBOARD_LOG_DIR"),
        help="Directory to save tensorboard logs",
    )
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        default=-1,
        help="Checkpoint frequency",
    )

    args = parser.parse_args()

    checkpoint_frequency_default = 10000
    assert args.storage_uri is not None
    assert args.training_data_uri is not None
    assert args.validation_data_uri is not None

    if args.models_dir is None:
        args.models_dir = os.path.join(args.storage_uri, "models")
    if args.checkpoint_dir is not None and args.checkpoint_frequency == -1:
        logger.warning(
            "Checkpoint frequency set to default value of {}".format(
                checkpoint_frequency_default
            )
        )
        args.checkpoint_frequency = checkpoint_frequency_default
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(args.storage_uri, "checkpoints")
    if args.tensorboard_log_dir is None:
        args.tensorboard_log_dir = os.path.join(args.storage_uri, "tensorboard_logs")

    # Set logging level
    if args.log_convention == "json":
        configure_logger(level=args.log_level)
    else:
        logging.basicConfig(level=args.log_level)

    config_args = load_args_from_yaml(args.config_file, args.config_key)
    train_d4pg(
        config_key=args.config_key,
        training_data_uri=args.training_data_uri,
        validation_data_uri=args.validation_data_uri,
        models_dir=args.models_dir,
        checkpoint_dir=args.checkpoint_dir,
        tensorboard_log_dir=args.tensorboard_log_dir,
        args=config_args,
    )
