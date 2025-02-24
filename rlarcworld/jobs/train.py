import os
import argparse
import yaml
import json

from torch.utils.tensorboard import SummaryWriter

from torchrl.data.replay_buffers import (
    TensorDictReplayBuffer,
    LazyMemmapStorage,
    PrioritizedSampler,
)
from torch.utils.data import DataLoader
from rlarcworld.arc_dataset import ArcDataset, ArcSampleTransformer
from rlarcworld.algorithms.d4pg import D4PG
from rlarcworld.enviroments.arc_batch_grid_env import ArcBatchGridEnv
from rlarcworld.agent.actor import ArcActorNetwork
from rlarcworld.agent.critic import ArcCriticNetwork
from rlarcworld.enviroments.wrappers.rewards import PixelAwareRewardWrapper
from rlarcworld.utils import BetaScheduler, enable_cuda
import torch
import numpy as np

import logging


logger = logging.getLogger(__name__)


def load_args_from_yaml(file_path, key):
    with open(file_path, "r") as f:
        args_dict = yaml.safe_load(f)
    return args_dict[key]


def check_args(args):
    assert args["grid_size"] == 30
    assert args["color_values"] == 11
    if args["n_steps"] > 1:
        assert args["num_atoms"].get("n_reward") is not None
        assert args["v_min"].get("n_reward") is not None
        assert args["v_max"].get("n_reward") is not None
    else:
        assert args["num_atoms"].get("n_reward") is None
        assert args["v_min"].get("n_reward") is None
        assert args["v_max"].get("n_reward") is None
    assert args["num_atoms"].get("binary") is not None
    assert args["v_min"].get("binary") is not None
    assert args["v_max"].get("binary") is not None
    assert args["num_atoms"].get("pixel_wise") is not None
    assert args["v_min"].get("pixel_wise") is not None
    assert args["v_max"].get("pixel_wise") is not None


def train_d4pg(config_key, path_uri, training_data_uri, validation_data_uri, args):
    logger.info(f"D4PG train configuration: \n{json.dumps(args, indent=4)}")
    device_configs = enable_cuda()
    logger.info(f"Device configurations: \n{json.dumps(device_configs, indent=4)}")
    check_args(args)

    # Tensorboard
    log_dir = os.path.join(path_uri, "tensorboard_runs", config_key)
    tb_writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard log directory: {log_dir}")

    # Saved models
    models_path = os.path.join(path_uri, "saved_models", config_key)
    logger.info(f"Saved models directory: {models_path}")

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
    logger.info("Load Datasets from {}".format(training_data_uri))
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

    logger.info("Load Datasets from {}".format(validation_data_uri))
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
        max_penality=args["max_penality"],
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
        save_path=models_path,
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
    args = parser.parse_args()

    assert args.storage_uri is not None
    assert args.training_data_uri is not None
    assert args.validation_data_uri is not None

    # Set logging level
    logging.basicConfig(level=args.log_level)

    config_args = load_args_from_yaml(args.config_file, args.config_key)
    train_d4pg(
        config_key=args.config_key,
        path_uri=args.storage_uri,
        training_data_uri=args.training_data_uri,
        validation_data_uri=args.validation_data_uri,
        args=config_args,
    )
