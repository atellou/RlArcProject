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
from rlarcworld.utils import BetaScheduler

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


def train_d4pg(config_key, args):
    logger.info(f"D4PG train configuration: \n{json.dumps(args, indent=4)}")
    check_args(args)
    path_uri = os.environ["GCP_STORAGE_URI"]
    logger.info(
        f"TensorBoard log directory: {os.path.join(path_uri, 'tensorboard_runs', config_key)}"
    )
    tb_writer = SummaryWriter(
        log_dir=os.path.join(path_uri, "tensorboard_runs", config_key)
    )
    logger.info(
        f"Saved models directory: {os.path.join(path_uri, 'saved_models', config_key)}"
    )
    models_path = os.path.join(path_uri, "saved_models", config_key)
    # Load Datasets
    dataset = ArcDataset(
        arc_dataset_dir=os.environ["AIP_TRAINING_DATA_URI"],
        keep_in_memory=False,
        transform=ArcSampleTransformer(
            (args["grid_size"], args["grid_size"]), examples_stack_dim=10
        ),
    )
    train_samples = DataLoader(dataset=dataset, batch_size=args["batch_size"])

    dataset = ArcDataset(
        arc_dataset_dir=os.environ["AIP_VALIDATION_DATA_URI"],
        keep_in_memory=False,
        transform=ArcSampleTransformer(
            (args["grid_size"], args["grid_size"]), examples_stack_dim=10
        ),
    )
    validation_samples = DataLoader(dataset=dataset, batch_size=args["batch_size"])

    # Set up environment
    env = ArcBatchGridEnv(
        grid_size=args["grid_size"],
        color_values=args["color_values"],
        n_steps=args["n_steps"],
        gamma=args["gamma"],
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
        grid_size=args["grid_size"], color_values=args["color_values"]
    )
    critic = ArcCriticNetwork(
        grid_size=args["grid_size"],
        color_values=args["color_values"],
        num_atoms=args["num_atoms"],
        v_min=args["v_min"],
        v_max=args["v_max"],
    )

    # Set up replay buffer
    beta_scheduler = BetaScheduler(start=args["beta"], end=1, steps=args["max_steps"])
    sampler = PrioritizedSampler(
        args["replay_buffer_size"],
        alpha=args["alpha"],
        beta_scheduler=beta_scheduler.beta_scheduler,
    )
    storage = LazyMemmapStorage(args["replay_buffer_size"])
    rb = TensorDictReplayBuffer(storage=storage, sampler=sampler)

    # Set up D4PG algorithm
    policy_lr = lambda t: args["policy_lr"] * (0.1 ** (t // 100000))
    critic_lr = lambda t: args["critic_lr"] * (0.1 ** (t // 100000))
    SummaryWriter(log_dir="runs/test_validation_d4pg")

    d4pg = D4PG(
        env=env,
        train_samples=train_samples,
        validation_samples=validation_samples,
        actor=actor,
        critic=critic,
        batch_size=args["batch_size"],
        replay_buffer=rb,
        target_update_frequency=args["target_update_frequency"],
        n_steps=args["n_steps"],
        gamma=args["gamma"],
        learning_rate_actor=policy_lr,
        learning_rate_critic=critic_lr,
        entropy_tau=args["entropy_tau"],
        clip_grad_norm=args["clip_grad_norm"],
        num_samples=args["num_samples"],
        num_workers=args["num_workers"],
        tb_writer=tb_writer,
        device=args["device"],
        seed=args["seed"],
        save_path=models_path,
    )

    # Set up training loop
    d4pg.fit(
        epoch=args["max_epochs"],
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
    args = parser.parse_args()

    # Set logging level
    logging.basicConfig(level=args.log_level)

    # Load configuration
    config_args = load_args_from_yaml(args.config_file, args.config_key)
    train_d4pg(args.config_key, config_args)
