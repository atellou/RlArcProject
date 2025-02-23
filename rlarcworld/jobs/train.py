# train_d4pg.py

import argparse
import yaml

from torchrl.data.replay_buffers import (
    TensorDictReplayBuffer,
    LazyMemmapStorage,
    PrioritizedSampler,
)
from torchrl.data import DataLoader
from rlarcworld.algorithms.d4pg import D4PG
from rlarcworld.enviroments.arc_batch_grid_env import ArcBatchGridEnv
from rlarcworld.agent.actor import ArcActorNetwork
from rlarcworld.agent.critic import ArcCriticNetwork
from rlarcworld.enviroments.wrappers.rewards import PixelAwareRewardWrapper


def load_args_from_yaml(file_path, key):
    with open(file_path, "r") as f:
        args_dict = yaml.safe_load(f)
    return args_dict[key]


def check_args(args):
    assert args.grid_size == 30
    assert args.color_values == 11
    if args.n_steps > 1:
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


def train_d4pg(args):
    check_args(args)
    # Set up environment
    env = ArcBatchGridEnv(
        grid_size=args.grid_size,
        color_values=args.color_values,
        n_steps=args.n_steps,
        gamma=args.gamma,
    )
    env = PixelAwareRewardWrapper(
        env,
        max_penality=args.max_penality,
        n_steps=args.n_steps,
        gamma=args.gamma,
        apply_clamp=args.apply_clamp,
        v_min=args.v_min,
        v_max=args.v_max,
    )

    # Set up actor and critic networks
    actor = ArcActorNetwork(size=args.grid_size, color_values=args.color_values)
    critic = ArcCriticNetwork(
        size=args.grid_size,
        color_values=args.color_values,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
    )

    # Set up replay buffer
    sampler = PrioritizedSampler(args.replay_buffer_size, 1.1, 1.0)
    storage = LazyMemmapStorage(args.replay_buffer_size)
    rb = TensorDictReplayBuffer(storage=storage, sampler=sampler)

    # Set up D4PG algorithm
    d4pg = D4PG(
        env=env,
        actor=actor,
        critic=critic,
        batch_size=args.batch_size,
        replay_buffer=rb,
        target_update_frequency=args.target_update_frequency,
        n_steps=args.n_steps,
        gamma=args.gamma,
        policy_lr=args.policy_lr,
        critic_lr=args.critic_lr,
        alpha=args.alpha,
        entropy_tau=args.entropy_tau,
        clip_grad_norm=args.clip_grad_norm,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
    )

    # Set up training loop
    for epoch in range(args.max_epochs):
        d4pg.fit(
            max_steps=args.max_steps,
            validation_steps_frequency=args.validation_steps_frequency,
            validation_steps_per_train_step=args.validation_steps_per_train_step,
            validation_steps_per_episode=args.validation_steps_per_episode,
            logger_frequency=args.logger_frequency,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train D4PG algorithm")
    parser.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--config_key",
        type=str,
        default="default",
        help="Key to use in the YAML configuration file",
    )
    args = parser.parse_args()

    config_args = load_args_from_yaml(args.config_file, args.config_key)
    train_d4pg(config_args)
