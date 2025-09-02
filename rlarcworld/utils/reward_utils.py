"""Utility functions for reward processing in RL training."""

import torch
from typing import Dict, Any, Optional


def process_rewards(
    env: Any,
    reward: torch.Tensor,
    n_steps: int = 1,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Process and format rewards from the environment.

    Args:
        env: The environment instance
        reward: Raw reward tensor from the environment
        n_steps: Number of steps for n-step returns (if supported)
        device: Device to move tensors to

    Returns:
        Dictionary of processed rewards with different formats
    """
    reward_dict = {
        "pixel_wise": reward.unsqueeze(-1),
        "binary": env.get_source_env_reward().unsqueeze(-1),
    }

    # Add n-step reward if configured and supported by the environment
    if n_steps > 1 and hasattr(env, "n_step_reward"):
        n_step_reward = env.n_step_reward()
        if device is not None:
            n_step_reward = n_step_reward.to(device)
        reward_dict["n_reward"] = n_step_reward.unsqueeze(-1)

    return reward_dict


def get_on_step_rewards(n_steps: int = 1, device: Optional[torch.device] = None):
    """Create a default on_step callback that processes rewards.

    Args:
        n_steps: Number of steps for n-step returns
        device: Device to move tensors to

    Returns:
        An on_step callback function
    """

    def on_step(
        env, state, action_probs, reward, observation, terminated, truncated, info
    ):
        # Process rewards
        reward = process_rewards(env=env, reward=reward, n_steps=n_steps, device=device)
        return (
            env,
            state,
            action_probs,
            reward,
            observation,
            terminated,
            truncated,
            info,
        )

    return on_step
