"""Utility functions for reward processing in RL training."""

import logging

import torch

from rlarcworld.environments.wrappers.rewards import PixelAwareRewardWrapper

logger = logging.getLogger(__name__)


class RewardProcessor:
    def __init__(
        self,
        n_steps: int = 1,
        device: torch.device | None = None,
        v_min_n_step: float | None = None,
        v_max_n_step: float | None = None,
    ):
        self.n_steps = n_steps
        self.device = device
        self.v_min_n_step = v_min_n_step
        self.v_max_n_step = v_max_n_step

    def process_rewards(
        self,
        env: PixelAwareRewardWrapper,
        reward: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Process and format rewards from the environment.

        Args:
            env: The environment instance
            reward: Raw reward tensor from the environment
            n_steps: Number of steps for n-step returns (if supported)
            device: Device to move tensors to
            v_min_n_step: Minimum value for n-step returns
            v_max_n_step: Maximum value for n-step returns

        Returns:
            Dictionary of processed rewards with different formats
        """

        if self.n_steps > 1:
            assert (
                self.v_min_n_step is not None and self.v_max_n_step is not None
            ), "v_min_n_step and v_max_n_step must be provided for n-step rewards"
        logger.debug(
            "Processing rewards: n_steps=%s, device=%s, reward_shape=%s",
            self.n_steps,
            str(self.device),
            tuple(reward.shape) if hasattr(reward, "shape") else None,
        )
        reward_dict = {
            "pixel_wise": reward.unsqueeze(-1),
            "binary": env.get_source_env_reward().unsqueeze(-1),
        }

        # Add n-step reward if configured and supported by the environment
        if self.n_steps > 1 and hasattr(env, "n_step_reward"):
            n_step_reward = env.n_step_reward(self.v_min_n_step, self.v_max_n_step)
            if self.device is not None:
                n_step_reward = n_step_reward.to(self.device)
            reward_dict["n_reward"] = n_step_reward.unsqueeze(-1)
            logger.debug(
                "Added n_step_reward with shape %s", tuple(n_step_reward.shape)
            )

        return reward_dict

    def get_on_step_rewards(self):
        """Create a default on_step callback that processes rewards.

        Returns:
            An on_step callback function
        """

        def on_step(
            env, state, action_probs, reward, observation, terminated, truncated, info
        ):
            # Process rewards
            reward = self.process_rewards(env=env, reward=reward)
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
