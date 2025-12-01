"""PixelWiseArcStepper class to handle environment stepping and episode simulation."""

import logging
from typing import Any, Callable, Tuple

import torch

from rlarcworld.arc_dataset import ArcDataset, ArcSampleTransformer, ComposeTransforms
from rlarcworld.common.configs import NStepMinVMaxConfig
from rlarcworld.common.enums import ArcEnvRunningMode, RewardKeys
from rlarcworld.common.types import RewardDict
from rlarcworld.environments.arc_batch_grid_env import ArcBatchGridEnv
from rlarcworld.environments.steppers.core import EnvStepper
from rlarcworld.environments.wrappers.pixel_aware_reward import PixelAwareRewardWrapper
from rlarcworld.models.actor import ArcActorNetwork
from rlarcworld.utils.augmentation import ArcAugmentationTransformer
from rlarcworld.utils.device import CudaConfig

logger = logging.getLogger(__name__)


class PixelWiseArcStepper:
    """PixelWiseArcStepper to handle environment stepping and episode simulation."""

    def __init__(
        self,
        max_steps_per_epoch: int,
        dataset_path: str,
        grid_size: int,
        color_values: int,
        env_mode: ArcEnvRunningMode,
        actor: ArcActorNetwork,
        config: CudaConfig,
        n_step_config: NStepMinVMaxConfig | None = None,
    ) -> None:
        """Creates a stepper for the environment and dataset.

        Args:
            max_steps_per_epoch (int): Maximum steps per epoch.
            dataset_path (str): Path to the dataset.
            grid_size (int): Size of the ARC grid.
            color_values (int): Number of possible colors.
            env_mode (ArcEnvRunningMode): Environment running mode.
            actor (ArcActorNetwork): Actor network.
            config (dict): Configuration dictionary.
            n_step_config (NStepMinVMaxConfig | None):
                Minimum value for n-step returns.
        """
        # Initialize dataset and environment
        logger.info("Setting up dataset and environment...")
        self.env_mode = env_mode
        self.device = config["device"]
        self.n_step_config = (
            n_step_config if n_step_config is not None else NStepMinVMaxConfig()
        )

        # Create augmentation transformer
        augmentation = ArcAugmentationTransformer(
            augmentations=["rotate", "reflect", "permute_colors"],
            max_examples=None,
            color_permutation_range=(1, 12),
            augmentation_probability=0.5,  # Apply augmentations 50% of the time
            shuffle_probability=0.5,  # Shuffle examples 50% of the time
        )

        # IMPORTANT: ArcSampleTransformer must come FIRST to convert lists to tensors
        # and apply the padding and zero-based correction.
        # Then augmentation can operate on the tensor data.
        transform = ComposeTransforms(
            [
                ArcSampleTransformer(
                    output_size=(grid_size, grid_size),
                    examples_stack_dim=10,
                    padding_constant_value=-1,
                    zero_based_correction=1,
                ),
                augmentation,
            ]
        )
        self.dataset: ArcDataset = ArcDataset(
            dataset_path, keep_in_memory=False, transform=transform
        )

        base_env = ArcBatchGridEnv(
            grid_size=grid_size,
            color_values=color_values,
            device=self.device,
        )
        # Wrap with pixel-aware rewards to expose get_source_env_reward and n-step API
        self.environment = PixelAwareRewardWrapper(
            base_env,
            max_penalty=1.0,
            n_steps=self.n_step_config.n_steps,
            gamma=base_env.gamma,
        )

        self.core = EnvStepper(
            actor=actor,
            device=self.device,
            env=self.environment,
            sampler=iter(self.dataset),
            max_steps=max_steps_per_epoch,
            env_sample_reset=self._reset_env,
            amp_enabled=config.get("amp_enabled", True),
            on_step=self._get_on_step_rewards(),
        )

    def _reset_env(self, env, sample, **kwargs):
        """Reset the environment with a sample from the dataset.

        Args:
            env: The environment to reset.
            sample: The sample from the dataset.
            **kwargs: Additional keyword arguments.
        """
        options: dict = kwargs.setdefault("options", {"mode": self.env_mode.value})
        running_mode = options.setdefault("mode", self.env_mode.value)
        if running_mode != self.env_mode.value:
            raise ValueError(f"Unsupported running mode: {running_mode}")

        # Accept single-sample outputs from dataset and add batch dim if needed
        task = sample["task"]
        examples = sample["examples"]
        # Ensure 'input' and 'output' have shape (B, H, W)
        inp = task["input"]
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)
        batched_task = {"input": inp}

        if running_mode == ArcEnvRunningMode.TRAIN.value:
            out = task["output"]
            if out.ndim == 2:
                out = out.unsqueeze(0)
            batched_task["output"] = out

        # Ensure examples have shape (B, E, 2, H, W)
        if examples.ndim == 4:
            examples = examples.unsqueeze(0)
        args = {"batch": batched_task, "examples": examples}
        options.update(args)
        env.reset(**kwargs)

    def create_step_stream_from_samples(self, **kwargs):
        """Create a step stream from samples."""
        return self.core.create_step_stream_from_samples(**kwargs)

    # ==================== Reward Processing ====================

    def process_rewards(
        self,
        env: PixelAwareRewardWrapper,
        reward: torch.Tensor,
    ) -> RewardDict:
        """Process and format rewards from the environment.

        Args:
            env: The environment instance
            reward: Raw reward tensor from the environment

        Returns:
            Dictionary of processed rewards with different formats
        """
        # Prepare base rewards with consistent device placement
        pixel_wise_reward = reward.unsqueeze(-1)
        binary_reward = env.get_source_env_reward().unsqueeze(-1)

        if self.device is not None:
            pixel_wise_reward = pixel_wise_reward.to(self.device)
            binary_reward = binary_reward.to(self.device)

        rewards: RewardDict = {
            RewardKeys.PIXEL_WISE.value: pixel_wise_reward,
            RewardKeys.BINARY.value: binary_reward,
        }

        # Add n-step reward if configured and supported by the environment
        if env.n_steps > 1:
            pixel_n_step = env.n_step_reward()
            binary_n_step = env.get_source_env_n_step_reward()
            if self.device is not None:
                pixel_n_step = pixel_n_step.to(self.device)
                binary_n_step = binary_n_step.to(self.device)

            pixel_wise_minmax = self.n_step_config.get_pixel_wise_minmax()
            if pixel_wise_minmax is not None:
                pixel_n_step = torch.clamp(pixel_n_step, **pixel_wise_minmax)
            binary_minmax = self.n_step_config.get_binary_minmax()
            if binary_minmax is not None:
                binary_n_step = torch.clamp(binary_n_step, **binary_minmax)
            rewards[RewardKeys.PIXEL_WISE_N_REWARD.value] = pixel_n_step.unsqueeze(-1)
            rewards[RewardKeys.BINARY_N_REWARD.value] = binary_n_step.unsqueeze(-1)

        return rewards

    def _get_on_step_rewards(self) -> Callable[..., Tuple[Any, ...]]:
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
