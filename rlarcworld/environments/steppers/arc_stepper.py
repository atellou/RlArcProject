"""ArcStepper class to handle environment stepping and episode simulation."""

import logging
from typing import Any, Callable, Tuple

import torch

from rlarcworld.agent.actor import ArcActorNetwork
from rlarcworld.arc_dataset import ArcDataset, ArcSampleTransformer, ComposeTransforms
from rlarcworld.common.enums import ArcEnvRunningMode, RewardKeys
from rlarcworld.common.types import REWARD_REQUIRED_KEYS, RewardDict
from rlarcworld.environments.arc_batch_grid_env import ArcBatchGridEnv
from rlarcworld.environments.steppers.core import EnvStepper
from rlarcworld.environments.wrappers.pixel_aware_reward import PixelAwareRewardWrapper
from rlarcworld.utils.augmentation import ArcAugmentationTransformer
from rlarcworld.utils.device import CudaConfig

logger = logging.getLogger(__name__)


class ArcStepper:
    """ArcStepper class to handle environment stepping and episode simulation."""

    def __init__(
        self,
        max_steps_per_epoch: int,
        dataset_path: str,
        grid_size: int,
        color_values: int,
        env_mode: ArcEnvRunningMode,
        actor: ArcActorNetwork,
        config: CudaConfig,
        n_steps: int = 1,
        v_min_n_step: float | None = None,
        v_max_n_step: float | None = None,
    ) -> None:
        """Creates a stepper for the environment and dataset.

        Args:
            max_steps_per_epoch (int): Maximum steps per epoch.
            dataset_path (str): Path to the dataset.
            grid_size (int): Size of the ARC grid.
            color_values (int): Number of possible colors.
            n_steps (int): Number of steps for n-step returns.
            env_mode (ArcEnvRunningMode): Environment running mode.
            actor (ArcActorNetwork): Actor network.
            config (dict): Configuration dictionary.
            v_min_n_step (float | None): Minimum value for n-step returns.
            v_max_n_step (float | None): Maximum value for n-step returns.
        """
        # Initialize dataset and environment
        logger.info("Setting up dataset and environment...")
        self.env_mode = env_mode
        self.device = config["device"]
        self.v_min_n_step = v_min_n_step
        self.v_max_n_step = v_max_n_step
        self.n_steps = n_steps

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
            n_steps=self.n_steps,
        )
        # Wrap with pixel-aware rewards to expose get_source_env_reward and n-step API
        self.environment = PixelAwareRewardWrapper(
            base_env,
            max_penalty=1.0,
            n_steps=base_env.n_steps,
            gamma=base_env.gamma,
            apply_clamp=False,
        )

        self.core = EnvStepper(
            actor=actor,
            device=self.device,
            env=self.environment,
            sampler=iter(self.dataset),
            max_steps=max_steps_per_epoch,
            env_sample_reset=self.reset_env,
            amp_enabled=config.get("amp_enabled", True),
            on_step=self.get_on_step_rewards(),
        )

    def reset_env(self, env, sample, **kwargs):
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
        kwargs["options"].update(args)
        env.reset(**kwargs)

    def create_step_stream_from_samples(self, **kwargs):
        """Create a step stream from samples."""
        return self.core.create_step_stream_from_samples(**kwargs)

    def process_rewards(
        self,
        env: PixelAwareRewardWrapper,
        reward: torch.Tensor,
    ) -> RewardDict:
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
        if env.n_steps > 1:
            assert (
                self.v_min_n_step is not None and self.v_max_n_step is not None
            ), "v_min_n_step and v_max_n_step must be provided for n-step rewards"
        logger.debug(
            "Processing rewards: n_steps=%s, device=%s, reward_shape=%s",
            env.n_steps,
            str(self.device),
            tuple(reward.shape) if hasattr(reward, "shape") else None,
        )

        # Prepare base rewards with consistent device placement
        pixel_wise_reward = reward.unsqueeze(-1)
        binary_reward = env.get_source_env_reward().unsqueeze(-1)

        if self.device is not None:
            pixel_wise_reward = pixel_wise_reward.to(self.device)
            binary_reward = binary_reward.to(self.device)

        reward_dict: RewardDict = {
            RewardKeys.PIXEL_WISE.value: pixel_wise_reward,
            RewardKeys.BINARY.value: binary_reward,
        }

        # Add n-step reward if configured and supported by the environment
        if env.n_steps > 1 and hasattr(env, "n_step_reward"):
            n_step_reward = env.n_step_reward(self.v_min_n_step, self.v_max_n_step)
            if self.device is not None:
                n_step_reward = n_step_reward.to(self.device)
            reward_dict[RewardKeys.N_REWARD.value] = n_step_reward.unsqueeze(-1)
            logger.debug(
                "Added n_step_reward with shape %s", tuple(n_step_reward.shape)
            )

        # Sanity-check required keys are present
        assert set(reward_dict.keys()).issuperset(REWARD_REQUIRED_KEYS), (
            "Reward dict missing required keys: expected at least "
            f"{REWARD_REQUIRED_KEYS}, got {set(reward_dict.keys())}"
        )

        return reward_dict

    def get_on_step_rewards(self) -> Callable[..., Tuple[Any, ...]]:
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
