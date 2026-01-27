"""PixelWiseArcStepper class to handle environment stepping and episode simulation."""

import logging
from dataclasses import dataclass
from typing import Callable

import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from rlarcworld.arc_dataset import ArcDataset, ArcSampleTransformer, ComposeTransforms
from rlarcworld.common.configs import NStepMinVMaxConfig
from rlarcworld.common.enums import ArcEnvRunningMode, RewardKeys
from rlarcworld.common.types import RewardDict
from rlarcworld.environments.arc_batch_grid_env import ArcBatchGridEnv
from rlarcworld.environments.steppers.core import EnvStepper, OnEnvStepArgs
from rlarcworld.environments.wrappers.curriculum import (
    ArcActionCurriculum,
    PixelLocationStrategy,
)
from rlarcworld.environments.wrappers.pixel_aware_reward import PixelAwareRewardWrapper
from rlarcworld.models.actor import ViTARCActor
from rlarcworld.utils.augmentation import ArcAugmentationTransformer
from rlarcworld.utils.device import CudaConfig

logger = logging.getLogger(__name__)


@dataclass
class ArcActionCurriculumConfig:
    """Configuration for the curriculum."""

    enabled: bool = False
    replace_probability: float = 1.0
    pixel_location_strategy: PixelLocationStrategy = PixelLocationStrategy.RANDOM

    def __init__(
        self,
        enabled: bool,
        replace_probability: float,
        pixel_location_strategy: PixelLocationStrategy,
    ):
        """Initialize the curriculum configuration."""
        self.enabled = enabled
        self.replace_probability = replace_probability
        self.pixel_location_strategy = pixel_location_strategy


class PixelWiseArcStepper:
    """PixelWiseArcStepper to handle environment stepping and episode simulation."""

    def __init__(
        self,
        max_steps_per_episode: int,
        dataset: str | list[str],
        grid_size: int,
        color_values: int,
        actor: ViTARCActor,
        device_config: CudaConfig,
        batch_size: int,
        *,
        augmentation_enabled: bool = False,
        max_examples: int = 10,
        dataset_shuffle_seed: int | None = None,
        augmentation_generator: torch.Generator | None = None,
        max_penalty: float = 1.0,
        step_cost: float = 0.0,
        perfect_no_submit_penalty_coef: float = 1.0,
        zero_improvement_penalty: float = 0.0,
        shaping_clamp_min: float = 0.0,
        shaping_clamp_min_tau_steps: int = 0,
        n_step_config: NStepMinVMaxConfig | None = None,
        curriculum: ArcActionCurriculumConfig | None = None,
    ) -> None:
        """Creates a stepper for the environment and dataset.

        Args:
            max_steps_per_episode (int): Maximum steps per episode.
            dataset (str | list[str]): Path to the dataset.
            grid_size (int): Size of the ARC grid.
            color_values (int): Number of possible colors.
            actor (ViTARCActor): Actor network.
            device_config (CudaConfig): Device configuration dictionary.
            batch_size (int): Batch size for the dataloader.
            augmentation_enabled (bool): Whether to enable data augmentation.
            max_examples (int): Maximum number of examples to use.
            dataset_shuffle_seed (int | None):
                Seed for shuffling the dataset.
            augmentation_generator (torch.Generator | None):
                Generator for deterministic randomness.
            max_penalty (float): Maximum penalty for the environment reward.
            step_cost (float): Per-step time cost applied in train mode.
            perfect_no_submit_penalty_coef (float):
                Coefficient for perfect no-submit penalty.
            zero_improvement_penalty (float): Coefficient for zero improvement penalty.
            shaping_clamp_min (float): Minimum value for shaping reward clamping.
            shaping_clamp_min_tau_steps (int): Number of steps for shaping clamp min tau
            n_step_config (NStepMinVMaxConfig | None):
                Minimum value for n-step returns.
            curriculum (ArcActionCurriculumConfig | None):
                Curriculum configuration.
        """
        # Initialize dataset and environment
        logger.info("Setting up dataset and environment...")
        self.device = device_config["device"]
        self.n_step_config = (
            n_step_config if n_step_config is not None else NStepMinVMaxConfig()
        )
        self.batch_size = batch_size
        self.max_examples = max_examples

        # IMPORTANT: ArcSampleTransformer must come FIRST to convert lists to tensors
        # and apply the padding and zero-based correction.
        # Then augmentation can operate on the tensor data.
        transform: list[Callable[[dict], dict]] = [
            ArcSampleTransformer(
                output_size=(grid_size, grid_size),
                padding_constant_value=-1,
                zero_based_correction=1,
            )
        ]

        # Create augmentation transformer
        if augmentation_enabled:
            transform.append(
                ArcAugmentationTransformer(
                    augmentations=["rotate", "reflect", "permute_colors"],
                    max_examples=max_examples,
                    color_permutation_range=(1, 12),
                    augmentation_probability=0.5,  # Apply augmentations 50% of the time
                    shuffle_probability=0.5,  # Shuffle examples 50% of the time
                    generator=augmentation_generator,
                )
            )

        self.dataset: ArcDataset = ArcDataset(
            dataset,
            keep_in_memory=False,
            transform=ComposeTransforms(transform),
            shuffle_seed=dataset_shuffle_seed,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_fn,
        )

        env_device = self.device
        if isinstance(env_device, str) and (
            env_device.startswith("cuda") or env_device == "mps"
        ):
            env_device = "cpu"

        base_env = ArcBatchGridEnv(
            grid_size=grid_size,
            color_values=color_values,
            device=env_device,
            max_penalty=max_penalty,
            step_cost=step_cost,
        )
        # Wrap with pixel-aware rewards to expose get_source_env_reward and n-step API
        pixel_aware_env_args = {
            "max_penalty": max_penalty,
            "perfect_no_submit_penalty_coef": perfect_no_submit_penalty_coef,
            "zero_improvement_penalty": zero_improvement_penalty,
            "shaping_clamp_min": shaping_clamp_min,
            "shaping_clamp_min_tau_steps": shaping_clamp_min_tau_steps,
            "n_steps": self.n_step_config.n_steps,
            "gamma": base_env.gamma,
        }
        if curriculum is not None:
            self.environment = ArcActionCurriculum(
                env=base_env,
                enabled=curriculum.enabled,
                replace_probability=curriculum.replace_probability,
                pixel_location_strategy=curriculum.pixel_location_strategy,
                **pixel_aware_env_args,
            )
        else:
            self.environment = PixelAwareRewardWrapper(
                base_env,
                max_penalty=max_penalty,
                perfect_no_submit_penalty_coef=perfect_no_submit_penalty_coef,
                zero_improvement_penalty=zero_improvement_penalty,
                shaping_clamp_min=shaping_clamp_min,
                shaping_clamp_min_tau_steps=shaping_clamp_min_tau_steps,
                n_steps=self.n_step_config.n_steps,
                gamma=base_env.gamma,
            )

        self.core = EnvStepper(
            actor=actor,
            device=self.device,
            env=self.environment,
            sampler=self.dataloader,
            max_steps=max_steps_per_episode,
            env_sample_reset=self._reset_env,
            amp_enabled=device_config.get("amp_enabled", True),
            on_step=self._get_on_step_rewards(),
        )

    def curriculum_set_replace_probability(self, probability: float):
        """Setter for the curriculum."""
        if isinstance(self.environment, ArcActionCurriculum):
            self.environment.replace_probability = probability
        else:
            raise AttributeError("Environment is not an ArcActionCurriculum")

    def curriculum_set_enabled(self, enabled: bool):
        """Setter for the curriculum."""
        if isinstance(self.environment, ArcActionCurriculum):
            self.environment.enabled = enabled
        else:
            raise AttributeError("Environment is not an ArcActionCurriculum")

    def _collate_fn(self, samples: list[dict]) -> dict:
        """Collate function for the dataloader."""
        tasks_inp = []
        tasks_out = []
        examples_list = []
        for s in samples:
            t = s["task"]
            tasks_inp.append(t["input"])
            tasks_out.append(t["output"])
            examples_list.append(s["examples"])

        batch_in = torch.stack(tasks_inp, dim=0)
        batch_out = torch.stack(tasks_out, dim=0)

        padded_examples = []
        for ex in examples_list:
            if isinstance(ex, torch.Tensor) and ex.dim() == 4:
                ex = ex[: self.max_examples]
                if ex.shape[0] < self.max_examples:
                    pad_shape = (self.max_examples - ex.shape[0], *ex.shape[1:])
                    pad = torch.zeros(pad_shape, dtype=ex.dtype, device=ex.device)
                    ex = torch.cat([ex, pad], dim=0)
                padded_examples.append(ex)
            else:
                raise ValueError(
                    "Example of type %s and shape %s, skipping.",
                    type(ex),
                    ex.shape if isinstance(ex, torch.Tensor) else len(ex),
                )

        batch_examples = torch.stack(padded_examples, dim=0)
        return {
            "task": {"input": batch_in, "output": batch_out},
            "examples": batch_examples,
        }

    def _reset_env(self, env, sample, **kwargs):
        """Reset the environment with a sample from the dataset.

        Args:
            env: The environment to reset.
            sample: The sample from the dataset.
            **kwargs: Additional keyword arguments.
        """
        options: dict = kwargs.setdefault("options", {})
        options["mode"] = ArcEnvRunningMode.TRAIN

        # Handle both dict samples and TensorDict states
        if isinstance(sample, TensorDict):
            # Sample is a state from replay buffer
            # Extract grid as input (could be "grid" or "initial" key)
            grid = sample.get("grid", sample.get("initial"))
            if grid is None:
                raise ValueError("Sample TensorDict must have 'grid' or 'initial' key")
            inp = grid
            # For TensorDict, target must be present for TRAIN mode
            out = sample.get("target")
            if out is None:
                raise ValueError(
                    "Sample TensorDict must have 'target' key for TRAIN mode. "
                    "PixelWiseArcStepper requires target for pixel-wise reward."
                )
            examples = sample.get("examples")
        else:
            # Sample is a dict with task/examples structure
            task: dict[str, torch.Tensor] = sample["task"]
            examples = sample["examples"]
            inp = task["input"]
            out = task.get("output")
            if out is None:
                raise ValueError(
                    "Sample task must have 'output' key for TRAIN mode. "
                    "PixelWiseArcStepper requires target for pixel-wise reward."
                )

        # Ensure 'input' has shape (B, H, W)
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)
        elif inp.ndim == 4:  # (B, C, H, W) -> (B, H, W)
            inp = inp[:, 0, :, :]

        # Ensure 'output' has shape (B, H, W)
        if out.ndim == 2:
            out = out.unsqueeze(0)
        elif out.ndim == 4:  # (B, C, H, W) -> (B, H, W)
            out = out[:, 0, :, :]
        batched_task = {"input": inp, "output": out}

        # Ensure examples have shape (B, E, 2, H, W)
        if examples is not None and examples.ndim == 4:
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

    def _get_on_step_rewards(self) -> Callable[[OnEnvStepArgs], OnEnvStepArgs]:
        """Create a default on_step callback that processes rewards.

        Returns:
            An on_step callback function
        """

        def on_step(
            env_step_args: OnEnvStepArgs,
        ):
            # Process rewards
            env_step_args.reward = self.process_rewards(
                env=env_step_args.environment, reward=env_step_args.reward
            )
            return env_step_args

        return on_step
