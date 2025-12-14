"""ArcWorldModelStepper class to handle environment stepping.

This class handles environment stepping,
world model training, and imagination (Dyna-style).
"""

import logging
from typing import Any, Callable, Generator, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from rlarcworld.common.configs import NStepMinVMaxConfig
from rlarcworld.common.enums import ArcEnvRunningMode, RewardKeys
from rlarcworld.common.protocols import ActorCriticCoreProto, RewardWorldModelProto
from rlarcworld.common.types import RewardDict
from rlarcworld.environments.arc_batch_grid_env import ArcBatchGridEnv
from rlarcworld.environments.steppers.core import EnvStepper
from rlarcworld.environments.wrappers.reward_world_model import RewardWorldModelWrapper
from rlarcworld.models.actor import ViTARCActor
from rlarcworld.runners.core import OnStepContext
from rlarcworld.utils.device import CudaConfig
from rlarcworld.utils.torch_queue import TorchQueue

logger = logging.getLogger(__name__)


class ArcWorldModelStepper:
    """Trainer that handles World Model updates and imagination-based policy training.

    This class:
    1. Updates the RewardWorldModel using true rewards from real environment
    2. Generates imagined rollouts for policy training (Dyna-style)
    3. Does NOT use a dataset - works from provided samples (e.g., from replay buffer)

    Usage in training loop:
        - Call `update_model(state, true_reward)` to train the world model
        - Call `imagine_and_train(sample, d4pg_core)` for imagination rollouts
        - Or use as callback via `get_training_callback(d4pg_core)`
    """

    def __init__(
        self,
        grid_size: int,
        color_values: int,
        actor: ViTARCActor,
        config: CudaConfig,
        # World model
        reward_model: RewardWorldModelProto,
        reward_model_lr: float = 1e-4,
        # Imagination params
        step_horizon: int = 5,
        gamma: float = 0.99,
        n_step_config: NStepMinVMaxConfig | None = None,
        # Smooth transition params
        grid_diff_epsilon: float = 0.01,
        smooth_transition: bool = True,
        transition_steepness: float = 100.0,
        # Training params
        model_update_frequency: int = 10,
        imagination_ratio: int = 5,
        imagination_frequency: int = 10,
    ) -> None:
        """Creates a World Model trainer for imagination-based policy training.

        Args:
            grid_size: Size of the grid.
            color_values: Number of possible colors.
            actor: Actor network (shared with D4PGCore).
            config: Configuration dictionary.
            reward_model: RewardWorldModel to train and use for imagination.
            reward_model_lr: Learning rate for world model optimizer.
            step_horizon: Planning horizon for imagined rollouts.
            gamma: Discount factor.
            n_step_config: N-step configuration.
            grid_diff_epsilon: Threshold for "perfect" grid detection.
            smooth_transition: Use smooth sigmoid transition.
            transition_steepness: Steepness of smooth transition.
            model_update_frequency: Update world model every N training steps.
            imagination_ratio: Number of imagined rollouts per real step.
            imagination_frequency: Run imagination every N training steps.
        """
        self.model = reward_model
        self.device = config["device"]
        self.model_update_frequency = model_update_frequency
        self.imagination_ratio = imagination_ratio
        self.imagination_frequency = imagination_frequency
        self.n_step_config = (
            n_step_config if n_step_config is not None else NStepMinVMaxConfig()
        )

        # Metrics tracking using tensors for compute optimization
        self._max_tracked_losses = 1000
        self._model_losses = TensorDict(
            {
                RewardKeys.BINARY.value: TorchQueue(
                    torch.tensor([]),
                    queue_size=self._max_tracked_losses,
                ),
                RewardKeys.PIXEL_WISE.value: TorchQueue(
                    torch.tensor([]),
                    queue_size=self._max_tracked_losses,
                ),
            }
        )
        self._imagination_count: int = 0

        # Env Definition
        base_env = ArcBatchGridEnv(
            grid_size=grid_size,
            color_values=color_values,
            device=self.device,
        )

        self.environment = RewardWorldModelWrapper(
            env=base_env,
            model=reward_model,
            step_horizon=step_horizon,
            n_steps=self.n_step_config.n_steps,
            gamma=gamma,
            grid_diff_epsilon=grid_diff_epsilon,
            smooth_transition=smooth_transition,
            transition_steepness=transition_steepness,
        )

        self.stepper = EnvStepper(
            actor=actor,
            device=self.device,
            env=self.environment,
            sampler=None,
            max_steps=step_horizon,
            env_sample_reset=self._reset_env,
            amp_enabled=config.get("amp_enabled", True),
            on_step=self._get_on_step_rewards(),
        )

        # World model optimizer
        self.model_optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=reward_model_lr
        )

    def _reset_env(self, env, sample, **kwargs):
        """Reset the environment with a sample.

        Args:
            env: The environment to reset.
            sample: Either a dict with task/examples or a TensorDict state.
            **kwargs: Additional keyword arguments.
        """
        options: dict = kwargs.setdefault("options", {})
        options["mode"] = ArcEnvRunningMode.INFERENCE

        # Handle both dict samples and TensorDict states
        if isinstance(sample, TensorDict):
            # Sample is a state from replay buffer
            # Extract grid as input (could be "grid" or "initial" key)
            grid = sample.get("grid", sample.get("initial"))
            if grid is None:
                raise ValueError("Sample TensorDict must have 'grid' or 'initial' key")
            inp = grid
            examples = sample.get("examples")
        else:
            # Sample is a dict with task/examples structure
            task: dict[str, torch.Tensor] = sample["task"]
            examples = sample["examples"]
            inp = task["input"]

        # Ensure 'input' has shape (B, H, W)
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)
        elif inp.ndim == 4:  # (B, C, H, W) -> (B, H, W)
            inp = inp[:, 0, :, :]
        batched_task = {"input": inp}

        # Ensure examples have shape (B, E, 2, H, W)
        if examples is not None and examples.ndim == 4:
            examples = examples.unsqueeze(0)

        args = {"batch": batched_task, "examples": examples}
        options.update(args)
        env.reset(**kwargs)

    def create_step_stream(
        self, sample, **kwargs
    ) -> Generator[tuple[bool, int, TensorDict], None, None]:
        """Create a step stream from reset.

        Args:
            sample: Starting state (dict with task/examples or TensorDict state)
            **kwargs: Additional arguments passed to reset

        Yields:
            Tuples of (done, step_number, step_data)
        """
        return self.stepper.stream_on_reset(sample, **kwargs)

    # ==================== Reward Processing ====================

    def process_rewards(self, env: RewardWorldModelWrapper) -> RewardDict:
        """Process and format rewards from the environment.

        Args:
            env: The environment instance

        Returns:
            Dictionary of processed rewards with different formats
        """
        # Prepare base rewards with consistent device placement
        rewards = TensorDict(
            {
                RewardKeys.PIXEL_WISE.value: env.pixel_wise_model_reward,
                RewardKeys.BINARY.value: env.binary_model_reward,
            },
            batch_size=[],
        ).to(self.device)

        # Add n-step reward if configured and supported by the environment
        if env.n_steps > 1:
            pixel_n_step = env.n_pixel_wise_reward
            binary_n_step = env.n_binary_reward
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
            reward = self.process_rewards(env=env)
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

    # ==================== World Model Training ====================

    def compute_model_loss(
        self,
        pred: TensorDict,
        true_reward: TensorDict,
    ) -> tuple[torch.Tensor, dict]:
        """Compute world model loss.

        Args:
            pred: Predicted rewards from environment {binary, pixel_wise}
            true_reward: True rewards from environment {binary, pixel_wise}

        Returns:
            Total loss and dict of component losses
        """
        # Binary classification loss (cross-entropy)
        # true_reward["binary"] is in [0, 1], pred["binary"] is [P(0), P(1)]
        binary_target = true_reward[RewardKeys.BINARY.value].long()
        if binary_target.dim() > 1:
            binary_target = binary_target.squeeze(-1)
        # Clamp to valid class indices
        binary_target = binary_target.clamp(0, 1)
        binary_loss = F.cross_entropy(pred[RewardKeys.BINARY.value], binary_target)

        # Pixel-wise regression loss (MSE)
        # Normalize true pixel_wise to [0, 1] percentage
        pixel_target = true_reward[RewardKeys.PIXEL_WISE.value].float()
        if pixel_target.dim() > 1:
            pixel_target = pixel_target.squeeze(-1)
        # Clamp to valid range [0, 1]
        pixel_target = pixel_target.clamp(0, 1)
        pixel_pred = pred[RewardKeys.PIXEL_WISE.value]
        pixel_loss = F.mse_loss(pixel_pred, pixel_target)

        # Total loss
        total_loss = binary_loss + pixel_loss

        return total_loss, {
            RewardKeys.BINARY.value: binary_loss.item(),
            RewardKeys.PIXEL_WISE.value: pixel_loss.item(),
        }

    def update_model(
        self,
        state: TensorDict,
        true_reward: TensorDict,
    ) -> dict:
        """Compute loss and update world model.

        Args:
            state: Environment state TensorDict
            true_reward: True rewards from real environment {binary, pixel_wise}

        Returns:
            Dict of loss values
        """
        self.model_optimizer.zero_grad()
        loss, loss_dict = self.compute_model_loss(
            self.model.forward(state), true_reward
        )
        loss.backward()
        self.model_optimizer.step()
        self.push_model_losses(loss_dict)
        return loss_dict

    # ==================== Imagination Rollouts ====================

    def _imagine_and_train(
        self,
        core: ActorCriticCoreProto,
        sample: TensorDict | dict,
        training: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Run imagination rollout and train policy on imagined data.

        Args:
            core: ActorCriticCore for policy training
            sample: Starting state for imagination
            training: Whether to update networks

        Returns:
            (actor_loss, critic_loss) from imagined training
        """
        total_actor_loss = torch.tensor(0.0, device=self.device)
        total_critic_loss = torch.tensor(0.0, device=self.device)
        step_count = 0

        for done, step_idx, step_data in self.create_step_stream(sample):
            # Train on imagined step
            actor_loss, critic_loss, _td_error = core.compute_loss(
                step_data, training=training
            )

            if actor_loss is not None:
                total_actor_loss += actor_loss
            if critic_loss is not None:
                total_critic_loss += critic_loss
            step_count += 1

            if done:
                break

        self._imagination_count += 1

        if step_count > 0:
            return total_actor_loss / step_count, total_critic_loss / step_count
        return None, None

    # ==================== Callback Integration ====================

    def training_callback(self, ctx: OnStepContext):
        """Callback for integration with Trainer.

        This callback:
        1. Updates the world model with true rewards every model_update_frequency
        2. Runs imagination rollouts for policy training every imagination_frequency

        Args:
            ctx: OnStepContext containing training information
        """
        if ctx.batch is None:
            return

        # 1. Update world model with true rewards
        if ctx.iteration % self.model_update_frequency == 0:
            loss_dict = self.update_model(
                state=ctx.batch["state"],
                true_reward=ctx.batch["reward"],
            )
            logger.debug(
                f"World model updated. Iteration {ctx.iteration}, loss: {loss_dict}"
            )

        # 2. Run imagination rollouts
        if ctx.iteration % self.imagination_frequency == 0:
            for _ in range(self.imagination_ratio):
                starting_state = ctx.batch["state"]

                # Run imagination and train
                self._imagine_and_train(
                    core=ctx.core,
                    sample=starting_state,
                    training=True,
                )

            logger.debug(
                f"Imagination rollouts at iteration {ctx.iteration}, "
                f"total imaginations: {self._imagination_count}"
            )

    # ==================== Metrics ====================

    def push_model_losses(self, loss_dict: dict[str, float]) -> None:
        """Push a new loss dictionary to the model losses queue."""
        for key in loss_dict.keys():
            self._model_losses[key] = self._model_losses[key].push(
                torch.tensor([loss_dict[key]])
            )

    def get_average_model_losses(self, reset: bool = True) -> dict[str, float]:
        """Get average world model losses.

        Args:
            reset: If True, reset the loss queues after computing average

        Returns:
            Dict mapping loss keys to their average values
        """
        avg = {
            key: self._model_losses[key].mean().item()
            for key in self._model_losses.keys()
        }
        if reset:
            self._reset_loss_queues()
        return avg

    def _reset_loss_queues(self) -> None:
        """Reset the loss tracking queues."""
        self._model_losses = TensorDict(
            {
                RewardKeys.BINARY.value: TorchQueue(
                    torch.tensor([]),
                    queue_size=self._max_tracked_losses,
                ),
                RewardKeys.PIXEL_WISE.value: TorchQueue(
                    torch.tensor([]),
                    queue_size=self._max_tracked_losses,
                ),
            }
        )

    def get_loss_history(self) -> TensorDict:
        """Get the raw loss history as TensorDict.

        Returns:
            TensorDict with loss queues for each loss type
        """
        return self._model_losses.clone()

    @property
    def imagination_count(self) -> int:
        """Total number of imagination rollouts performed."""
        return self._imagination_count

    # ==================== Checkpointing ====================

    def checkpoint(self) -> dict:
        """Return checkpoint dict for world model trainer."""
        return {
            "reward_model": self.model.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "imagination_count": self._imagination_count,
            "model_losses": {
                key: self._model_losses[key].clone()
                for key in self._model_losses.keys()
            },
        }

    def load(self, checkpoint: dict) -> None:
        """Load from checkpoint."""
        if "reward_model" in checkpoint:
            self.model.load_state_dict(checkpoint["reward_model"])
        if "model_optimizer" in checkpoint:
            self.model_optimizer.load_state_dict(checkpoint["model_optimizer"])
        if "imagination_count" in checkpoint:
            self._imagination_count = checkpoint["imagination_count"]
        if "model_losses" in checkpoint:
            self._model_losses = TensorDict(
                {
                    key: checkpoint["model_losses"][key].to(self.device)
                    for key in checkpoint["model_losses"].keys()
                }
            )
