"""Curriculum learning utilities for ARC environments.

This module provides curriculum learning wrappers and strategies for
training agents on ARC (Abstract Reasoning Corpus) tasks. The curriculum
helps guide the learning process by modifying actions based on the
current state and target configuration.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import torch
from tensordict import TensorDict

from rlarcworld.common.enums import ActionKeys
from rlarcworld.environments.arc_batch_grid_env import ArcBatchGridEnv
from rlarcworld.environments.wrappers.pixel_aware_reward import PixelAwareRewardWrapper

logger = logging.getLogger(__name__)


class CurriculumBaseEnv(Protocol):
    """Base protocol for curriculum learning environments."""

    def step(self, action):
        """Step the environment."""
        ...

    def get_source_env_reward(self) -> torch.Tensor:
        """Returns the last reward from the source environment."""
        ...

    def get_current_grid(self) -> torch.Tensor:
        """Return the current grid."""
        ...

    def get_target(self) -> torch.Tensor:
        """Return the target grid."""
        ...

    def get_state(self, **kwargs) -> dict:
        """Return the state of the environment.

        Args:
            **kwargs: Additional keyword arguments passed to the parent
                environment's get_state method.

        Returns:
            The state of the environment, which is a dictionary containing the
            current grid, last grid, initial grid, index, terminated, and
            examples.
        """
        ...


class PixelLocationStrategy(StrEnum):
    """Location selection strategy."""

    RANDOM = "random"
    TOP_LEFT = "top_left"
    BOTTOM_RIGHT = "bottom_right"


@dataclass
class ActionStrategy:
    """Action selection strategy.

    Holds the probabilities of each action type.

    Args:
        no_change: Probability of taking no action
        no_change_submit: Probability of taking no action and submitting
        improve: Probability of improving the grid
        degrade: Probability of degrading the grid
        degrade_submit: Probability of degrading the grid and submitting
        on_perfect_match_submit: Probability of submitting when grid is perfect

    """

    no_change: float = 0.1
    no_change_submit: float = 0.05
    improve: float = 0.7
    degrade: float = 0.1
    degrade_submit: float = 0.05
    on_perfect_match_submit: float = 0.9

    def __post_init__(self) -> None:
        """Check that the probabilities sum to 1."""
        total = (
            float(self.no_change)
            + float(self.no_change_submit)
            + float(self.improve)
            + float(self.degrade)
            + float(self.degrade_submit)
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"ActionStrategy probabilities must sum to 1.0. Got {total}."
            )


class ArcActions:
    """Class to generate actions for ARC environments."""

    def __init__(
        self,
        color_values: int,
        pixel_location_strategy: PixelLocationStrategy = PixelLocationStrategy.RANDOM,
        action_strategy: ActionStrategy = ActionStrategy(),
    ) -> None:
        """Initialize the action generator.

        Args:
            color_values: Number of color values available in the environment.
            pixel_location_strategy: Strategy for selecting pixel locations.
            action_strategy: Strategy for determining action types.
        """
        self.color_values = color_values
        self.pixel_location_strategy = pixel_location_strategy
        self.action_strategy = action_strategy
        self._strategy_counts: dict[str, float] = {
            "oracle_calls": 0,
            "samples_total": 0,
            "no_change": 0,
            "no_change_submit": 0,
            "improve": 0,
            "degrade": 0,
            "degrade_submit": 0,
            "perfect_submit_override": 0,
        }

    def reset_stats(self) -> None:
        """Reset internal action counters."""
        for k in self._strategy_counts.keys():
            self._strategy_counts[k] = 0

    def get_stats(self, *, reset: bool = False) -> dict[str, float]:
        """Get internal action counters.

        Args:
            reset: If True, reset counters after reading.
        """
        out = dict(self._strategy_counts)
        if reset:
            self.reset_stats()
        return out

    def _get_locations(
        self, current: torch.Tensor, target: torch.Tensor, to_improve: bool
    ) -> torch.Tensor:
        """Get the locations of the pixels that need to be improved.

        Args:
            current (torch.Tensor): Current grid of the form (B, H, W)
            target (torch.Tensor): Target grid of the form (B, H, W)
            to_improve (bool): Whether to get the locations of the pixels to improve

        Returns:
            locations (torch.Tensor): Locations of the pixels to improve (B, 2)
            on_pixel_diff (torch.Tensor):
                Grids with 1 or 0 mismatches with the target (B)
        """
        bsz, height, width = current.shape
        device = current.device

        mismatch = current.long() != target.long()
        on_pixel_diff = torch.sum(mismatch, dim=(1, 2)).long() <= 1
        if to_improve:
            all_mismatch = torch.nonzero(mismatch, as_tuple=False)
        else:
            all_mismatch = torch.nonzero(~mismatch, as_tuple=False)
        locations = torch.zeros((bsz, 2), dtype=torch.long, device=device)
        for b in range(bsz):
            miss_locations_b = all_mismatch[torch.where(all_mismatch[:, 0] == b)]
            if miss_locations_b.numel() == 0:
                y_rand = torch.randint(0, height, (1,), device=device).item()
                x_rand = torch.randint(0, width, (1,), device=device).item()
                locations[b] = torch.tensor(
                    [y_rand, x_rand], dtype=torch.long, device=device
                )
            else:
                miss_locations_b = miss_locations_b[:, 1:]
                match self.pixel_location_strategy:
                    case PixelLocationStrategy.RANDOM:
                        locations[b] = miss_locations_b[
                            torch.randint(
                                0, miss_locations_b.shape[0], (1,), device=device
                            ).item()
                        ]
                    case PixelLocationStrategy.TOP_LEFT:
                        locations[b] = miss_locations_b[0]
                    case PixelLocationStrategy.BOTTOM_RIGHT:
                        locations[b] = miss_locations_b[-1]

        return locations, on_pixel_diff

    def improve_grid(
        self, current: torch.Tensor, target: torch.Tensor, on_match_submit: bool = False
    ) -> TensorDict:
        """Generates actions that improves a grid.

        Args:
            current (torch.Tensor): Current grid of the form (B, H, W)
            target (torch.Tensor): Target grid of the form (B, H, W)
            on_match_submit (bool): Whether to submit on match
        """
        device = current.device
        locations, on_pixel_diff = self._get_locations(current, target, to_improve=True)
        y = locations[:, 0].long()
        x = locations[:, 1].long()
        b = torch.arange(current.shape[0], device=device)
        color = target[b, y, x].long()
        submit_tensor = torch.zeros_like(x)
        if on_match_submit:
            submit_tensor = on_pixel_diff.long()

        return TensorDict(
            {
                ActionKeys.X_LOCATION.value: x,
                ActionKeys.Y_LOCATION.value: y,
                ActionKeys.COLOR_VALUES.value: color,
                ActionKeys.SUBMIT.value: submit_tensor,
            },
            batch_size=[current.shape[0]],
        )

    def degrade_grid(
        self, current: torch.Tensor, target: torch.Tensor, submit: bool = False
    ) -> TensorDict:
        """Generates actions that degrades a grid.

        Args:
            current (torch.Tensor): Current grid of the form (B, H, W)
            target (torch.Tensor): Target grid of the form (B, H, W)
            submit (bool): Whether to submit
        """
        device = current.device
        locations, _ = self._get_locations(current, target, to_improve=False)
        y = locations[:, 0].long()
        x = locations[:, 1].long()
        b = torch.arange(current.shape[0], device=device)
        color = target[b, y, x].long()
        color = torch.where(
            color < self.color_values - 1,
            color + 1,
            color - 1,
        )
        submit_tensor = torch.zeros_like(x)
        if submit:
            submit_tensor = torch.ones_like(x)

        return TensorDict(
            {
                ActionKeys.X_LOCATION.value: x,
                ActionKeys.Y_LOCATION.value: y,
                ActionKeys.COLOR_VALUES.value: color,
                ActionKeys.SUBMIT.value: submit_tensor,
            },
            batch_size=[current.shape[0]],
        )

    def no_change(self, current: torch.Tensor, submit: bool = False) -> TensorDict:
        """Generate actions that do not change the grid.

        Args:
            current (torch.Tensor): Current grid of the form (B, H, W)
            submit (bool): Whether to submit
        """
        bsz, height, width = current.shape
        device = current.device
        y = torch.randint(0, height, (bsz,), device=device)
        x = torch.randint(0, width, (bsz,), device=device)
        b = torch.arange(bsz, device=device)
        color = current[b, y, x].long()
        submit_tensor = torch.zeros_like(x)
        if submit:
            submit_tensor = torch.ones_like(x)

        return TensorDict(
            {
                ActionKeys.X_LOCATION.value: x,
                ActionKeys.Y_LOCATION.value: y,
                ActionKeys.COLOR_VALUES.value: color,
                ActionKeys.SUBMIT.value: submit_tensor,
            },
            batch_size=[current.shape[0]],
        )

    def oracle_action(self, current: torch.Tensor, target: torch.Tensor) -> TensorDict:
        """Generate actions that improve the grid."""
        device = current.device
        bsz = current.shape[0]

        # Stats: count every sample produced by the oracle.
        self._strategy_counts["oracle_calls"] += 1
        self._strategy_counts["samples_total"] += int(bsz)

        perfect = (current.long() == target.long()).all(dim=(1, 2))

        probs = torch.tensor(
            [
                self.action_strategy.no_change,
                self.action_strategy.no_change_submit,
                self.action_strategy.improve,
                self.action_strategy.degrade,
                self.action_strategy.degrade_submit,
            ],
            device=device,
            dtype=torch.float32,
        )
        # categories: 0=no_change, 1=no_change_submit, 2=improve,
        # 3=degrade, 4=degrade_submit
        cat = torch.multinomial(probs, num_samples=bsz, replacement=True)

        # Force submit behavior on perfect grids (independent of sampled category)
        if self.action_strategy.on_perfect_match_submit > 0:
            submit_mask = perfect & (
                torch.rand((bsz,), device=device)
                < float(self.action_strategy.on_perfect_match_submit)
            )
        else:
            submit_mask = torch.zeros((bsz,), device=device, dtype=torch.bool)

        out = TensorDict(
            {
                ActionKeys.X_LOCATION.value: torch.zeros(
                    (bsz,), device=device, dtype=torch.long
                ),
                ActionKeys.Y_LOCATION.value: torch.zeros(
                    (bsz,), device=device, dtype=torch.long
                ),
                ActionKeys.COLOR_VALUES.value: torch.zeros(
                    (bsz,), device=device, dtype=torch.long
                ),
                ActionKeys.SUBMIT.value: torch.zeros(
                    (bsz,), device=device, dtype=torch.long
                ),
            },
            batch_size=[bsz],
        )

        def _assign(mask: torch.Tensor, td: TensorDict) -> None:
            if not bool(mask.any().item()):
                return
            out[ActionKeys.X_LOCATION.value][mask] = td[ActionKeys.X_LOCATION.value][:]
            out[ActionKeys.Y_LOCATION.value][mask] = td[ActionKeys.Y_LOCATION.value][:]
            out[ActionKeys.COLOR_VALUES.value][mask] = td[
                ActionKeys.COLOR_VALUES.value
            ][:]
            out[ActionKeys.SUBMIT.value][mask] = td[ActionKeys.SUBMIT.value][:]

        def _gen_and_assign(mask: torch.Tensor, fn) -> None:
            if not bool(mask.any().item()):
                return
            td = fn()
            _assign(mask, td)

        # Generate only what we need, per selected subset.
        m0 = cat == 0
        m1 = cat == 1
        m2 = cat == 2
        m3 = cat == 3
        m4 = cat == 4

        # Stats: base counts before perfect-grid overrides.
        self._strategy_counts["no_change"] += int(m0.sum().item())
        self._strategy_counts["no_change_submit"] += int(m1.sum().item())
        self._strategy_counts["improve"] += int(m2.sum().item())
        self._strategy_counts["degrade"] += int(m3.sum().item())
        self._strategy_counts["degrade_submit"] += int(m4.sum().item())

        _gen_and_assign(m0, lambda: self.no_change(current[m0], submit=False))
        _gen_and_assign(m1, lambda: self.no_change(current[m1], submit=True))
        _gen_and_assign(
            m2,
            lambda: self.improve_grid(current[m2], target[m2], on_match_submit=False),
        )
        _gen_and_assign(
            m3, lambda: self.degrade_grid(current[m3], target[m3], submit=False)
        )
        _gen_and_assign(
            m4, lambda: self.degrade_grid(current[m4], target[m4], submit=True)
        )

        # Override submit on perfect grids according to on_perfect_match_submit
        out[ActionKeys.SUBMIT.value] = torch.where(
            submit_mask,
            torch.ones((bsz,), device=device, dtype=torch.long),
            out[ActionKeys.SUBMIT.value],
        )

        # Safety: never "improve" a perfect grid by changing pixels when we submit.
        # If we're submitting on a perfect grid, convert to no-op submit.
        if bool(submit_mask.any().item()):
            # Stats: these elements were force-overridden to no_change_submit.
            n_override = int(submit_mask.sum().item())
            self._strategy_counts["perfect_submit_override"] += n_override

            # Remove previous category counts for overridden elements and add them
            # into the final executed category: no_change_submit.
            if n_override > 0:
                sub_cat = cat[submit_mask]
                # categories: 0=no_change, 1=no_change_submit, 2=improve,
                # 3=degrade, 4=degrade_submit
                self._strategy_counts["no_change"] -= int((sub_cat == 0).sum().item())
                self._strategy_counts["no_change_submit"] -= int(
                    (sub_cat == 1).sum().item()
                )
                self._strategy_counts["improve"] -= int((sub_cat == 2).sum().item())
                self._strategy_counts["degrade"] -= int((sub_cat == 3).sum().item())
                self._strategy_counts["degrade_submit"] -= int(
                    (sub_cat == 4).sum().item()
                )
                self._strategy_counts["no_change_submit"] += n_override

            td_no_submit2 = self.no_change(current[submit_mask], submit=True)
            _assign(submit_mask, td_no_submit2)

        for k, v in out.items():
            assert v.shape[0] == bsz, f"Key {k} has shape {v.shape}, expected {bsz}"

        return out


class ArcActionCurriculum(PixelAwareRewardWrapper):
    """Curriculum for training ARC environments.

    Implements an object able to modify the actions
    for the environment to generate meaningful transitions
    """

    def __init__(
        self,
        env: ArcBatchGridEnv,
        *,
        enabled: bool = True,
        replace_probability: float = 1.0,
        pixel_location_strategy: PixelLocationStrategy = PixelLocationStrategy.RANDOM,
        **kwargs,
    ):
        """Initialize the curriculum.

        Args:
            env (gym.Env): The environment to wrap.
            color_values (int): Number of possible colors.
            enabled (bool): Whether to enable the curriculum. Defaults to True.
            pixel_location_strategy (PixelLocationStrategy): Strategy for selecting
                pixel locations when modifying the grid. Defaults to RANDOM.
            replace_probability (float): Probability of replacing the given action
                with a curriculum-generated action. Defaults to 1.0.
            **kwargs: Additional keyword arguments passed to the parent environment.
        """
        super().__init__(env, **kwargs)
        self.color_values = env.color_values
        self.enabled = enabled
        self._replace_probability = float(replace_probability)
        self.arc_actions = ArcActions(
            color_values=self.color_values,
            pixel_location_strategy=pixel_location_strategy,
        )

    @property
    def replace_probability(self) -> float:
        """Return the replace probability."""
        return self._replace_probability

    @replace_probability.setter
    def replace_probability(self, value: float):
        """Set the replace probability."""
        self._replace_probability = float(value)

    def step(
        self, action: TensorDict
    ) -> tuple[TensorDict, torch.Tensor, bool, bool, dict]:
        """Step the environment."""

        def _infer_batch_size(td: TensorDict | None) -> int:
            if td is None:
                return 1
            try:
                if (
                    hasattr(td, "batch_size")
                    and td.batch_size is not None
                    and len(td.batch_size) > 0
                ):
                    return int(td.batch_size[0])
            except Exception:
                pass
            try:
                for v in td.values():
                    if hasattr(v, "shape") and len(v.shape) > 0:
                        return int(v.shape[0])
            except Exception:
                pass
            return 1

        curriculum_action = None
        if self.enabled and self.replace_probability > 0:
            if (
                self.replace_probability >= 1.0
                or torch.rand(()) < self.replace_probability
            ):
                grid = self.env.get_current_grid()
                target = self.env.get_target()
                curriculum_action = self.arc_actions.oracle_action(grid, target)

        if curriculum_action is not None:
            observation, reward, terminated, truncated, info = super().step(
                curriculum_action
            )
            info["curriculum_action"] = curriculum_action
            bsz = _infer_batch_size(curriculum_action)
            info["curriculum_used"] = torch.ones((bsz,), dtype=torch.long)
        else:
            observation, reward, terminated, truncated, info = super().step(action)
            bsz = _infer_batch_size(action)
            info["curriculum_used"] = torch.zeros((bsz,), dtype=torch.long)

        return observation, reward, terminated, truncated, info
