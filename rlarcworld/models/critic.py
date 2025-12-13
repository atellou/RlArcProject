"""Critic network for the ARC environment.

This module contains the implementation of the critic network for the ARC
environment. The critic network is a neural network that takes as input the
state of the environment and the action taken in the environment and outputs a
distribution over the possible rewards.

The critic network is composed of several convolutional layers, a GRU layer,
and a linear layer. The convolutional layers are used to process the input
state, the GRU layer is used to process the output of the convolutional layers,
and the linear layer is used to output a distribution over the possible rewards.

The critic network is trained using the distributional reinforcement learning
algorithm.

"""

import logging

import torch
import torch.utils.checkpoint as checkpoint
from tensordict import TensorDict

from rlarcworld.common.enums import ActionKeys, StateKeys
from rlarcworld.common.types import STATE_REQUIRED_KEYS
from rlarcworld.models.nn_modules import (
    CriticHeadFromEncoder,
    RlArcWorldViTARC,
    RlArcWorldViTARCConfig,
)
from rlarcworld.utils.device import enable_cuda

logger = logging.getLogger(__name__)


class VitARCCritic(torch.nn.Module):
    """Critic network for the ARC environment."""

    def __init__(
        self,
        color_values: int,
        rl_vit_arc_config: RlArcWorldViTARCConfig,
        num_atoms: dict[str, int],
        v_min: dict[str, int],
        v_max: dict[str, int],
        config=None,
    ):
        """Initialize the critic network for the ARC environment.

        Args:
            grid_size (int): Size of the grid.
            color_values (int): The number of color values in the grid.
            num_atoms (Dict[str, int]): The number of atoms in the distribution
                for each reward type.
            v_min (Dict[str, int]): The minimum value for each reward type.
            v_max (Dict[str, int]): The maximum value for each reward type.
            rl_vit_arc_config (RlArcWorldViTARCConfig): The ViTARC configuration.
            config (CudaConfig, optional): CUDA configuration. Defaults to None.
        """
        super().__init__()
        self.num_atoms = num_atoms
        self.grid_size = rl_vit_arc_config.grid_size
        self.color_values = color_values
        self.v_min = v_min
        self.v_max = v_max
        self.config = enable_cuda() if config is None else config
        self.device = self.config["device"]
        for key, min_val in v_min.items():
            assert min_val < self.v_max[key], (
                f"v_min[{key}]={min_val} is not lower"
                + " than v_max[{key}]={self.v_max[key]}"
            )
        self.z_atoms = TensorDict(
            {
                key: torch.linspace(v_min[key], v_max[key], value)
                for key, value in num_atoms.items()
            }
        ).to(self.device)
        self.no_scale_keys = [
            ActionKeys.X_LOCATION.value,
            ActionKeys.Y_LOCATION.value,
            ActionKeys.COLOR_VALUES.value,
            ActionKeys.SUBMIT.value,
        ]

        # Shared ViTARC encoder for state processing
        self.rl_vit_arc = RlArcWorldViTARC(rl_vit_arc_config)

        # Critic head that converts encoder output + actions into Q-distributions
        self.critic_head = CriticHeadFromEncoder(
            hidden_size=rl_vit_arc_config.d_model,
            grid_size=self.grid_size,
            color_values=self.color_values,
            num_atoms=self.num_atoms,
            num_heads=rl_vit_arc_config.num_heads,
            dropout=rl_vit_arc_config.dropout,
        )

        self.to(self.device)

        # Log configuration and parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"ViTARCActor initialized with best ViTARC config:\n"
            f"  - APE: {rl_vit_arc_config.ape_type}\n"
            f"  - RPE: {rl_vit_arc_config.rpe_type} (largest gain)\n"
            f"  - PEmixer: {rl_vit_arc_config.ape_mixer} (second largest gain)\n"
            f"  - OPE: {rl_vit_arc_config.use_ope} (third largest gain)\n"
            f"  - Params: {total_params:,} total, {trainable_params:,} trainable"
        )

    @torch.jit.export
    def input_val(self, state: TensorDict):
        r"""Validates the input state TensorDict.

        Ensures that the input state is of type TensorDict
        required keys:
            - "last_grid"
            - "grid"
            - "examples"
            - "initial"
            - "index"
            - "terminated"

        Args:
            state (TensorDict): The input state to validate.

        Raises:
            TypeError: If the input state is not a TensorDict.
            ValueError: If the state keys do not match the required keys.
        """
        assert isinstance(state, TensorDict), TypeError(
            "Input State must be a TensorDict"
        )
        in_keys = set(STATE_REQUIRED_KEYS)
        assert set(state.keys()) == in_keys, ValueError(
            f"State keys must be {in_keys}. Keys found {set(state.keys())}"
        )

    @torch.jit.export
    def output_val(self, distribution: TensorDict):
        r"""Validates the output distribution TensorDict.

        Ensures that the output distribution is of type TensorDict and contains the
        required keys: "pixel_wise" and "binary".

        Args:
            distribution (TensorDict): The output distribution to validate.

        Raises:
            TypeError: If the output distribution is not a TensorDict.
            ValueError: If the distribution keys do not match the required keys.
        """
        assert isinstance(distribution, TensorDict), TypeError(
            "Distribution must be a TensorDict"
        )
        in_keys = set(self.num_atoms.keys())
        assert set(distribution.keys()) == in_keys, ValueError(
            "Distribution keys must be {}. Keys found {}".format(
                in_keys, set(distribution.keys())
            )
        )

    def forward(self, state: TensorDict, action: TensorDict) -> TensorDict:
        r"""Forward pass of the critic network.

        Args:
            state (TensorDict): The state of the environment.
            action (TensorDict): The action taken in the environment.

        Returns:
            TensorDict: The output distribution of the critic network.

        Raises:
            TypeError: If the input state or action is not a TensorDict.
            ValueError: If the state keys do not match the required keys.
            AssertionError: If a NaN is detected in the output.
        """
        current_device = next(self.parameters()).device
        state = state.clone().to(current_device)

        # Only present in training
        state.pop(StateKeys.TERMINATED.value, None)

        # Validate input
        self.input_val(state)

        # Encode state using shared ViTARC encoder
        if self.config.get("use_checkpointing"):
            vitarc_output = checkpoint.checkpoint(
                self.rl_vit_arc, state, use_reentrant=False
            )
        else:
            vitarc_output = self.rl_vit_arc(state)

        # Use critic head to produce per-reward distributions
        action = action.to(current_device)
        q_dist = self.critic_head(vitarc_output, action)

        self.output_val(q_dist)

        return q_dist.auto_batch_size_()
