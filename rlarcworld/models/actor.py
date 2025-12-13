"""ViTARC Actor Adapter for RL.

This module wraps the ViTARC encoder to work as an actor network in the RL
framework. It uses the best-performing configuration from the ViTARC paper:

- **2D-RPE (Two-slope-Alibi)**: Largest performance gain - 2D relative position
  bias using Manhattan distance with different slopes for left/right directions
- **PEmixer (weighted_sum_no_norm_vec)**: Learnable vector weights for balancing
  positional encoding and token embeddings
- **OPE (Object-based Positional Encoding)**: Object indices from contour detection
- **2D Sinusoidal APE**: Separate x/y absolute positional embeddings

Reference: arXiv:2410.06405 - "Tackling the Abstraction and Reasoning Corpus
with Vision Transformers: the Importance of 2D Representation, Positions, and Objects"

No text tokenizer is needed - grid values (0-10) are used directly as token IDs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from tensordict import TensorDict

from rlarcworld.common.enums import StateKeys
from rlarcworld.common.types import ACTION_KEYS_SET, STATE_REQUIRED_KEYS
from rlarcworld.models.vit_arc import (
    ViTARCForConditionalGeneration,
)
from rlarcworld.nn_modules import (
    ActionHeadFromEncoder,
    RlArcWorldViTARC,
    RlArcWorldViTARCConfig,
)
from rlarcworld.utils.device import enable_cuda

if TYPE_CHECKING:
    from rlarcworld.utils.device import CudaConfig


logger = logging.getLogger(__name__)


# =============================================================================
# ViTARC Actor - Main Actor Class
# =============================================================================


class ViTARCActor(nn.Module):
    """ViTARC-based Actor for ARC RL.

    Uses the best-performing configuration from the ViTARC paper (arXiv:2410.06405)
    """

    def __init__(
        self,
        color_values: int,
        rl_vit_arc_config: RlArcWorldViTARCConfig,
        config: CudaConfig | None = None,
    ):
        """Initialize the ViTARC actor.

        Args:
            color_values: Number of colors (default 11 for ARC).
            rl_vit_arc_config: Configuration for the ViTARC model.
            config: CUDA configuration for device management.
        """
        super().__init__()

        self.rl_vit_arc = RlArcWorldViTARC(rl_vit_arc_config)

        # Action heads
        self.action_head_keys = set(ACTION_KEYS_SET)
        self.action_heads = ActionHeadFromEncoder(
            hidden_size=rl_vit_arc_config.d_model,
            grid_size=rl_vit_arc_config.grid_size,
            color_values=color_values,
            num_heads=rl_vit_arc_config.num_heads,
            dropout=rl_vit_arc_config.dropout,
        )

        # Device configuration
        self.config = enable_cuda() if config is None else config
        self.device = self.config["device"]
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

    def input_val(self, state: TensorDict) -> None:
        """Validate that the state contains exactly the required keys."""
        assert isinstance(state, TensorDict), TypeError(
            "Input State must be a TensorDict"
        )
        assert set(state.keys()) == STATE_REQUIRED_KEYS, ValueError(
            f"State keys must be {STATE_REQUIRED_KEYS}. Keys found {set(state.keys())}"
        )

    def output_val(self, action: TensorDict) -> None:
        """Validate that the action dict contains exactly the expected heads."""
        assert isinstance(action, TensorDict), TypeError("Action must be a TensorDict")
        assert set(action.keys()) == self.action_head_keys, ValueError(
            f"Action keys must be {self.action_head_keys}. "
            f"Keys found {set(action.keys())}"
        )

    def forward(self, state: TensorDict) -> TensorDict:
        """Forward pass of the ViTARC actor.

        Args:
            state: TensorDict containing:
                - grid: Current grid [B, 1, H, W]
                - last_grid: Previous grid [B, 1, H, W]
                - initial: Initial grid [B, 1, H, W]
                - index: Step index grid [B, 1, H, W]
                - examples: Example pairs [B, num_examples, 2, H, W]

        Returns:
            TensorDict with action probabilities for each action type.
        """
        current_device = next(self.parameters()).device
        state = state.clone().to(current_device)

        # Extract and remove optional keys (not used in ViTARC encoding)
        state.pop(StateKeys.TERMINATED.value, None)

        self.input_val(state)

        # VitArc
        vitarc_output = self.rl_vit_arc(state)

        # Extract action logits from encoder hidden states
        action_logits = self.action_heads(
            vitarc_output.hidden_states, vitarc_output.attention_mask
        )

        # Apply softmax to get probabilities
        action_probs = TensorDict(
            {
                key: torch.softmax(logits, dim=-1)
                for key, logits in action_logits.items()
            }
        )

        self.output_val(action_probs)
        return action_probs.auto_batch_size_()


class ViTARCActorFromPretrained(ViTARCActor):
    """ViTARC Actor that can load from pretrained ViTARC weights.

    Use this if you have pretrained ViTARC weights from supervised
    training on ARC tasks.
    """

    @classmethod
    def from_pretrained_vitarc(
        cls,
        vitarc_model: ViTARCForConditionalGeneration,
        grid_size: int = 30,
        color_values: int = 11,
        config: CudaConfig | None = None,
    ) -> ViTARCActorFromPretrained:
        """Create actor from pretrained ViTARC model.

        Args:
            vitarc_model: Pretrained ViTARCForConditionalGeneration.
            grid_size: Grid size.
            color_values: Number of colors.
            config: CUDA configuration.

        Returns:
            ViTARCActor with encoder weights from pretrained model.
        """
        # Extract config from pretrained model
        t5_config = vitarc_model.config

        # Create actor with matching config
        rl_vit_arc_config = RlArcWorldViTARCConfig(
            grid_size=grid_size,
            d_model=t5_config.d_model,
            d_ff=t5_config.d_ff,
            num_layers=t5_config.num_layers,
            num_heads=t5_config.num_heads,
            dropout=t5_config.dropout_rate,
            ape_type=getattr(t5_config, "ape_type", "SinusoidalAPE2D"),
            rpe_type=getattr(t5_config, "rpe_type", "Two-slope-Alibi"),
            use_ope=getattr(t5_config, "use_OPE", True),
            ape_mixer=getattr(t5_config, "ape_mixer", "default"),
        )

        actor = cls(
            color_values=color_values,
            rl_vit_arc_config=rl_vit_arc_config,
            config=config,
        )

        # Copy encoder weights
        actor.rl_vit_arc.encoder.load_state_dict(vitarc_model.encoder.state_dict())
        actor.rl_vit_arc.embed_tokens.load_state_dict(vitarc_model.shared.state_dict())

        logger.info("Loaded encoder weights from pretrained ViTARC model")
        return actor
