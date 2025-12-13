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
from transformers.models.t5.configuration_t5 import T5Config

from rlarcworld.common.enums import StateKeys
from rlarcworld.common.types import (
    ACTION_KEYS_SET,
    GRID_ENCODER_KEYS_SET,
    STATE_REQUIRED_KEYS,
)
from rlarcworld.models.grid_encoder import GridEncoder
from rlarcworld.models.vit_arc import (
    CustomT5Stack,
    ViTARCForConditionalGeneration,
)
from rlarcworld.nn_modules import (
    ActionHeadFromEncoder,
    ExampleCrossAttention,
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

    Uses the best-performing configuration from the ViTARC paper (arXiv:2410.06405):

    1. **2D-RPE (Two-slope-Alibi)**: Largest performance gain. Uses 2D Manhattan
       distance with different slopes for left/right directions in raster order.

    2. **PEmixer (weighted_sum_no_norm_vec)**: Second largest gain. Learnable
       vector weights α and β for balancing token and position embeddings:
       output = α ⊙ tokens + β ⊙ positions

    3. **OPE (Object-based Positional Encoding)**: Third largest gain. Extends
       2D APE with object index: E_pos = concat(E_obj, E_x, E_y)

    4. **2D Sinusoidal APE**: Base. Separate sinusoidal embeddings for x and y.

    Architecture:
    1. Encode current grid as token sequence (color values 0-10 as token IDs)
    2. Extract object indices using OpenCV contour detection
    3. Encode with ViTARC encoder (2D APE + 2D RPE + OPE + PEmixer)
    4. Extract action distributions via cross-attention heads
    """

    # Best configuration from ViTARC paper ablation study
    BEST_APE_TYPE = "SinusoidalAPE2D"
    BEST_RPE_TYPE = "Two-slope-Alibi"  # Largest gain
    BEST_APE_MIXER = "weighted_sum_no_norm_vec"  # Second largest gain
    BEST_USE_OPE = True  # Third largest gain

    def __init__(
        self,
        grid_size: int,
        color_values: int,
        d_model: int = 128,
        d_ff: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        ape_type: str | None = None,
        rpe_type: str | None = None,
        use_ope: bool | None = None,
        ape_mixer: str | None = None,
        config: CudaConfig | None = None,
        grids_to_encode: list[str] | None = None,
    ):
        """Initialize the ViTARC actor.

        Args:
            grid_size: Size of the ARC grid (default 30).
            color_values: Number of colors (default 11 for ARC).
            d_model: Model hidden dimension.
            d_ff: Feed-forward dimension.
            num_layers: Number of encoder layers.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            ape_type: Absolute positional encoding type. Default: best from paper.
            rpe_type: Relative positional encoding type. Default: best from paper.
            use_ope: Whether to use object-based positional encoding. Default: True.
            ape_mixer: Strategy for mixing embeddings. Default: best from paper.
            config: CUDA configuration.
            grids_to_encode: List of grids to encode.
        """
        super().__init__()
        if grids_to_encode is None:
            grids_to_encode = [StateKeys.GRID.value]
        else:
            if not set(grids_to_encode).issubset(GRID_ENCODER_KEYS_SET):
                raise ValueError(
                    f"Grid encoder keys must be in {GRID_ENCODER_KEYS_SET}"
                )
            if StateKeys.GRID.value not in grids_to_encode:
                raise ValueError(f"{StateKeys.GRID.value} must be in grids_to_encode")
        self.grids_to_encode = grids_to_encode
        self.grid_size = grid_size
        self.color_values = color_values
        self.action_head_keys = set(ACTION_KEYS_SET)

        # Use best config from paper if not specified
        ape_type = ape_type or self.BEST_APE_TYPE
        rpe_type = rpe_type or self.BEST_RPE_TYPE
        ape_mixer = ape_mixer or self.BEST_APE_MIXER
        use_ope = use_ope if use_ope is not None else self.BEST_USE_OPE

        # Grid encoder (direct grid values as tokens, no text tokenizer)
        self.grid_encoder = GridEncoder(grid_size, use_ope=use_ope)
        self.encoder_ids = self.grid_encoder.encoder_ids
        self.use_ope = use_ope

        # Create T5Config for ViTARC
        self.t5_config = T5Config(
            vocab_size=self.encoder_ids.VOCAB_SIZE,
            d_model=d_model,
            d_ff=d_ff,
            d_kv=d_model // num_heads,
            num_layers=num_layers,
            num_decoder_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout,
            feed_forward_proj="gated-gelu",
            is_encoder_decoder=True,
            pad_token_id=self.encoder_ids.PAD_ID,
            eos_token_id=self.encoder_ids.EOS_ID,
            decoder_start_token_id=self.encoder_ids.BOS_ID,
        )

        # Add custom ViTARC config attributes (best config from paper)
        self.t5_config.ape_type = ape_type
        self.t5_config.rpe_type = rpe_type
        self.t5_config.rpe_abs = True
        self.t5_config.use_OPE = use_ope
        self.t5_config.ape_mixer = ape_mixer
        # rows/cols for 2D positional encoding grid dimensions
        self.t5_config.rows = grid_size + 3  # +3 for BOS/EOS tokens
        self.t5_config.cols = grid_size

        # Token embeddings
        self.embed_tokens = nn.Embedding(self.encoder_ids.VOCAB_SIZE, d_model)

        # Temporal embedding (0=initial, 1=last, 2=current)
        self.use_temporal = (
            StateKeys.INITIAL.value in grids_to_encode
            or StateKeys.LAST_GRID.value in grids_to_encode
        )
        if self.use_temporal:
            self.temporal_embed = nn.Embedding(3, d_model)

        # Example cross-attention (if examples are used)
        self.use_examples = StateKeys.EXAMPLES.value in grids_to_encode
        if self.use_examples:
            self.example_cross_attention = ExampleCrossAttention(
                d_model=d_model,
                num_heads=num_heads,
                num_layers=2,
                dropout=dropout,
            )

        # ViTARC encoder (using CustomT5Stack with all enhancements)
        encoder_config = T5Config(**self.t5_config.to_dict())
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = CustomT5Stack(encoder_config, self.embed_tokens)

        # Action heads
        self.action_heads = ActionHeadFromEncoder(
            hidden_size=d_model,
            grid_size=grid_size,
            color_values=color_values,
            num_heads=num_heads,
            dropout=dropout,
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
            f"  - APE: {ape_type}\n"
            f"  - RPE: {rpe_type} (largest gain)\n"
            f"  - PEmixer: {ape_mixer} (second largest gain)\n"
            f"  - OPE: {use_ope} (third largest gain)\n"
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

        # Get grids
        grid = state[StateKeys.GRID.value]
        initial = state.get(StateKeys.INITIAL.value)
        last_grid = state.get(StateKeys.LAST_GRID.value)
        examples = state.get(StateKeys.EXAMPLES.value)

        # Encode temporal grids: <s> [initial] <sep> [last] <sep> [current] </s>
        if self.use_temporal:
            input_ids, attention_mask, temporal_ids, object_idx = (
                self.grid_encoder.encode_temporal_grids(
                    current=grid,
                    initial=initial
                    if StateKeys.INITIAL.value in self.grids_to_encode
                    else None,
                    last_grid=last_grid
                    if StateKeys.LAST_GRID.value in self.grids_to_encode
                    else None,
                )
            )
        else:
            input_ids, attention_mask, object_idx = self.grid_encoder.encode_grid(grid)
            temporal_ids = None

        # Encode temporal grids with ViTARC encoder (2D APE + 2D RPE + OPE + PEmixer)
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            object_idx=object_idx,
            return_dict=True,
        )
        hidden_states = encoder_outputs.last_hidden_state

        # Add temporal embedding if using temporal grids
        if self.use_temporal and temporal_ids is not None:
            hidden_states = hidden_states + self.temporal_embed(temporal_ids)

        # Encode examples and apply cross-attention
        if self.use_examples and examples is not None:
            batch_size, num_examples = examples.shape[:2]
            example_hidden_list = []
            example_mask_list = []

            for ex_idx in range(num_examples):
                ex_pair = examples[:, ex_idx]  # [B, 2, H, W]
                ex_ids, ex_mask, ex_obj_idx = self.grid_encoder.encode_example_pair(
                    ex_pair
                )

                # Encode through ViTARC encoder (shared weights, with OPE)
                ex_output = self.encoder(
                    input_ids=ex_ids,
                    attention_mask=ex_mask,
                    object_idx=ex_obj_idx,
                    return_dict=True,
                )
                example_hidden_list.append(ex_output.last_hidden_state)
                example_mask_list.append(ex_mask)

            # Concatenate all examples: [B, num_ex * ex_seq_len, d_model]
            example_hidden = torch.cat(example_hidden_list, dim=1)
            example_mask = torch.cat(example_mask_list, dim=1)

            # Apply cross-attention: grid attends to examples
            hidden_states = self.example_cross_attention(
                grid_hidden=hidden_states,
                example_hidden=example_hidden,
                grid_mask=attention_mask,
                example_mask=example_mask,
            )

        # Extract action logits from encoder hidden states
        action_logits = self.action_heads(hidden_states, attention_mask)

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
        actor = cls(
            grid_size=grid_size,
            color_values=color_values,
            d_model=t5_config.d_model,
            d_ff=t5_config.d_ff,
            num_layers=t5_config.num_layers,
            num_heads=t5_config.num_heads,
            dropout=t5_config.dropout_rate,
            ape_type=getattr(t5_config, "ape_type", "SinusoidalAPE2D"),
            rpe_type=getattr(t5_config, "rpe_type", "Two-slope-Alibi"),
            use_ope=getattr(t5_config, "use_OPE", True),
            ape_mixer=getattr(t5_config, "ape_mixer", "default"),
            config=config,
        )

        # Copy encoder weights
        actor.encoder.load_state_dict(vitarc_model.encoder.state_dict())
        actor.embed_tokens.load_state_dict(vitarc_model.shared.state_dict())

        logger.info("Loaded encoder weights from pretrained ViTARC model")
        return actor
