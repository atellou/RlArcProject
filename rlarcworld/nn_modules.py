"""Neural network building blocks for ARC agents.

This module defines reusable neural network components used by the
actor and critic, including CNN feature extractors, multi-head
attention, and cross-attention classifier heads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDict
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.t5.configuration_t5 import T5Config

from rlarcworld.common.enums import ActionKeys, RewardKeys, StateKeys
from rlarcworld.common.types import ACTION_KEYS_SET, GRID_ENCODER_KEYS_SET
from rlarcworld.models.grid_encoder import GridEncoder
from rlarcworld.models.vit_arc import CustomT5Stack

logger = logging.getLogger(__name__)


@dataclass
class RlArcWorldViTARCConfig:
    """Configuration for the RlArcWorld ViTARC nn module/encoder.

    A configuration class for the ViTARC used in RlArcWorld.
    This configuration defines the hyperparameters for the ViTARC-based neural network.

    Attributes:
        grid_size: Size of the input grid (height = width).
        d_model: Model dimension for embeddings and hidden layers.
        d_ff: Feed-forward dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        ape_type: Type of absolute position encoding ('none', 'learnable', 'sinusoidal')
        rpe_type: Type of relative position encoding ('none', 'learnable', 'sinusoidal')
        use_ope: Whether to use object position encoding.
        ape_mixer: Mixer type for absolute position encoding.
        grids_to_encode: List of grid keys to encode.
        color_values: Number of color values in the dataset.
    """

    grid_size: int
    d_model: int = 128
    d_ff: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    ape_type: str | None = None
    rpe_type: str | None = None
    use_ope: bool | None = None
    ape_mixer: str | None = None
    grids_to_encode: list[str] | None = None
    color_values: int | None = None


@dataclass
class ViTARCEncoderOutput:
    """Structured output for ViTARC encoder modules.

    Exposes both the conditioned hidden states and the intermediate artifacts
    necessary for downstream consumers to reuse embeddings or perform custom
    pooling/attention.

    Returns:
        hidden_states: The conditioned hidden states from the encoder.
        attention_mask: The attention mask used during encoding.
        encoder_outputs: The full encoder outputs for potential reuse.
        input_ids: The input token IDs.
        temporal_ids: Optional temporal position IDs.
        object_idx: Optional object indices for object-aware processing.
        example_hidden: Optional example-level hidden states.
        example_mask: Optional example-level attention mask.
    """

    hidden_states: torch.Tensor
    attention_mask: torch.Tensor | None
    encoder_outputs: BaseModelOutput
    input_ids: torch.Tensor
    temporal_ids: torch.Tensor | None
    object_idx: torch.Tensor | None
    example_hidden: torch.Tensor | None
    example_mask: torch.Tensor | None


# =============================================================================
# Example Cross-Attention - For conditioning on example pairs
# =============================================================================


class ExampleCrossAttention(nn.Module):
    """Cross-attention module for conditioning grid encoding on examples.

    Allows the grid representation to attend to encoded example pairs,
    learning the transformation pattern demonstrated by the examples.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize example cross-attention.

        Args:
            d_model: Model hidden dimension.
            num_heads: Number of attention heads.
            num_layers: Number of cross-attention layers.
            dropout: Dropout probability.
        """
        super().__init__()
        self.d_model = d_model

        # Cross-attention layers (grid attends to examples)
        self.cross_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    d_model, num_heads, dropout=dropout, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )

        # Layer norms for each cross-attention
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)]
        )

        # Feed-forward after cross-attention
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        grid_hidden: torch.Tensor,
        example_hidden: torch.Tensor,
        grid_mask: torch.Tensor | None = None,
        example_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply cross-attention from grid to examples.

        Args:
            grid_hidden: Grid encoder output [B, grid_seq_len, d_model].
            example_hidden: Example encoder output [B, num_ex * ex_seq_len, d_model].
            grid_mask: Optional attention mask for grid [B, grid_seq_len].
            example_mask: Optional attention mask for examples [B, num_ex * ex_seq_len].

        Returns:
            Updated grid hidden states [B, grid_seq_len, d_model].
        """
        # Convert masks to key_padding_mask format
        example_key_padding_mask = None
        if example_mask is not None:
            example_key_padding_mask = ~example_mask.bool()

        hidden = grid_hidden
        for cross_attn, layer_norm in zip(
            self.cross_attn_layers, self.layer_norms, strict=False
        ):
            # Cross-attention: grid queries attend to example keys/values
            attn_out, _ = cross_attn(
                query=hidden,
                key=example_hidden,
                value=example_hidden,
                key_padding_mask=example_key_padding_mask,
            )
            hidden = layer_norm(hidden + attn_out)

        # Feed-forward
        hidden = self.ffn_norm(hidden + self.ffn(hidden))

        return hidden


class ActionHeadFromEncoder(nn.Module):
    """Converts ViTARC encoder outputs to action distributions.

    Uses cross-attention with learnable queries to extract action
    information from the encoder hidden states.
    """

    def __init__(
        self,
        hidden_size: int,
        grid_size: int,
        color_values: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize action heads.

        Args:
            hidden_size: ViTARC hidden dimension.
            grid_size: Grid size for location actions.
            color_values: Number of colors.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.hidden_size = hidden_size

        # Learnable queries for each action type
        self.action_queries = nn.ParameterDict(
            {
                ActionKeys.X_LOCATION.value: nn.Parameter(
                    torch.randn(1, 1, hidden_size) * 0.02
                ),
                ActionKeys.Y_LOCATION.value: nn.Parameter(
                    torch.randn(1, 1, hidden_size) * 0.02
                ),
                ActionKeys.COLOR_VALUES.value: nn.Parameter(
                    torch.randn(1, 1, hidden_size) * 0.02
                ),
                ActionKeys.SUBMIT.value: nn.Parameter(
                    torch.randn(1, 1, hidden_size) * 0.02
                ),
            }
        )

        # Cross-attention to extract action info from encoder states
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        # Layer norm before classification
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Classification heads
        self.classifiers = nn.ModuleDict(
            {
                ActionKeys.X_LOCATION.value: nn.Linear(hidden_size, grid_size),
                ActionKeys.Y_LOCATION.value: nn.Linear(hidden_size, grid_size),
                ActionKeys.COLOR_VALUES.value: nn.Linear(hidden_size, color_values),
                ActionKeys.SUBMIT.value: nn.Linear(hidden_size, 2),
            }
        )

    def forward(
        self, encoder_hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> dict[str, torch.Tensor]:
        """Compute action logits from encoder hidden states.

        Args:
            encoder_hidden_states: [B, seq_len, hidden_size].
            attention_mask: Optional attention mask [B, seq_len].

        Returns:
            Dict mapping action keys to logit tensors.
        """
        batch_size = encoder_hidden_states.size(0)

        # Convert attention mask to key_padding_mask format if provided
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        action_logits = {}
        for action_key, query in self.action_queries.items():
            # Expand query for batch
            q = query.expand(batch_size, -1, -1)

            # Cross-attention: query attends to encoder states
            attended, _ = self.cross_attention(
                q,
                encoder_hidden_states,
                encoder_hidden_states,
                key_padding_mask=key_padding_mask,
            )

            # Normalize and classify
            attended = self.layer_norm(attended.squeeze(1))
            logits = self.classifiers[action_key](attended)
            action_logits[action_key] = logits

        return action_logits


# =============================================================================
# Critic head for ViTARC encoder (distributional Q-values)
# =============================================================================


class CriticHeadFromEncoder(nn.Module):
    """Converts ViTARC encoder outputs + actions to distributional Q-values.

    Uses learnable queries (one per reward type) that attend over the
    encoder hidden states *plus* an appended action token, similar in
    spirit to ``ActionHeadFromEncoder``.
    """

    def __init__(
        self,
        hidden_size: int,
        grid_size: int,
        color_values: int,
        num_atoms: dict[str, int],
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialize critic head from encoder.

        Args:
            hidden_size: Hidden dimension of the encoder.
            grid_size: Size of the grid (used for x, y coordinates).
            color_values: Number of possible color values.
            num_atoms: Dictionary mapping reward type to number of atoms
                for distributional Q-learning.
            num_heads: Number of attention heads. Defaults to 4.
            dropout: Dropout probability. Defaults to 0.1.

        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_atoms = num_atoms

        # Action features: concat x, y, color, submit probabilities
        action_input_dim = (grid_size * 2) + color_values + 2
        self.action_projection = nn.Linear(action_input_dim, hidden_size)

        # Learnable queries for each reward type
        self.value_queries = nn.ParameterDict(
            {
                key: nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
                for key in self.num_atoms.keys()
            }
        )

        # Cross-attention: queries attend to encoder hidden states (+ action token)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        # Layer norm before classification
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Classification heads for distributional Q-values
        self.classifiers = nn.ModuleDict(
            {key: nn.Linear(hidden_size, num) for key, num in self.num_atoms.items()}
        )

    def forward(
        self,
        encoder_output: ViTARCEncoderOutput,
        action: TensorDict,
    ) -> TensorDict:
        """Compute per-reward distributions from encoder output and action.

        Args:
            encoder_output: ``ViTARCEncoderOutput`` with hidden states and mask.
            action: TensorDict with action distributions for
                ``x_location``, ``y_location``, ``color_values``, ``submit``.

        Returns:
            TensorDict mapping reward keys to probability distributions
            over ``num_atoms[key]`` atoms.
        """
        hidden_states = encoder_output.hidden_states  # [B, seq_len, H]
        attention_mask = encoder_output.attention_mask  # [B, seq_len] or None

        batch_size = hidden_states.size(0)

        # Concatenate action distributions in a fixed order and project to token
        action_concat = torch.cat(
            [
                action[ActionKeys.X_LOCATION.value],
                action[ActionKeys.Y_LOCATION.value],
                action[ActionKeys.COLOR_VALUES.value],
                action[ActionKeys.SUBMIT.value],
            ],
            dim=1,
        )
        action_embed = self.action_projection(action_concat).unsqueeze(1)

        # Append action token to encoder sequence
        hidden_states = torch.cat([hidden_states, action_embed], dim=1)

        # Extend attention mask for action token if provided
        if attention_mask is not None:
            action_mask = torch.ones(
                batch_size,
                1,
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([attention_mask, action_mask], dim=1)

        # Convert attention mask to key_padding_mask format
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        # Cross-attention: one learnable query per reward type
        q_logits: dict[str, torch.Tensor] = {}
        for reward_key, query in self.value_queries.items():
            q = query.expand(batch_size, -1, -1)  # [B, 1, H]

            attended, _ = self.cross_attention(
                q,
                hidden_states,
                hidden_states,
                key_padding_mask=key_padding_mask,
            )

            attended = self.layer_norm(attended.squeeze(1))  # [B, H]
            logits = self.classifiers[reward_key](attended)  # [B, num_atoms]
            q_logits[reward_key] = logits

        # Convert logits to probability distributions over atoms
        q_dist = TensorDict(
            {key: torch.softmax(logits, dim=-1) for key, logits in q_logits.items()}
        )

        return q_dist


# =============================================================================
# Reward Head - For reward prediction from encoder states
# =============================================================================


class RewardHeadFromEncoder(nn.Module):
    """Converts ViTARC encoder outputs to reward predictions.

    Uses learnable queries (one per reward type) that attend over the
    encoder hidden states to predict rewards. Similar to CriticHeadFromEncoder
    but without action conditioning.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialize reward head from encoder.

        Args:
            hidden_size: Hidden dimension of the encoder.
            num_heads: Number of attention heads. Defaults to 4.
            dropout: Dropout probability. Defaults to 0.1.
        """
        super().__init__()
        self.hidden_size = hidden_size

        # Learnable queries for each reward type
        self.reward_queries = nn.ParameterDict(
            {
                RewardKeys.BINARY.value: nn.Parameter(
                    torch.randn(1, 1, hidden_size) * 0.02
                ),
                RewardKeys.PIXEL_WISE.value: nn.Parameter(
                    torch.randn(1, 1, hidden_size) * 0.02
                ),
            }
        )

        # Cross-attention: queries attend to encoder hidden states
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        # Layer norm before classification
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Classification heads for reward predictions
        # Binary: 2 classes (incorrect, correct)
        # Pixel-wise: 1 output (regression)
        self.classifiers = nn.ModuleDict(
            {
                RewardKeys.BINARY.value: nn.Linear(hidden_size, 2),
                RewardKeys.PIXEL_WISE.value: nn.Linear(hidden_size, 1),
            }
        )

    def forward(
        self,
        encoder_output: ViTARCEncoderOutput,
    ) -> TensorDict:
        """Compute reward predictions from encoder output.

        Args:
            encoder_output: ``ViTARCEncoderOutput`` with hidden states and mask.

        Returns:
            TensorDict with:
                - ``binary``: Softmax probabilities [B, 2] for (incorrect, correct)
                - ``pixel_wise``: Sigmoid output [B] for percentage of wrong pixels
        """
        hidden_states = encoder_output.hidden_states  # [B, seq_len, H]
        attention_mask = encoder_output.attention_mask  # [B, seq_len] or None

        batch_size = hidden_states.size(0)

        # Convert attention mask to key_padding_mask format
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()

        # Cross-attention: one learnable query per reward type
        reward_outputs: dict[str, torch.Tensor] = {}
        for reward_key, query in self.reward_queries.items():
            q = query.expand(batch_size, -1, -1)  # [B, 1, H]

            attended, _ = self.cross_attention(
                q,
                hidden_states,
                hidden_states,
                key_padding_mask=key_padding_mask,
            )

            attended = self.layer_norm(attended.squeeze(1))  # [B, H]
            logits = self.classifiers[reward_key](attended)  # [B, num_classes]
            reward_outputs[reward_key] = logits

        # Process outputs into reward predictions
        # Binary: softmax classification (better calibrated probabilities)
        binary_prob = torch.softmax(reward_outputs[RewardKeys.BINARY.value], dim=-1)

        # Pixel-wise: sigmoid to predict percentage of wrong pixels in [0, 1]
        pixel_wise_percentage = torch.sigmoid(
            reward_outputs[RewardKeys.PIXEL_WISE.value].squeeze(-1)
        )

        return TensorDict(
            {
                RewardKeys.BINARY.value: binary_prob,
                RewardKeys.PIXEL_WISE.value: pixel_wise_percentage,
            }
        ).auto_batch_size_()


# =============================================================================
# VitArc Adaptation for RlArcWorld
# =============================================================================


class RlArcWorldViTARC(nn.Module):
    """ViTARC-based for RlArcWorld.

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

    def __init__(self, config: RlArcWorldViTARCConfig, verbose: bool = False):
        """Initialize the ViTARC model adapted for RlArcWorld.

        Args:
            config: ViTARC configuration defining model hyperparameters for RlArcWorld.
            verbose: Whether to print initialization parameter information.
        """
        super().__init__()
        if config.grids_to_encode is None:
            config.grids_to_encode = [StateKeys.GRID.value]
        else:
            if not set(config.grids_to_encode).issubset(GRID_ENCODER_KEYS_SET):
                raise ValueError(
                    f"Grid encoder keys must be in {GRID_ENCODER_KEYS_SET}"
                )
            if StateKeys.GRID.value not in config.grids_to_encode:
                raise ValueError(f"{StateKeys.GRID.value} must be in grids_to_encode")
        self.grids_to_encode = config.grids_to_encode
        self.action_head_keys = set(ACTION_KEYS_SET)

        # Use best config from paper if not specified
        ape_type = config.ape_type or self.BEST_APE_TYPE
        rpe_type = config.rpe_type or self.BEST_RPE_TYPE
        ape_mixer = config.ape_mixer or self.BEST_APE_MIXER
        use_ope = config.use_ope if config.use_ope is not None else self.BEST_USE_OPE

        # Grid encoder (direct grid values as tokens, no text tokenizer)
        self.grid_encoder = GridEncoder(config.grid_size, use_ope=use_ope)
        self.encoder_ids = self.grid_encoder.encoder_ids
        self.use_ope = use_ope

        # Create T5Config for ViTARC
        self.t5_config = T5Config(
            vocab_size=self.encoder_ids.VOCAB_SIZE,
            d_model=config.d_model,
            d_ff=config.d_ff,
            d_kv=config.d_model // config.num_heads,
            num_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout_rate=config.dropout,
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
        self.t5_config.rows = config.grid_size + 3  # +3 for BOS/EOS tokens
        self.t5_config.cols = config.grid_size

        # Token embeddings
        self.embed_tokens = nn.Embedding(self.encoder_ids.VOCAB_SIZE, config.d_model)

        # Temporal embedding (0=initial, 1=last, 2=current)
        self.use_temporal = (
            StateKeys.INITIAL.value in self.grids_to_encode
            or StateKeys.LAST_GRID.value in self.grids_to_encode
        )
        if self.use_temporal:
            self.temporal_embed = nn.Embedding(3, config.d_model)

        # Example cross-attention (if examples are used)
        self.use_examples = StateKeys.EXAMPLES.value in self.grids_to_encode
        if self.use_examples:
            self.example_cross_attention = ExampleCrossAttention(
                d_model=config.d_model,
                num_heads=config.num_heads,
                num_layers=2,
                dropout=config.dropout,
            )

        # ViTARC encoder (using CustomT5Stack with all enhancements)
        encoder_config = T5Config(**self.t5_config.to_dict())
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = CustomT5Stack(encoder_config, self.embed_tokens)

        # Log configuration and parameter count
        if verbose:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            logger.info(
                f"ViTARCActor initialized with best ViTARC config:\n"
                f"  - APE: {ape_type}\n"
                f"  - RPE: {rpe_type} (largest gain)\n"
                f"  - PEmixer: {ape_mixer} (second largest gain)\n"
                f"  - OPE: {use_ope} (third largest gain)\n"
                f"  - Params: {total_params:,} total, {trainable_params:,} trainable"
            )

    def forward(self, state: TensorDict) -> ViTARCEncoderOutput:
        """Forward pass that encodes ARC state information.

        Args:
            state: TensorDict containing:
                - grid: Current grid [B, 1, H, W]
                - initial: Initial grid [B, 1, H, W]
                - last_grid: Previous grid [B, 1, H, W]
                - examples: Example pairs [B, num_examples, 2, H, W]

        Returns:
            ViTARCEncoderOutput containing encoded representations.
        """
        # Get grids
        grid = state[StateKeys.GRID.value]
        initial = state.get(StateKeys.INITIAL.value)
        last_grid = state.get(StateKeys.LAST_GRID.value)
        examples = state.get(StateKeys.EXAMPLES.value)

        example_hidden: torch.Tensor | None = None
        example_mask: torch.Tensor | None = None

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

        return ViTARCEncoderOutput(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            input_ids=input_ids,
            temporal_ids=temporal_ids,
            object_idx=object_idx,
            example_hidden=example_hidden,
            example_mask=example_mask,
        )
