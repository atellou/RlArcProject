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
from transformers.models.t5.configuration_t5 import T5Config

from rlarcworld.common.configs import RlArcWorldViTARCConfig, ViTARCEncoderOutput
from rlarcworld.common.enums import ActionKeys, RewardKeys, StateKeys
from rlarcworld.common.types import ACTION_KEYS_SET
from rlarcworld.models.grid_encoder import GridEncoder
from rlarcworld.models.vit_arc import CustomT5Stack, FixedAbsolutePositionalEmbedding

logger = logging.getLogger(__name__)

# =============================================================================
# Learned Compression - Perceiver-style attention-based compression
# =============================================================================


class LearnedCompression(nn.Module):
    """Perceiver-style learned compression using cross-attention.

    Uses learnable query tokens that attend to the full sequence,
    allowing the model to learn what information to preserve during compression.
    This is more flexible than pooling as it can adaptively focus on
    task-relevant features.
    """

    def __init__(
        self,
        d_model: int,
        num_compressed_tokens: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize learned compression module.

        Args:
            d_model: Model hidden dimension
            num_compressed_tokens: Number of tokens in compressed output
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.num_compressed_tokens = num_compressed_tokens

        # Learnable compression queries
        self.query_embeddings = nn.Embedding(num_compressed_tokens, d_model)
        nn.init.xavier_uniform_(self.query_embeddings.weight)

        # Cross-attention: queries attend to input sequence
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norm and feedforward for post-processing
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compress input sequence using learned queries.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            mask: Optional attention mask of shape [batch, seq_len]

        Returns:
            Compressed tensor of shape [batch, num_compressed_tokens, d_model]
        """
        batch_size = x.shape[0]
        device = x.device

        # Ensure input has correct shape [batch, seq_len, d_model]
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor, got shape {x.shape}")
        if x.shape[2] != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {x.shape[2]}")

        # Get compression queries from embedding layer
        query_indices = torch.arange(
            self.num_compressed_tokens, dtype=torch.long, device=device
        )
        queries = self.query_embeddings(query_indices).unsqueeze(
            0
        )  # [1, num_tokens, d_model]
        queries = queries.repeat(batch_size, 1, 1)  # [batch, num_tokens, d_model]

        # Cross-attention: queries attend to input
        # key_padding_mask expects True for positions to ignore
        key_padding_mask = None
        if mask is not None:
            # mask is 1 for valid positions, 0 for padding
            # key_padding_mask needs True for padding
            key_padding_mask = mask == 0

        compressed, _ = self.cross_attention(
            query=queries,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
        )

        # Residual connection + layer norm
        compressed = self.norm1(queries + compressed)

        # Feedforward + residual
        compressed = self.norm2(compressed + self.ffn(compressed))

        return compressed


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
                    embed_dim=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
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


@dataclass
class CriticResponseManager:
    """Manages responses from the critic head."""

    q_logits: TensorDict
    _q_dist: TensorDict | None = None
    _q_log_dist: TensorDict | None = None

    @property
    def q_dist(self) -> TensorDict:
        """Compute the Q-distribution."""
        if self._q_dist is None:
            self._q_dist = TensorDict(
                {
                    key: torch.softmax(logits.float(), dim=-1)
                    for key, logits in self.q_logits.items()
                }
            )
        return self._q_dist

    @property
    def q_log_dist(self) -> TensorDict:
        """Compute the Logarithmic Q-distribution."""
        if self._q_log_dist is None:
            self._q_log_dist = TensorDict(
                {
                    key: torch.log_softmax(logits.float(), dim=-1)
                    for key, logits in self.q_logits.items()
                }
            )
        return self._q_log_dist

    def keys(self) -> list[str]:
        """Return the keys of the Q-distribution."""
        return list(self.q_logits.keys())


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
    ) -> CriticResponseManager:
        """Compute per-reward distributions from encoder output and action.

        Args:
            encoder_output: ``ViTARCEncoderOutput`` with hidden states and mask.
            action: TensorDict with action distributions for
                ``x_location``, ``y_location``, ``color_values``, ``submit``.
        """
        hidden_states = encoder_output.hidden_states.float()  # [B, seq_len, H]
        attention_mask = encoder_output.attention_mask  # [B, seq_len] or None

        batch_size = hidden_states.size(0)

        # Flatten action distributions into a single feature vector and project
        # into the encoder hidden size.
        action_features = torch.cat(
            [action[key] for key in ACTION_KEYS_SET], dim=-1
        ).float()
        action_token = self.action_projection(action_features).unsqueeze(1)  # [B, 1, H]

        # Append action token to sequence
        hidden_with_action = torch.cat([hidden_states, action_token], dim=1)

        # Convert attention mask to key_padding_mask format
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
            pad = torch.zeros((batch_size, 1), device=key_padding_mask.device).bool()
            key_padding_mask = torch.cat([key_padding_mask, pad], dim=1)

        q_logits: dict[str, torch.Tensor] = {}
        for reward_key, query in self.value_queries.items():
            q = query.expand(batch_size, -1, -1)  # [B, 1, H]
            attended, _ = self.cross_attention(
                q,
                hidden_with_action,
                hidden_with_action,
                key_padding_mask=key_padding_mask,
            )
            attended = self.layer_norm(attended.squeeze(1))  # [B, H]
            logits = self.classifiers[reward_key](attended)  # [B, num_atoms]
            q_logits[reward_key] = logits

        return CriticResponseManager(q_logits=TensorDict(q_logits).auto_batch_size_())


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
        hidden_states = encoder_output.hidden_states.float()  # [B, seq_len, H]
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
        binary_prob = torch.softmax(
            reward_outputs[RewardKeys.BINARY.value].float(), dim=-1
        )

        # Pixel-wise: sigmoid to predict percentage of wrong pixels in [0, 1]
        pixel_wise_percentage = torch.sigmoid(
            reward_outputs[RewardKeys.PIXEL_WISE.value].float().squeeze(-1)
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

    def __init__(
        self,
        config: RlArcWorldViTARCConfig,
        verbose: bool = False,
        checkpoint: bool = False,
    ):
        """Initialize the ViTARC model adapted for RlArcWorld.

        Args:
            config: ViTARC configuration defining model hyperparameters for RlArcWorld.
            verbose: Whether to print initialization parameter information.
            checkpoint: Whether to use checkpointing.
        """
        super().__init__()
        self.action_head_keys = set(ACTION_KEYS_SET)
        self.config = config

        # Use best config from paper if not specified
        ape_type = config.ape_type or self.BEST_APE_TYPE
        rpe_type = config.rpe_type or self.BEST_RPE_TYPE
        ape_mixer = config.ape_mixer or self.BEST_APE_MIXER
        use_ope = config.use_ope if config.use_ope is not None else self.BEST_USE_OPE

        # Grid encoder (direct grid values as tokens, no text tokenizer)
        self.grid_encoder = GridEncoder(config.grid_size, use_ope=use_ope)
        self.encoder_ids = self.grid_encoder.encoder_ids
        self.use_ope = use_ope

        # Optional index grid usage (timestamps of cell modifications)
        self.use_index_grid = config.use_index_grid

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
        self.t5_config.trainable_ape = config.trainable_ape
        # rows/cols for 2D positional encoding grid dimensions
        self.t5_config.rows = config.grid_size + 3  # +3 for BOS/EOS tokens
        self.t5_config.cols = config.grid_size
        # Compression tokens for 1D Alibi RPE
        self.t5_config.grid_compression_tokens = config.grid_compression_tokens
        self.t5_config.example_compression_tokens = config.example_compression_tokens

        # Token embeddings
        self.embed_tokens = nn.Embedding(self.encoder_ids.VOCAB_SIZE, config.d_model)

        # Index/timestamp embedding (same dimension as tokens so it can be added)
        if self.use_index_grid:
            self.index_embed = FixedAbsolutePositionalEmbedding(config.d_model)

        # Temporal embedding (0=initial, 1=last, 2=current)
        self.use_temporal = config.use_temporal
        if self.use_temporal:
            self.temporal_embed = nn.Embedding(3, config.d_model)

        # Grid compression (optional)
        self.grid_compression_tokens = config.grid_compression_tokens
        self.compression_method = config.compression_method
        if self.grid_compression_tokens is not None:
            if self.compression_method == "learned":
                self.grid_compressor = LearnedCompression(
                    d_model=config.d_model,
                    num_compressed_tokens=self.grid_compression_tokens,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                )
            else:  # pooling
                self.grid_projection = nn.Sequential(
                    nn.Linear(config.d_model, config.d_model),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )
                self.grid_pooling = nn.AdaptiveAvgPool1d(self.grid_compression_tokens)

        # Example cross-attention (if examples are used)
        self.use_examples = config.use_examples
        self.example_compression_tokens = None
        if self.use_examples:
            # Compression for example sequence length
            # Reduces memory usage in cross-attention
            self.example_compression_tokens = config.example_compression_tokens
            self.max_learnable_examples = config.max_learnable_examples
            if self.compression_method == "learned":
                if self.max_learnable_examples < 1:
                    raise ValueError(
                        "max_learnable_examples must be at least 1 "
                        "for learned compression"
                    )
                self.example_compressor = LearnedCompression(
                    d_model=config.d_model,
                    num_compressed_tokens=self.example_compression_tokens,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                )
            else:  # pooling
                self.example_projection = nn.Sequential(
                    nn.Linear(config.d_model, config.d_model),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )
                # Pool to configurable number of tokens per example
                self.example_pooling = nn.AdaptiveAvgPool1d(
                    self.example_compression_tokens
                )

            self.example_cross_attention = ExampleCrossAttention(
                d_model=config.d_model,
                num_heads=config.num_heads,
                num_layers=2,
                dropout=config.dropout,
            )
        # Example embedding cache for performance
        self._example_cache: dict[int, torch.Tensor] = {}
        self._cache_valid = False

        # ViTARC encoder (using CustomT5Stack with all enhancements)
        self.t5_config.max_grids = 1
        if self.use_temporal:
            self.t5_config.max_grids += 1
        if self.use_examples:
            self.t5_config.max_grids += 1
        encoder_config = T5Config(**self.t5_config.to_dict())
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = CustomT5Stack(encoder_config, self.embed_tokens)
        # Enable internal gradient checkpointing for memory savings
        if checkpoint:
            self.encoder.gradient_checkpointing_enable()
            self.encoder.config.use_cache = False

        # Log configuration and parameter count
        if verbose:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            logger.info(
                f"RlArcWorldViTARC config:\n"
                f"  - APE: {ape_type}\n"
                f"  - RPE: {rpe_type} (largest gain)\n"
                f"  - PEmixer: {ape_mixer} (second largest gain)\n"
                f"  - OPE: {use_ope} (third largest gain)\n"
                f"  - Examples: {self.use_examples}\n"
                f"  - Grid compression: {self.grid_compression_tokens}\n"
                f"  - Example compression: {self.example_compression_tokens}\n"
                f"  - Params: {total_params:,} total, {trainable_params:,} trainable"
            )

    def clear_example_cache(self) -> None:
        """Clear example embedding cache after model weight updates."""
        self._example_cache.clear()

    def enable_example_cache(self) -> None:
        """Enable example embedding cache for reuse across steps."""
        self._cache_valid = True

    def _encode_example_batch(
        self,
        ex_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of example pairs and return concatenated hidden states.

        Args:
            ex_batch: Example batch tensor with shape [B, E, 2, H, W]

        Returns:
            Concatenated hidden states with shape [B, E*seq_len, d_model]
        """
        bsz, ex_count = ex_batch.shape[:2]
        if ex_count == 0:
            # Return a correctly-shaped empty tensor for concatenation.
            device = ex_batch.device
            return torch.empty((bsz, 0, self.t5_config.d_model), device=device)

        # Skip fully padded examples (all zeros) to save encoder compute.
        # Keep shape contract by scattering valid encodings back into a full tensor.
        with torch.no_grad():
            valid_ex = ex_batch.ne(0).any(dim=(2, 3, 4))
        if not valid_ex.any():
            device = ex_batch.device
            return torch.empty((bsz, 0, self.t5_config.d_model), device=device)

        flat_valid = valid_ex.reshape(-1)
        ex_flat_all = ex_batch.reshape(bsz * ex_count, *ex_batch.shape[2:])
        ex_flat = ex_flat_all[flat_valid]
        ex_ids, ex_mask, ex_obj_idx = self.grid_encoder.encode_example_pair(ex_flat)

        # compression
        if self.grid_compression_tokens is not None:
            ex_embeds = self.embed_tokens(ex_ids)
            if self.compression_method == "learned":
                ex_embeds = self.example_compressor(ex_embeds, mask=ex_mask)
            else:
                ex_embeds = self.example_projection(ex_embeds)
                ex_embeds = self.example_pooling(ex_embeds.transpose(1, 2)).transpose(
                    1, 2
                )
            ex_mask = torch.ones(
                ex_ids.size(0),
                self.example_compression_tokens,
                dtype=ex_mask.dtype,
                device=ex_mask.device,
            )
            if ex_obj_idx is not None and self.use_ope:
                ex_obj_idx = (
                    torch.nn.functional.adaptive_max_pool1d(
                        ex_obj_idx.unsqueeze(1).float(),
                        output_size=self.example_compression_tokens,
                    )
                    .squeeze(1)
                    .long()
                )
            else:
                ex_obj_idx = None
            ex_output = self.encoder(
                inputs_embeds=ex_embeds,
                attention_mask=ex_mask,
                object_idx=ex_obj_idx,
                return_dict=True,
            )
        else:
            ex_output = self.encoder(
                input_ids=ex_ids,
                attention_mask=ex_mask,
                object_idx=ex_obj_idx,
                return_dict=True,
            )

        h_valid = ex_output.last_hidden_state  # [N, seq_len, d_model]
        seq_len = int(h_valid.size(1))
        d_model = int(h_valid.size(2))
        h_all = torch.zeros(
            (bsz * ex_count, seq_len, d_model),
            device=h_valid.device,
            dtype=h_valid.dtype,
        )
        h_all[flat_valid] = h_valid
        return h_all.reshape(bsz, ex_count * seq_len, d_model)

    def _encode_examples(self, examples: torch.Tensor) -> torch.Tensor:
        """Encode example pairs and return concatenated hidden states.

        Handles compression and caching. Expects shape [B, num_examples, 2, H, W].
        """
        batch_size, num_examples = examples.shape[:2]
        # Cache key: cheap fingerprint based on the *first* example (index 0).
        # We assume each sample has at least one example; padded examples are all-zero.
        if examples.numel() > 0:
            ex0_first = examples[0, 0].reshape(-1)
            ex0_last = examples[-1, 0].reshape(-1)
            cache_key = hash(
                (
                    tuple(examples.shape),
                    float(ex0_first.sum().item()),
                    float((ex0_first.float().square().sum()).item()),
                    tuple(int(x) for x in ex0_first[:16].tolist()),
                    float(ex0_last.sum().item()),
                    float((ex0_last.float().square().sum()).item()),
                    tuple(int(x) for x in ex0_last[:16].tolist()),
                )
            )
        else:
            cache_key = hash((tuple(examples.shape), 0))

        cache_enabled = self._cache_valid

        logger.debug("Example cache MISS (key=%s)", cache_key)
        learnable_examples = min(int(self.max_learnable_examples), int(num_examples))

        hidden_learnable = self._encode_example_batch(examples[:, :learnable_examples])

        if learnable_examples >= int(num_examples):
            return hidden_learnable

        if cache_enabled and cache_key in self._example_cache:
            logger.debug("Example cache HIT (key=%s)", cache_key)
            hidden_frozen = self._example_cache[cache_key]
        else:
            with torch.no_grad():
                hidden_frozen = self._encode_example_batch(
                    examples[:, learnable_examples:]
                )
            if cache_enabled:
                self._example_cache[cache_key] = hidden_frozen.detach()

        return torch.cat([hidden_learnable, hidden_frozen], dim=1)

    def forward(
        self,
        state: TensorDict,
        *,
        example_hidden: torch.Tensor | None = None,
    ) -> ViTARCEncoderOutput:
        """Forward pass that encodes ARC state information.

        Args:
            state: TensorDict containing:
                - grid: Current grid [B, 1, H, W]
                - initial: Initial grid [B, 1, H, W]
                - last_grid: Previous grid [B, 1, H, W]
                - examples: Example pairs [B, num_examples, 2, H, W]
            example_hidden:
                Precomputed example hidden states [B, num_examples, seq_len, d_model]

        Returns:
            ViTARCEncoderOutput containing encoded representations.
        """
        # Get grids
        grid = state[StateKeys.GRID.value]
        initial = state.get(StateKeys.INITIAL.value)
        last_grid = state.get(StateKeys.LAST_GRID.value)
        index_grid = state.get(StateKeys.INDEX.value)

        index_ids: torch.Tensor | None = None

        # Encode temporal grids: <s> [initial] <sep> [last] <sep> [current] </s>
        if self.use_temporal:
            input_ids, attention_mask, temporal_ids, object_idx = (
                self.grid_encoder.encode_temporal_grids(
                    current=grid, initial=initial, last_grid=last_grid
                )
            )
        else:
            input_ids, attention_mask, object_idx = self.grid_encoder.encode_grid(grid)
            temporal_ids = None

            # Build index_ids aligned with input_ids: <s> [index_grid] </s>
            if self.use_index_grid and index_grid is not None:
                # index_grid comes as [B, H, W]; flatten to match grid tokens
                index_2d = (
                    index_grid.squeeze(1) if index_grid.dim() == 4 else index_grid
                )
                index_tokens = self.grid_encoder.flatten_index_grid(index_2d)
                batch_size = index_tokens.size(0)
                device = index_tokens.device
                zero_pad = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
                index_ids = torch.cat([zero_pad, index_tokens, zero_pad], dim=1)

        # Apply grid compression BEFORE encoder if enabled
        # This reduces memory in the encoder's self-attention
        if self.grid_compression_tokens is not None:
            # Get initial embeddings from input_ids
            inputs_embeds = self.embed_tokens(input_ids)

            # Debug: Check shape before compression
            if inputs_embeds.numel() == 0:
                logger.error(f"Empty inputs_embeds! input_ids shape: {input_ids.shape}")
                raise ValueError(
                    f"Empty embeddings from input_ids with shape {input_ids.shape}"
                )

            # Compress: [B, seq_len, d_model] -> [B, K, d_model]
            # where K = grid_compression_tokens (e.g., 256)
            if self.compression_method == "learned":
                inputs_embeds = self.grid_compressor(inputs_embeds, mask=attention_mask)
            else:  # pooling
                inputs_embeds = self.grid_projection(inputs_embeds)
                # Pool: [B, d_model, seq_len] -> [B, d_model, K]
                inputs_embeds = self.grid_pooling(inputs_embeds.transpose(1, 2))
                # Reshape: [B, d_model, K] -> [B, K, d_model]
                inputs_embeds = inputs_embeds.transpose(1, 2)

            # Update attention mask to match compressed sequence
            if attention_mask is not None:
                batch_size = attention_mask.size(0)
                attention_mask = torch.ones(
                    batch_size,
                    self.grid_compression_tokens,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

            # Compress object_idx to match compressed sequence if OPE is enabled
            if object_idx is not None and self.use_ope:
                # Use adaptive max pooling to preserve object information
                # [B, seq_len] -> [B, 1, seq_len] -> [B, 1, K] -> [B, K]
                object_idx_compressed = (
                    torch.nn.functional.adaptive_max_pool1d(
                        object_idx.unsqueeze(1).float(),
                        output_size=self.grid_compression_tokens,
                    )
                    .squeeze(1)
                    .long()
                )
            else:
                object_idx_compressed = None

            # Note: With compression, 1D Alibi RPE is used instead of 2D RPE
            # The encoder will automatically select the 1D distance matrix
            # based on the compressed sequence length (grid_compression_tokens)
            # OPE is still used via object_idx_compressed for spatial awareness

            # Encode with compressed inputs
            encoder_outputs = self.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                object_idx=object_idx_compressed,
                return_dict=True,
            )
        else:
            # Encode normally without compression
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                object_idx=object_idx,
                return_dict=True,
            )

        hidden_states = encoder_outputs.last_hidden_state

        # Add index embedding if enabled and available (Markov timestamp grid)
        if self.use_index_grid and index_ids is not None:
            if self.grid_compression_tokens is not None:
                # Compress index_ids to match compressed sequence
                # Use adaptive max pooling to preserve important indices
                # [B, seq_len] -> [B, 1, seq_len] -> [B, 1, K] -> [B, K]
                index_ids_compressed = (
                    torch.nn.functional.adaptive_max_pool1d(
                        index_ids.unsqueeze(1).float(),
                        output_size=self.grid_compression_tokens,
                    )
                    .squeeze(1)
                    .long()
                )
                hidden_states = hidden_states + self.index_embed(index_ids_compressed)
            else:
                hidden_states = hidden_states + self.index_embed(index_ids)

        # Add temporal embedding if using temporal grids
        if self.use_temporal and temporal_ids is not None:
            if self.grid_compression_tokens is not None:
                # Compress temporal_ids to match compressed sequence
                # Use adaptive max pooling to preserve temporal information
                # [B, seq_len] -> [B, 1, seq_len] -> [B, 1, K] -> [B, K]
                temporal_ids_compressed = (
                    torch.nn.functional.adaptive_max_pool1d(
                        temporal_ids.unsqueeze(1).float(),
                        output_size=self.grid_compression_tokens,
                    )
                    .squeeze(1)
                    .long()
                )
                hidden_states = hidden_states + self.temporal_embed(
                    temporal_ids_compressed
                )
            else:
                hidden_states = hidden_states + self.temporal_embed(temporal_ids)

        # Pre-compute examples outside checkpoint (or if passed in)
        if example_hidden is None and self.use_examples:
            examples = state.get(StateKeys.EXAMPLES.value)
            if examples is None:
                raise ValueError("Examples must be provided if use_examples is True")
            example_hidden = self._encode_examples(examples)

        if example_hidden is not None:
            hidden_states = self.example_cross_attention(
                grid_hidden=hidden_states,
                example_hidden=example_hidden,
                grid_mask=attention_mask,
            )

        return ViTARCEncoderOutput(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            input_ids=input_ids,
            temporal_ids=temporal_ids,
            object_idx=object_idx,
            example_hidden=example_hidden,
            example_mask=None,
            index_ids=index_ids,
        )
