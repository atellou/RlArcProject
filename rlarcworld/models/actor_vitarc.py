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

import cv2
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from transformers.models.t5.configuration_t5 import T5Config

from rlarcworld.common.enums import ActionKeys, GridEncoderIdsType, StateKeys
from rlarcworld.common.types import (
    ACTION_KEYS_SET,
    GRID_ENCODER_KEYS_SET,
    STATE_REQUIRED_KEYS,
)
from rlarcworld.models.vit_arc import (
    CustomT5Stack,
    ViTARCForConditionalGeneration,
)
from rlarcworld.utils.device import enable_cuda

if TYPE_CHECKING:
    from rlarcworld.utils.device import CudaConfig

logger = logging.getLogger(__name__)

# =============================================================================
# Object Index Extractor - For Object-based Positional Encoding (OPE)
# =============================================================================


class ObjectIndexExtractor:
    """Extracts object indices from grids using contour detection.

    This provides the Object-based Positional Encoding (OPE). The object index is
    incorporated into the positional encoding as:
        E_pos = concat(E_obj, E_x, E_y)

    This allows the model to learn object-level abstractions while still
    having access to pixel-level (x, y) coordinates.
    """

    def __init__(
        self,
        padding_id: int | None = GridEncoderIdsType.PAD_ID,
    ):
        """Initialize the object index extractor.

        Args:
            padding_id: Padding token ID.
        """
        super().__init__()
        self.padding_id = padding_id

    def _extract_objects_opencv(self, grid: torch.Tensor) -> torch.Tensor:
        """Extract object indices using OpenCV contour detection.

        Args:
            grid: Single grid as torch tensor [H, W].

        Returns:
            Object indices [H, W] where each pixel has its object ID.
        """
        height, width = grid.shape
        object_indices = torch.zeros((height, width), dtype=torch.int64)
        background_id = torch.bincount(grid.view(-1))

        if self.padding_id is not None and background_id.argmax() == self.padding_id:
            background_id = torch.topk(background_id, k=2)[1][1]
        else:
            background_id = background_id.argmax()

        # Process each non-background color separately
        unique_colors = torch.unique(grid)
        current_obj_id = 0

        for color in unique_colors:
            if color in (background_id, self.padding_id):  # Skip background
                continue

            # Create binary mask for this color
            mask = np.where(grid == color, 1, 0).astype(np.uint8)

            # Find contours (connected components of this color)
            contours, _ = cv2.findContours(
                mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                # Bounding box approach (as in ViTARC paper)
                x, y, w, h = cv2.boundingRect(contour)
                # Mark all pixels in bounding box with same color as this object
                box = mask[y : y + h, x : x + w]
                object_indices[y : y + h, x : x + w] = torch.from_numpy(
                    current_obj_id * box
                )

                current_obj_id += 1

        return object_indices.to(grid.device)

    def __call__(self, grid: torch.Tensor) -> torch.Tensor:
        """Extract object indices from a batch of grids.

        Args:
            grid: Grid tensor [B, H, W] with color values.

        Returns:
            Object indices [B, H*W] where each pixel has its object ID.
        """
        return torch.stack([self._extract_objects_opencv(g).flatten() for g in grid])


# =============================================================================
# Grid Encoder - Direct grid-to-token conversion (no text tokenizer needed)
# =============================================================================


class GridEncoder:
    """Encodes ARC grids directly as token sequences.

    No text tokenizer needed - grid color values (0-10) are used directly
    as token IDs. This matches ViTARC's pixel-level tokenization where each
    grid cell becomes one token.

    Vocabulary:
        - 0-10: ARC color values
        - 11: <pad>
        - 12: <s> (BOS)
        - 13: </s> (EOS)
        - 14: <arc_endxgrid> (End of row)
        - 15: <arc_endygrid> (End of column)
        - 16: <arc_endgrid> (End of grid)
        - 17: <sep> (separator between grids)
        - 18: <mask> (End of ARC)
    """

    def __init__(self, grid_size: int = 30, use_ope: bool = False):
        """Initialize the grid encoder.

        Args:
            grid_size: Maximum grid dimension (default 30 for ARC).
            use_ope: Whether to use Object-based Positional Encoding (OPE).
        """
        self.grid_size = grid_size
        self.use_ope = use_ope
        self.encoder_ids = GridEncoderIdsType
        self.valid_non_encoded_ids = list(
            range(self.encoder_ids.COLOR_START_ID, self.encoder_ids.COLOR_END_ID + 1)
        ) + [self.encoder_ids.PAD_ID]
        self.use_ope = use_ope
        if use_ope:
            self.object_extractor = ObjectIndexExtractor(
                padding_id=self.encoder_ids.PAD_ID
            )

    def flatten_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """Flatten a grid to a token sequence with row/column end markers.

        For a grid of shape [B, H, W], produces tokens in row-major order:
            [row0_pixels] <endxgrid> [row1_pixels] <endxgrid> ... <endygrid> <endgrid>

        Args:
            grid: Grid tensor [B, 1, H, W] or [B, H, W].

        Returns:
            Token IDs [B, (H+1)*(W+1)].
        """
        if grid.dim() == 4:
            grid = grid.squeeze(1)

        batch_size, height, width = grid.shape
        if height != self.grid_size or width != self.grid_size:
            raise ValueError(
                "Expected grid of shapes [Batch Size, 1, Grid Size, Grid Size]"
                " or [Batch Size, Grid Size, Grid Size]"
            )
        unique_colors = torch.unique(grid)
        if not all(color in self.valid_non_encoded_ids for color in unique_colors):
            raise ValueError(
                "Grid contains invalid color values. Expected "
                f"{self.valid_non_encoded_ids}, got: {unique_colors.tolist()}"
            )

        device = grid.device

        # Build output with end markers
        # Each row: ENDXGRID at end. Last row: ENDYGRID. Corner: ENDGRID
        # Result shape: [B, H+1, W+1]
        out_h, out_w = height + 1, width + 1
        output = torch.full(
            (batch_size, out_h, out_w),
            self.encoder_ids.PAD_ID,
            dtype=torch.long,
            device=device,
        )

        # Copy grid values
        output[:, :height, :width] = grid.long()

        # Add ENDXGRID at end of each row (column index = width)
        output[:, :height, width] = self.encoder_ids.ENDXGRID_ID

        # Add ENDYGRID at end of each column (row index = height)
        output[:, height, :width] = self.encoder_ids.ENDYGRID_ID

        # Add ENDGRID at corner (bottom-right)
        output[:, height, width] = self.encoder_ids.ENDGRID_ID

        return output.view(batch_size, -1)

    def encode_grid(
        self,
        grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Encode current grid state for the encoder.

        Format: <s> [grid pixels] </s>

        Args:
            grid: Grid to encode [B, 1, H, W] or [B, H, W].

        Returns:
            Tuple of (input_ids, attention_mask, object_idx or None).
        """
        batch_size = grid.size(0)
        device = grid.device

        # Flatten grid
        grid_tokens = self.flatten_grid(grid)

        # Create special tokens
        bos = torch.full(
            (batch_size, 1), self.encoder_ids.BOS_ID, dtype=torch.long, device=device
        )
        eos = torch.full(
            (batch_size, 1), self.encoder_ids.EOS_ID, dtype=torch.long, device=device
        )

        # Concatenate: <s> [grid] </s>
        input_ids = torch.cat([bos, grid_tokens, eos], dim=1)
        attention_mask = (input_ids != self.encoder_ids.PAD_ID).long()

        # Extract object indices if extractor is available
        object_idx = None
        if self.use_ope:
            zero_pad = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            grid_2d = grid.squeeze(1) if grid.dim() == 4 else grid
            grid_object_idx = self._get_object_idx(grid_2d, grid_tokens)
            object_idx = torch.cat([zero_pad, grid_object_idx, zero_pad], dim=1)

        return input_ids, attention_mask, object_idx

    def encode_temporal_grids(
        self,
        current: torch.Tensor,
        initial: torch.Tensor | None = None,
        last_grid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Encode temporal sequence of grids.

        Format: <s> [initial] <sep> [last] <sep> [current] </s>

        If initial or last_grid are None, they are omitted from the sequence.

        Args:
            current: Current grid [B, 1, H, W].
            initial: Initial grid [B, 1, H, W] (optional).
            last_grid: Previous grid [B, 1, H, W] (optional).

        Returns:
            Tuple of:
                - input_ids [B, total_seq_len]
                - attention_mask [B, total_seq_len]
                - temporal_ids [B, total_seq_len] (0=initial, 1=last, 2=current)
                - object_idx [B, total_seq_len] or None
        """
        batch_size = current.size(0)
        device = current.device

        # Flatten all grids
        current_tokens = self.flatten_grid(current)
        seq_len_per_grid = current_tokens.size(1)

        # Build sequence parts
        bos = torch.full(
            (batch_size, 1), self.encoder_ids.BOS_ID, dtype=torch.long, device=device
        )
        sep = torch.full(
            (batch_size, 1), self.encoder_ids.SEP_ID, dtype=torch.long, device=device
        )
        eos = torch.full(
            (batch_size, 1), self.encoder_ids.EOS_ID, dtype=torch.long, device=device
        )

        # Build token sequence, temporal IDs, and object IDs
        tokens_list = [bos]
        temporal_list = [torch.zeros(batch_size, 1, dtype=torch.long, device=device)]
        zero_pad = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        object_list = [zero_pad] if self.use_ope else None

        if initial is not None:
            initial_tokens = self.flatten_grid(initial)
            tokens_list.extend([initial_tokens, sep])
            temporal_list.extend(
                [
                    torch.zeros(
                        batch_size, seq_len_per_grid, dtype=torch.long, device=device
                    ),
                    torch.zeros(batch_size, 1, dtype=torch.long, device=device),
                ]
            )
            if object_list is not None:
                initial_2d = initial.squeeze(1) if initial.dim() == 4 else initial
                initial_obj = self._get_object_idx(initial_2d, initial_tokens)
                object_list.extend([initial_obj, zero_pad])

        if last_grid is not None:
            if initial is not None and torch.equal(initial, last_grid):
                last_tokens = torch.full(
                    (batch_size, seq_len_per_grid),
                    self.encoder_ids.EMPTY_ID,
                    dtype=torch.long,
                    device=device,
                )
                # Empty grid has no objects
                last_obj = torch.zeros(
                    batch_size, seq_len_per_grid, dtype=torch.long, device=device
                )
            else:
                last_tokens = self.flatten_grid(last_grid)
                last_2d = last_grid.squeeze(1) if last_grid.dim() == 4 else last_grid
                last_obj = (
                    self._get_object_idx(last_2d, last_tokens)
                    if object_list is not None
                    else None
                )
            tokens_list.extend([last_tokens, sep])
            temporal_list.extend(
                [
                    torch.ones(
                        batch_size, seq_len_per_grid, dtype=torch.long, device=device
                    ),
                    torch.ones(batch_size, 1, dtype=torch.long, device=device),
                ]
            )
            if object_list is not None:
                object_list.extend([last_obj, zero_pad])

        # Always add current grid
        tokens_list.extend([current_tokens, eos])
        temporal_list.extend(
            [
                torch.full(
                    (batch_size, seq_len_per_grid), 2, dtype=torch.long, device=device
                ),
                torch.full((batch_size, 1), 2, dtype=torch.long, device=device),
            ]
        )
        if object_list is not None:
            current_2d = current.squeeze(1) if current.dim() == 4 else current
            current_obj = self._get_object_idx(current_2d, current_tokens)
            object_list.extend([current_obj, zero_pad])

        input_ids = torch.cat(tokens_list, dim=1)
        temporal_ids = torch.cat(temporal_list, dim=1)
        attention_mask = (input_ids != self.encoder_ids.PAD_ID).long()
        object_idx = torch.cat(object_list, dim=1) if object_list is not None else None

        return input_ids, attention_mask, temporal_ids, object_idx

    def encode_example_pair(
        self, example: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Encode an example input-output pair.

        Format: <s> [input_grid] <sep> [output_grid] </s>

        Args:
            example: Example pair [B, 2, H, W] where dim 1 is (input, output).

        Returns:
            Tuple of (input_ids, attention_mask).
        """
        batch_size = example.size(0)
        device = example.device

        # Split into input and output grids
        input_grid = example[:, 0]  # [B, H, W]
        output_grid = example[:, 1]  # [B, H, W]

        # Add batch dim for flatten_grid (expects [B, H, W])
        input_tokens = self.flatten_grid(input_grid.unsqueeze(1))
        output_tokens = self.flatten_grid(output_grid.unsqueeze(1))

        # Create special tokens
        zero_pad = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        bos = torch.full(
            (batch_size, 1), self.encoder_ids.BOS_ID, dtype=torch.long, device=device
        )
        sep = torch.full(
            (batch_size, 1), self.encoder_ids.SEP_ID, dtype=torch.long, device=device
        )
        eos = torch.full(
            (batch_size, 1), self.encoder_ids.EOS_ID, dtype=torch.long, device=device
        )

        # Concatenate: <s> [input] <sep> [output] </s>
        input_ids: torch.Tensor = torch.cat(
            [bos, input_tokens, sep, output_tokens, eos], dim=1
        )
        attention_mask: torch.Tensor = (input_ids != self.encoder_ids.PAD_ID).long()

        if self.use_ope:
            input_object_idx = self._get_object_idx(input_grid, input_tokens)
            output_object_idx = self._get_object_idx(output_grid, output_tokens)
            object_idx = torch.cat(
                [zero_pad, input_object_idx, zero_pad, output_object_idx, zero_pad],
                dim=1,
            )
            return input_ids, attention_mask, object_idx

        return input_ids, attention_mask, None

    def _get_object_idx(
        self, grid: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor | None:
        """Extract object indices from a batch of grids.

        Args:
            grid: Grid tensor [B, H, W] with color values.
            input_ids: Input IDs [B, seq_len].

        Returns:
            Object indices [B, H*W] where each pixel has its object ID.
        """
        device = grid.device
        grid_2d = grid.squeeze(1) if grid.dim() == 4 else grid
        if self.use_ope:
            object_idx = self.object_extractor(grid_2d)

            # Pad object_idx to match input_ids length
            batch_size, seq_len = input_ids.shape
            obj_seq_len = object_idx.size(1)
            if obj_seq_len < seq_len:
                padding = torch.zeros(
                    (batch_size, seq_len - obj_seq_len),
                    dtype=object_idx.dtype,
                    device=device,
                )
                object_idx = torch.cat([object_idx, padding], dim=1)
            elif obj_seq_len > seq_len:
                object_idx = object_idx[:, :seq_len]
            return object_idx
        return None


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
