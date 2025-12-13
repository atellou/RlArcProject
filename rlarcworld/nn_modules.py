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
import torch.nn.functional as F
from tensordict import TensorDict
from torchvision import models
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.t5.configuration_t5 import T5Config

from rlarcworld.common.enums import ActionKeys, StateKeys
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


def resolve_submodule(root: nn.Module, path: str) -> nn.Module | None:
    """Resolve a submodule by dotted path for sequential modules.

    Examples:
        - "features" -> root.features
        - "features.0" -> root.features[0]
        - "features.0.0" -> root.features[0][0]
        - "classifier" -> root.classifier
    """
    cur: nn.Module | torch.nn.Module = root
    if not path:
        return cur
    for part in path.split("."):
        if part.isdigit():
            idx = int(part)
            try:
                cur = cur[idx]
            except Exception:
                return None
        else:
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
    return cur


class CnnPreTrainedModule(nn.Module):
    """Pre-trained CNN module."""

    def __init__(
        self,
        model: str = "efficientnet_v2_s",
        model_weights: str | None = "DEFAULT",
        norm_type: str = "group_norm",
        freeze: list[str] | None = None,
        do_not_freeze: list[str] | None = None,
        embedding_size: int = 128,
    ):
        """Initialize the pre-trained CNN module.

        Args:
            model (str, optional): The name of the pre-trained model.
             Defaults to "efficientnet_v2_s".
            model_weights (str, optional): The weights of the pre-trained model.
             Defaults to "DEFAULT".
            norm_type (str, optional): The type of normalization to use.
             Defaults to "group_norm".
            freeze (list[str], optional): List of layers to freeze.
             Defaults to None.
            do_not_freeze (list[str], optional): List of layers to unfreeze.
             Defaults to None.
            embedding_size (int, optional): The size of the embedding.
             Defaults to 128.
        """
        super().__init__()
        self.base_model = getattr(models, model)(weights=model_weights)

        # Adjust for 1-channel input
        cnn1_out_channels: int = self.base_model.features[0][0].out_channels
        self.base_model.features[0][0] = nn.Conv2d(
            1, cnn1_out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.embedding_layer = nn.Linear(
            self.base_model.classifier[1].in_features,
            embedding_size,
        )
        self.base_model.classifier = nn.Identity()

        # Freeze all layers except conv1 and embedding_layer
        if freeze is None:
            freeze = ["ALL"]
        if "ALL" in freeze:
            for param in self.base_model.parameters():
                param.requires_grad = False
        else:
            for layer in freeze:
                layer = layer.strip()
                if not layer:
                    continue
                target = resolve_submodule(self.base_model, layer)
                if target is not None:
                    for param in target.parameters():
                        param.requires_grad = False
                else:
                    logger.warning(
                        "Layer path '%s' not found in model to freeze", layer
                    )

        if do_not_freeze is not None:
            for layer in do_not_freeze:
                layer = layer.strip()
                if not layer:
                    continue
                target = resolve_submodule(self.base_model, layer)
                if target is not None:
                    for param in target.parameters():
                        param.requires_grad = True
                else:
                    logger.warning(
                        "Layer path '%s' not found in model to unfreeze", layer
                    )

        for param in self.base_model.features[0][0].parameters():
            param.requires_grad = True
        for param in self.embedding_layer.parameters():
            param.requires_grad = True

        # CRITICAL: Replace BatchNorm with GroupNorm for small batch stability
        # BatchNorm is unstable with small batches (batch_size=4) and causes NaN
        # GroupNorm is batch-size independent and more stable
        if norm_type == "group_norm":
            self._replace_batchnorm_with_groupnorm(self.base_model)
        elif norm_type == "batch_norm":
            for module in self.base_model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    for param in module.parameters():
                        param.requires_grad = True
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")

        self.add_module("base_model", self.base_model)
        self.add_module("embedding_layer", self.embedding_layer)

    def _replace_batchnorm_with_groupnorm(self, module: nn.Module) -> None:
        """Replace all BatchNorm2d layers with GroupNorm recursively.

        Args:
            module: The module to process
        """
        for name, child in list(module.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                # Replace with GroupNorm (num_groups=8 is a good default)
                num_channels = child.num_features
                num_groups = min(8, num_channels)  # Ensure divisible
                while num_channels % num_groups != 0:
                    num_groups -= 1
                group_norm = nn.GroupNorm(
                    num_groups=num_groups,
                    num_channels=num_channels,
                    eps=child.eps,
                    affine=True,
                )
                # Copy affine parameters if they exist
                if child.affine:
                    with torch.no_grad():
                        group_norm.weight.copy_(child.weight)
                        group_norm.bias.copy_(child.bias)
                setattr(module, name, group_norm)
                logger.debug(
                    f"Replaced BatchNorm2d with GroupNorm "
                    f"(channels={num_channels}, groups={num_groups})"
                )
            else:
                # Recursively process child modules
                self._replace_batchnorm_with_groupnorm(child)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the output embedding for a given input image.

        Args:
            x (torch.Tensor): The input image, of shape (N, C, H, W).

        Returns:
            torch.Tensor: The output embedding, of shape (N, embedding_size).
        """
        x = self.base_model(x)
        embedding = self.embedding_layer(x)
        return embedding


class MultiHeadAttention(nn.Module):
    """Computes multi-head attention. Supports nested or padded tensors.

    Ref:https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize the multi-head attention module.

        Args:
            E_q (int): Size of embedding dim for query
            E_k (int): Size of embedding dim for key
            E_v (int): Size of embedding dim for value
            E_total (int): Total embedding dim of combined heads post input projection.
             Each head has dim E_total // nheads
            nheads (int): Number of heads
            dropout (float, optional): Dropout probability. Default: 0.0
            bias (bool, optional): Whether to add bias to input projection.
             Default: True
            device (torch.device, optional): Device to use. Default: None
            dtype (torch.dtype, optional): Data type to use. Default: None
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape
             (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout, is_causal=is_causal
        )
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output


class CnnAttention(nn.Module):
    """CnnAttention module.

    The module consists of two CNN embeddings for time steps t0 and t1,
    and a MultiHeadAttention module.
    """

    def __init__(
        self,
        embedding_size: int = 128,
        nheads: int = 4,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """Initializes the CnnAttention module.

        Args:
            embedding_size (int, optional): The size of the embedding for the Cnn
             modules and the multi-head attention module. Defaults to 128.
            nheads (int, optional): The number of attention heads in
             the multi-head attention module. Defaults to 4.
            dropout (float, optional): The dropout probability for
             the multi-head attention module. Defaults to 0.0.
            bias (bool, optional): Whether to include a bias term in
             the multi-head attention module. Defaults to True.

        """
        super().__init__()
        self.cnn_embedding_t0 = CnnPreTrainedModule()
        self.cnn_embedding_t1 = CnnPreTrainedModule()

        self.mha = MultiHeadAttention(
            E_q=embedding_size,
            E_k=embedding_size,
            E_v=embedding_size,
            E_total=embedding_size,
            nheads=nheads,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Runs the following process:
            1. Flattens the input
            2. Extract features from t=0 and t=1 using two different Cnn modules
            3. Run multi-head attention on the two feature vectors
            4. Reshape the output to match the input shape

        Args:
            x (torch.Tensor): input of shape
             (``N``, ``num_examples``, ``times_state``, ``height``, ``width``)

        Returns:
            x (torch.Tensor): output of shape (``N``, ``num_examples``, ``E_out``)
        """
        batch_size, num_examples, times_state, height, width = x.shape
        x = x.view(batch_size * num_examples, times_state, height, width)
        t0 = self.cnn_embedding_t0(x[:, 0].unsqueeze(1))
        t1 = self.cnn_embedding_t1(x[:, 1].unsqueeze(1))
        x = self.mha(query=t1, key=t0, value=t0)
        return x.view(batch_size, num_examples, -1)


class CrossAttentionClassifier(nn.Module):
    """CrossAttentionClassifier for the ARC environment."""

    def __init__(
        self,
        output_classes: dict,
        embedding_size: int = 128,
        nheads: int = 4,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """Initialize a CrossAttentionClassifier.

        Args:
            output_classes (dict): A dictionary mapping output
             types to the number of classes in each type.
            embedding_size (int, optional): The size of the
             embedding dimension. Defaults to 128.
            nheads (int, optional): The number of attention heads in
             the multi-head attention module. Defaults to 4.
            dropout (float, optional): The dropout probability for the
             multi-head attention module. Defaults to 0.0.
            bias (bool, optional): Whether to include a bias term in
             the multi-head attention module. Defaults to True.
        """
        super().__init__()
        self.output_classes = output_classes
        # Learnable query vector (1 per sample)
        self.query = nn.Parameter(
            torch.randn(1, 1, embedding_size)
        )  # Shape: [1, 1, 128]

        # Multihead cross-attention
        self.cross_attention = MultiHeadAttention(
            E_q=embedding_size,
            E_k=embedding_size,
            E_v=embedding_size,
            E_total=embedding_size,
            nheads=nheads,
            dropout=dropout,
            bias=bias,
        )

        # MLP Classifier
        self.classifier = torch.nn.ModuleDict(
            {
                k: nn.Linear(embedding_size, num_classes)
                for k, num_classes in output_classes.items()
            }
        )

    def forward(self, embeddings: torch.Tensor) -> TensorDict:
        """Computes the forward pass of the CrossAttentionClassifier.

        The forward pass consists of the following steps:

        1. Expand the query vector to match the batch size
        2. Apply cross-attention (query attends to the embeddings)
        3. Remove the sequence dimension (1)
        4. Pass through the classifier

        Args:
            embeddings (torch.Tensor): input embeddings of shape
             (Batch Size, Sequence Length, Embedding Dimension)

        Returns:
            TensorDict of logits: A dictionary of output logits,
            mapping each output type to a tensor of shape (Batch Size, Num_classes)

        """
        batch_size = embeddings.size(0)

        # Expand query to match batch size: [batch_size, 1, 128]
        query = self.query.expand(batch_size, -1, -1)

        # Apply cross-attention (query attends to the embeddings)
        attended_output = self.cross_attention(
            query, embeddings, embeddings
        )  # Shape: [batch_size, 1, 128]

        # Remove the sequence dimension (1) -> [batch_size, 128]
        attended_output = attended_output.squeeze(1)

        # Pass through classifier
        logits = TensorDict(
            {k: self.classifier[k](attended_output) for k in self.classifier.keys()}
        )

        return logits


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
