"""Neural network building blocks for ARC agents.

This module defines reusable neural network components used by the
actor and critic, including CNN feature extractors, multi-head
attention, and cross-attention classifier heads.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torchvision import models

logger = logging.getLogger(__name__)


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
        model: str = "efficientnet_b0",
        model_weights: str | None = "DEFAULT",
        freeze: list[str] | None = None,
        do_not_freeze: list[str] | None = None,
        embedding_size: int = 128,
    ):
        """Initialize the pre-trained CNN module.

        Args:
            model (str, optional): The name of the pre-trained model.
             Defaults to "efficientnet_b0".
            model_weights (str, optional): The weights of the pre-trained model.
             Defaults to "DEFAULT".
            freeze (list[str], optional): List of layers to freeze.
             Defaults to None.
            do_not_freeze (list[str], optional): List of layers to unfreeze.
             Defaults to None.
            embedding_size (int, optional): The size of the embedding.
             Defaults to 128.
        """
        super().__init__()
        self.base_model = getattr(models, model)(weights=model_weights)
        self.base_model.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=1, padding=1, bias=False
        )  # Adjust for 1-channel input
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
        self._replace_batchnorm_with_groupnorm(self.base_model)

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
