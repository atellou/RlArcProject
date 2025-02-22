import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


import logging

logger = logging.getLogger(__name__)


# Define Pretrained CNN Model (ResNet50, ~23M parameters)
class ResNetModule(nn.Module):
    def __init__(
        self,
        resnet_version="resnet50",
        resnet_weights="ResNet50_Weights.DEFAULT",
        freeze: str = "ALL",
        do_not_freeze: str = None,
        embedding_size=256,
    ):
        """
        Initialize a ResNet-based model for learning an embedding from images.

        Args:
            resnet_version (str): A string specifying the ResNet version to use
                (e.g. "resnet50", "resnet101", etc.). Defaults to "resnet50".
            resnet_weights (str): A string specifying the ResNet weights to use.
                Defaults to "ResNet50_Weights.DEFAULT".
            freeze (bool): Comma separated string specifying which layers to freeze. Defaults to "ALL".
            embedding_size (int): The size of the output embedding. Defaults to
                256.
        """
        super(ResNetModule, self).__init__()
        self.base_model = getattr(models, resnet_version)(weights=resnet_weights)
        self.base_model.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )  # Adjust for 1-channel input
        self.embedding_layer = nn.Linear(
            self.base_model.fc.in_features, embedding_size
        )  # ResNet50 output -> embedding
        self.base_model.fc = nn.Identity()  # Remove classification head

        # Freeze all layers except conv1 and embedding_layer
        freeze = freeze.split(",")
        if "ALL" in freeze:
            for param in self.base_model.parameters():
                param.requires_grad = False
        else:
            for layer in freeze:
                if not hasattr(self.base_model, layer):
                    for param in getattr(self.base_model, layer).parameters():
                        param.requires_grad = False

        if do_not_freeze is not None:
            do_not_freeze = do_not_freeze.split(",")
            for layer in do_not_freeze:
                if hasattr(self.base_model, layer):
                    for param in getattr(self.base_model, layer).parameters():
                        param.requires_grad = True
                else:
                    logging.warning(f"Layer {layer} not found in model")

        for param in self.base_model.conv1.parameters():
            param.requires_grad = True
        for param in self.embedding_layer.parameters():
            param.requires_grad = True
        for module in self.base_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = True

    def forward(self, x):
        """
        Compute the output embedding for a given input image.

        Args:
            x (torch.Tensor): The input image, of shape (N, C, H, W).

        Returns:
            torch.Tensor: The output embedding, of shape (N, embedding_size).
        """
        x = self.base_model(x)
        embedding = self.embedding_layer(x)
        return embedding


class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.
    Ref:https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection. Each head
            has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True,
        device=None,
        dtype=None,
    ):
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
        attn_mask=None,
        is_causal=False,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (``N``, ``L_q``, ``E_qk``)
            key (torch.Tensor): key of shape (``N``, ``L_kv``, ``E_qk``)
            value (torch.Tensor): value of shape (``N``, ``L_kv``, ``E_v``)
            attn_mask (torch.Tensor, optional): attention mask of shape (``N``, ``L_q``, ``L_kv``) to pass to SDPA. Default: None
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


class ResNetAttention(nn.Module):
    def __init__(self, embedding_size=256, nheads=8, dropout=0.0, bias=True):
        """
        Initializes the ResNetAttention module.

        Args:
            embedding_size (int, optional): The size of the embedding for the ResNet modules and the multi-head attention. Defaults to 256.
            nheads (int, optional): The number of attention heads in the multi-head attention module. Defaults to 8.
            dropout (float, optional): The dropout probability for the multi-head attention module. Defaults to 0.0.
            bias (bool, optional): Whether to include a bias term in the multi-head attention module. Defaults to True.

        The module consists of two ResNet embeddings for time steps t0 and t1, and a MultiHeadAttention module.
        """

        super(ResNetAttention, self).__init__()
        self.resnet_embedding_t0 = ResNetModule(
            resnet_version="resnet18",
            resnet_weights="ResNet18_Weights.DEFAULT",
            do_not_freeze="layer4",
            embedding_size=embedding_size,
        )
        self.resnet_embedding_t1 = ResNetModule(
            resnet_version="resnet18",
            resnet_weights="ResNet18_Weights.DEFAULT",
            do_not_freeze="layer4",
            embedding_size=embedding_size,
        )

        self.mha = MultiHeadAttention(
            E_q=embedding_size,
            E_k=embedding_size,
            E_v=embedding_size,
            E_total=embedding_size,
            nheads=nheads,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, x):
        """
        Forward pass; runs the following process:
            1. Flatten the input
            2. Extract features from t=0 and t=1 using two different ResNets
            3. Run multi-head attention on the two feature vectors
            4. Reshape the output to match the input shape

        Args:
            x (torch.Tensor): input of shape (``N``, ``num_examples``, ``times_state``, ``height``, ``width``)

        Returns:
            x (torch.Tensor): output of shape (``N``, ``num_examples``, ``E_out``)
        """
        batch_size, num_examples, times_state, height, width = x.shape
        x = x.view(batch_size * num_examples, times_state, height, width)
        t0 = self.resnet_embedding_t0(x[:, 0].unsqueeze(1))
        t1 = self.resnet_embedding_t1(x[:, 1].unsqueeze(1))
        x = self.mha(t0, t1, t1)
        return x.view(batch_size, num_examples, -1)
