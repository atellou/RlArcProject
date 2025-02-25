import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from tensordict import TensorDict

import logging

logger = logging.getLogger(__name__)


class CnnPreTrainedModule(nn.Module):
    def __init__(
        self,
        model="efficientnet_b0",
        model_weights="EfficientNet_B0_Weights.DEFAULT",
        freeze: str = "ALL",
        do_not_freeze: str = None,
        embedding_size=128,
    ):
        super(CnnPreTrainedModule, self).__init__()

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

        for param in self.base_model.features[0][0].parameters():
            param.requires_grad = True
        for param in self.embedding_layer.parameters():
            param.requires_grad = True
        for module in self.base_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = True

        self.add_module("base_model", self.base_model)
        self.add_module("embedding_layer", self.embedding_layer)

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


class CnnAttention(nn.Module):
    def __init__(self, embedding_size=128, nheads=4, dropout=0.0, bias=True):
        """
        Initializes the CnnAttention module.

        Args:
            embedding_size (int, optional): The size of the embedding for the Cnn modules and the multi-head attention. Defaults to 128.
            nheads (int, optional): The number of attention heads in the multi-head attention module. Defaults to 8.
            dropout (float, optional): The dropout probability for the multi-head attention module. Defaults to 0.0.
            bias (bool, optional): Whether to include a bias term in the multi-head attention module. Defaults to True.

        The module consists of two CNN embeddings for time steps t0 and t1, and a MultiHeadAttention module.
        """

        super(CnnAttention, self).__init__()
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

    def forward(self, x):
        """
        Forward pass; runs the following process:
            1. Flatten the input
            2. Extract features from t=0 and t=1 using two different Cnn modules
            3. Run multi-head attention on the two feature vectors
            4. Reshape the output to match the input shape

        Args:
            x (torch.Tensor): input of shape (``N``, ``num_examples``, ``times_state``, ``height``, ``width``)

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
    def __init__(
        self,
        output_classes: dict,
        embedding_size=128,
        nheads=4,
        dropout=0.0,
        bias=True,
    ):
        """
        Initialize a CrossAttentionClassifier.

        Args:
            output_classes (dict): A dictionary mapping output types to the number of classes in each type.
            embedding_size (int, optional): The size of the embedding dimension. Defaults to 128.
            num_heads (int, optional): The number of attention heads in the multi-head attention module. Defaults to 4.
            dropout (float, optional): The dropout probability for the multi-head attention module. Defaults to 0.0.
            bias (bool, optional): Whether to include a bias term in the multi-head attention module. Defaults to True.

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

    def forward(self, embeddings):
        """
        Computes the forward pass of the CrossAttentionClassifier.

        Args:
            embeddings (torch.Tensor): input embeddings of shape (Batch Size, Sequence Length, Embedding Dimension)

        Returns:
            TensorDict of logits: A dictionary of output logits, mapping each output type to a tensor of shape (Batch Size, Num_classes)

        The forward pass consists of the following steps:

        1. Expand the query vector to match the batch size
        2. Apply cross-attention (query attends to the embeddings)
        3. Remove the sequence dimension (1)
        4. Pass through the classifier
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


CnnPreTrainedModule()
