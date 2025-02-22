import torch
import torch.nn as nn
from torchvision import models

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


class ResNetGruTransformer(nn.Module):
    def __init__(self, embedding_size=256, num_classes=10):
        super(ResNetGruTransformer, self).__init__()
        self.resnet_embedding_t0 = ResNetModule(
            resnet_version="resnet18",
            resnet_weights="ResNet18_Weights.DEFAULT",
            embedding_size=256,
        )
        self.resnet_embedding_t1 = ResNetModule(
            resnet_version="resnet18",
            resnet_weights="ResNet18_Weights.DEFAULT",
            embedding_size=256,
        )
        self.lstm = torch.nn.GRU(
            input_size=256,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.embedding_layer = nn.Linear(512, embedding_size)
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x, return_embedding=False):
        B, N, T, H, W = x.shape  # B=batch, N=examples, T=time steps (2), H=W=30
        x = x.view(B * N, T, H, W)  # Flatten batch and example dims
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)
        x = x.view(B, N, -1)  # Reshape back to (B, N, Features)
        x = x.mean(
            dim=1
        )  # Aggregate over N (few-shot examples) to maintain batch integrity
        _, (h_n, _) = self.lstm(x.unsqueeze(1))  # Pass through LSTM
        embedding = self.embedding_layer(h_n[-1])

        if return_embedding:
            return embedding
        return self.fc(embedding)


import torch
import torch.nn as nn


class LSTMTransformer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        lstm_hidden_dim=256,
        transformer_dim=256,
        num_heads=4,
        num_layers=2,
    ):
        super().__init__()

        # LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=lstm_hidden_dim, batch_first=True
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=num_heads
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, t0, t1):
        # Stack inputs as a sequence: (batch_size, seq_len=2, embed_dim)
        x = torch.stack([t0, t1], dim=1)

        # Pass through LSTM
        _, (h_n, _) = self.lstm(x)  # h_n is (1, batch_size, lstm_hidden_dim)
        lstm_output = h_n.squeeze(0)  # (batch_size, lstm_hidden_dim)

        # Add sequence dimension for Transformer (batch_size, seq_len=1, embed_dim)
        transformer_input = lstm_output.unsqueeze(1)

        # Pass through Transformer
        transformer_output = self.transformer(
            transformer_input
        )  # (batch_size, seq_len=1, embed_dim)

        # Remove sequence dimension
        return transformer_output.squeeze(1)  # (batch_size, embed_dim)
