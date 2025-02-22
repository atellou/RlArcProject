import torch
import torch.nn as nn
from torchvision import models


# Define Pretrained CNN Model (ResNet50, ~23M parameters)
class ResNetModule(nn.Module):
    def __init__(
        self,
        resnet_version="resnet50",
        resnet_weights="ResNet50_Weights.DEFAULT",
        freeze: str = "ALL",
        do_not_freeze: str = "",
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
        self.base_model.fc = nn.Identity()  # Remove classification head
        self.embedding_layer = nn.Linear(
            2048, embedding_size
        )  # ResNet50 output -> embedding

        # Freeze all layers except conv1 and embedding_layer
        freeze = freeze.split(",")
        if "ALL" in freeze:
            for param in self.base_model.parameters():
                param.requires_grad = False
        elif freeze:
            for layer in freeze:
                for param in getattr(self.base_model, layer).parameters():
                    param.requires_grad = False

        do_not_freeze = do_not_freeze.split(",")
        if do_not_freeze:
            for layer in do_not_freeze:
                for param in getattr(self.base_model, layer).parameters():
                    param.requires_grad = True

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
