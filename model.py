"""Module defining the Model class for Debiasify.

This module implements a dual-branch network based on a pretrained ResNet18.
It extracts both shallow (bias-prone) and deep (robust) feature representations,
and provides two classification heads (shallow and deep) to facilitate the
self-distillation and unsupervised debiasing strategy as described in the Debiasify paper.

The architecture:
  - Shared initial layers: conv1, bn1, relu, maxpool, layer1, layer2 (shallow branch).
  - Alignment layer: maps shallow features from 128 to 512 dimensions.
  - Shallow branch: Global-average pooled and aligned features get classified via shallow_classifier.
  - Deep branch: The output from layer2 is further processed by layer3 and layer4, then pooled and classified via deep_classifier.
  
Public methods:
  - forward(x): returns a dictionary with keys "shallow_features", "deep_features", "logits_shallow", and "logits_deep".
  - get_shallow_features(x): returns the aligned shallow feature vector.
  - get_deep_features(x): returns the deep feature vector.
  
Configuration parameters are read from the provided config dictionary.
"""

from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Model(nn.Module):
    """Debiasify Model class implementing a dual-branch ResNet18.

    Attributes:
        conv1 (nn.Module): Convolutional layer from ResNet18.
        bn1 (nn.Module): BatchNorm layer following conv1.
        relu (nn.Module): ReLU activation.
        maxpool (nn.Module): Max pooling layer.
        layer1 (nn.Module): First residual block.
        layer2 (nn.Module): Second residual block (end of shallow branch).
        layer3 (nn.Module): Third residual block (deep branch begins).
        layer4 (nn.Module): Fourth residual block (deep branch ends).
        avgpool (nn.Module): Global average pooling.
        alignment (nn.Module): Alignment layer mapping shallow features (128) to deep features (512).
        shallow_classifier (nn.Module): Fully connected classifier for shallow features.
        deep_classifier (nn.Module): Fully connected classifier for deep features.
    """

    def __init__(self, config: Dict[str, Any], num_classes: int = 2) -> None:
        """
        Initializes the Model.

        Args:
            config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
            num_classes (int, optional): Number of classes for classification.
                                         Defaults to 2.
        """
        super(Model, self).__init__()

        model_config: Dict[str, Any] = config.get("model", {})
        backbone: str = model_config.get("backbone", "ResNet18")
        pretrained: bool = model_config.get("pretrained", True)

        if backbone.lower() != "resnet18":
            raise ValueError("Currently, only ResNet18 backbone is supported.")

        # Load the pretrained ResNet18 backbone.
        resnet = models.resnet18(pretrained=pretrained)

        # Extract initial layers.
        self.conv1 = resnet.conv1         # Output: 64 channels.
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # Shallow branch: layer1 and layer2. In ResNet18, layer1 outputs 64 and layer2 outputs 128 channels.
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2  # End of second module -> shallow branch.

        # Deep branch: layer3 and layer4. In ResNet18, final output channels from layer4 is 512.
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4  # Final module.

        # Global average pooling layer.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Alignment layer: maps shallow features from 128 (from layer2) to 512 dimensions.
        self.alignment = nn.Sequential(
            nn.Linear(128, 512, bias=True),
            nn.ReLU(inplace=True)
        )

        # Classifier heads.
        self.shallow_classifier = nn.Linear(512, num_classes)
        self.deep_classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Processes input x through shared initial layers, then branches into the shallow
        and deep paths. Returns a dictionary with both feature representations and logits.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - "shallow_features": Aligned shallow feature vector (batch_size, 512).
                - "deep_features": Deep feature vector (batch_size, 512).
                - "logits_shallow": Logits from the shallow classifier (batch_size, num_classes).
                - "logits_deep": Logits from the deep classifier (batch_size, num_classes).
        """
        # Shared initial processing.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x_shallow = self.layer2(x)  # Feature map from shallow branch.

        # Shallow branch.
        shallow_pool = self.avgpool(x_shallow)         # Shape: (batch, 128, 1, 1)
        shallow_pool = torch.flatten(shallow_pool, 1)    # Shape: (batch, 128)
        shallow_features = self.alignment(shallow_pool)  # Shape: (batch, 512)
        logits_shallow = self.shallow_classifier(shallow_features)

        # Deep branch.
        x_deep = self.layer3(x_shallow)
        x_deep = self.layer4(x_deep)
        deep_pool = self.avgpool(x_deep)                 # Shape: (batch, 512, 1, 1)
        deep_features = torch.flatten(deep_pool, 1)        # Shape: (batch, 512)
        logits_deep = self.deep_classifier(deep_features)

        return {
            "shallow_features": shallow_features,
            "deep_features": deep_features,
            "logits_shallow": logits_shallow,
            "logits_deep": logits_deep,
        }

    def get_shallow_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes and returns the aligned shallow feature vector from input x.

        This method processes the input through the shared initial layers and the shallow branch,
        performs global average pooling, and applies the alignment layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Aligned shallow feature vector of shape (batch_size, 512).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x_shallow = self.layer2(x)
        pool = self.avgpool(x_shallow)
        pool = torch.flatten(pool, 1)
        aligned_features = self.alignment(pool)
        return aligned_features

    def get_deep_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes and returns the deep feature vector from input x.

        This method processes the input through the shared initial layers, then through
        the deep branch (layer3 and layer4), and finally applies global average pooling.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Deep feature vector of shape (batch_size, 512).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x_shallow = self.layer2(x)
        x_deep = self.layer3(x_shallow)
        x_deep = self.layer4(x_deep)
        pool = self.avgpool(x_deep)
        deep_features = torch.flatten(pool, 1)
        return deep_features
