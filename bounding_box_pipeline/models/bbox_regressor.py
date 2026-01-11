"""
3D Bounding Box Regression Neural Network.

Convolutional network for predicting 3D bounding boxes from volumetric data.
"""

from typing import Literal, Tuple

import torch
from torch import nn


class BBoxRegressor3D(nn.Module):
    """
    3D Convolutional Neural Network for bounding box regression.

    Architecture:
    - 5 convolutional blocks with BatchNorm and MaxPool
    - Progressive channel expansion (32 -> 64 -> 128 -> 256 -> 512)
    - Fully connected regression head with dropout
    - Sigmoid activation for normalized [0, 1] coordinate output

    Input: (B, 1, D, H, W) where D=H=W=128 by default
    Output: (B, 6) normalized coordinates [z1, y1, x1, z2, y2, x2]
    """

    def __init__(
        self,
        input_size: Tuple[int, int, int] = (128, 128, 128),
        in_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.5,
    ) -> None:
        """
        Initialize the regressor.

        Args:
            input_size: Expected input volume size (D, H, W).
            in_channels: Number of input channels.
            base_channels: Base number of channels (doubled at each level).
            dropout: Dropout probability in FC layers.
        """
        super().__init__()

        self.input_size = input_size
        c = base_channels  # Base channels

        # Convolutional feature extractor
        # Each block: Conv3d -> BatchNorm -> ReLU -> MaxPool (2x downsample)
        self.features = nn.Sequential(
            # Block 1: 128 -> 64
            nn.Conv3d(in_channels, c, kernel_size=3, padding=1),
            nn.BatchNorm3d(c),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Block 2: 64 -> 32
            nn.Conv3d(c, c * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(c * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Block 3: 32 -> 16
            nn.Conv3d(c * 2, c * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(c * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Block 4: 16 -> 8
            nn.Conv3d(c * 4, c * 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(c * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Block 5: 8 -> 4
            nn.Conv3d(c * 8, c * 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(c * 16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Calculate flattened size after conv layers
        # For 128^3 input: 128 -> 64 -> 32 -> 16 -> 8 -> 4
        final_size = input_size[0] // 32  # 5 pooling layers with stride 2
        flattened_size = (c * 16) * (final_size ** 3)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6),
            nn.Sigmoid(),  # Output in [0, 1] for normalized coordinates
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, 1, D, H, W).

        Returns:
            Predicted bbox coordinates (B, 6) in [0, 1] range.
        """
        features = self.features(x)
        bbox = self.regressor(features)
        return bbox

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BBoxRegressorLite(nn.Module):
    """
    Lightweight version of BBoxRegressor3D.

    Uses fewer channels for faster training and lower memory usage.
    """

    def __init__(
        self,
        input_size: Tuple[int, int, int] = (128, 128, 128),
        in_channels: int = 1,
        dropout: float = 0.3,
    ) -> None:
        """Initialize lightweight regressor."""
        super().__init__()

        self.input_size = input_size

        # Lighter feature extractor
        self.features = nn.Sequential(
            # Block 1: 128 -> 64
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Block 2: 64 -> 32
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Block 3: 32 -> 16
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Block 4: 16 -> 8
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # Block 5: 8 -> 4
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(2),  # Always output 2x2x2
        )

        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 6),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.features(x)
        bbox = self.regressor(features)
        return bbox


def create_regressor(
    variant: Literal["standard", "lite"] = "standard",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create bbox regressor.

    Args:
        variant: Model variant - "standard" or "lite".
        **kwargs: Additional arguments for model constructor.

    Returns:
        Instantiated model.
    """
    if variant == "standard":
        return BBoxRegressor3D(**kwargs)
    elif variant == "lite":
        return BBoxRegressorLite(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")
