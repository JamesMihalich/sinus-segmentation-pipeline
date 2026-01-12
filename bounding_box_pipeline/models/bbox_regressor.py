"""
3D Bounding Box Regression Neural Network.

Convolutional network for predicting 3D bounding boxes from volumetric data.
"""

from typing import Literal, Tuple

import torch
from torch import nn


class ResidualBlock3D(nn.Module):
    """
    3D Residual block with optional downsampling.

    Uses pre-activation design (BN -> ReLU -> Conv) for better gradient flow.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        stride = 2 if downsample else 1

        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Skip connection with optional projection
        if downsample or in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + identity


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


class BBoxRegressorResidual(nn.Module):
    """
    Residual 3D CNN for bounding box regression.

    Uses residual blocks for better gradient flow and improved training.
    Recommended for better IoU performance.
    """

    def __init__(
        self,
        input_size: Tuple[int, int, int] = (128, 128, 128),
        in_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.3,
    ) -> None:
        """
        Initialize the residual regressor.

        Args:
            input_size: Expected input volume size (D, H, W).
            in_channels: Number of input channels.
            base_channels: Base number of channels.
            dropout: Dropout probability in FC layers.
        """
        super().__init__()

        self.input_size = input_size
        c = base_channels

        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c, kernel_size=3, padding=1),
            nn.BatchNorm3d(c),
            nn.ReLU(inplace=True),
        )

        # Residual blocks with downsampling
        # 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.layer1 = ResidualBlock3D(c, c, downsample=True)       # 128->64
        self.layer2 = ResidualBlock3D(c, c * 2, downsample=True)   # 64->32
        self.layer3 = ResidualBlock3D(c * 2, c * 4, downsample=True)  # 32->16
        self.layer4 = ResidualBlock3D(c * 4, c * 8, downsample=True)  # 16->8
        self.layer5 = ResidualBlock3D(c * 8, c * 16, downsample=True) # 8->4

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool3d(1)

        # Regression head with smoother dimension reduction
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 16, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.gap(x)
        bbox = self.regressor(x)
        return bbox

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_regressor(
    variant: Literal["standard", "lite", "residual"] = "standard",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create bbox regressor.

    Args:
        variant: Model variant - "standard", "lite", or "residual".
        **kwargs: Additional arguments for model constructor.

    Returns:
        Instantiated model.
    """
    if variant == "standard":
        return BBoxRegressor3D(**kwargs)
    elif variant == "lite":
        return BBoxRegressorLite(**kwargs)
    elif variant == "residual":
        return BBoxRegressorResidual(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}")
