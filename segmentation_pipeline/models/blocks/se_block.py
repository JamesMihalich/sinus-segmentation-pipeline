"""
Squeeze-and-Excitation block for 3D feature recalibration.

Implements channel attention mechanism for 3D convolutional networks.
"""

import torch
from torch import nn


class ChannelSELayer3D(nn.Module):
    """
    Squeeze-and-Excitation block for 3D feature maps.

    Performs channel-wise recalibration by:
    1. Squeeze: Global average pooling to get channel descriptors
    2. Excitation: FC layers to learn channel interdependencies
    3. Scale: Multiply input by learned channel weights

    Reference: Hu et al., "Squeeze-and-Excitation Networks" (2018)
    """

    def __init__(
        self,
        num_channels: int,
        reduction_ratio: int = 8,
    ) -> None:
        """
        Initialize SE block.

        Args:
            num_channels: Number of input/output channels.
            reduction_ratio: Reduction factor for bottleneck FC layer.
                Higher = fewer parameters, lower capacity.
                Common values: 2 (aggressive), 8 (standard), 16 (light).
        """
        super().__init__()

        num_channels_reduced = max(1, num_channels // reduction_ratio)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SE attention to input.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Recalibrated tensor of same shape.
        """
        batch_size, num_channels, D, H, W = x.size()

        # Squeeze: global average pooling
        squeeze = self.avg_pool(x).view(batch_size, num_channels)

        # Excitation: FC -> ReLU -> FC -> Sigmoid
        excitation = self.relu(self.fc1(squeeze))
        excitation = self.sigmoid(self.fc2(excitation))

        # Scale: multiply channels by learned weights
        scale = excitation.view(batch_size, num_channels, 1, 1, 1)

        return x * scale
