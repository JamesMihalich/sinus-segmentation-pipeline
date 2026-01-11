"""
Residual block with GroupNorm and SE attention for 3D segmentation.
"""

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .se_block import ChannelSELayer3D


def _pick_groups(channels: int, target_groups: int = 8) -> int:
    """
    Find largest group divisor for GroupNorm.

    Args:
        channels: Number of channels.
        target_groups: Desired number of groups.

    Returns:
        Valid number of groups that divides channels.
    """
    for g in range(min(target_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ResNetBlockSE(nn.Module):
    """
    Residual block with GroupNorm and Squeeze-and-Excitation attention.

    Architecture:
        x -> Conv3d -> GroupNorm -> ReLU -> Conv3d -> GroupNorm -> SE -> (+identity) -> ReLU

    Supports optional downsampling via stride and channel projection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        se_reduction_ratio: int = 8,
        num_groups: int = 8,
        include_gn_in_proj: bool = True,
        use_interpolation_safeguard: bool = False,
    ) -> None:
        """
        Initialize residual block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride for first conv (1 or 2 for downsampling).
            se_reduction_ratio: Reduction ratio for SE block.
            num_groups: Target number of groups for GroupNorm.
            include_gn_in_proj: Include GroupNorm in projection path.
            use_interpolation_safeguard: Add trilinear interpolation fallback
                for shape mismatches (useful for variable input sizes).
        """
        super().__init__()

        self.use_interpolation_safeguard = use_interpolation_safeguard

        # Compute valid group count
        g_out = _pick_groups(out_channels, num_groups)

        # Main path
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.gn1 = nn.GroupNorm(num_groups=g_out, num_channels=out_channels)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.gn2 = nn.GroupNorm(num_groups=g_out, num_channels=out_channels)

        # SE attention
        self.se = ChannelSELayer3D(out_channels, reduction_ratio=se_reduction_ratio)

        # Projection for residual connection when dimensions change
        self.proj = None
        if stride != 1 or in_channels != out_channels:
            if include_gn_in_proj:
                self.proj = nn.Sequential(
                    nn.Conv3d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.GroupNorm(num_groups=g_out, num_channels=out_channels),
                )
            else:
                self.proj = nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C_in, D, H, W).

        Returns:
            Output tensor of shape (B, C_out, D', H', W').
        """
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.se(out)

        # Residual connection
        if self.proj is not None:
            identity = self.proj(identity)

        # Handle shape mismatch (safeguard for odd dimensions)
        if self.use_interpolation_safeguard and out.shape != identity.shape:
            identity = F.interpolate(
                identity,
                size=out.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        out = out + identity
        out = self.act(out)

        return out
