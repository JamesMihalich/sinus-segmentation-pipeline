"""
3D Residual U-Net with Squeeze-and-Excitation blocks.

Configurable skip connection mode (concatenation or addition).
"""

from typing import List, Literal, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .blocks.residual_block import ResNetBlockSE


class ResidualUnetSE3D(nn.Module):
    """
    3D Residual U-Net with Squeeze-and-Excitation attention.

    A encoder-decoder architecture with:
    - Residual blocks at each level
    - SE attention for channel recalibration
    - Configurable skip connection mode (concat or additive)
    - 4 encoder levels with 2x downsampling

    Args:
        in_channels: Number of input channels (default: 1 for grayscale).
        out_channels: Number of output channels (default: 1 for binary seg).
        base_channels: Base number of channels (doubled at each level).
        skip_mode: How to combine skip connections - "concat" or "additive".
        se_reduction_ratio: Reduction ratio for SE blocks.
        use_interpolation_safeguard: Add interpolation for shape mismatches.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        skip_mode: Literal["concat", "additive"] = "concat",
        se_reduction_ratio: int = 8,
        use_interpolation_safeguard: bool = False,
    ) -> None:
        """Initialize U-Net architecture."""
        super().__init__()

        self.skip_mode = skip_mode
        self.use_interpolation_safeguard = use_interpolation_safeguard

        # Channel progression: [16, 32, 64, 128] for base_channels=16
        chs = [base_channels * (2**i) for i in range(4)]

        # Common block kwargs
        block_kwargs = {
            "se_reduction_ratio": se_reduction_ratio,
            "include_gn_in_proj": skip_mode == "concat",
            "use_interpolation_safeguard": use_interpolation_safeguard,
        }

        # Encoder
        self.enc0 = ResNetBlockSE(in_channels, chs[0], stride=1, **block_kwargs)
        self.down0 = nn.Conv3d(chs[0], chs[0], kernel_size=3, stride=2, padding=1)

        self.enc1 = ResNetBlockSE(chs[0], chs[1], stride=1, **block_kwargs)
        self.down1 = nn.Conv3d(chs[1], chs[1], kernel_size=3, stride=2, padding=1)

        self.enc2 = ResNetBlockSE(chs[1], chs[2], stride=1, **block_kwargs)
        self.down2 = nn.Conv3d(chs[2], chs[2], kernel_size=3, stride=2, padding=1)

        self.enc3 = ResNetBlockSE(chs[2], chs[3], stride=1, **block_kwargs)

        # Bottleneck
        self.bottleneck = ResNetBlockSE(chs[3], chs[3], stride=1, **block_kwargs)

        # Decoder
        self.up2 = nn.ConvTranspose3d(chs[3], chs[2], kernel_size=2, stride=2)
        if skip_mode == "concat":
            self.dec2 = ResNetBlockSE(chs[2] * 2, chs[2], stride=1, **block_kwargs)
        else:
            self.dec2 = ResNetBlockSE(chs[2], chs[2], stride=1, **block_kwargs)

        self.up1 = nn.ConvTranspose3d(chs[2], chs[1], kernel_size=2, stride=2)
        if skip_mode == "concat":
            self.dec1 = ResNetBlockSE(chs[1] * 2, chs[1], stride=1, **block_kwargs)
        else:
            self.dec1 = ResNetBlockSE(chs[1], chs[1], stride=1, **block_kwargs)

        self.up0 = nn.ConvTranspose3d(chs[1], chs[0], kernel_size=2, stride=2)
        if skip_mode == "concat":
            self.dec0 = ResNetBlockSE(chs[0] * 2, chs[0], stride=1, **block_kwargs)
        else:
            self.dec0 = ResNetBlockSE(chs[0], chs[0], stride=1, **block_kwargs)

        # Output layer
        self.final_conv = nn.Conv3d(chs[0], out_channels, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming init for conv layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _match_and_combine(
        self,
        upsampled: torch.Tensor,
        encoder_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Match spatial dimensions and combine features.

        Args:
            upsampled: Upsampled decoder features.
            encoder_features: Encoder skip connection features.

        Returns:
            Combined features.
        """
        # Handle shape mismatch from odd dimensions
        if self.use_interpolation_safeguard and upsampled.shape[2:] != encoder_features.shape[2:]:
            upsampled = F.interpolate(
                upsampled,
                size=encoder_features.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        if self.skip_mode == "concat":
            return torch.cat([upsampled, encoder_features], dim=1)
        else:
            return upsampled + encoder_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.

        Args:
            x: Input tensor of shape (B, C_in, D, H, W).

        Returns:
            Output logits of shape (B, C_out, D, H, W).
        """
        # Encoder path
        e0 = self.enc0(x)
        d0 = self.down0(e0)

        e1 = self.enc1(d0)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        e3 = self.enc3(d2)

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder path
        u2 = self.up2(b)
        combined2 = self._match_and_combine(u2, e2)
        z2 = self.dec2(combined2)

        u1 = self.up1(z2)
        combined1 = self._match_and_combine(u1, e1)
        z1 = self.dec1(combined1)

        u0 = self.up0(z1)
        combined0 = self._match_and_combine(u0, e0)
        z0 = self.dec0(combined0)

        return self.final_conv(z0)

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_unet(
    preset: Literal["standard", "additive", "lightweight"] = "standard",
    **kwargs,
) -> ResidualUnetSE3D:
    """
    Create U-Net with preset configurations.

    Args:
        preset: Configuration preset.
            - "standard": Concat skips, SE ratio 8 (original definition.py)
            - "additive": Add skips, SE ratio 2, interpolation safeguard
            - "lightweight": Smaller base channels for memory constrained

    Returns:
        Configured ResidualUnetSE3D model.
    """
    presets = {
        "standard": {
            "skip_mode": "concat",
            "se_reduction_ratio": 8,
            "use_interpolation_safeguard": False,
        },
        "additive": {
            "skip_mode": "additive",
            "se_reduction_ratio": 2,
            "use_interpolation_safeguard": True,
        },
        "lightweight": {
            "base_channels": 8,
            "skip_mode": "concat",
            "se_reduction_ratio": 8,
            "use_interpolation_safeguard": False,
        },
    }

    config = presets.get(preset, presets["standard"])
    config.update(kwargs)

    return ResidualUnetSE3D(**config)
