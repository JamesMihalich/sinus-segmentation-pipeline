"""
2D Attention Residual U-Net for medical image segmentation.

Architecture based on:
- Residual blocks in encoder/decoder
- Attention gates on skip connections
- 4 encoding/decoding stages with bottleneck

Input: (B, 1, 512, 512) -> Output: (B, 1, 512, 512)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualBlock2D(nn.Module):
    """
    Residual block with two conv layers and skip connection.

    Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Skip connection with 1x1 conv if channels change
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class AttentionGate2D(nn.Module):
    """
    Attention gate for skip connections.

    Learns to focus on relevant spatial regions by combining
    gating signal (from decoder) with skip features (from encoder).
    """

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: Optional[int] = None,
    ):
        super().__init__()

        if inter_channels is None:
            inter_channels = skip_channels // 2

        # Transform gate signal
        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )

        # Transform skip connection
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )

        # Attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        gate: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            gate: Gating signal from decoder (lower resolution).
            skip: Skip connection from encoder (higher resolution).

        Returns:
            Attention-weighted skip features.
        """
        # Transform both inputs
        g = self.W_g(gate)
        x = self.W_x(skip)

        # Upsample gate to match skip resolution if needed
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)

        # Compute attention
        attention = self.relu(g + x)
        attention = self.psi(attention)

        # Apply attention to skip features
        return skip * attention


class AttentionResUNet2D(nn.Module):
    """
    2D Attention Residual U-Net.

    Architecture:
        Encoder: 4 residual blocks with max pooling
        Bottleneck: 1024 channels at 32x32
        Decoder: 4 stages with attention gates and transposed convolutions

    Input shape: (B, in_channels, 512, 512)
    Output shape: (B, out_channels, 512, 512)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 64,
    ):
        """
        Args:
            in_channels: Number of input channels (1 for grayscale).
            out_channels: Number of output channels (1 for binary segmentation).
            base_features: Base number of features (doubles each encoder stage).
        """
        super().__init__()

        f = base_features  # 64

        # Encoder
        self.enc1 = ResidualBlock2D(in_channels, f)      # -> 64
        self.enc2 = ResidualBlock2D(f, f * 2)            # -> 128
        self.enc3 = ResidualBlock2D(f * 2, f * 4)        # -> 256
        self.enc4 = ResidualBlock2D(f * 4, f * 8)        # -> 512

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ResidualBlock2D(f * 8, f * 16)  # -> 1024

        # Decoder upsampling
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)

        # Attention gates
        self.att4 = AttentionGate2D(gate_channels=f * 8, skip_channels=f * 8)
        self.att3 = AttentionGate2D(gate_channels=f * 4, skip_channels=f * 4)
        self.att2 = AttentionGate2D(gate_channels=f * 2, skip_channels=f * 2)
        self.att1 = AttentionGate2D(gate_channels=f, skip_channels=f)

        # Decoder residual blocks (after concatenation)
        self.dec4 = ResidualBlock2D(f * 16, f * 8)   # 512+512 -> 512
        self.dec3 = ResidualBlock2D(f * 8, f * 4)    # 256+256 -> 256
        self.dec2 = ResidualBlock2D(f * 4, f * 2)    # 128+128 -> 128
        self.dec1 = ResidualBlock2D(f * 2, f)        # 64+64 -> 64

        # Output
        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Output tensor of shape (B, out_channels, H, W).
        """
        # Encoder path
        e1 = self.enc1(x)           # (B, 64, 512, 512)
        e2 = self.enc2(self.pool(e1))  # (B, 128, 256, 256)
        e3 = self.enc3(self.pool(e2))  # (B, 256, 128, 128)
        e4 = self.enc4(self.pool(e3))  # (B, 512, 64, 64)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # (B, 1024, 32, 32)

        # Decoder path with attention
        d4 = self.up4(b)                    # (B, 512, 64, 64)
        e4_att = self.att4(gate=d4, skip=e4)
        d4 = torch.cat([d4, e4_att], dim=1)  # (B, 1024, 64, 64)
        d4 = self.dec4(d4)                   # (B, 512, 64, 64)

        d3 = self.up3(d4)                    # (B, 256, 128, 128)
        e3_att = self.att3(gate=d3, skip=e3)
        d3 = torch.cat([d3, e3_att], dim=1)  # (B, 512, 128, 128)
        d3 = self.dec3(d3)                   # (B, 256, 128, 128)

        d2 = self.up2(d3)                    # (B, 128, 256, 256)
        e2_att = self.att2(gate=d2, skip=e2)
        d2 = torch.cat([d2, e2_att], dim=1)  # (B, 256, 256, 256)
        d2 = self.dec2(d2)                   # (B, 128, 256, 256)

        d1 = self.up1(d2)                    # (B, 64, 512, 512)
        e1_att = self.att1(gate=d1, skip=e1)
        d1 = torch.cat([d1, e1_att], dim=1)  # (B, 128, 512, 512)
        d1 = self.dec1(d1)                   # (B, 64, 512, 512)

        # Output
        return self.out_conv(d1)             # (B, out_channels, 512, 512)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = AttentionResUNet2D(in_channels=1, out_channels=1)
    print(f"Parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(2, 1, 512, 512)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
