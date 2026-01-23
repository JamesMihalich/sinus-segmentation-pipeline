"""
Enhanced 3D Residual U-Net with Attention Gates and Deep Supervision.

Improvements over base ResidualUnetSE3D:
- Attention gates on skip connections (spatial attention)
- Deep supervision with auxiliary outputs
- Increased base channels (32 vs 16)
- Dropout regularization
- Deeper bottleneck (2 blocks)
"""

from typing import Dict, List, Literal, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .blocks.residual_block import ResNetBlockSE


class AttentionGate3D(nn.Module):
    """
    3D Attention Gate for skip connections.

    Learns spatial attention weights by combining gating signal (from decoder)
    with skip features (from encoder). Helps focus on relevant regions.

    Reference: Oktay et al., "Attention U-Net", MIDL 2018
    """

    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: Optional[int] = None,
    ):
        """
        Args:
            gate_channels: Channels in gating signal (from decoder).
            skip_channels: Channels in skip connection (from encoder).
            inter_channels: Intermediate channels (default: skip_channels // 2).
        """
        super().__init__()

        if inter_channels is None:
            inter_channels = max(skip_channels // 2, 1)

        # Transform gating signal
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, inter_channels), num_channels=inter_channels),
        )

        # Transform skip connection
        self.W_x = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, inter_channels), num_channels=inter_channels),
        )

        # Attention coefficients
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        gate: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply attention to skip connection.

        Args:
            gate: Gating signal from decoder (B, C_gate, D, H, W).
            skip: Skip features from encoder (B, C_skip, D', H', W').

        Returns:
            Attention-weighted skip features (B, C_skip, D', H', W').
        """
        g = self.W_g(gate)
        x = self.W_x(skip)

        # Upsample gate if needed to match skip resolution
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(
                g, size=x.shape[2:], mode="trilinear", align_corners=False
            )

        # Compute attention weights
        attention = self.relu(g + x)
        attention = self.psi(attention)

        return skip * attention


class DeepSupervisionHead(nn.Module):
    """
    Auxiliary segmentation head for deep supervision.

    Produces predictions at intermediate decoder stages,
    upsampled to match the original resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
    ):
        """
        Args:
            in_channels: Input feature channels.
            out_channels: Output segmentation channels.
            scale_factor: Upsampling factor to match original resolution.
        """
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce upsampled segmentation prediction."""
        out = self.conv(x)
        if self.scale_factor > 1:
            out = F.interpolate(
                out,
                scale_factor=self.scale_factor,
                mode="trilinear",
                align_corners=False,
            )
        return out


class ResidualUnetSE3DEnhanced(nn.Module):
    """
    Enhanced 3D Residual U-Net with Attention Gates and Deep Supervision.

    Architecture improvements:
    - Attention gates on all skip connections
    - Deep supervision with auxiliary outputs at decoder stages
    - Increased base channels (32) for more capacity
    - Dropout after bottleneck for regularization
    - Deeper bottleneck (2 residual blocks)

    Args:
        in_channels: Number of input channels (default: 1).
        out_channels: Number of output channels (default: 1).
        base_channels: Base feature channels (default: 32).
        skip_mode: Skip connection mode - "concat" or "additive".
        se_reduction_ratio: SE block reduction ratio.
        dropout_rate: Dropout rate after bottleneck (0 to disable).
        deep_supervision: Enable auxiliary outputs for training.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        skip_mode: Literal["concat", "additive"] = "concat",
        se_reduction_ratio: int = 8,
        dropout_rate: float = 0.15,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()

        self.skip_mode = skip_mode
        self.deep_supervision = deep_supervision

        # Channel progression: [32, 64, 128, 256] for base_channels=32
        chs = [base_channels * (2**i) for i in range(4)]

        # Common block kwargs
        block_kwargs = {
            "se_reduction_ratio": se_reduction_ratio,
            "include_gn_in_proj": skip_mode == "concat",
            "use_interpolation_safeguard": True,
        }

        # ============== Encoder ==============
        self.enc0 = ResNetBlockSE(in_channels, chs[0], stride=1, **block_kwargs)
        self.down0 = nn.Conv3d(chs[0], chs[0], kernel_size=3, stride=2, padding=1)

        self.enc1 = ResNetBlockSE(chs[0], chs[1], stride=1, **block_kwargs)
        self.down1 = nn.Conv3d(chs[1], chs[1], kernel_size=3, stride=2, padding=1)

        self.enc2 = ResNetBlockSE(chs[1], chs[2], stride=1, **block_kwargs)
        self.down2 = nn.Conv3d(chs[2], chs[2], kernel_size=3, stride=2, padding=1)

        self.enc3 = ResNetBlockSE(chs[2], chs[3], stride=1, **block_kwargs)

        # ============== Bottleneck (deeper: 2 blocks) ==============
        self.bottleneck1 = ResNetBlockSE(chs[3], chs[3], stride=1, **block_kwargs)
        self.bottleneck2 = ResNetBlockSE(chs[3], chs[3], stride=1, **block_kwargs)

        # Dropout after bottleneck
        self.dropout = nn.Dropout3d(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        # ============== Attention Gates ==============
        self.att2 = AttentionGate3D(gate_channels=chs[3], skip_channels=chs[2])
        self.att1 = AttentionGate3D(gate_channels=chs[2], skip_channels=chs[1])
        self.att0 = AttentionGate3D(gate_channels=chs[1], skip_channels=chs[0])

        # ============== Decoder ==============
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

        # ============== Output ==============
        self.final_conv = nn.Conv3d(chs[0], out_channels, kernel_size=1)

        # ============== Deep Supervision Heads ==============
        if deep_supervision:
            self.ds2 = DeepSupervisionHead(chs[2], out_channels, scale_factor=4)
            self.ds1 = DeepSupervisionHead(chs[1], out_channels, scale_factor=2)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming init."""
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
        """Match spatial dimensions and combine features."""
        if upsampled.shape[2:] != encoder_features.shape[2:]:
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

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C_in, D, H, W).

        Returns:
            Dictionary with:
                - "out": Main output (B, C_out, D, H, W)
                - "ds1": Deep supervision output 1 (if enabled)
                - "ds2": Deep supervision output 2 (if enabled)
        """
        # ============== Encoder ==============
        e0 = self.enc0(x)
        d0 = self.down0(e0)

        e1 = self.enc1(d0)
        d1 = self.down1(e1)

        e2 = self.enc2(d1)
        d2 = self.down2(e2)

        e3 = self.enc3(d2)

        # ============== Bottleneck ==============
        b = self.bottleneck1(e3)
        b = self.bottleneck2(b)
        b = self.dropout(b)

        # ============== Decoder with Attention ==============
        # Stage 2
        u2 = self.up2(b)
        e2_att = self.att2(gate=u2, skip=e2)
        combined2 = self._match_and_combine(u2, e2_att)
        z2 = self.dec2(combined2)

        # Stage 1
        u1 = self.up1(z2)
        e1_att = self.att1(gate=u1, skip=e1)
        combined1 = self._match_and_combine(u1, e1_att)
        z1 = self.dec1(combined1)

        # Stage 0
        u0 = self.up0(z1)
        e0_att = self.att0(gate=u0, skip=e0)
        combined0 = self._match_and_combine(u0, e0_att)
        z0 = self.dec0(combined0)

        # ============== Output ==============
        out = self.final_conv(z0)

        result = {"out": out}

        # Deep supervision outputs
        if self.deep_supervision and self.training:
            result["ds2"] = self.ds2(z2)
            result["ds1"] = self.ds1(z1)

        return result

    def forward_inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference (returns only main output).

        Args:
            x: Input tensor (B, C_in, D, H, W).

        Returns:
            Output tensor (B, C_out, D, H, W).
        """
        return self.forward(x)["out"]

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DeepSupervisionLoss(nn.Module):
    """
    Combined loss for deep supervision training.

    Computes weighted sum of losses at main output and auxiliary outputs.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        weights: Tuple[float, float, float] = (1.0, 0.5, 0.25),
    ):
        """
        Args:
            base_loss: Base loss function (e.g., BCEWithLogitsLoss, DiceLoss).
            weights: Weights for (main, ds1, ds2) outputs.
        """
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted loss.

        Args:
            outputs: Dictionary with "out", "ds1", "ds2" tensors.
            target: Ground truth tensor.

        Returns:
            Combined loss value.
        """
        loss = self.weights[0] * self.base_loss(outputs["out"], target)

        if "ds1" in outputs:
            # Match target size to ds1 if needed
            ds1_target = target
            if outputs["ds1"].shape != target.shape:
                ds1_target = F.interpolate(
                    target.float(),
                    size=outputs["ds1"].shape[2:],
                    mode="nearest",
                )
            loss = loss + self.weights[1] * self.base_loss(outputs["ds1"], ds1_target)

        if "ds2" in outputs:
            ds2_target = target
            if outputs["ds2"].shape != target.shape:
                ds2_target = F.interpolate(
                    target.float(),
                    size=outputs["ds2"].shape[2:],
                    mode="nearest",
                )
            loss = loss + self.weights[2] * self.base_loss(outputs["ds2"], ds2_target)

        return loss


def create_enhanced_unet(
    preset: Literal["standard", "large", "lightweight"] = "standard",
    **kwargs,
) -> ResidualUnetSE3DEnhanced:
    """
    Create enhanced U-Net with preset configurations.

    Args:
        preset: Configuration preset.
            - "standard": 32 base channels, deep supervision on
            - "large": 48 base channels for more capacity
            - "lightweight": 24 base channels for memory constrained

    Returns:
        Configured ResidualUnetSE3DEnhanced model.
    """
    presets = {
        "standard": {
            "base_channels": 32,
            "dropout_rate": 0.15,
            "deep_supervision": True,
        },
        "large": {
            "base_channels": 48,
            "dropout_rate": 0.2,
            "deep_supervision": True,
        },
        "lightweight": {
            "base_channels": 24,
            "dropout_rate": 0.1,
            "deep_supervision": True,
        },
    }

    config = presets.get(preset, presets["standard"])
    config.update(kwargs)

    return ResidualUnetSE3DEnhanced(**config)


if __name__ == "__main__":
    # Test the model
    model = create_enhanced_unet("standard")
    print(f"Parameters: {model.get_num_parameters():,}")

    # Test forward pass
    x = torch.randn(1, 1, 64, 64, 64)

    # Training mode (with deep supervision)
    model.train()
    outputs = model(x)
    print(f"Training outputs: {list(outputs.keys())}")
    print(f"  Main output: {outputs['out'].shape}")
    if "ds1" in outputs:
        print(f"  DS1 output: {outputs['ds1'].shape}")
    if "ds2" in outputs:
        print(f"  DS2 output: {outputs['ds2'].shape}")

    # Inference mode
    model.eval()
    with torch.no_grad():
        out = model.forward_inference(x)
    print(f"Inference output: {out.shape}")
