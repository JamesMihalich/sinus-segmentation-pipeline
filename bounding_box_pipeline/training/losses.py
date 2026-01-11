"""
Loss functions for bounding box regression.

Provides smooth L1 loss and IoU-based losses.
"""

import torch
from torch import nn

from ..utils.bbox_utils import compute_iou_tensor


class SmoothL1IoULoss(nn.Module):
    """
    Combined Smooth L1 and IoU loss for bounding box regression.

    Smooth L1 is robust to outliers, while IoU directly optimizes
    the overlap metric we care about.
    """

    def __init__(
        self,
        smooth_l1_weight: float = 1.0,
        iou_weight: float = 0.0,
        beta: float = 1.0,
    ) -> None:
        """
        Initialize loss function.

        Args:
            smooth_l1_weight: Weight for Smooth L1 component.
            iou_weight: Weight for IoU component.
            beta: Beta parameter for Smooth L1 (transition point).
        """
        super().__init__()
        self.smooth_l1_weight = smooth_l1_weight
        self.iou_weight = iou_weight
        self.smooth_l1 = nn.SmoothL1Loss(beta=beta)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            pred: Predicted boxes (B, 6).
            target: Target boxes (B, 6).

        Returns:
            Scalar loss value.
        """
        loss = 0.0

        if self.smooth_l1_weight > 0:
            loss += self.smooth_l1_weight * self.smooth_l1(pred, target)

        if self.iou_weight > 0:
            iou = compute_iou_tensor(pred, target)
            iou_loss = 1.0 - iou.mean()
            loss += self.iou_weight * iou_loss

        return loss


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss for bounding box regression.

    GIoU adds a penalty term for the smallest enclosing box,
    providing gradients even when boxes don't overlap.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        """Initialize GIoU loss."""
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GIoU loss.

        Args:
            pred: Predicted boxes (B, 6).
            target: Target boxes (B, 6).

        Returns:
            Scalar loss value.
        """
        # Extract coordinates
        pred_min = pred[:, :3]
        pred_max = pred[:, 3:]
        target_min = target[:, :3]
        target_max = target[:, 3:]

        # Intersection
        inter_min = torch.max(pred_min, target_min)
        inter_max = torch.min(pred_max, target_max)
        inter_dims = torch.clamp(inter_max - inter_min, min=0)
        intersection = inter_dims.prod(dim=1)

        # Union
        pred_vol = (pred_max - pred_min).prod(dim=1)
        target_vol = (target_max - target_min).prod(dim=1)
        union = pred_vol + target_vol - intersection + self.eps

        # IoU
        iou = intersection / union

        # Enclosing box
        enclose_min = torch.min(pred_min, target_min)
        enclose_max = torch.max(pred_max, target_max)
        enclose_vol = (enclose_max - enclose_min).prod(dim=1) + self.eps

        # GIoU
        giou = iou - (enclose_vol - union) / enclose_vol

        return (1.0 - giou).mean()
