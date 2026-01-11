"""
Loss functions for medical image segmentation.

Includes Dice loss and combined loss functions commonly used in segmentation.
"""

from typing import Optional, Tuple

import torch
from torch import nn


def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    reduce_batch: bool = True,
) -> torch.Tensor:
    """
    Compute Dice coefficient (F1 score) for binary segmentation.

    Args:
        pred: Predicted probabilities, shape (B, C, D, H, W) or (B, C, ...).
        target: Ground truth binary mask, same shape as pred.
        eps: Small constant for numerical stability.
        reduce_batch: If True, return mean over batch. If False, return per-sample.

    Returns:
        Dice coefficient in range [0, 1]. Higher is better.
    """
    # Sum over spatial dimensions, keep batch and channel
    dims = tuple(range(2, pred.dim()))

    intersection = (pred * target).sum(dims)
    pred_sum = pred.sum(dims)
    target_sum = target.sum(dims)

    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)

    if reduce_batch:
        return dice.mean()
    return dice


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.

    Loss = 1 - Dice coefficient

    Handles sigmoid internally if needed.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        apply_sigmoid: bool = False,
    ) -> None:
        """
        Initialize Dice loss.

        Args:
            eps: Small constant for numerical stability.
            apply_sigmoid: If True, apply sigmoid to inputs.
                Set False if using with BCEWithLogitsLoss combination.
        """
        super().__init__()
        self.eps = eps
        self.apply_sigmoid = apply_sigmoid

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            logits: Raw logits or probabilities, shape (B, C, D, H, W).
            targets: Ground truth binary mask, same shape.

        Returns:
            Scalar loss value.
        """
        if self.apply_sigmoid:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.sigmoid(logits)  # Always apply sigmoid for dice calc

        return 1.0 - dice_coefficient(probs, targets, self.eps)


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy and Dice loss.

    Loss = bce_weight * BCE + dice_weight * Dice

    This combination is effective because:
    - BCE provides stable gradients early in training
    - Dice directly optimizes the evaluation metric
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        """
        Initialize combined loss.

        Args:
            bce_weight: Weight for BCE loss component.
            dice_weight: Weight for Dice loss component.
            eps: Small constant for Dice calculation.
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(eps=eps)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            logits: Raw logits, shape (B, C, D, H, W).
            targets: Ground truth binary mask, same shape.

        Returns:
            Scalar loss value.
        """
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.

    FL(p) = -alpha * (1-p)^gamma * log(p)

    Focuses learning on hard examples by down-weighting easy ones.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        """
        Initialize Focal loss.

        Args:
            alpha: Weighting factor for positive class.
            gamma: Focusing parameter. Higher = more focus on hard examples.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Focal loss.

        Args:
            logits: Raw logits.
            targets: Binary ground truth.

        Returns:
            Scalar loss value.
        """
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma

        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_weight * focal_weight * bce_loss

        return loss.mean()
