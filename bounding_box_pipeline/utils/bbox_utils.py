"""
Bounding box utilities for 3D volumes.

Provides functions for extracting, normalizing, and computing metrics
on 3D bounding boxes.
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_relative_bbox(
    mask_volume: np.ndarray,
    return_absolute: bool = False,
) -> Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """
    Extract normalized bounding box from a 3D binary mask.

    Finds the axis-aligned bounding box containing all non-zero voxels
    and normalizes coordinates to [0, 1] range relative to volume shape.

    Args:
        mask_volume: 3D binary mask array.
        return_absolute: If True, also return absolute pixel coordinates.

    Returns:
        Normalized bbox as [z1, y1, x1, z2, y2, x2] in [0, 1] range.
        Returns None if mask is empty.
        If return_absolute=True, returns (normalized, absolute) tuple.
    """
    # Find non-zero coordinates
    nonzero_coords = np.argwhere(mask_volume > 0)

    if len(nonzero_coords) == 0:
        logger.warning("Empty mask - no bounding box found")
        return None

    # Get min/max for each dimension
    min_coords = nonzero_coords.min(axis=0)
    max_coords = nonzero_coords.max(axis=0)

    # Absolute coordinates: [z1, y1, x1, z2, y2, x2]
    absolute_bbox = np.concatenate([min_coords, max_coords]).astype(np.float32)

    # Normalize to [0, 1] range
    shape = np.array(mask_volume.shape, dtype=np.float32)
    normalized_bbox = np.concatenate([
        min_coords / shape,
        max_coords / shape,
    ]).astype(np.float32)

    if return_absolute:
        return normalized_bbox, absolute_bbox
    return normalized_bbox


def normalize_bbox(
    bbox: np.ndarray,
    volume_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Normalize absolute bbox coordinates to [0, 1] range.

    Args:
        bbox: Absolute coordinates [z1, y1, x1, z2, y2, x2].
        volume_shape: Shape of the volume (D, H, W).

    Returns:
        Normalized coordinates in [0, 1] range.
    """
    shape = np.array(list(volume_shape) * 2, dtype=np.float32)
    return bbox.astype(np.float32) / shape


def denormalize_bbox(
    bbox: np.ndarray,
    volume_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Convert normalized bbox coordinates to absolute pixel coordinates.

    Args:
        bbox: Normalized coordinates [z1, y1, x1, z2, y2, x2] in [0, 1].
        volume_shape: Shape of the volume (D, H, W).

    Returns:
        Absolute pixel coordinates.
    """
    shape = np.array(list(volume_shape) * 2, dtype=np.float32)
    return (bbox * shape).astype(np.int32)


def compute_iou(
    box1: np.ndarray,
    box2: np.ndarray,
) -> float:
    """
    Compute 3D Intersection over Union (IoU) between two boxes.

    Args:
        box1: First box [z1, y1, x1, z2, y2, x2].
        box2: Second box [z1, y1, x1, z2, y2, x2].

    Returns:
        IoU value in [0, 1].
    """
    # Extract coordinates
    z1_1, y1_1, x1_1, z2_1, y2_1, x2_1 = box1
    z1_2, y1_2, x1_2, z2_2, y2_2, x2_2 = box2

    # Compute intersection
    inter_z1 = max(z1_1, z1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x1 = max(x1_1, x1_2)
    inter_z2 = min(z2_1, z2_2)
    inter_y2 = min(y2_1, y2_2)
    inter_x2 = min(x2_1, x2_2)

    # Intersection dimensions (clamp to 0)
    inter_d = max(0, inter_z2 - inter_z1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_w = max(0, inter_x2 - inter_x1)

    intersection = inter_d * inter_h * inter_w

    # Individual volumes
    vol1 = (z2_1 - z1_1) * (y2_1 - y1_1) * (x2_1 - x1_1)
    vol2 = (z2_2 - z1_2) * (y2_2 - y1_2) * (x2_2 - x1_2)

    # Union
    union = vol1 + vol2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def compute_iou_batch(
    boxes1: Union[np.ndarray, torch.Tensor],
    boxes2: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Compute mean IoU for batches of boxes.

    Args:
        boxes1: First batch of boxes (N, 6).
        boxes2: Second batch of boxes (N, 6).

    Returns:
        Mean IoU across batch.
    """
    # Convert to numpy if needed
    if isinstance(boxes1, torch.Tensor):
        boxes1 = boxes1.detach().cpu().numpy()
    if isinstance(boxes2, torch.Tensor):
        boxes2 = boxes2.detach().cpu().numpy()

    # Handle single box
    if boxes1.ndim == 1:
        boxes1 = boxes1[np.newaxis, :]
    if boxes2.ndim == 1:
        boxes2 = boxes2[np.newaxis, :]

    # Compute IoU for each pair
    ious = []
    for b1, b2 in zip(boxes1, boxes2):
        ious.append(compute_iou(b1, b2))

    return float(np.mean(ious))


def compute_iou_tensor(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute IoU using PyTorch tensors (differentiable).

    Args:
        pred: Predicted boxes (B, 6) - normalized coordinates.
        target: Target boxes (B, 6) - normalized coordinates.
        eps: Small epsilon for numerical stability.

    Returns:
        IoU tensor (B,).
    """
    # Extract min/max coordinates
    pred_min = pred[:, :3]  # z1, y1, x1
    pred_max = pred[:, 3:]  # z2, y2, x2
    target_min = target[:, :3]
    target_max = target[:, 3:]

    # Intersection
    inter_min = torch.max(pred_min, target_min)
    inter_max = torch.min(pred_max, target_max)
    inter_dims = torch.clamp(inter_max - inter_min, min=0)
    intersection = inter_dims[:, 0] * inter_dims[:, 1] * inter_dims[:, 2]

    # Volumes
    pred_dims = pred_max - pred_min
    target_dims = target_max - target_min
    pred_vol = pred_dims[:, 0] * pred_dims[:, 1] * pred_dims[:, 2]
    target_vol = target_dims[:, 0] * target_dims[:, 1] * target_dims[:, 2]

    # Union
    union = pred_vol + target_vol - intersection + eps

    return intersection / union


def bbox_center(bbox: np.ndarray) -> np.ndarray:
    """
    Compute center point of bounding box.

    Args:
        bbox: Box coordinates [z1, y1, x1, z2, y2, x2].

    Returns:
        Center coordinates [z, y, x].
    """
    return (bbox[:3] + bbox[3:]) / 2


def bbox_size(bbox: np.ndarray) -> np.ndarray:
    """
    Compute size of bounding box.

    Args:
        bbox: Box coordinates [z1, y1, x1, z2, y2, x2].

    Returns:
        Size [depth, height, width].
    """
    return bbox[3:] - bbox[:3]


def bbox_volume(bbox: np.ndarray) -> float:
    """
    Compute volume of bounding box.

    Args:
        bbox: Box coordinates [z1, y1, x1, z2, y2, x2].

    Returns:
        Volume (product of dimensions).
    """
    size = bbox_size(bbox)
    return float(np.prod(size))


def expand_bbox(
    bbox: np.ndarray,
    margin: Union[float, Tuple[float, float, float]],
    volume_shape: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Expand bounding box by a margin.

    Args:
        bbox: Box coordinates [z1, y1, x1, z2, y2, x2].
        margin: Margin to add (single value or per-dimension).
        volume_shape: If provided, clip to volume bounds.

    Returns:
        Expanded bbox.
    """
    if isinstance(margin, (int, float)):
        margin = (margin, margin, margin)

    margin = np.array(margin, dtype=bbox.dtype)
    expanded = np.concatenate([
        bbox[:3] - margin,
        bbox[3:] + margin,
    ])

    if volume_shape is not None:
        # Clip to volume bounds
        expanded[:3] = np.maximum(expanded[:3], 0)
        expanded[3:] = np.minimum(expanded[3:], np.array(volume_shape))

    return expanded
