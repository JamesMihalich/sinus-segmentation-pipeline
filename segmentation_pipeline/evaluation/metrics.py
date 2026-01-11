"""
Segmentation evaluation metrics.

Includes overlap metrics (Dice, IoU) and distance metrics (Hausdorff).
"""

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import generate_binary_structure, binary_erosion
from scipy.spatial import cKDTree


def dice_score(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute Dice coefficient (F1 score).

    Args:
        prediction: Binary prediction mask.
        ground_truth: Binary ground truth mask.
        eps: Small constant for numerical stability.

    Returns:
        Dice score in range [0, 1].
    """
    pred_flat = prediction.astype(bool).flatten()
    gt_flat = ground_truth.astype(bool).flatten()

    intersection = np.logical_and(pred_flat, gt_flat).sum()
    pred_sum = pred_flat.sum()
    gt_sum = gt_flat.sum()

    return float((2.0 * intersection + eps) / (pred_sum + gt_sum + eps))


def iou_score(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute Intersection over Union (Jaccard index).

    Args:
        prediction: Binary prediction mask.
        ground_truth: Binary ground truth mask.
        eps: Small constant for numerical stability.

    Returns:
        IoU score in range [0, 1].
    """
    pred_flat = prediction.astype(bool).flatten()
    gt_flat = ground_truth.astype(bool).flatten()

    intersection = np.logical_and(pred_flat, gt_flat).sum()
    union = np.logical_or(pred_flat, gt_flat).sum()

    return float(intersection / (union + eps))


def precision_score(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute precision (positive predictive value).

    Args:
        prediction: Binary prediction mask.
        ground_truth: Binary ground truth mask.
        eps: Small constant for numerical stability.

    Returns:
        Precision in range [0, 1].
    """
    pred_flat = prediction.astype(bool).flatten()
    gt_flat = ground_truth.astype(bool).flatten()

    true_positives = np.logical_and(pred_flat, gt_flat).sum()
    pred_positives = pred_flat.sum()

    return float(true_positives / (pred_positives + eps))


def recall_score(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute recall (sensitivity, true positive rate).

    Args:
        prediction: Binary prediction mask.
        ground_truth: Binary ground truth mask.
        eps: Small constant for numerical stability.

    Returns:
        Recall in range [0, 1].
    """
    pred_flat = prediction.astype(bool).flatten()
    gt_flat = ground_truth.astype(bool).flatten()

    true_positives = np.logical_and(pred_flat, gt_flat).sum()
    gt_positives = gt_flat.sum()

    return float(true_positives / (gt_positives + eps))


def hausdorff_95(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    Compute 95th percentile Hausdorff Distance.

    Uses KDTree for efficient nearest-neighbor computation.

    Args:
        prediction: Binary prediction mask.
        ground_truth: Binary ground truth mask.
        spacing: Voxel spacing (x, y, z) in mm.

    Returns:
        HD95 in mm, or NaN if either mask is empty.
    """
    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)

    # Handle empty masks
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan

    # Extract surface voxels (boundary)
    struct = generate_binary_structure(3, 1)
    pred_border = pred ^ binary_erosion(pred, structure=struct)
    gt_border = gt ^ binary_erosion(gt, structure=struct)

    # Get coordinates scaled by spacing
    pred_coords = np.argwhere(pred_border) * np.array(spacing)
    gt_coords = np.argwhere(gt_border) * np.array(spacing)

    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return np.nan

    # Build KD-trees for fast nearest-neighbor lookup
    tree_pred = cKDTree(pred_coords)
    tree_gt = cKDTree(gt_coords)

    # Distances from GT to nearest Pred
    dist_gt_to_pred, _ = tree_pred.query(gt_coords)

    # Distances from Pred to nearest GT
    dist_pred_to_gt, _ = tree_gt.query(pred_coords)

    # 95th percentile of both directions
    hd95_gt_to_pred = np.percentile(dist_gt_to_pred, 95)
    hd95_pred_to_gt = np.percentile(dist_pred_to_gt, 95)

    return float(max(hd95_gt_to_pred, hd95_pred_to_gt))


def average_surface_distance(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """
    Compute Average Symmetric Surface Distance.

    Args:
        prediction: Binary prediction mask.
        ground_truth: Binary ground truth mask.
        spacing: Voxel spacing in mm.

    Returns:
        ASSD in mm, or NaN if either mask is empty.
    """
    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)

    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan

    # Extract surface voxels
    struct = generate_binary_structure(3, 1)
    pred_border = pred ^ binary_erosion(pred, structure=struct)
    gt_border = gt ^ binary_erosion(gt, structure=struct)

    pred_coords = np.argwhere(pred_border) * np.array(spacing)
    gt_coords = np.argwhere(gt_border) * np.array(spacing)

    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return np.nan

    tree_pred = cKDTree(pred_coords)
    tree_gt = cKDTree(gt_coords)

    dist_gt_to_pred, _ = tree_pred.query(gt_coords)
    dist_pred_to_gt, _ = tree_gt.query(pred_coords)

    # Average of both directions
    return float(
        (dist_gt_to_pred.sum() + dist_pred_to_gt.sum())
        / (len(gt_coords) + len(pred_coords))
    )


def compute_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    compute_distances: bool = True,
) -> Dict[str, float]:
    """
    Compute comprehensive segmentation metrics.

    Args:
        prediction: Binary prediction mask.
        ground_truth: Binary ground truth mask.
        spacing: Voxel spacing for distance metrics.
        compute_distances: Whether to compute HD95 (slower).

    Returns:
        Dictionary with all metric values.
    """
    metrics = {
        "dice": dice_score(prediction, ground_truth),
        "iou": iou_score(prediction, ground_truth),
        "precision": precision_score(prediction, ground_truth),
        "recall": recall_score(prediction, ground_truth),
    }

    if compute_distances:
        metrics["hd95"] = hausdorff_95(prediction, ground_truth, spacing)
        metrics["assd"] = average_surface_distance(
            prediction, ground_truth, spacing
        )

    return metrics


def compute_volume_difference(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """
    Compute volume-based metrics.

    Args:
        prediction: Binary prediction mask.
        ground_truth: Binary ground truth mask.
        spacing: Voxel spacing in mm.

    Returns:
        Dictionary with volume metrics.
    """
    voxel_volume = spacing[0] * spacing[1] * spacing[2]  # mm^3

    pred_volume = prediction.astype(bool).sum() * voxel_volume
    gt_volume = ground_truth.astype(bool).sum() * voxel_volume

    volume_diff = pred_volume - gt_volume
    volume_diff_percent = (volume_diff / gt_volume * 100) if gt_volume > 0 else np.nan

    return {
        "pred_volume_mm3": float(pred_volume),
        "gt_volume_mm3": float(gt_volume),
        "volume_diff_mm3": float(volume_diff),
        "volume_diff_percent": float(volume_diff_percent),
    }
