"""
Evaluation metrics for bounding box regression.

Provides IoU computation and batch evaluation utilities.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def compute_iou(
    box1: np.ndarray,
    box2: np.ndarray,
) -> float:
    """
    Compute 3D Intersection over Union between two boxes.

    Args:
        box1: First box [z1, y1, x1, z2, y2, x2].
        box2: Second box [z1, y1, x1, z2, y2, x2].

    Returns:
        IoU value in [0, 1].
    """
    # Extract coordinates
    z1_1, y1_1, x1_1, z2_1, y2_1, x2_1 = box1
    z1_2, y1_2, x1_2, z2_2, y2_2, x2_2 = box2

    # Intersection bounds
    inter_z1 = max(z1_1, z1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x1 = max(x1_1, x1_2)
    inter_z2 = min(z2_1, z2_2)
    inter_y2 = min(y2_1, y2_2)
    inter_x2 = min(x2_1, x2_2)

    # Intersection dimensions
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

    return float(intersection / union)


def compute_iou_batch(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
) -> float:
    """
    Compute mean IoU for a batch of boxes.

    Args:
        pred: Predicted boxes (B, 6).
        target: Target boxes (B, 6).

    Returns:
        Mean IoU across batch.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    if pred.ndim == 1:
        pred = pred[np.newaxis, :]
    if target.ndim == 1:
        target = target[np.newaxis, :]

    ious = []
    for p, t in zip(pred, target):
        ious.append(compute_iou(p, t))

    return float(np.mean(ious))


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for bbox prediction.

    Args:
        pred: Predicted box [z1, y1, x1, z2, y2, x2].
        target: Target box [z1, y1, x1, z2, y2, x2].

    Returns:
        Dictionary with IoU, center error, size error, etc.
    """
    iou = compute_iou(pred, target)

    # Center error (Euclidean distance between centers)
    pred_center = (pred[:3] + pred[3:]) / 2
    target_center = (target[:3] + target[3:]) / 2
    center_error = np.linalg.norm(pred_center - target_center)

    # Size error (relative difference in dimensions)
    pred_size = pred[3:] - pred[:3]
    target_size = target[3:] - target[:3]
    size_error = np.abs(pred_size - target_size).mean()

    # Relative size error
    relative_size_error = np.abs(pred_size - target_size) / (target_size + 1e-6)
    mean_relative_size_error = relative_size_error.mean()

    return {
        "iou": iou,
        "center_error": float(center_error),
        "size_error": float(size_error),
        "relative_size_error": float(mean_relative_size_error),
    }


class BBoxEvaluator:
    """
    Batch evaluator for bounding box predictions.
    """

    def __init__(
        self,
        prediction_dir: Union[str, Path],
        ground_truth_dir: Union[str, Path],
    ) -> None:
        """
        Initialize evaluator.

        Args:
            prediction_dir: Directory with prediction files.
            ground_truth_dir: Directory with ground truth files.
        """
        self.prediction_dir = Path(prediction_dir)
        self.ground_truth_dir = Path(ground_truth_dir)

    def _load_bbox(
        self,
        path: Path,
        keys: List[str] = None,
    ) -> Optional[np.ndarray]:
        """Load bbox from NPZ file."""
        if keys is None:
            keys = ["predicted_bbox", "label", "bbox"]

        with np.load(path) as data:
            for key in keys:
                if key in data:
                    return data[key]
        return None

    def evaluate_all(
        self,
        pattern: str = "*_prediction.npz",
    ) -> pd.DataFrame:
        """
        Evaluate all predictions.

        Args:
            pattern: Glob pattern for prediction files.

        Returns:
            DataFrame with per-sample metrics.
        """
        pred_files = sorted(self.prediction_dir.glob(pattern))
        logger.info(f"Found {len(pred_files)} prediction files")

        results = []

        for pred_path in pred_files:
            # Find corresponding ground truth
            stem = pred_path.stem.replace("_prediction", "")
            gt_path = self.ground_truth_dir / f"{stem}.npz"

            if not gt_path.exists():
                logger.warning(f"No GT found for {pred_path.name}")
                continue

            try:
                pred_bbox = self._load_bbox(pred_path, ["predicted_bbox", "bbox"])
                gt_bbox = self._load_bbox(gt_path, ["label", "bbox"])

                if pred_bbox is None or gt_bbox is None:
                    continue

                metrics = compute_metrics(pred_bbox, gt_bbox)
                metrics["sample_id"] = stem
                results.append(metrics)

            except Exception as e:
                logger.error(f"Error evaluating {pred_path.name}: {e}")

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        cols = ["sample_id"] + [c for c in df.columns if c != "sample_id"]
        return df[cols]

    def compute_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute summary statistics."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols].agg(["mean", "std", "min", "max"])

    def save_results(
        self,
        df: pd.DataFrame,
        output_path: Union[str, Path],
    ) -> Path:
        """Save results to CSV."""
        output_path = Path(output_path)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")
        return output_path
