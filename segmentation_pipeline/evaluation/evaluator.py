"""
Batch evaluation of segmentation predictions.

Computes metrics across multiple prediction-ground truth pairs.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..utils.io import load_npz
from .metrics import compute_metrics

logger = logging.getLogger(__name__)


# Default key search order
PREDICTION_KEYS = ["mask", "prediction", "pred", "label"]
GROUND_TRUTH_KEYS = ["label", "mask", "ground_truth", "gt", "segmentation"]


def _load_array(
    path: Path,
    priority_keys: List[str],
) -> np.ndarray:
    """Load array with key priority."""
    data = np.load(path)

    for key in priority_keys:
        if key in data:
            return data[key]

    # Fallback to first key
    if len(data.files) > 0:
        return data[data.files[0]]

    raise KeyError(f"No valid key found in {path}")


class SegmentationEvaluator:
    """
    Batch evaluator for segmentation predictions.

    Matches prediction files with ground truth files and computes metrics.
    """

    def __init__(
        self,
        prediction_dir: Union[str, Path],
        ground_truth_dirs: Union[str, Path, List[Path]],
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        prediction_suffix: str = "_prediction",
        gt_patterns: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize evaluator.

        Args:
            prediction_dir: Directory containing prediction files.
            ground_truth_dirs: Directory or list of directories with GT files.
            spacing: Voxel spacing for distance metrics.
            prediction_suffix: Suffix to strip from prediction names.
            gt_patterns: List of filename patterns to try for GT matching.
        """
        self.prediction_dir = Path(prediction_dir)
        self.spacing = spacing
        self.prediction_suffix = prediction_suffix

        # Handle single or multiple GT directories
        if isinstance(ground_truth_dirs, (str, Path)):
            self.gt_dirs = [Path(ground_truth_dirs)]
        else:
            self.gt_dirs = [Path(d) for d in ground_truth_dirs]

        # Default GT filename patterns
        self.gt_patterns = gt_patterns or [
            "{id}.npz",
            "{id}_cropped_mask.npz",
            "{id}_label.npz",
            "{id}_mask.npz",
        ]

    def _find_ground_truth(self, patient_id: str) -> Optional[Path]:
        """Find ground truth file for a patient ID."""
        for gt_dir in self.gt_dirs:
            for pattern in self.gt_patterns:
                gt_path = gt_dir / pattern.format(id=patient_id)
                if gt_path.exists():
                    return gt_path
        return None

    def _extract_patient_id(self, prediction_path: Path) -> str:
        """Extract patient ID from prediction filename."""
        name = prediction_path.stem
        # Remove prediction suffix
        if name.endswith(self.prediction_suffix):
            name = name[: -len(self.prediction_suffix)]
        # Remove other common suffixes
        for suffix in ["_cropped_img", "_cropped"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return name

    def evaluate_single(
        self,
        prediction_path: Path,
        ground_truth_path: Path,
    ) -> Dict[str, float]:
        """
        Evaluate a single prediction-GT pair.

        Args:
            prediction_path: Path to prediction file.
            ground_truth_path: Path to ground truth file.

        Returns:
            Dictionary of metric values.
        """
        pred = _load_array(prediction_path, PREDICTION_KEYS)
        gt = _load_array(ground_truth_path, GROUND_TRUTH_KEYS)

        # Validate shapes
        if pred.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}"
            )

        return compute_metrics(pred, gt, spacing=self.spacing)

    def evaluate_all(
        self,
        pattern: str = "*_prediction.npz",
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Evaluate all predictions in directory.

        Args:
            pattern: Glob pattern for prediction files.
            show_progress: Show progress bar.

        Returns:
            DataFrame with per-patient metrics.
        """
        pred_files = list(self.prediction_dir.glob(pattern))
        logger.info(f"Found {len(pred_files)} prediction files")

        results = []
        iterator = pred_files
        if show_progress:
            iterator = tqdm(pred_files, desc="Evaluating")

        for pred_path in iterator:
            patient_id = self._extract_patient_id(pred_path)
            gt_path = self._find_ground_truth(patient_id)

            if gt_path is None:
                logger.warning(f"No GT found for {patient_id}")
                continue

            try:
                metrics = self.evaluate_single(pred_path, gt_path)
                metrics["patient_id"] = patient_id
                results.append(metrics)

            except Exception as e:
                logger.error(f"Error evaluating {patient_id}: {e}")
                continue

        if not results:
            logger.warning("No valid evaluations completed")
            return pd.DataFrame()

        df = pd.DataFrame(results)

        # Reorder columns
        cols = ["patient_id"] + [c for c in df.columns if c != "patient_id"]
        df = df[cols]

        return df

    def compute_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute summary statistics from results.

        Args:
            df: Results DataFrame from evaluate_all().

        Returns:
            Summary DataFrame with mean, std, min, max.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        summary = df[numeric_cols].agg(["mean", "std", "min", "max"])

        return summary

    def save_results(
        self,
        df: pd.DataFrame,
        output_path: Union[str, Path],
        include_summary: bool = True,
    ) -> Path:
        """
        Save evaluation results to CSV.

        Args:
            df: Results DataFrame.
            output_path: Output CSV path.
            include_summary: Append summary statistics.

        Returns:
            Path to saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if include_summary:
            summary = self.compute_summary(df)

            # Print summary
            logger.info("\nEvaluation Summary:")
            logger.info("=" * 50)
            logger.info(f"\nPer-metric statistics:\n{summary.to_string()}")

        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")

        return output_path


def evaluate_directory(
    prediction_dir: Union[str, Path],
    ground_truth_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to evaluate predictions in a directory.

    Args:
        prediction_dir: Directory with predictions.
        ground_truth_dir: Directory with ground truth.
        output_path: Optional path to save CSV results.
        spacing: Voxel spacing.
        show_progress: Show progress bar.

    Returns:
        DataFrame with evaluation results.
    """
    evaluator = SegmentationEvaluator(
        prediction_dir=prediction_dir,
        ground_truth_dirs=ground_truth_dir,
        spacing=spacing,
    )

    results = evaluator.evaluate_all(show_progress=show_progress)

    if output_path and len(results) > 0:
        evaluator.save_results(results, output_path)

    return results
