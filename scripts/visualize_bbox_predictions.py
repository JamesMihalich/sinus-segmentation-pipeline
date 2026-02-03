#!/usr/bin/env python3
"""
Visualize bounding box predictions on test patients.

Generates visualizations in all 3 planes (axial, coronal, sagittal) showing:
- Original image slice
- Predicted bounding box (red)
- Ground truth bounding box (green)

Usage:
    python scripts/visualize_bbox_predictions.py \
        --bbox-checkpoint bbox_model.pt \
        --data-dir /path/to/nifti \
        --split-manifest manifest.csv \
        --output-dir ./bbox_visualizations
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bounding_box_pipeline.inference.predictor import BBoxPredictor
from bounding_box_pipeline.models import create_regressor
from bounding_box_pipeline.utils.bbox_utils import get_relative_bbox, denormalize_bbox

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_nifti(path: Path) -> np.ndarray:
    """Load NIfTI file and return volume."""
    import nibabel as nib
    nii = nib.load(path)
    return nii.get_fdata()


def load_test_patients_from_manifest(manifest_path: Path) -> List[str]:
    """Load test patient IDs from manifest CSV."""
    test_patients = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "test (held-out)":
                test_patients.append(row["patient_id"])
    return test_patients


def find_patient_files(
    patient_id: str,
    data_dir: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Find image and label files for a patient."""
    image_patterns = [f"{patient_id}.nii", f"{patient_id}.nii.gz"]
    label_patterns = [f"{patient_id}_label.nii", f"{patient_id}_label.nii.gz"]

    image_path = None
    for pattern in image_patterns:
        candidate = data_dir / pattern
        if candidate.exists():
            image_path = candidate
            break

    label_path = None
    for pattern in label_patterns:
        candidate = data_dir / pattern
        if candidate.exists():
            label_path = candidate
            break

    return image_path, label_path


def get_gt_bbox(label_volume: np.ndarray) -> Optional[np.ndarray]:
    """Extract ground truth bounding box from label mask."""
    nonzero = np.argwhere(label_volume > 0)
    if len(nonzero) == 0:
        return None

    min_coords = nonzero.min(axis=0)
    max_coords = nonzero.max(axis=0)

    # Return as [z1, y1, x1, z2, y2, x2]
    return np.concatenate([min_coords, max_coords])


def draw_bbox_on_ax(
    ax,
    bbox: np.ndarray,
    plane: str,
    slice_idx: int,
    color: str,
    label: str,
    linestyle: str = "-",
):
    """Draw bounding box rectangle on axis if it intersects the slice."""
    z1, y1, x1, z2, y2, x2 = bbox

    if plane == "axial":  # XY plane, slice along Z
        if z1 <= slice_idx <= z2:
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none',
                linestyle=linestyle, label=label
            )
            ax.add_patch(rect)
    elif plane == "coronal":  # XZ plane, slice along Y
        if y1 <= slice_idx <= y2:
            rect = patches.Rectangle(
                (x1, z1), x2 - x1, z2 - z1,
                linewidth=2, edgecolor=color, facecolor='none',
                linestyle=linestyle, label=label
            )
            ax.add_patch(rect)
    elif plane == "sagittal":  # YZ plane, slice along X
        if x1 <= slice_idx <= x2:
            rect = patches.Rectangle(
                (y1, z1), y2 - y1, z2 - z1,
                linewidth=2, edgecolor=color, facecolor='none',
                linestyle=linestyle, label=label
            )
            ax.add_patch(rect)


def create_visualization(
    volume: np.ndarray,
    pred_bbox: np.ndarray,
    gt_bbox: Optional[np.ndarray],
    patient_id: str,
    output_dir: Path,
    window_level: float = 600,
    window_width: float = 1250,
):
    """Create visualization with all 3 planes at multiple slices."""

    # Apply windowing for display
    lower = window_level - window_width / 2
    upper = window_level + window_width / 2
    display_vol = np.clip(volume, lower, upper)
    display_vol = (display_vol - lower) / window_width

    D, H, W = volume.shape

    # Calculate slice positions - use bbox centers and edges
    pred_center = (pred_bbox[:3] + pred_bbox[3:]) // 2

    if gt_bbox is not None:
        gt_center = (gt_bbox[:3] + gt_bbox[3:]) // 2
    else:
        gt_center = pred_center

    # Create figure with multiple slices per plane
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    fig.suptitle(f"Patient: {patient_id}\nPred BBox (red): {pred_bbox.tolist()}\nGT BBox (green): {gt_bbox.tolist() if gt_bbox is not None else 'N/A'}",
                 fontsize=12, y=0.98)

    # Axial slices (Z axis) - row 0
    z_slices = [
        int(D * 0.25),
        int(pred_center[0]),
        int(gt_center[0]) if gt_bbox is not None else int(D * 0.5),
        int(D * 0.75),
        int(pred_bbox[0]),  # pred bbox start
    ]
    z_slices = [max(0, min(D-1, z)) for z in z_slices]

    for i, z in enumerate(z_slices):
        ax = axes[0, i]
        ax.imshow(display_vol[z, :, :], cmap='gray', aspect='auto')
        ax.set_title(f"Axial Z={z}")
        draw_bbox_on_ax(ax, pred_bbox, "axial", z, 'red', 'Predicted')
        if gt_bbox is not None:
            draw_bbox_on_ax(ax, gt_bbox, "axial", z, 'lime', 'Ground Truth', '--')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    # Coronal slices (Y axis) - row 1
    y_slices = [
        int(H * 0.25),
        int(pred_center[1]),
        int(gt_center[1]) if gt_bbox is not None else int(H * 0.5),
        int(H * 0.75),
        int(pred_bbox[1]),  # pred bbox start
    ]
    y_slices = [max(0, min(H-1, y)) for y in y_slices]

    for i, y in enumerate(y_slices):
        ax = axes[1, i]
        ax.imshow(display_vol[:, y, :], cmap='gray', aspect='auto')
        ax.set_title(f"Coronal Y={y}")
        draw_bbox_on_ax(ax, pred_bbox, "coronal", y, 'red', 'Predicted')
        if gt_bbox is not None:
            draw_bbox_on_ax(ax, gt_bbox, "coronal", y, 'lime', 'Ground Truth', '--')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')

    # Sagittal slices (X axis) - row 2
    x_slices = [
        int(W * 0.25),
        int(pred_center[2]),
        int(gt_center[2]) if gt_bbox is not None else int(W * 0.5),
        int(W * 0.75),
        int(pred_bbox[2]),  # pred bbox start
    ]
    x_slices = [max(0, min(W-1, x)) for x in x_slices]

    for i, x in enumerate(x_slices):
        ax = axes[2, i]
        ax.imshow(display_vol[:, :, x], cmap='gray', aspect='auto')
        ax.set_title(f"Sagittal X={x}")
        draw_bbox_on_ax(ax, pred_bbox, "sagittal", x, 'red', 'Predicted')
        if gt_bbox is not None:
            draw_bbox_on_ax(ax, gt_bbox, "sagittal", x, 'lime', 'Ground Truth', '--')
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')

    # Add legend
    legend_elements = [
        patches.Patch(edgecolor='red', facecolor='none', label='Predicted BBox'),
        patches.Patch(edgecolor='lime', facecolor='none', linestyle='--', label='Ground Truth BBox'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # Save
    output_path = output_dir / f"{patient_id}_bbox_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def compute_bbox_metrics(pred_bbox: np.ndarray, gt_bbox: np.ndarray) -> Dict:
    """Compute metrics between predicted and ground truth bboxes."""
    # Centers
    pred_center = (pred_bbox[:3] + pred_bbox[3:]) / 2
    gt_center = (gt_bbox[:3] + gt_bbox[3:]) / 2
    center_error = np.linalg.norm(pred_center - gt_center)

    # Sizes
    pred_size = pred_bbox[3:] - pred_bbox[:3]
    gt_size = gt_bbox[3:] - gt_bbox[:3]

    # IoU
    inter_min = np.maximum(pred_bbox[:3], gt_bbox[:3])
    inter_max = np.minimum(pred_bbox[3:], gt_bbox[3:])
    inter_size = np.maximum(0, inter_max - inter_min)
    intersection = np.prod(inter_size)

    pred_vol = np.prod(pred_size)
    gt_vol = np.prod(gt_size)
    union = pred_vol + gt_vol - intersection

    iou = intersection / union if union > 0 else 0

    # Per-axis errors
    z_error = pred_center[0] - gt_center[0]
    y_error = pred_center[1] - gt_center[1]
    x_error = pred_center[2] - gt_center[2]

    return {
        "iou": iou,
        "center_error": center_error,
        "z_error": z_error,
        "y_error": y_error,
        "x_error": x_error,
        "pred_size": pred_size.tolist(),
        "gt_size": gt_size.tolist(),
        "size_ratio": (pred_size / gt_size).tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize bounding box predictions on test patients"
    )

    parser.add_argument(
        "--bbox-checkpoint",
        type=Path,
        required=True,
        help="Path to bounding box model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing NIfTI files",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        required=True,
        help="Path to CSV manifest with splits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./bbox_visualizations"),
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--bbox-model",
        type=str,
        choices=["standard", "lite", "pooling"],
        default="pooling",
        help="Bounding box model variant",
    )
    parser.add_argument(
        "--window-level",
        type=float,
        default=600,
        help="CT window level for display",
    )
    parser.add_argument(
        "--window-width",
        type=float,
        default=1250,
        help="CT window width for display",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate inputs
    if not args.bbox_checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.bbox_checkpoint}")
        return 1
    if not args.split_manifest.exists():
        logger.error(f"Manifest not found: {args.split_manifest}")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load test patients
    test_patients = load_test_patients_from_manifest(args.split_manifest)
    logger.info(f"Found {len(test_patients)} test patients")

    # Initialize model
    logger.info(f"Loading bounding box model ({args.bbox_model})...")
    bbox_model = create_regressor(variant=args.bbox_model)
    bbox_predictor = BBoxPredictor.from_checkpoint(
        model=bbox_model,
        checkpoint_path=args.bbox_checkpoint,
        device=args.device,
        window_level=args.window_level,
        window_width=args.window_width,
    )

    # Process each patient
    all_metrics = []

    for patient_id in test_patients:
        logger.info(f"Processing {patient_id}...")

        image_path, label_path = find_patient_files(patient_id, args.data_dir)

        if image_path is None:
            logger.warning(f"Image not found for {patient_id}")
            continue

        # Load volume
        volume = load_nifti(image_path)

        # Load ground truth if available
        gt_bbox = None
        if label_path is not None:
            label_volume = load_nifti(label_path)
            gt_bbox = get_gt_bbox(label_volume)

        # Predict bounding box
        pred_bbox, normalized_bbox = bbox_predictor.predict_single(
            volume, return_normalized=True
        )

        # Compute metrics if GT available
        metrics = {"patient_id": patient_id}
        if gt_bbox is not None:
            bbox_metrics = compute_bbox_metrics(pred_bbox, gt_bbox)
            metrics.update(bbox_metrics)
            logger.info(f"  IoU: {bbox_metrics['iou']:.4f}, Center Error: {bbox_metrics['center_error']:.1f} voxels")

        all_metrics.append(metrics)

        # Create visualization
        output_path = create_visualization(
            volume=volume,
            pred_bbox=pred_bbox,
            gt_bbox=gt_bbox,
            patient_id=patient_id,
            output_dir=args.output_dir,
            window_level=args.window_level,
            window_width=args.window_width,
        )
        logger.info(f"  Saved: {output_path}")

    # Save metrics summary
    if all_metrics:
        import pandas as pd
        df = pd.DataFrame(all_metrics)
        metrics_path = args.output_dir / "bbox_metrics.csv"
        df.to_csv(metrics_path, index=False)
        logger.info(f"\nMetrics saved to: {metrics_path}")

        # Print summary
        if "iou" in df.columns:
            print("\n" + "=" * 50)
            print("BOUNDING BOX METRICS SUMMARY")
            print("=" * 50)
            print(f"Mean IoU: {df['iou'].mean():.4f}")
            print(f"Mean Center Error: {df['center_error'].mean():.1f} voxels")
            print(f"\nPer-axis center errors (mean):")
            print(f"  Z: {df['z_error'].mean():+.1f}")
            print(f"  Y: {df['y_error'].mean():+.1f}")
            print(f"  X: {df['x_error'].mean():+.1f}")
            print("\nPer-patient IoU:")
            for _, row in df.iterrows():
                print(f"  {row['patient_id']}: {row['iou']:.4f}")

    logger.info(f"\nVisualizations saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
