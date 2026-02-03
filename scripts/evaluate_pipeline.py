#!/usr/bin/env python3
"""
End-to-end pipeline evaluation: Bounding Box → Segmentation.

Runs the complete pipeline on test patients from a manifest:
1. Load original NIfTI volume
2. Predict bounding box
3. Crop volume using predicted bbox
4. Run segmentation on cropped volume
5. Map segmentation back to original coordinates
6. Evaluate against ground truth

Usage:
    python scripts/evaluate_pipeline.py \
        --bbox-checkpoint bbox_model.pt \
        --seg-checkpoint seg_model.pt \
        --data-dir /path/to/nifti \
        --gt-dir /path/to/ground_truth \
        --split-manifest manifest.csv \
        --output-dir ./pipeline_results
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from bounding_box_pipeline.inference.predictor import BBoxPredictor
from bounding_box_pipeline.models import create_regressor
from bounding_box_pipeline.utils.bbox_utils import expand_bbox
from segmentation_pipeline.inference.predictor import VolumePredictor
from segmentation_pipeline.models import create_unet, create_enhanced_unet
from segmentation_pipeline.evaluation.metrics import compute_metrics
from segmentation_pipeline.utils.io import save_npz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load NIfTI file and return volume and affine."""
    import nibabel as nib
    nii = nib.load(path)
    return nii.get_fdata(), nii.affine


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
    gt_dir: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Find image and ground truth files for a patient."""
    # Common naming patterns
    image_patterns = [
        f"{patient_id}.nii",
        f"{patient_id}.nii.gz",
    ]
    gt_patterns = [
        f"{patient_id}_label.nii",
        f"{patient_id}_label.nii.gz",
        f"{patient_id}.npz",
        f"{patient_id}_cropped_mask.npz",
    ]

    image_path = None
    for pattern in image_patterns:
        candidate = data_dir / pattern
        if candidate.exists():
            image_path = candidate
            break

    gt_path = None
    for pattern in gt_patterns:
        candidate = gt_dir / pattern
        if candidate.exists():
            gt_path = candidate
            break

    return image_path, gt_path


def crop_volume_with_bbox(
    volume: np.ndarray,
    bbox: np.ndarray,
    margin: float = 0.0,
) -> Tuple[np.ndarray, Tuple[slice, slice, slice]]:
    """
    Crop volume using bounding box coordinates.

    Args:
        volume: Input volume (D, H, W).
        bbox: Absolute bbox [z1, y1, x1, z2, y2, x2].
        margin: Margin to add around bbox (fraction of bbox size).

    Returns:
        Tuple of (cropped_volume, slices_used).
    """
    bbox = bbox.astype(np.float32)

    # Optionally expand bbox with margin (convert fraction to pixels)
    if margin > 0:
        bbox_size = bbox[3:] - bbox[:3]  # [d, h, w]
        pixel_margin = (bbox_size * margin).astype(int)
        bbox = expand_bbox(bbox.astype(int), margin=tuple(pixel_margin), volume_shape=volume.shape)

    bbox = bbox.astype(int)
    z1, y1, x1, z2, y2, x2 = bbox

    # Clamp to volume bounds
    z1 = max(0, z1)
    y1 = max(0, y1)
    x1 = max(0, x1)
    z2 = min(volume.shape[0], z2)
    y2 = min(volume.shape[1], y2)
    x2 = min(volume.shape[2], x2)

    slices = (slice(z1, z2), slice(y1, y2), slice(x1, x2))
    cropped = volume[slices].copy()

    return cropped, slices


def map_mask_to_original(
    mask: np.ndarray,
    original_shape: Tuple[int, int, int],
    slices: Tuple[slice, slice, slice],
) -> np.ndarray:
    """Map cropped mask back to original volume coordinates."""
    full_mask = np.zeros(original_shape, dtype=np.uint8)
    full_mask[slices] = mask
    return full_mask


def load_ground_truth(gt_path: Path) -> np.ndarray:
    """Load ground truth from NIfTI or NPZ."""
    if gt_path.suffix in [".nii", ".gz"]:
        import nibabel as nib
        return nib.load(gt_path).get_fdata().astype(np.uint8)
    else:
        data = np.load(gt_path)
        for key in ["label", "mask", "ground_truth", "gt"]:
            if key in data:
                return data[key].astype(np.uint8)
        return data[data.files[0]].astype(np.uint8)


def run_pipeline(
    bbox_predictor: BBoxPredictor,
    seg_predictor: VolumePredictor,
    image_path: Path,
    gt_path: Path,
    bbox_margin: float = 0.05,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict:
    """
    Run end-to-end pipeline on a single patient.

    Returns:
        Dictionary with metrics and intermediate results.
    """
    # Load image
    volume, affine = load_nifti(image_path)
    original_shape = volume.shape

    # Step 1: Predict bounding box
    absolute_bbox, normalized_bbox = bbox_predictor.predict_single(
        volume, return_normalized=True
    )

    # Step 2: Crop volume
    cropped_volume, crop_slices = crop_volume_with_bbox(
        volume, absolute_bbox, margin=bbox_margin
    )

    # Step 3: Run segmentation on cropped volume
    cropped_mask, probabilities = seg_predictor.predict_single(
        cropped_volume, return_probabilities=True
    )

    # Step 4: Map back to original coordinates
    full_mask = map_mask_to_original(cropped_mask, original_shape, crop_slices)

    # Load ground truth and compute metrics
    gt_mask = load_ground_truth(gt_path)

    # Handle shape mismatches (GT might be in different space)
    if gt_mask.shape != full_mask.shape:
        logger.warning(
            f"Shape mismatch: pred {full_mask.shape} vs gt {gt_mask.shape}"
        )
        # Try to evaluate on cropped region only
        gt_cropped = gt_mask[crop_slices] if gt_mask.shape == original_shape else gt_mask
        if gt_cropped.shape == cropped_mask.shape:
            metrics = compute_metrics(cropped_mask, gt_cropped, spacing=spacing)
            metrics["evaluated_on"] = "cropped"
        else:
            metrics = {"error": "shape_mismatch"}
    else:
        metrics = compute_metrics(full_mask, gt_mask, spacing=spacing)
        metrics["evaluated_on"] = "full"

    # Add bbox info
    metrics["bbox_z1"] = int(absolute_bbox[0])
    metrics["bbox_y1"] = int(absolute_bbox[1])
    metrics["bbox_x1"] = int(absolute_bbox[2])
    metrics["bbox_z2"] = int(absolute_bbox[3])
    metrics["bbox_y2"] = int(absolute_bbox[4])
    metrics["bbox_x2"] = int(absolute_bbox[5])
    metrics["cropped_shape"] = cropped_volume.shape

    return {
        "metrics": metrics,
        "full_mask": full_mask,
        "cropped_mask": cropped_mask,
        "probabilities": probabilities,
        "bbox": absolute_bbox,
        "crop_slices": crop_slices,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline evaluation: BBox → Segmentation"
    )

    # Required arguments
    parser.add_argument(
        "--bbox-checkpoint",
        type=Path,
        required=True,
        help="Path to bounding box model checkpoint",
    )
    parser.add_argument(
        "--seg-checkpoint",
        type=Path,
        required=True,
        help="Path to segmentation model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing NIfTI image files",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        required=True,
        help="Directory containing ground truth files",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        required=True,
        help="Path to CSV manifest with train/val/test splits",
    )

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./pipeline_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--bbox-model",
        type=str,
        choices=["standard", "lite", "pooling"],
        default="pooling",
        help="Bounding box model variant",
    )
    parser.add_argument(
        "--seg-model",
        type=str,
        choices=["standard", "additive", "enhanced"],
        default="standard",
        help="Segmentation model variant (standard=concat skips, additive=add skips)",
    )
    parser.add_argument(
        "--bbox-window-level",
        type=float,
        default=600,
        help="CT window level for bbox model",
    )
    parser.add_argument(
        "--bbox-window-width",
        type=float,
        default=1250,
        help="CT window width for bbox model",
    )
    parser.add_argument(
        "--bbox-margin",
        type=float,
        default=0.05,
        help="Margin to add around predicted bbox (fraction)",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=[0.33, 0.33, 0.33],
        help="Voxel spacing for distance metrics (z, y, x)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save prediction masks to output directory",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate inputs
    if not args.bbox_checkpoint.exists():
        logger.error(f"BBox checkpoint not found: {args.bbox_checkpoint}")
        return 1
    if not args.seg_checkpoint.exists():
        logger.error(f"Segmentation checkpoint not found: {args.seg_checkpoint}")
        return 1
    if not args.split_manifest.exists():
        logger.error(f"Manifest not found: {args.split_manifest}")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load test patients from manifest
    test_patients = load_test_patients_from_manifest(args.split_manifest)
    logger.info(f"Found {len(test_patients)} test patients in manifest")

    if not test_patients:
        logger.error("No test patients found in manifest")
        return 1

    # Initialize bounding box model
    logger.info(f"Loading bounding box model ({args.bbox_model})...")
    bbox_model = create_regressor(variant=args.bbox_model)
    bbox_predictor = BBoxPredictor.from_checkpoint(
        model=bbox_model,
        checkpoint_path=args.bbox_checkpoint,
        device=args.device,
        window_level=args.bbox_window_level,
        window_width=args.bbox_window_width,
    )

    # Initialize segmentation model
    logger.info(f"Loading segmentation model ({args.seg_model})...")
    if args.seg_model == "enhanced":
        seg_model = create_enhanced_unet()
    elif args.seg_model == "additive":
        seg_model = create_unet("additive")
    else:
        seg_model = create_unet("standard")

    seg_predictor = VolumePredictor.from_checkpoint(
        model=seg_model,
        checkpoint_path=args.seg_checkpoint,
        device=args.device,
    )

    # Run pipeline on each test patient
    results = []
    for patient_id in tqdm(test_patients, desc="Processing patients"):
        image_path, gt_path = find_patient_files(
            patient_id, args.data_dir, args.gt_dir
        )

        if image_path is None:
            logger.warning(f"Image not found for patient {patient_id}")
            continue
        if gt_path is None:
            logger.warning(f"Ground truth not found for patient {patient_id}")
            continue

        try:
            result = run_pipeline(
                bbox_predictor=bbox_predictor,
                seg_predictor=seg_predictor,
                image_path=image_path,
                gt_path=gt_path,
                bbox_margin=args.bbox_margin,
                spacing=tuple(args.spacing),
            )

            metrics = result["metrics"]
            metrics["patient_id"] = patient_id
            results.append(metrics)

            # Save predictions if requested
            if args.save_predictions:
                pred_dir = args.output_dir / "predictions"
                pred_dir.mkdir(exist_ok=True)
                save_npz(
                    pred_dir / f"{patient_id}_prediction.npz",
                    mask=result["full_mask"],
                    cropped_mask=result["cropped_mask"],
                    bbox=result["bbox"],
                )

            logger.info(
                f"{patient_id}: Dice={metrics.get('dice', 'N/A'):.4f}"
                if isinstance(metrics.get('dice'), float) else f"{patient_id}: {metrics}"
            )

        except Exception as e:
            logger.error(f"Error processing {patient_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not results:
        logger.error("No patients were successfully processed")
        return 1

    # Create results DataFrame
    df = pd.DataFrame(results)

    # Reorder columns
    cols = ["patient_id"] + [c for c in df.columns if c != "patient_id"]
    df = df[cols]

    # Save results
    output_csv = args.output_dir / "pipeline_evaluation.csv"
    df.to_csv(output_csv, index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nProcessed: {len(results)} / {len(test_patients)} test patients")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Filter to main metrics only
    main_metrics = [c for c in numeric_cols if c in ["dice", "iou", "hd95", "assd"]]
    if main_metrics:
        summary = df[main_metrics].agg(["mean", "std", "min", "max"])
        print("\nMetric Statistics:")
        print(summary.to_string())

    print(f"\nResults saved to: {output_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
