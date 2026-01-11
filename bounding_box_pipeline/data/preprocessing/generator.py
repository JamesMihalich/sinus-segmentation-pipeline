"""
Dataset generator for bounding box localization.

Converts NIfTI image-label pairs to NPZ files with normalized bounding boxes.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from ...utils.bbox_utils import get_relative_bbox
from ...utils.io import save_npz
from .windowing import apply_ct_window, resize_volume

logger = logging.getLogger(__name__)


def process_single_pair(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    target_shape: Tuple[int, int, int] = (128, 128, 128),
    window_level: float = 600.0,
    window_width: float = 1250.0,
) -> Optional[Path]:
    """
    Process a single image-label pair to NPZ format.

    Extracts bounding box from label BEFORE resizing to preserve accuracy,
    then windows and resizes the image.

    Args:
        image_path: Path to image NIfTI file.
        label_path: Path to label/mask NIfTI file.
        output_path: Path for output NPZ file.
        target_shape: Target volume shape for resizing.
        window_level: CT window level.
        window_width: CT window width.

    Returns:
        Output path if successful, None otherwise.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel required: pip install nibabel")

    try:
        # Load data
        label_nii = nib.load(label_path)
        image_nii = nib.load(image_path)

        label_data = label_nii.get_fdata()
        image_data = image_nii.get_fdata()

        original_shape = image_data.shape

        # Extract bounding box BEFORE resizing (preserves accuracy)
        bbox = get_relative_bbox(label_data)

        if bbox is None:
            logger.warning(f"Empty mask, skipping: {label_path.name}")
            return None

        # Apply CT windowing
        windowed = apply_ct_window(
            image_data,
            window_level=window_level,
            window_width=window_width,
        )

        # Resize to target shape
        resized = resize_volume(windowed, target_shape)

        # Save NPZ
        save_npz(
            output_path,
            image=resized.astype(np.uint8),
            label=bbox,
            original_shape=np.array(original_shape),
            compressed=True,
        )

        logger.info(f"Processed: {image_path.name} -> {output_path.name}")
        return output_path

    except Exception as e:
        logger.error(f"Error processing {image_path.name}: {e}")
        return None


def create_localization_dataset(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    target_shape: Tuple[int, int, int] = (128, 128, 128),
    window_level: float = 600.0,
    window_width: float = 1250.0,
    label_suffix: str = "_label",
    pattern: str = "*.nii",
) -> List[Path]:
    """
    Create localization dataset from NIfTI image-label pairs.

    Scans input directory for label files, finds corresponding images,
    extracts bounding boxes, and saves as NPZ files.

    Args:
        input_dir: Directory containing NIfTI files.
        output_dir: Output directory for NPZ files.
        target_shape: Target volume shape for resizing.
        window_level: CT window level.
        window_width: CT window width.
        label_suffix: Suffix identifying label files.
        pattern: Glob pattern for NIfTI files.

    Returns:
        List of successfully created output paths.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find label files
    label_files = sorted(input_dir.glob(f"*{label_suffix}.nii"))

    if not label_files:
        # Try .nii.gz extension
        label_files = sorted(input_dir.glob(f"*{label_suffix}.nii.gz"))

    logger.info(f"Found {len(label_files)} label files")
    logger.info(f"Target shape: {target_shape}")
    logger.info(f"Window: level={window_level}, width={window_width}")

    results = []

    for label_path in label_files:
        # Find corresponding image file
        stem = label_path.name.replace(f"{label_suffix}.nii", "").replace(
            f"{label_suffix}.nii.gz", ""
        )

        # Try different image file patterns
        image_path = None
        for ext in [".nii", ".nii.gz"]:
            candidate = input_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            logger.warning(f"No image found for {label_path.name}")
            continue

        # Output path
        output_path = output_dir / f"{stem}.npz"

        result = process_single_pair(
            image_path=image_path,
            label_path=label_path,
            output_path=output_path,
            target_shape=target_shape,
            window_level=window_level,
            window_width=window_width,
        )

        if result:
            results.append(result)

    logger.info(f"Successfully processed {len(results)}/{len(label_files)} pairs")
    return results


def verify_dataset(
    dataset_dir: Union[str, Path],
    pattern: str = "*.npz",
) -> dict:
    """
    Verify dataset integrity and print statistics.

    Args:
        dataset_dir: Directory containing NPZ files.
        pattern: Glob pattern for files.

    Returns:
        Dictionary with dataset statistics.
    """
    dataset_dir = Path(dataset_dir)
    files = sorted(dataset_dir.glob(pattern))

    stats = {
        "total_files": len(files),
        "valid_files": 0,
        "invalid_files": [],
        "image_shapes": [],
        "label_shapes": [],
    }

    for f in files:
        try:
            with np.load(f) as data:
                if "image" not in data or "label" not in data:
                    stats["invalid_files"].append(str(f))
                    continue

                img_shape = data["image"].shape
                lbl_shape = data["label"].shape

                stats["image_shapes"].append(img_shape)
                stats["label_shapes"].append(lbl_shape)
                stats["valid_files"] += 1

        except Exception as e:
            stats["invalid_files"].append(f"{f}: {e}")

    # Summary
    if stats["image_shapes"]:
        shapes = set(stats["image_shapes"])
        stats["unique_image_shapes"] = [list(s) for s in shapes]

    logger.info(f"Dataset verification:")
    logger.info(f"  Total files: {stats['total_files']}")
    logger.info(f"  Valid files: {stats['valid_files']}")
    logger.info(f"  Invalid files: {len(stats['invalid_files'])}")

    return stats
