"""
Volume cropping utilities for medical image preprocessing.

Provides bounding-box based cropping and dataset preparation.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import nibabel as nib
import numpy as np

from ...utils.io import save_npz
from ...utils.volume_ops import get_bbox_slices
from .windowing import apply_ct_window

logger = logging.getLogger(__name__)


def crop_to_mask_bbox(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int = 5,
) -> Tuple[np.ndarray, np.ndarray, Tuple[slice, ...]]:
    """
    Crop image and mask to the bounding box of the mask.

    Args:
        image: Image volume.
        mask: Mask/label volume.
        padding: Padding around bounding box.

    Returns:
        Tuple of (cropped_image, cropped_mask, slices used).

    Raises:
        ValueError: If mask is empty.
    """
    slices = get_bbox_slices(mask, padding=padding)

    cropped_image = image[slices]
    cropped_mask = mask[slices]

    return cropped_image, cropped_mask, slices


def process_nifti_pair_to_npz(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    output_path: Union[str, Path],
    window_level: float = -300.0,
    window_width: float = 1000.0,
    padding: int = 10,
) -> Optional[Path]:
    """
    Process a NIfTI image-mask pair to cropped, windowed NPZ.

    Args:
        image_path: Path to image NIfTI file.
        mask_path: Path to mask NIfTI file.
        output_path: Path for output NPZ file.
        window_level: CT window level.
        window_width: CT window width.
        padding: Padding around bounding box.

    Returns:
        Output path if successful, None otherwise.
    """
    image_path = Path(image_path)
    mask_path = Path(mask_path)
    output_path = Path(output_path)

    try:
        # Load data
        image_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)

        image_data = image_nii.get_fdata().astype(np.float32)
        mask_data = mask_nii.get_fdata().astype(np.uint8)

        # Validate mask
        nonzero_ratio = np.count_nonzero(mask_data) / mask_data.size
        if nonzero_ratio > 0.9:
            logger.warning(
                f"Mask seems inverted (>90% nonzero): {mask_path.name}"
            )
            return None

        if nonzero_ratio == 0:
            logger.warning(f"Empty mask: {mask_path.name}")
            return None

        # Crop to bounding box
        cropped_image, cropped_mask, slices = crop_to_mask_bbox(
            image_data, mask_data, padding=padding
        )

        # Apply CT windowing
        windowed_image = apply_ct_window(
            cropped_image,
            window_level=window_level,
            window_width=window_width,
            output_range=(0, 255),
            output_dtype=np.uint8,
        )

        # Log info
        logger.info(f"Processing {image_path.name}")
        logger.info(f"  Original: {image_data.shape}")
        logger.info(f"  Cropped: {windowed_image.shape}")

        # Save
        save_npz(
            output_path,
            image=windowed_image,
            label=cropped_mask,
            compressed=True,
        )

        return output_path

    except Exception as e:
        logger.error(f"Error processing {image_path.name}: {e}")
        return None


def process_dataset_to_npz(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    window_level: float = -300.0,
    window_width: float = 1000.0,
    padding: int = 10,
    image_pattern: str = "H*.nii",
    label_suffix: str = "_label",
) -> List[Path]:
    """
    Process all image-mask pairs in a directory to NPZ format.

    Args:
        input_dir: Directory containing NIfTI files.
        output_dir: Output directory for NPZ files.
        window_level: CT window level.
        window_width: CT window width.
        padding: Padding around bounding box.
        image_pattern: Glob pattern for image files.
        label_suffix: Suffix for label files.

    Returns:
        List of successfully processed output paths.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all image files (excluding labels)
    images = sorted(input_dir.glob(image_pattern))
    images = [f for f in images if label_suffix not in f.name]

    logger.info(f"Output directory: {output_dir}")
    logger.info(
        f"Window: level={window_level}, width={window_width}, padding={padding}"
    )
    logger.info(f"Found {len(images)} images to process")

    processed = []

    for image_path in images:
        # Find corresponding mask
        mask_filename = f"{image_path.stem}{label_suffix}.nii"
        mask_path = input_dir / mask_filename

        if not mask_path.exists():
            logger.warning(f"Missing mask for {image_path.name}")
            continue

        # Output filename
        output_filename = image_path.stem + ".npz"
        output_path = output_dir / output_filename

        result = process_nifti_pair_to_npz(
            image_path=image_path,
            mask_path=mask_path,
            output_path=output_path,
            window_level=window_level,
            window_width=window_width,
            padding=padding,
        )

        if result:
            processed.append(result)

    logger.info(f"Processed {len(processed)}/{len(images)} pairs")
    return processed
