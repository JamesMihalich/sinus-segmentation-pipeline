"""
Volume resampling utilities for resolution normalization.

Provides isotropic resampling and header alignment functions.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import nibabel as nib
import nibabel.processing as nip
import numpy as np

logger = logging.getLogger(__name__)


def resample_to_isotropic(
    image: nib.Nifti1Image,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    order: int = 3,
) -> nib.Nifti1Image:
    """
    Resample a NIfTI image to isotropic resolution.

    Args:
        image: Input NIfTI image.
        target_spacing: Target voxel size in mm (x, y, z).
        order: Interpolation order (0=nearest, 1=linear, 3=cubic).

    Returns:
        Resampled NIfTI image.
    """
    return nip.resample_to_output(image, voxel_sizes=target_spacing, order=order)


def resample_image_and_mask(
    image: nib.Nifti1Image,
    mask: nib.Nifti1Image,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """
    Resample both image and mask to target spacing.

    Uses cubic interpolation for image (smooth) and nearest neighbor for mask
    (preserves binary values).

    Args:
        image: Input image.
        mask: Input mask/label.
        target_spacing: Target voxel size.

    Returns:
        Tuple of (resampled_image, resampled_mask).
    """
    # Cubic for image (smooth transitions)
    resampled_image = nip.resample_to_output(
        image, voxel_sizes=target_spacing, order=3
    )

    # Nearest neighbor for mask (preserve binary values)
    resampled_mask = nip.resample_to_output(
        mask, voxel_sizes=target_spacing, order=0
    )

    return resampled_image, resampled_mask


def fix_header_mismatch(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    save_mask: bool = True,
) -> Optional[nib.Nifti1Image]:
    """
    Fix header mismatch between image and mask.

    Copies image's affine and header to mask while preserving mask data.

    Args:
        image_path: Path to image NIfTI file.
        mask_path: Path to mask NIfTI file.
        save_mask: If True, save corrected mask to original path.

    Returns:
        Corrected mask NIfTI image, or None if no mismatch.
    """
    image_path = Path(image_path)
    mask_path = Path(mask_path)

    image = nib.load(image_path)
    mask = nib.load(mask_path)

    image_zoom = image.header.get_zooms()[:3]
    mask_zoom = mask.header.get_zooms()[:3]

    # Check for mismatch
    has_mismatch = (
        not np.allclose(image_zoom, mask_zoom, atol=1e-6)
        or not np.array_equal(image.affine, mask.affine)
    )

    if not has_mismatch:
        return None

    logger.info(f"Fixing header mismatch: {mask_path.name}")
    logger.info(f"  Image resolution: {image_zoom}")
    logger.info(f"  Mask resolution: {mask_zoom}")

    # Create new mask with image's affine and header
    new_mask = nib.Nifti1Image(
        mask.get_fdata(), image.affine, header=image.header
    )
    new_mask.set_data_dtype(np.uint8)

    if save_mask:
        nib.save(new_mask, mask_path)
        logger.info(f"  Saved corrected mask")

    return new_mask


def fix_header_mismatches(
    directory: Union[str, Path],
    image_pattern: str = "H*.nii",
    label_suffix: str = "_label",
) -> int:
    """
    Fix all header mismatches in a directory.

    Args:
        directory: Directory containing NIfTI files.
        image_pattern: Glob pattern for image files.
        label_suffix: Suffix that distinguishes label files.

    Returns:
        Number of files fixed.
    """
    directory = Path(directory)
    image_files = sorted(directory.glob(image_pattern))

    # Filter out label files
    image_files = [f for f in image_files if label_suffix not in f.name]

    fixed_count = 0
    logger.info(f"Checking {len(image_files)} pairs for header mismatches...")

    for image_path in image_files:
        mask_path = image_path.parent / f"{image_path.stem}{label_suffix}.nii"

        if not mask_path.exists():
            continue

        result = fix_header_mismatch(image_path, mask_path, save_mask=True)
        if result is not None:
            fixed_count += 1

    logger.info(f"Fixed {fixed_count} header mismatches")
    return fixed_count


def resample_dataset(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    target_spacing: Tuple[float, float, float] = (0.33, 0.33, 0.33),
    image_pattern: str = "H*.nii",
    label_suffix: str = "_label",
) -> List[Path]:
    """
    Resample all image-mask pairs in a directory to target spacing.

    Args:
        input_dir: Input directory with NIfTI files.
        output_dir: Output directory for resampled files.
        target_spacing: Target isotropic spacing in mm.
        image_pattern: Glob pattern for image files.
        label_suffix: Suffix for label files.

    Returns:
        List of successfully processed image paths.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files (excluding labels)
    images = sorted(input_dir.glob(image_pattern))
    images = [f for f in images if label_suffix not in f.name]

    logger.info(f"Target spacing: {target_spacing} mm")
    logger.info(f"Processing {len(images)} patients...")

    processed = []

    for image_path in images:
        mask_path = image_path.parent / f"{image_path.stem}{label_suffix}.nii"

        if not mask_path.exists():
            logger.warning(f"No mask for {image_path.name}, skipping")
            continue

        try:
            # Load
            image = nib.load(image_path)
            mask = nib.load(mask_path)

            # Resample
            resampled_image, resampled_mask = resample_image_and_mask(
                image, mask, target_spacing
            )

            # Save
            nib.save(resampled_image, output_dir / image_path.name)
            nib.save(resampled_mask, output_dir / mask_path.name)

            logger.info(
                f"Resampled {image_path.stem}: {image.shape} -> {resampled_image.shape}"
            )
            processed.append(image_path)

        except Exception as e:
            logger.error(f"Failed on {image_path.stem}: {e}")

    logger.info(f"Successfully resampled {len(processed)}/{len(images)} pairs")
    return processed
