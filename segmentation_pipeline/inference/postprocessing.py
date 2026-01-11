"""
Post-processing utilities for segmentation predictions.

Includes thresholding, connected component analysis, and morphological operations.
"""

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage
from skimage import measure


def threshold_predictions(
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Apply threshold to probability map.

    Args:
        probabilities: Probability map in range [0, 1].
        threshold: Threshold value.

    Returns:
        Binary mask as uint8.
    """
    return (probabilities > threshold).astype(np.uint8)


def keep_largest_component(
    mask: np.ndarray,
    connectivity: int = 1,
) -> np.ndarray:
    """
    Keep only the largest connected component.

    Args:
        mask: Binary mask.
        connectivity: Connectivity for labeling (1, 2, or 3).

    Returns:
        Binary mask with only largest component.
    """
    labels = measure.label(mask, connectivity=connectivity)

    if labels.max() == 0:
        return mask

    counts = np.bincount(labels.ravel())
    counts[0] = 0  # Ignore background

    largest_label = counts.argmax()
    return (labels == largest_label).astype(mask.dtype)


def remove_small_components(
    mask: np.ndarray,
    min_size: int,
    connectivity: int = 1,
) -> np.ndarray:
    """
    Remove connected components smaller than min_size.

    Args:
        mask: Binary mask.
        min_size: Minimum component size in voxels.
        connectivity: Connectivity for labeling.

    Returns:
        Filtered binary mask.
    """
    labels = measure.label(mask, connectivity=connectivity)
    counts = np.bincount(labels.ravel())

    # Zero out small components
    for label_id in range(1, len(counts)):
        if counts[label_id] < min_size:
            labels[labels == label_id] = 0

    return (labels > 0).astype(mask.dtype)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in binary mask.

    Args:
        mask: Binary mask.

    Returns:
        Mask with holes filled.
    """
    return ndimage.binary_fill_holes(mask).astype(mask.dtype)


def apply_morphological_closing(
    mask: np.ndarray,
    radius: int = 1,
) -> np.ndarray:
    """
    Apply morphological closing (dilation then erosion).

    Useful for filling small gaps in segmentation.

    Args:
        mask: Binary mask.
        radius: Radius of structuring element.

    Returns:
        Closed mask.
    """
    struct = ndimage.generate_binary_structure(3, 1)
    struct = ndimage.iterate_structure(struct, radius)

    dilated = ndimage.binary_dilation(mask, structure=struct)
    closed = ndimage.binary_erosion(dilated, structure=struct)

    return closed.astype(mask.dtype)


def apply_morphological_opening(
    mask: np.ndarray,
    radius: int = 1,
) -> np.ndarray:
    """
    Apply morphological opening (erosion then dilation).

    Useful for removing small noise.

    Args:
        mask: Binary mask.
        radius: Radius of structuring element.

    Returns:
        Opened mask.
    """
    struct = ndimage.generate_binary_structure(3, 1)
    struct = ndimage.iterate_structure(struct, radius)

    eroded = ndimage.binary_erosion(mask, structure=struct)
    opened = ndimage.binary_dilation(eroded, structure=struct)

    return opened.astype(mask.dtype)


def smooth_mask(
    mask: np.ndarray,
    sigma: float = 1.0,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Smooth mask boundaries using Gaussian blur.

    Args:
        mask: Binary mask.
        sigma: Gaussian blur sigma.
        threshold: Threshold for re-binarization.

    Returns:
        Smoothed binary mask.
    """
    smoothed = ndimage.gaussian_filter(mask.astype(np.float32), sigma=sigma)
    return (smoothed > threshold).astype(mask.dtype)


def postprocess_prediction(
    probabilities: np.ndarray,
    threshold: float = 0.5,
    keep_largest: bool = True,
    min_component_size: Optional[int] = None,
    fill_holes_flag: bool = False,
    closing_radius: Optional[int] = None,
) -> np.ndarray:
    """
    Apply full post-processing pipeline to predictions.

    Args:
        probabilities: Probability map from model.
        threshold: Probability threshold.
        keep_largest: Keep only largest component.
        min_component_size: Remove components smaller than this.
        fill_holes_flag: Fill holes in mask.
        closing_radius: Apply morphological closing with this radius.

    Returns:
        Post-processed binary mask.
    """
    # Threshold
    mask = threshold_predictions(probabilities, threshold)

    # Morphological operations
    if closing_radius is not None and closing_radius > 0:
        mask = apply_morphological_closing(mask, radius=closing_radius)

    # Remove small components
    if min_component_size is not None and min_component_size > 0:
        mask = remove_small_components(mask, min_size=min_component_size)

    # Keep largest component
    if keep_largest:
        mask = keep_largest_component(mask)

    # Fill holes
    if fill_holes_flag:
        mask = fill_holes(mask)

    return mask
