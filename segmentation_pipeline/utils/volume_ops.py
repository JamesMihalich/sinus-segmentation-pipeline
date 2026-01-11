"""
Volume operations for 3D medical imaging data.

Includes cropping, padding, bounding box extraction, and morphological operations.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from skimage import measure


def get_bbox_slices(
    volume: np.ndarray,
    padding: int = 0,
    return_coords: bool = False,
) -> Union[Tuple[slice, ...], Tuple[Tuple[slice, ...], np.ndarray, np.ndarray]]:
    """
    Calculate 3D bounding box slices for non-zero region.

    Args:
        volume: 3D array to find bounding box of.
        padding: Padding to add around bounding box.
        return_coords: If True, also return min/max coordinates.

    Returns:
        Tuple of slice objects for each dimension.
        If return_coords=True, also returns (min_coords, max_coords).

    Raises:
        ValueError: If volume is empty (all zeros).
    """
    nonzero_coords = np.argwhere(volume > 0)

    if nonzero_coords.size == 0:
        raise ValueError("Volume is empty (all zeros)")

    min_coords = nonzero_coords.min(axis=0)
    max_coords = nonzero_coords.max(axis=0)

    slices = []
    for axis in range(3):
        start = max(0, min_coords[axis] - padding)
        stop = min(volume.shape[axis], max_coords[axis] + 1 + padding)
        slices.append(slice(start, stop))

    slices = tuple(slices)

    if return_coords:
        return slices, min_coords, max_coords
    return slices


def crop_to_nonzero(
    volume: np.ndarray,
    padding: int = 0,
    return_slices: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[slice, ...]]]:
    """
    Crop volume to bounding box of non-zero region.

    Args:
        volume: 3D array to crop.
        padding: Padding to add around bounding box.
        return_slices: If True, also return the slice objects used.

    Returns:
        Cropped volume.
        If return_slices=True, returns tuple of (cropped_volume, slices).
    """
    assert volume.ndim == 3, f"Expected 3D volume, got {volume.ndim}D"

    slices = get_bbox_slices(volume, padding=padding)
    cropped = volume[slices]

    if return_slices:
        return cropped, slices
    return cropped


def crop_pair_to_mask(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop both image and mask to the mask's bounding box.

    Args:
        image: Image volume.
        mask: Mask/label volume.
        padding: Padding around bounding box.

    Returns:
        Tuple of (cropped_image, cropped_mask).
    """
    slices = get_bbox_slices(mask, padding=padding)
    return image[slices], mask[slices]


def pad_to_shape(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int],
    mode: str = "constant",
    constant_value: float = 0,
) -> np.ndarray:
    """
    Pad volume to target shape, centered.

    Args:
        volume: 3D array to pad.
        target_shape: Desired output shape.
        mode: Padding mode ('constant', 'edge', 'reflect', etc.).
        constant_value: Value for constant padding.

    Returns:
        Padded volume.

    Raises:
        ValueError: If volume is larger than target in any dimension.
    """
    current_shape = volume.shape
    pad_widths = []

    for curr, target in zip(current_shape, target_shape):
        if curr > target:
            raise ValueError(
                f"Volume dimension {curr} exceeds target {target}. "
                "Use cropping instead."
            )
        total_pad = target - curr
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_widths.append((pad_before, pad_after))

    if mode == "constant":
        return np.pad(volume, pad_widths, mode=mode, constant_values=constant_value)
    return np.pad(volume, pad_widths, mode=mode)


def pad_to_minimum(
    volume: np.ndarray,
    min_shape: Tuple[int, int, int],
    mode: str = "constant",
    constant_value: float = 0,
) -> np.ndarray:
    """
    Pad volume to minimum shape if needed.

    Args:
        volume: 3D array to pad.
        min_shape: Minimum shape for each dimension.
        mode: Padding mode.
        constant_value: Value for constant padding.

    Returns:
        Padded volume (or original if already large enough).
    """
    target_shape = tuple(
        max(curr, min_dim) for curr, min_dim in zip(volume.shape, min_shape)
    )

    if target_shape == volume.shape:
        return volume

    return pad_to_shape(volume, target_shape, mode=mode, constant_value=constant_value)


def pad_to_divisible(
    volume: np.ndarray,
    divisor: int = 8,
    mode: str = "constant",
    constant_value: float = 0,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Pad volume so each dimension is divisible by divisor.

    Useful for U-Net architectures that require specific divisibility.

    Args:
        volume: 3D array to pad.
        divisor: Number each dimension should be divisible by.
        mode: Padding mode.
        constant_value: Value for constant padding.

    Returns:
        Tuple of (padded_volume, original_shape).
    """
    original_shape = volume.shape
    target_shape = tuple(
        ((dim + divisor - 1) // divisor) * divisor for dim in original_shape
    )

    if target_shape == original_shape:
        return volume, original_shape

    padded = pad_to_shape(volume, target_shape, mode=mode, constant_value=constant_value)
    return padded, original_shape


def remove_padding(
    volume: np.ndarray,
    original_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Remove padding to restore original shape.

    Assumes padding was applied symmetrically (centered).

    Args:
        volume: Padded 3D array.
        original_shape: Original shape before padding.

    Returns:
        Volume cropped to original shape.
    """
    current_shape = volume.shape
    slices = []

    for curr, orig in zip(current_shape, original_shape):
        start = (curr - orig) // 2
        stop = start + orig
        slices.append(slice(start, stop))

    return volume[tuple(slices)]


def keep_largest_component(
    volume: np.ndarray,
    connectivity: int = 1,
) -> np.ndarray:
    """
    Keep only the largest connected component in a binary volume.

    Args:
        volume: Binary 3D array.
        connectivity: Connectivity for labeling (1, 2, or 3).

    Returns:
        Binary volume with only largest component.
    """
    labels = measure.label(volume, connectivity=connectivity)

    if labels.max() == 0:
        return volume

    counts = np.bincount(labels.ravel())
    counts[0] = 0  # Ignore background

    largest_label = counts.argmax()
    return (labels == largest_label).astype(volume.dtype)


def remove_small_components(
    volume: np.ndarray,
    min_size: int,
    connectivity: int = 1,
) -> np.ndarray:
    """
    Remove connected components smaller than min_size.

    Args:
        volume: Binary 3D array.
        min_size: Minimum component size to keep.
        connectivity: Connectivity for labeling.

    Returns:
        Binary volume with small components removed.
    """
    labels = measure.label(volume, connectivity=connectivity)
    counts = np.bincount(labels.ravel())

    # Set labels of small components to 0
    small_labels = np.where(counts < min_size)[0]
    for label_id in small_labels:
        if label_id != 0:  # Don't touch background
            labels[labels == label_id] = 0

    return (labels > 0).astype(volume.dtype)


def compute_volume_stats(volume: np.ndarray) -> dict:
    """
    Compute basic statistics for a volume.

    Args:
        volume: 3D array.

    Returns:
        Dictionary with shape, dtype, min, max, mean, std.
    """
    return {
        "shape": volume.shape,
        "dtype": str(volume.dtype),
        "min": float(volume.min()),
        "max": float(volume.max()),
        "mean": float(volume.mean()),
        "std": float(volume.std()),
        "nonzero_count": int(np.count_nonzero(volume)),
        "nonzero_ratio": float(np.count_nonzero(volume) / volume.size),
    }
