"""
CT windowing and volume resizing utilities.

Provides functions for preprocessing medical imaging volumes.
"""

from typing import Tuple

import numpy as np
from scipy.ndimage import zoom


def apply_ct_window(
    image_data: np.ndarray,
    window_level: float,
    window_width: float,
    output_range: Tuple[float, float] = (0, 255),
    output_dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """
    Apply CT windowing to image data.

    CT windowing enhances visualization of specific tissue types by
    mapping a range of Hounsfield units to the display range.

    Args:
        image_data: Input image array (typically in Hounsfield units).
        window_level: Window center (level).
        window_width: Window width.
        output_range: Output value range (default 0-255 for uint8).
        output_dtype: Output data type.

    Returns:
        Windowed image array.

    Example:
        # Soft tissue window
        windowed = apply_ct_window(ct_data, window_level=40, window_width=400)

        # Lung window
        windowed = apply_ct_window(ct_data, window_level=-600, window_width=1500)

        # Bone window
        windowed = apply_ct_window(ct_data, window_level=400, window_width=1800)
    """
    # Calculate window bounds
    lower = window_level - (window_width / 2)
    upper = window_level + (window_width / 2)

    # Clip to window range
    windowed = np.clip(image_data, lower, upper)

    # Normalize to output range
    windowed = (windowed - lower) / (upper - lower)
    windowed = windowed * (output_range[1] - output_range[0]) + output_range[0]

    return windowed.astype(output_dtype)


def resize_volume(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int],
    order: int = 1,
    mode: str = "constant",
) -> np.ndarray:
    """
    Resize 3D volume to target shape using spline interpolation.

    Args:
        volume: Input 3D array.
        target_shape: Desired output shape (D, H, W).
        order: Interpolation order (0=nearest, 1=linear, 3=cubic).
        mode: How to handle boundaries ('constant', 'nearest', etc.).

    Returns:
        Resized volume.
    """
    current_shape = np.array(volume.shape)
    target_shape = np.array(target_shape)

    # Calculate zoom factors
    zoom_factors = target_shape / current_shape

    # Apply zoom
    resized = zoom(volume, zoom_factors, order=order, mode=mode)

    return resized


def normalize_volume(
    volume: np.ndarray,
    target_range: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Normalize volume to target range.

    Args:
        volume: Input array.
        target_range: Output (min, max) range.

    Returns:
        Normalized array.
    """
    v_min = volume.min()
    v_max = volume.max()

    if v_max - v_min < 1e-6:
        return np.full_like(volume, target_range[0], dtype=np.float32)

    normalized = (volume - v_min) / (v_max - v_min)
    normalized = normalized * (target_range[1] - target_range[0]) + target_range[0]

    return normalized.astype(np.float32)
