"""
CT windowing utilities for medical image preprocessing.

Applies window/level transformations to normalize CT intensity values.
"""

from typing import Tuple, Union

import numpy as np


def apply_ct_window(
    image: np.ndarray,
    window_level: float,
    window_width: float,
    output_range: Tuple[float, float] = (0, 255),
    output_dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """
    Apply CT windowing (Width/Level) and normalize to output range.

    Args:
        image: Input CT image (typically int16 or float32).
        window_level: Center of the window (in Hounsfield Units).
        window_width: Width of the window (in HU).
        output_range: Output value range (min, max).
        output_dtype: Output data type.

    Returns:
        Windowed and normalized image.

    Examples:
        Bone window: level=400, width=1500
        Soft tissue: level=40, width=400
        Lung window: level=-600, width=1500
        Brain window: level=40, width=80
    """
    lower_bound = window_level - (window_width / 2)
    upper_bound = window_level + (window_width / 2)

    # Clip to window
    windowed = np.clip(image, lower_bound, upper_bound)

    # Normalize to 0-1
    if window_width > 0:
        normalized = (windowed - lower_bound) / window_width
    else:
        normalized = windowed - lower_bound

    # Scale to output range
    out_min, out_max = output_range
    scaled = normalized * (out_max - out_min) + out_min

    return scaled.astype(output_dtype)


def get_window_preset(preset: str) -> Tuple[float, float]:
    """
    Get predefined CT window settings.

    Args:
        preset: Name of window preset.

    Returns:
        Tuple of (window_level, window_width).

    Available presets:
        - bone: Bone visualization
        - soft_tissue: Soft tissue visualization
        - lung: Lung parenchyma
        - brain: Brain tissue
        - liver: Liver window
        - mediastinum: Chest soft tissue
        - abdomen: Abdominal organs
    """
    presets = {
        "bone": (400, 1500),
        "soft_tissue": (40, 400),
        "lung": (-600, 1500),
        "brain": (40, 80),
        "liver": (60, 150),
        "mediastinum": (50, 350),
        "abdomen": (40, 400),
        "custom_airways": (-300, 1000),  # Custom for airway segmentation
    }

    if preset not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

    return presets[preset]


def apply_preset_window(
    image: np.ndarray,
    preset: str,
    output_range: Tuple[float, float] = (0, 255),
    output_dtype: np.dtype = np.uint8,
) -> np.ndarray:
    """
    Apply a predefined CT window preset.

    Args:
        image: Input CT image.
        preset: Window preset name.
        output_range: Output value range.
        output_dtype: Output data type.

    Returns:
        Windowed image.
    """
    level, width = get_window_preset(preset)
    return apply_ct_window(
        image, level, width, output_range=output_range, output_dtype=output_dtype
    )


def normalize_intensity(
    image: np.ndarray,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
    output_range: Tuple[float, float] = (0, 1),
) -> np.ndarray:
    """
    Normalize intensity using percentile clipping.

    Args:
        image: Input image.
        percentile_low: Lower percentile for clipping.
        percentile_high: Upper percentile for clipping.
        output_range: Output value range.

    Returns:
        Normalized image.
    """
    low = np.percentile(image, percentile_low)
    high = np.percentile(image, percentile_high)

    clipped = np.clip(image, low, high)

    if high > low:
        normalized = (clipped - low) / (high - low)
    else:
        normalized = clipped - low

    out_min, out_max = output_range
    return normalized * (out_max - out_min) + out_min


def standardize_volume(
    image: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Z-score standardization (zero mean, unit variance).

    Args:
        image: Input image.
        eps: Small value to prevent division by zero.

    Returns:
        Standardized image with mean=0 and std=1.
    """
    mean = image.mean()
    std = image.std()

    if std > eps:
        return (image - mean) / std
    else:
        return image - mean
