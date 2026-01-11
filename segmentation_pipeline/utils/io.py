"""
I/O utilities for loading and saving medical imaging data.

Consolidates NPZ and NIfTI file handling with smart key detection.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np


# Default key priority for NPZ files
DEFAULT_IMAGE_KEYS = ["image.npy", "image", "img", "volume", "data"]
DEFAULT_LABEL_KEYS = ["label.npy", "label", "mask", "segmentation", "seg"]


def load_npz(
    path: Union[str, Path],
    key: Optional[str] = None,
    priority_keys: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Load array from NPZ file with intelligent key detection.

    Args:
        path: Path to NPZ file.
        key: Specific key to load. If provided, uses this directly.
        priority_keys: List of keys to try in order. Falls back to first available.

    Returns:
        Loaded numpy array.

    Raises:
        KeyError: If specified key not found.
        ValueError: If file contains no arrays.
    """
    path = Path(path)
    data = np.load(path)

    # If specific key requested, use it
    if key is not None:
        if key in data:
            return data[key]
        raise KeyError(f"Key '{key}' not found in {path.name}. Available: {data.files}")

    # Try priority keys in order
    if priority_keys:
        for k in priority_keys:
            if k in data:
                return data[k]

    # Fallback: return first available array
    if len(data.files) > 0:
        return data[data.files[0]]

    raise ValueError(f"No arrays found in {path.name}")


def load_npz_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load image array from NPZ file using image key priority.

    Args:
        path: Path to NPZ file.

    Returns:
        Image array.
    """
    return load_npz(path, priority_keys=DEFAULT_IMAGE_KEYS)


def load_npz_label(path: Union[str, Path]) -> np.ndarray:
    """
    Load label/mask array from NPZ file using label key priority.

    Args:
        path: Path to NPZ file.

    Returns:
        Label/mask array.
    """
    return load_npz(path, priority_keys=DEFAULT_LABEL_KEYS)


def load_npz_pair(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load both image and label from an NPZ file containing both.

    Args:
        path: Path to NPZ file.

    Returns:
        Tuple of (image, label) arrays.
    """
    path = Path(path)
    data = np.load(path)

    image = None
    label = None

    # Find image
    for k in DEFAULT_IMAGE_KEYS:
        if k in data:
            image = data[k]
            break

    # Find label
    for k in DEFAULT_LABEL_KEYS:
        if k in data:
            label = data[k]
            break

    if image is None:
        raise KeyError(f"No image key found in {path.name}. Available: {data.files}")
    if label is None:
        raise KeyError(f"No label key found in {path.name}. Available: {data.files}")

    return image, label


def save_npz(
    path: Union[str, Path],
    image: Optional[np.ndarray] = None,
    label: Optional[np.ndarray] = None,
    compressed: bool = True,
    **kwargs,
) -> Path:
    """
    Save arrays to NPZ file with standardized keys.

    Args:
        path: Output path.
        image: Image array (saved with key 'image').
        label: Label/mask array (saved with key 'label').
        compressed: Whether to use compression.
        **kwargs: Additional arrays to save.

    Returns:
        Path to saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    if image is not None:
        arrays["image"] = image
    if label is not None:
        arrays["label"] = label
    arrays.update(kwargs)

    if not arrays:
        raise ValueError("No arrays provided to save")

    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)

    return path


def load_nifti(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load NIfTI file and return data with affine matrix.

    Args:
        path: Path to NIfTI file (.nii or .nii.gz).

    Returns:
        Tuple of (data array, affine matrix).
    """
    path = Path(path)
    nii = nib.load(path)
    data = nii.get_fdata()
    affine = nii.affine
    return data, affine


def save_nifti(
    path: Union[str, Path],
    data: np.ndarray,
    affine: np.ndarray,
    dtype: Optional[np.dtype] = None,
) -> Path:
    """
    Save array as NIfTI file.

    Args:
        path: Output path.
        data: Data array.
        affine: 4x4 affine transformation matrix.
        dtype: Optional dtype to cast data to before saving.

    Returns:
        Path to saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if dtype is not None:
        data = data.astype(dtype)

    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, path)

    return path


def get_npz_info(path: Union[str, Path]) -> Dict[str, Tuple[tuple, np.dtype]]:
    """
    Get information about arrays in an NPZ file without fully loading them.

    Args:
        path: Path to NPZ file.

    Returns:
        Dictionary mapping keys to (shape, dtype) tuples.
    """
    path = Path(path)
    info = {}

    with np.load(path) as data:
        for key in data.files:
            arr = data[key]
            info[key] = (arr.shape, arr.dtype)

    return info


def get_nifti_info(path: Union[str, Path]) -> Dict[str, any]:
    """
    Get NIfTI header information without loading full data.

    Args:
        path: Path to NIfTI file.

    Returns:
        Dictionary with shape, dtype, spacing, and affine.
    """
    path = Path(path)
    nii = nib.load(path)
    header = nii.header

    return {
        "shape": tuple(header.get_data_shape()),
        "dtype": header.get_data_dtype(),
        "spacing": tuple(header.get_zooms()),
        "affine": nii.affine,
    }
