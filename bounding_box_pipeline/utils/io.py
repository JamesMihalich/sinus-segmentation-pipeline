"""
I/O utilities for loading and saving data.

Handles NPZ and NIfTI file formats commonly used in medical imaging.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def load_npz(
    path: Union[str, Path],
    keys: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Load arrays from NPZ file.

    Args:
        path: Path to NPZ file.
        keys: Specific keys to load. If None, loads all.

    Returns:
        Dictionary mapping keys to arrays.
    """
    path = Path(path)

    with np.load(path) as data:
        if keys is None:
            return {k: data[k] for k in data.files}
        return {k: data[k] for k in keys if k in data.files}


def load_npz_image(
    path: Union[str, Path],
    image_key: str = "image",
) -> np.ndarray:
    """
    Load image array from NPZ file.

    Args:
        path: Path to NPZ file.
        image_key: Key for image array.

    Returns:
        Image array.
    """
    with np.load(path) as data:
        if image_key in data:
            return data[image_key]
        # Try common alternatives
        for key in ["image", "img", "volume", "data"]:
            if key in data:
                return data[key]
        # Fallback to first array
        return data[data.files[0]]


def load_npz_label(
    path: Union[str, Path],
    label_key: str = "label",
) -> np.ndarray:
    """
    Load label array from NPZ file.

    Args:
        path: Path to NPZ file.
        label_key: Key for label array.

    Returns:
        Label array (bounding box coordinates).
    """
    with np.load(path) as data:
        if label_key in data:
            return data[label_key]
        # Try common alternatives
        for key in ["label", "bbox", "box", "target"]:
            if key in data:
                return data[key]
        raise KeyError(f"No label key found in {path}")


def save_npz(
    path: Union[str, Path],
    image: Optional[np.ndarray] = None,
    label: Optional[np.ndarray] = None,
    compressed: bool = True,
    **kwargs: np.ndarray,
) -> Path:
    """
    Save arrays to NPZ file.

    Args:
        path: Output path.
        image: Image array.
        label: Label array (bbox coordinates).
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

    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)

    return path


def load_nifti(
    path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load NIfTI file and return data with affine.

    Args:
        path: Path to NIfTI file.

    Returns:
        Tuple of (data array, affine matrix).
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel required: pip install nibabel")

    nii = nib.load(path)
    data = nii.get_fdata()
    affine = nii.affine

    return data, affine


def load_nifti_data(path: Union[str, Path]) -> np.ndarray:
    """
    Load NIfTI file and return only the data array.

    Args:
        path: Path to NIfTI file.

    Returns:
        Data array.
    """
    data, _ = load_nifti(path)
    return data


def save_nifti(
    path: Union[str, Path],
    data: np.ndarray,
    affine: Optional[np.ndarray] = None,
) -> Path:
    """
    Save array to NIfTI file.

    Args:
        path: Output path.
        data: Data array.
        affine: Affine matrix. If None, uses identity.

    Returns:
        Path to saved file.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel required: pip install nibabel")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if affine is None:
        affine = np.eye(4)

    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, path)

    return path
