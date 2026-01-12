"""
PyTorch dataset for bounding box localization.

Loads NPZ files containing image-bbox pairs for training.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def flip_bbox(bbox: np.ndarray, axis: int) -> np.ndarray:
    """
    Flip bounding box coordinates along an axis.

    Args:
        bbox: Normalized bbox [z1, y1, x1, z2, y2, x2] in [0, 1].
        axis: Axis to flip (0=z, 1=y, 2=x).

    Returns:
        Flipped bbox coordinates.
    """
    bbox = bbox.copy()
    # For normalized coordinates, flip is: new_coord = 1 - old_coord
    # Also need to swap min/max since they reverse after flip
    min_idx = axis
    max_idx = axis + 3
    new_min = 1.0 - bbox[max_idx]
    new_max = 1.0 - bbox[min_idx]
    bbox[min_idx] = new_min
    bbox[max_idx] = new_max
    return bbox


class LocalizationDataset(Dataset):
    """
    PyTorch Dataset for 3D bounding box localization.

    Loads NPZ files containing:
    - 'image': 3D volume (D, H, W) uint8
    - 'label': Normalized bbox coordinates (6,) float32
    """

    def __init__(
        self,
        file_paths: List[Path],
        normalize_image: bool = True,
        image_key: str = "image",
        label_key: str = "label",
        augment: bool = False,
        flip_prob: float = 0.5,
        intensity_shift_range: float = 0.1,
        intensity_scale_range: float = 0.1,
    ) -> None:
        """
        Initialize dataset.

        Args:
            file_paths: List of paths to NPZ files.
            normalize_image: Whether to normalize image to [0, 1].
            image_key: Key for image array in NPZ file.
            label_key: Key for label array in NPZ file.
            augment: Whether to apply data augmentation.
            flip_prob: Probability of flipping along each axis.
            intensity_shift_range: Max intensity shift (fraction of range).
            intensity_scale_range: Max intensity scale variation.
        """
        self.files = [Path(p) for p in file_paths]
        self.normalize_image = normalize_image
        self.image_key = image_key
        self.label_key = label_key
        self.augment = augment
        self.flip_prob = flip_prob
        self.intensity_shift_range = intensity_shift_range
        self.intensity_scale_range = intensity_scale_range

        # Validate files exist
        for f in self.files:
            if not f.exists():
                logger.warning(f"File not found: {f}")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, label_tensor).
            - image_tensor: Shape (1, D, H, W) float32
            - label_tensor: Shape (6,) float32 normalized bbox
        """
        try:
            data = np.load(self.files[idx])
            image = data[self.image_key]
            label = data[self.label_key]
        except Exception as e:
            logger.error(f"Error loading {self.files[idx]}: {e}")
            raise e

        # Convert to float
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        # Normalize image to [0, 1]
        if self.normalize_image and image.max() > 1:
            image = image / 255.0

        # Apply augmentation if enabled
        if self.augment:
            image, label = self._apply_augmentation(image, label)

        # Convert to tensor with channel dimension
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, D, H, W)
        label_tensor = torch.from_numpy(label)  # (6,)

        return image_tensor, label_tensor

    def _apply_augmentation(
        self,
        image: np.ndarray,
        label: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random augmentations to image and label.

        Args:
            image: 3D volume (D, H, W).
            label: Bbox coordinates (6,).

        Returns:
            Augmented (image, label) tuple.
        """
        # Random flips along each axis
        for axis in range(3):
            if np.random.random() < self.flip_prob:
                image = np.flip(image, axis=axis).copy()
                label = flip_bbox(label, axis)

        # Random intensity shift
        if self.intensity_shift_range > 0:
            shift = np.random.uniform(
                -self.intensity_shift_range,
                self.intensity_shift_range,
            )
            image = image + shift

        # Random intensity scale
        if self.intensity_scale_range > 0:
            scale = np.random.uniform(
                1.0 - self.intensity_scale_range,
                1.0 + self.intensity_scale_range,
            )
            image = image * scale

        # Clamp to valid range
        image = np.clip(image, 0.0, 1.0)

        return image, label

    def get_sample_info(self, idx: int) -> Dict:
        """
        Get metadata about a sample without full loading.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with file path and shape info.
        """
        path = self.files[idx]
        data = np.load(path)

        info = {
            "path": str(path),
            "filename": path.name,
            "image_shape": data[self.image_key].shape,
            "label_shape": data[self.label_key].shape,
            "label_values": data[self.label_key].tolist(),
        }

        if "original_shape" in data:
            info["original_shape"] = data["original_shape"].tolist()

        return info


def create_data_splits(
    file_paths: List[Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """
    Split file paths into train/val sets.

    Args:
        file_paths: List of all file paths.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_files, val_files).
    """
    assert abs(train_ratio + val_ratio - 1.0) < 1e-6

    # Shuffle with seed
    rng = np.random.default_rng(seed)
    indices = np.arange(len(file_paths))
    rng.shuffle(indices)

    # Calculate split point
    n = len(file_paths)
    train_end = int(n * train_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:]

    train_files = [file_paths[i] for i in train_indices]
    val_files = [file_paths[i] for i in val_indices]

    return train_files, val_files


def get_dataset_files(
    data_dir: Union[str, Path],
    pattern: str = "*.npz",
) -> List[Path]:
    """
    Get list of dataset files from directory.

    Args:
        data_dir: Directory containing NPZ files.
        pattern: Glob pattern.

    Returns:
        Sorted list of file paths.
    """
    data_dir = Path(data_dir)
    return sorted(data_dir.glob(pattern))
