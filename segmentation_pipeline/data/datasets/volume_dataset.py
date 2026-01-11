"""
PyTorch dataset for 3D volume segmentation.

Loads NPZ files containing image-label pairs for training.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ..augmentation.transforms import VolumeAugmentation, IdentityAugmentation

logger = logging.getLogger(__name__)


class VolumeDataset(Dataset):
    """
    PyTorch Dataset for 3D medical image volumes.

    Loads NPZ files containing 'image' and 'label' arrays.
    Supports random patch extraction and augmentation.
    """

    def __init__(
        self,
        file_paths: List[Path],
        patch_size: Optional[Tuple[int, int, int]] = None,
        augment: bool = False,
        aug_params: Optional[Dict] = None,
        normalize: bool = True,
        image_key: str = "image",
        label_key: str = "label",
    ) -> None:
        """
        Initialize dataset.

        Args:
            file_paths: List of paths to NPZ files.
            patch_size: If provided, extract patches of this size.
            augment: Whether to apply augmentation.
            aug_params: Parameters for VolumeAugmentation.
            normalize: Whether to normalize image to [0, 1].
            image_key: Key for image array in NPZ file.
            label_key: Key for label array in NPZ file.
        """
        self.files = [Path(p) for p in file_paths]
        self.patch_size = patch_size
        self.normalize = normalize
        self.image_key = image_key
        self.label_key = label_key

        # Setup augmentation
        if augment:
            params = aug_params if aug_params else {}
            self.aug = VolumeAugmentation(**params)
        else:
            self.aug = IdentityAugmentation()

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
            Tuple of (image_tensor, label_tensor) with shape (1, D, H, W).
        """
        try:
            data = np.load(self.files[idx])
            volume = data[self.image_key]
            mask = data[self.label_key]
        except Exception as e:
            logger.error(f"Error loading {self.files[idx]}: {e}")
            raise e

        # Convert to float
        volume = volume.astype(np.float32)
        mask = mask.astype(np.float32)

        # Normalize image
        if self.normalize:
            if volume.max() > 1:
                volume = volume / 255.0

        # Binarize mask
        mask = (mask > 0).astype(np.float32)

        # Extract patch or pad/crop to size
        if self.patch_size is not None:
            volume, mask = self._pad_or_crop(volume, mask, self.patch_size)

        # Apply augmentation
        volume, mask = self.aug(volume, mask)

        # Ensure correct dtype after augmentation
        volume = volume.astype(np.float32)
        mask = mask.astype(np.float32)

        # Convert to tensor with channel dimension
        volume_tensor = torch.from_numpy(volume).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return volume_tensor, mask_tensor

    def _pad_or_crop(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
        target_size: Tuple[int, int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pad or crop volume and mask to target size.

        Uses random cropping for training diversity.

        Args:
            volume: Image volume.
            mask: Mask volume.
            target_size: Target (D, H, W) size.

        Returns:
            Tuple of (resized_volume, resized_mask).
        """
        D, H, W = volume.shape
        td, th, tw = target_size

        # Random crop if larger
        if D > td or H > th or W > tw:
            sd = np.random.randint(0, max(1, D - td + 1)) if D > td else 0
            sh = np.random.randint(0, max(1, H - th + 1)) if H > th else 0
            sw = np.random.randint(0, max(1, W - tw + 1)) if W > tw else 0

            ed = min(sd + td, D)
            eh = min(sh + th, H)
            ew = min(sw + tw, W)

            volume = volume[sd:ed, sh:eh, sw:ew]
            mask = mask[sd:ed, sh:eh, sw:ew]

        # Pad if smaller
        D, H, W = volume.shape
        if D < td or H < th or W < tw:
            pad_d = max(0, td - D)
            pad_h = max(0, th - H)
            pad_w = max(0, tw - W)

            volume = np.pad(
                volume,
                (
                    (pad_d // 2, pad_d - pad_d // 2),
                    (pad_h // 2, pad_h - pad_h // 2),
                    (pad_w // 2, pad_w - pad_w // 2),
                ),
                mode="constant",
                constant_values=0,
            )
            mask = np.pad(
                mask,
                (
                    (pad_d // 2, pad_d - pad_d // 2),
                    (pad_h // 2, pad_h - pad_h // 2),
                    (pad_w // 2, pad_w - pad_w // 2),
                ),
                mode="constant",
                constant_values=0,
            )

        return volume, mask

    def get_sample_info(self, idx: int) -> Dict:
        """
        Get metadata about a sample without loading full data.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with file path and shape info.
        """
        path = self.files[idx]
        data = np.load(path)

        return {
            "path": str(path),
            "filename": path.name,
            "image_shape": data[self.image_key].shape,
            "label_shape": data[self.label_key].shape,
        }


def create_data_splits(
    file_paths: List[Path],
    train_ratio: float = 0.75,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split file paths into train/val/test sets.

    Args:
        file_paths: List of all file paths.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_files, val_files, test_files).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Shuffle with seed
    rng = np.random.default_rng(seed)
    indices = np.arange(len(file_paths))
    rng.shuffle(indices)

    # Calculate split points
    n = len(file_paths)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_files = [file_paths[i] for i in train_indices]
    val_files = [file_paths[i] for i in val_indices]
    test_files = [file_paths[i] for i in test_indices]

    return train_files, val_files, test_files
