"""
3D volume augmentation transforms for medical image segmentation.

Provides random augmentations that are applied consistently to image-mask pairs.
"""

import random
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import rotate, zoom


@dataclass
class AugmentationParams:
    """Parameters for volume augmentation."""

    flip_prob: float = 0.5
    rotate_prob: float = 0.4
    rotate_range: Tuple[float, float] = (-10.0, 10.0)
    scale_prob: float = 0.2
    scale_range: Tuple[float, float] = (0.95, 1.05)
    noise_prob: float = 0.4
    noise_std: float = 0.05
    brightness_prob: float = 0.4
    brightness_range: Tuple[float, float] = (0.9, 1.1)


class VolumeAugmentation:
    """
    3D volume augmentation for medical images.

    Applies random transformations to image-mask pairs consistently.
    Transformations are only applied during training.
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.4,
        rotate_range: Tuple[float, float] = (-10.0, 10.0),
        scale_prob: float = 0.2,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        noise_prob: float = 0.4,
        noise_std: float = 0.05,
        brightness_prob: float = 0.4,
        brightness_range: Tuple[float, float] = (0.9, 1.1),
    ) -> None:
        """
        Initialize augmentation with probability and range parameters.

        Args:
            flip_prob: Probability of applying random flip.
            rotate_prob: Probability of applying rotation.
            rotate_range: Range of rotation angles in degrees.
            scale_prob: Probability of applying scaling.
            scale_range: Range of scale factors.
            noise_prob: Probability of adding Gaussian noise.
            noise_std: Standard deviation of Gaussian noise.
            brightness_prob: Probability of brightness adjustment.
            brightness_range: Range of brightness multipliers.
        """
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.rotate_range = rotate_range
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.brightness_prob = brightness_prob
        self.brightness_range = brightness_range

    @classmethod
    def from_params(cls, params: AugmentationParams) -> "VolumeAugmentation":
        """Create instance from AugmentationParams dataclass."""
        return cls(
            flip_prob=params.flip_prob,
            rotate_prob=params.rotate_prob,
            rotate_range=params.rotate_range,
            scale_prob=params.scale_prob,
            scale_range=params.scale_range,
            noise_prob=params.noise_prob,
            noise_std=params.noise_std,
            brightness_prob=params.brightness_prob,
            brightness_range=params.brightness_range,
        )

    def __call__(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply augmentations to volume and mask.

        Args:
            volume: Image volume (D, H, W).
            mask: Mask volume (D, H, W).

        Returns:
            Tuple of (augmented_volume, augmented_mask).
        """
        # Random flips
        if random.random() < self.flip_prob:
            axis = random.choice([0, 1, 2])
            volume = np.flip(volume, axis=axis).copy()
            mask = np.flip(mask, axis=axis).copy()

        # Random rotation
        if random.random() < self.rotate_prob:
            angle = random.uniform(*self.rotate_range)
            axes = random.choice([(0, 1), (0, 2), (1, 2)])
            volume = rotate(volume, angle, axes=axes, reshape=False, order=1)
            mask = rotate(mask, angle, axes=axes, reshape=False, order=0)
            mask = (mask > 0.5).astype(np.float32)

        # Random scaling
        if random.random() < self.scale_prob:
            scale = random.uniform(*self.scale_range)
            original_shape = volume.shape
            volume = zoom(volume, scale, order=1)
            mask = zoom(mask, scale, order=0)
            volume = self._resize_to_shape(volume, original_shape)
            mask = self._resize_to_shape(mask, original_shape)
            mask = (mask > 0.5).astype(np.float32)

        # Gaussian noise (image only)
        if random.random() < self.noise_prob:
            noise = np.random.normal(0, self.noise_std, volume.shape)
            volume = volume + noise

        # Brightness adjustment (image only)
        if random.random() < self.brightness_prob:
            factor = random.uniform(*self.brightness_range)
            volume = volume * factor

        return volume, mask

    def _resize_to_shape(
        self,
        arr: np.ndarray,
        target_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Resize array to target shape via padding/cropping.

        Args:
            arr: Input array.
            target_shape: Target shape.

        Returns:
            Resized array.
        """
        current_shape = np.array(arr.shape)
        target = np.array(target_shape)

        # Pad if smaller
        pad_width = []
        for c, t in zip(current_shape, target):
            if c < t:
                diff = t - c
                pad_before = diff // 2
                pad_after = diff - pad_before
                pad_width.append((pad_before, pad_after))
            else:
                pad_width.append((0, 0))

        if any(p != (0, 0) for p in pad_width):
            arr = np.pad(arr, pad_width, mode="constant", constant_values=0)

        # Crop if larger
        current_shape = np.array(arr.shape)
        if np.any(current_shape > target):
            starts = (current_shape - target) // 2
            slices = tuple(slice(s, s + t) for s, t in zip(starts, target))
            arr = arr[slices]

        return arr


class IdentityAugmentation:
    """No-op augmentation for validation/test sets."""

    def __call__(
        self,
        volume: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return inputs unchanged."""
        return volume, mask
