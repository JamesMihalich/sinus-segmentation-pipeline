"""
Configuration management for the segmentation pipeline.

Provides dataclass-based configuration with YAML loading support.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


@dataclass
class AugmentationConfig:
    """Augmentation parameters for training."""

    flip_prob: float = 0.5
    rotate_prob: float = 0.4
    rotate_range: Tuple[float, float] = (-10.0, 10.0)
    scale_prob: float = 0.2
    scale_range: Tuple[float, float] = (0.95, 1.05)
    noise_prob: float = 0.4
    noise_std: float = 0.05
    brightness_prob: float = 0.4
    brightness_range: Tuple[float, float] = (0.9, 1.1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for VolumeAugmentation kwargs."""
        return {
            "flip_prob": self.flip_prob,
            "rotate_prob": self.rotate_prob,
            "rotate_range": self.rotate_range,
            "scale_prob": self.scale_prob,
            "scale_range": self.scale_range,
            "noise_prob": self.noise_prob,
            "noise_std": self.noise_std,
            "brightness_prob": self.brightness_prob,
            "brightness_range": self.brightness_range,
        }


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    architecture: str = "ResidualUnetSE3D"
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 16
    skip_mode: str = "concat"  # "concat" or "additive"
    se_reduction_ratio: int = 8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model kwargs."""
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "base_channels": self.base_channels,
            "skip_mode": self.skip_mode,
            "se_reduction_ratio": self.se_reduction_ratio,
        }


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    patch_size: Tuple[int, int, int] = (224, 224, 256)
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    num_workers: int = 4

    # Optimizer settings
    optimizer: str = "AdamW"
    gradient_clip_norm: float = 1.0

    # Scheduler settings
    scheduler: str = "ReduceLROnPlateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    scheduler_min_lr: float = 1e-6

    # Loss settings
    bce_weight: float = 0.5
    dice_weight: float = 0.5

    # Data split
    train_ratio: float = 0.75
    val_ratio: float = 0.15
    test_ratio: float = 0.10
    split_seed: int = 42


@dataclass
class InferenceConfig:
    """Inference parameters."""

    patch_size: Tuple[int, int, int] = (224, 224, 256)
    overlap: float = 0.5
    threshold: float = 0.5
    apply_postprocessing: bool = True
    keep_largest_component: bool = True


@dataclass
class PreprocessingConfig:
    """Data preprocessing parameters."""

    # CT windowing
    window_level: float = -300.0
    window_width: float = 1000.0

    # Resampling
    target_spacing: Tuple[float, float, float] = (0.33, 0.33, 0.33)

    # Cropping
    crop_padding: int = 10


@dataclass
class Config:
    """
    Main configuration class combining all config sections.

    Can be loaded from YAML or created programmatically.
    """

    # Paths
    data_root: Optional[Path] = None
    output_root: Optional[Path] = None
    checkpoint_path: Optional[Path] = None

    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.data_root, str):
            self.data_root = Path(self.data_root)
        if isinstance(self.output_root, str):
            self.output_root = Path(self.output_root)
        if isinstance(self.checkpoint_path, str):
            self.checkpoint_path = Path(self.checkpoint_path)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            Config instance.
        """
        path = Path(path)

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Handle environment variable expansion in paths
        for key in ["data_root", "output_root", "checkpoint_path"]:
            if key in data and data[key]:
                data[key] = os.path.expandvars(data[key])

        # Parse sub-configs
        model = ModelConfig(**data.pop("model", {}))
        training = TrainingConfig(**_parse_tuples(data.pop("training", {})))
        augmentation = AugmentationConfig(**_parse_tuples(data.pop("augmentation", {})))
        inference = InferenceConfig(**_parse_tuples(data.pop("inference", {})))
        preprocessing = PreprocessingConfig(
            **_parse_tuples(data.pop("preprocessing", {}))
        )

        return cls(
            model=model,
            training=training,
            augmentation=augmentation,
            inference=inference,
            preprocessing=preprocessing,
            **data,
        )

    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Output path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "data_root": str(self.data_root) if self.data_root else None,
            "output_root": str(self.output_root) if self.output_root else None,
            "checkpoint_path": (
                str(self.checkpoint_path) if self.checkpoint_path else None
            ),
            "model": {
                "architecture": self.model.architecture,
                "in_channels": self.model.in_channels,
                "out_channels": self.model.out_channels,
                "base_channels": self.model.base_channels,
                "skip_mode": self.model.skip_mode,
                "se_reduction_ratio": self.model.se_reduction_ratio,
            },
            "training": {
                "patch_size": list(self.training.patch_size),
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "epochs": self.training.epochs,
                "num_workers": self.training.num_workers,
                "optimizer": self.training.optimizer,
                "gradient_clip_norm": self.training.gradient_clip_norm,
                "scheduler": self.training.scheduler,
                "scheduler_factor": self.training.scheduler_factor,
                "scheduler_patience": self.training.scheduler_patience,
                "scheduler_min_lr": self.training.scheduler_min_lr,
                "bce_weight": self.training.bce_weight,
                "dice_weight": self.training.dice_weight,
                "train_ratio": self.training.train_ratio,
                "val_ratio": self.training.val_ratio,
                "test_ratio": self.training.test_ratio,
                "split_seed": self.training.split_seed,
            },
            "augmentation": {
                "flip_prob": self.augmentation.flip_prob,
                "rotate_prob": self.augmentation.rotate_prob,
                "rotate_range": list(self.augmentation.rotate_range),
                "scale_prob": self.augmentation.scale_prob,
                "scale_range": list(self.augmentation.scale_range),
                "noise_prob": self.augmentation.noise_prob,
                "noise_std": self.augmentation.noise_std,
                "brightness_prob": self.augmentation.brightness_prob,
                "brightness_range": list(self.augmentation.brightness_range),
            },
            "inference": {
                "patch_size": list(self.inference.patch_size),
                "overlap": self.inference.overlap,
                "threshold": self.inference.threshold,
                "apply_postprocessing": self.inference.apply_postprocessing,
                "keep_largest_component": self.inference.keep_largest_component,
            },
            "preprocessing": {
                "window_level": self.preprocessing.window_level,
                "window_width": self.preprocessing.window_width,
                "target_spacing": list(self.preprocessing.target_spacing),
                "crop_padding": self.preprocessing.crop_padding,
            },
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire config to nested dictionary."""
        return {
            "data_root": str(self.data_root) if self.data_root else None,
            "output_root": str(self.output_root) if self.output_root else None,
            "checkpoint_path": (
                str(self.checkpoint_path) if self.checkpoint_path else None
            ),
            "model": self.model.to_dict(),
            "training": {
                "patch_size": self.training.patch_size,
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "epochs": self.training.epochs,
            },
            "augmentation": self.augmentation.to_dict(),
        }


def _parse_tuples(d: Dict[str, Any]) -> Dict[str, Any]:
    """Convert lists to tuples for fields that expect tuples."""
    tuple_fields = {
        "patch_size",
        "rotate_range",
        "scale_range",
        "brightness_range",
        "target_spacing",
    }
    result = {}
    for k, v in d.items():
        if k in tuple_fields and isinstance(v, list):
            result[k] = tuple(v)
        else:
            result[k] = v
    return result
