"""
Configuration management for bounding box pipeline.

Provides dataclass-based configuration with YAML loading support.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    architecture: str = "BBoxRegressor3D"
    input_size: Tuple[int, int, int] = (128, 128, 128)
    in_channels: int = 1
    base_channels: int = 32
    dropout: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for model kwargs."""
        return {
            "input_size": self.input_size,
            "in_channels": self.in_channels,
            "base_channels": self.base_channels,
            "dropout": self.dropout,
        }


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    num_workers: int = 4

    # Optimizer
    optimizer: str = "Adam"

    # Scheduler
    scheduler: str = "ReduceLROnPlateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    scheduler_min_lr: float = 1e-6

    # Data split
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    split_seed: int = 42

    # Checkpointing
    save_every: int = 5
    snapshot_every: int = 5


@dataclass
class PreprocessingConfig:
    """Data preprocessing parameters."""

    # Target volume size
    target_shape: Tuple[int, int, int] = (128, 128, 128)

    # CT windowing
    window_level: float = 600.0
    window_width: float = 1250.0


@dataclass
class InferenceConfig:
    """Inference parameters."""

    target_shape: Tuple[int, int, int] = (128, 128, 128)


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
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

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
        model = ModelConfig(**_parse_tuples(data.pop("model", {})))
        training = TrainingConfig(**data.pop("training", {}))
        preprocessing = PreprocessingConfig(**_parse_tuples(data.pop("preprocessing", {})))
        inference = InferenceConfig(**_parse_tuples(data.pop("inference", {})))

        return cls(
            model=model,
            training=training,
            preprocessing=preprocessing,
            inference=inference,
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
                "input_size": list(self.model.input_size),
                "in_channels": self.model.in_channels,
                "base_channels": self.model.base_channels,
                "dropout": self.model.dropout,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "epochs": self.training.epochs,
                "num_workers": self.training.num_workers,
                "optimizer": self.training.optimizer,
                "scheduler": self.training.scheduler,
                "scheduler_factor": self.training.scheduler_factor,
                "scheduler_patience": self.training.scheduler_patience,
                "scheduler_min_lr": self.training.scheduler_min_lr,
                "train_ratio": self.training.train_ratio,
                "val_ratio": self.training.val_ratio,
                "split_seed": self.training.split_seed,
            },
            "preprocessing": {
                "target_shape": list(self.preprocessing.target_shape),
                "window_level": self.preprocessing.window_level,
                "window_width": self.preprocessing.window_width,
            },
            "inference": {
                "target_shape": list(self.inference.target_shape),
            },
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _parse_tuples(d: Dict[str, Any]) -> Dict[str, Any]:
    """Convert lists to tuples for fields that expect tuples."""
    tuple_fields = {"input_size", "target_shape"}
    result = {}
    for k, v in d.items():
        if k in tuple_fields and isinstance(v, list):
            result[k] = tuple(v)
        else:
            result[k] = v
    return result
