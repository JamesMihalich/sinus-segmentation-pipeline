#!/usr/bin/env python3
"""
Train bounding box regression model.

Usage:
    python train.py --data-dir /path/to/npz --output-dir ./training_logs
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bounding_box_pipeline.configs import Config
from bounding_box_pipeline.data.datasets import LocalizationDataset, create_data_splits, get_dataset_files
from bounding_box_pipeline.models import BBoxRegressor3D, BBoxRegressorResidual, create_regressor
from bounding_box_pipeline.training import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train bounding box regression model")

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing NPZ dataset files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./training_logs"),
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["standard", "residual", "lite"],
        default="residual",
        help="Model variant to use (default: residual)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config if provided
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Override with command line args
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr

    # Create run directory
    run_name = datetime.now().strftime("run-%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Save config
    config.to_yaml(output_dir / "config.yaml")

    # Load dataset files
    files = get_dataset_files(args.data_dir)
    if not files:
        logger.error("No NPZ files found in data directory")
        return 1

    logger.info(f"Found {len(files)} dataset files")

    # Split data
    train_files, val_files = create_data_splits(
        files,
        train_ratio=config.training.train_ratio,
        val_ratio=config.training.val_ratio,
        seed=config.training.split_seed,
    )

    logger.info(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # Create datasets (augmentation enabled for training only)
    train_ds = LocalizationDataset(train_files, augment=True)
    val_ds = LocalizationDataset(val_files, augment=False)

    # Create loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model using factory function
    model = create_regressor(
        variant=args.model,
        input_size=config.model.input_size,
        base_channels=config.model.base_channels,
        dropout=config.model.dropout,
    )
    logger.info(f"Model variant: {args.model}")
    logger.info(f"Model parameters: {model.get_num_parameters():,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        output_dir=output_dir,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        scheduler_factor=config.training.scheduler_factor,
        scheduler_patience=config.training.scheduler_patience,
        scheduler_min_lr=config.training.scheduler_min_lr,
    )

    # Resume if specified
    start_epoch = 1
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1
        logger.info(f"Resumed from epoch {start_epoch - 1}")

    # Train
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=config.training.epochs,
        start_epoch=start_epoch,
        snapshot_every=config.training.snapshot_every,
    )

    logger.info(f"Training complete. Best IoU: {trainer.best_iou:.4f}")
    logger.info(f"Results saved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
