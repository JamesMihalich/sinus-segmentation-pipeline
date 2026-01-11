#!/usr/bin/env python3
"""
Training script for 3D medical image segmentation.

Usage:
    python train.py --data-dir /path/to/npz --output-dir ./results
    python train.py --config config.yaml
"""

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from segmentation_pipeline.configs.config import Config
from segmentation_pipeline.data.datasets.volume_dataset import (
    VolumeDataset,
    create_data_splits,
)
from segmentation_pipeline.models.unet import ResidualUnetSE3D
from segmentation_pipeline.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation model")

    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing NPZ training files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
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
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--skip-mode",
        choices=["concat", "additive"],
        default="concat",
        help="Skip connection mode",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from checkpoint",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Override with CLI args
    if args.data_dir:
        config.data_root = args.data_dir
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr

    # Validate data directory
    if not config.data_root or not config.data_root.exists():
        logger.error(f"Data directory not found: {config.data_root}")
        return

    # Create output directory
    run_name = datetime.now().strftime("run-%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Save config
    config.to_yaml(output_dir / "config.yaml")

    # Find data files
    data_files = sorted(list(config.data_root.glob("*.npz")))
    logger.info(f"Found {len(data_files)} training files")

    if not data_files:
        logger.error("No .npz files found")
        return

    # Split data
    train_files, val_files, test_files = create_data_splits(
        data_files,
        train_ratio=config.training.train_ratio,
        val_ratio=config.training.val_ratio,
        test_ratio=config.training.test_ratio,
        seed=config.training.split_seed,
    )

    logger.info(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    # Save split manifest
    with open(output_dir / "data_split.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "split"])
        for p in train_files:
            writer.writerow([p.name, "train"])
        for p in val_files:
            writer.writerow([p.name, "validation"])
        for p in test_files:
            writer.writerow([p.name, "test"])

    # Create datasets
    train_ds = VolumeDataset(
        train_files,
        patch_size=config.training.patch_size,
        augment=True,
        aug_params=config.augmentation.to_dict(),
    )
    val_ds = VolumeDataset(
        val_files,
        patch_size=config.training.patch_size,
        augment=False,
    )

    # Create dataloaders
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

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = ResidualUnetSE3D(
        skip_mode=args.skip_mode,
        se_reduction_ratio=config.model.se_reduction_ratio,
    )
    logger.info(f"Model parameters: {model.get_num_parameters():,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        output_dir=output_dir,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        bce_weight=config.training.bce_weight,
        dice_weight=config.training.dice_weight,
        gradient_clip_norm=config.training.gradient_clip_norm,
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
    )

    logger.info("Training complete!")
    logger.info(f"Best val loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
