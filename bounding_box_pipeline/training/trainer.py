"""
Training pipeline for bounding box regression.

Provides training loop with logging, checkpointing, and visualization.
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..evaluation.metrics import compute_iou_batch
from ..visualization.plots import plot_training_curves
from ..visualization.snapshots import save_prediction_snapshot
from .losses import GIoULoss, SmoothL1IoULoss

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for bounding box regression models.

    Features:
    - Smooth L1 loss for robust training
    - IoU tracking for evaluation
    - Learning rate scheduling
    - Checkpoint saving
    - Visualization snapshots
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: Path,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 3,
        scheduler_min_lr: float = 1e-6,
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: Model to train.
            device: Device to train on.
            output_dir: Directory for checkpoints and logs.
            learning_rate: Initial learning rate.
            weight_decay: Weight decay for optimizer.
            scheduler_factor: LR reduction factor.
            scheduler_patience: Epochs before reducing LR.
            scheduler_min_lr: Minimum learning rate.
        """
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = self.output_dir / "snapshots"
        self.snapshot_dir.mkdir(exist_ok=True)

        # Loss function - GIoU provides gradients even when boxes don't overlap
        self.criterion = GIoULoss()

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",  # Minimize validation loss
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr,
        )

        # Tracking
        self.best_iou = 0.0
        self.metrics_path = self.output_dir / "training_log.csv"

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int,
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.
            total_epochs: Total number of epochs.

        Returns:
            Average training loss.
        """
        self.model.train()
        running_loss = 0.0

        iterator = tqdm(
            train_loader,
            desc=f"Train [{epoch}/{total_epochs}]",
            leave=False,
        )

        for images, labels in iterator:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(images)
            loss = self.criterion(predictions, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            iterator.set_postfix(loss=f"{loss.item():.4f}")

        return running_loss / len(train_loader.dataset)

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Tuple[float, float]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader.

        Returns:
            Tuple of (validation loss, validation IoU).
        """
        self.model.eval()
        loss_total = 0.0
        iou_total = 0.0
        n_samples = 0

        for images, labels in val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            predictions = self.model(images)
            loss = self.criterion(predictions, labels)

            iou = compute_iou_batch(predictions, labels)

            batch_size = images.size(0)
            loss_total += loss.item() * batch_size
            iou_total += iou * batch_size
            n_samples += batch_size

        return loss_total / n_samples, iou_total / n_samples

    def save_checkpoint(
        self,
        epoch: int,
        iou: float,
        filename: str = "checkpoint.pth",
    ) -> Path:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch.
            iou: Current validation IoU.
            filename: Checkpoint filename.

        Returns:
            Path to saved checkpoint.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "iou": iou,
        }

        path = self.output_dir / filename
        torch.save(checkpoint, path)
        return path

    def load_checkpoint(self, path: Path) -> int:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file.

        Returns:
            Epoch number from checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "iou" in checkpoint:
            self.best_iou = checkpoint["iou"]

        return checkpoint.get("epoch", 0)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        start_epoch: int = 1,
        snapshot_every: int = 5,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs to train.
            start_epoch: Starting epoch (for resuming).
            snapshot_every: Save visual snapshot every N epochs.

        Returns:
            Dictionary of training history.
        """
        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_iou": [],
            "lr": [],
        }

        # Initialize CSV log
        with open(self.metrics_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_iou", "lr"])

        epoch_iterator = tqdm(range(start_epoch, epochs + 1), desc="Epochs")

        for epoch in epoch_iterator:
            # Train
            train_loss = self.train_epoch(train_loader, epoch, epochs)

            # Validate
            val_loss, val_iou = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log history
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_iou"].append(val_iou)
            history["lr"].append(current_lr)

            # Save to CSV
            with open(self.metrics_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_iou:.4f}",
                    f"{current_lr:.8f}",
                ])

            # Update progress bar
            epoch_iterator.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_iou=f"{val_iou:.4f}",
                lr=f"{current_lr:.2e}",
            )

            # Save best model
            if val_iou > self.best_iou:
                self.best_iou = val_iou
                self.save_checkpoint(epoch, val_iou, "best_model.pth")
                logger.info(f"New best model at epoch {epoch} (IoU={val_iou:.4f})")

            # Save snapshots
            if epoch % snapshot_every == 0:
                save_prediction_snapshot(
                    self.model,
                    val_loader,
                    self.device,
                    self.snapshot_dir,
                    epoch,
                )

            # Update training curves plot
            plot_training_curves(history, self.output_dir / "training_curves.png")

        # Save final model
        self.save_checkpoint(epoch, val_iou, "last_model.pth")

        return history
