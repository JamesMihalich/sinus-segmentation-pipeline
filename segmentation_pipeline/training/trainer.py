"""
Training pipeline for 3D medical image segmentation.

Provides training loop with mixed precision, gradient clipping, and logging.
"""

import csv
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .losses import BCEDiceLoss, dice_coefficient

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for 3D segmentation models.

    Features:
    - Mixed precision training (AMP)
    - Gradient clipping
    - Learning rate scheduling
    - Checkpoint saving
    - Metrics logging to CSV
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: Path,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        gradient_clip_norm: float = 1.0,
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
            weight_decay: Weight decay for AdamW.
            bce_weight: Weight for BCE in combined loss.
            dice_weight: Weight for Dice in combined loss.
            gradient_clip_norm: Max norm for gradient clipping (0 to disable).
            scheduler_factor: LR reduction factor.
            scheduler_patience: Epochs to wait before reducing LR.
            scheduler_min_lr: Minimum learning rate.
        """
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.gradient_clip_norm = gradient_clip_norm

        # Create directories
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.csv"

        # Loss function
        self.criterion = BCEDiceLoss(bce_weight=bce_weight, dice_weight=dice_weight)

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr,
            threshold=0.001,
        )

        # Mixed precision
        self.scaler = torch.GradScaler(enabled=torch.cuda.is_available())

        # Tracking
        self.best_val_loss = math.inf
        self.current_epoch = 0

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        total_epochs: int,
        show_progress: bool = True,
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.
            total_epochs: Total number of epochs.
            show_progress: Whether to show progress bar.

        Returns:
            Average training loss.
        """
        self.model.train()
        running_loss = 0.0

        iterator = train_loader
        if show_progress:
            iterator = tqdm(
                train_loader,
                desc=f"Train [{epoch}/{total_epochs}]",
                leave=False,
            )

        for batch_idx, (volumes, masks) in enumerate(iterator):
            volumes = volumes.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Forward pass with mixed precision
            with torch.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available(),
            ):
                logits = self.model(volumes)
                loss = self.criterion(logits, masks)

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.gradient_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * volumes.size(0)

            if show_progress:
                avg_loss = running_loss / ((batch_idx + 1) * volumes.size(0))
                iterator.set_postfix(loss=f"{avg_loss:.4f}")

        return running_loss / len(train_loader.dataset)

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        show_progress: bool = True,
    ) -> Tuple[float, float]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader.
            show_progress: Whether to show progress bar.

        Returns:
            Tuple of (validation loss, validation dice).
        """
        self.model.eval()
        loss_total = 0.0
        dice_total = 0.0
        n_samples = 0

        iterator = val_loader
        if show_progress:
            iterator = tqdm(val_loader, desc="Valid", leave=False)

        for volumes, masks in iterator:
            volumes = volumes.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            logits = self.model(volumes)
            loss = self.criterion(logits, masks)

            probs = torch.sigmoid(logits)
            dice = dice_coefficient(probs, masks)

            batch_size = volumes.size(0)
            loss_total += loss.item() * batch_size
            dice_total += dice.item() * batch_size
            n_samples += batch_size

            if show_progress:
                iterator.set_postfix(
                    loss=f"{loss_total / n_samples:.4f}",
                    dice=f"{dice_total / n_samples:.4f}",
                )

        return loss_total / n_samples, dice_total / n_samples

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        filename: str = "checkpoint.pt",
    ) -> Path:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch.
            val_loss: Current validation loss.
            filename: Checkpoint filename.

        Returns:
            Path to saved checkpoint.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "val_loss": val_loss,
        }

        path = self.checkpoint_dir / filename
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

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        if "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        if "val_loss" in checkpoint:
            self.best_val_loss = checkpoint["val_loss"]

        return checkpoint.get("epoch", 0)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        start_epoch: int = 1,
        show_progress: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs to train.
            start_epoch: Starting epoch (for resuming).
            show_progress: Whether to show progress bars.

        Returns:
            Dictionary of training history.
        """
        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "lr": [],
        }

        # Initialize CSV
        with open(self.metrics_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "lr"])

        epoch_iterator = range(start_epoch, epochs + 1)
        if show_progress:
            epoch_iterator = tqdm(epoch_iterator, desc="Epochs")

        for epoch in epoch_iterator:
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(
                train_loader, epoch, epochs, show_progress
            )

            # Validate
            val_loss, val_dice = self.validate(val_loader, show_progress)

            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Log
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_dice"].append(val_dice)
            history["lr"].append(current_lr)

            # Save metrics
            with open(self.metrics_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_dice:.6f}",
                    f"{current_lr:.8f}",
                ])

            # Update progress
            if show_progress:
                epoch_iterator.set_postfix(
                    train_loss=f"{train_loss:.4f}",
                    val_loss=f"{val_loss:.4f}",
                    val_dice=f"{val_dice:.4f}",
                    lr=f"{current_lr:.2e}",
                )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, "best.pt")
                logger.info(f"New best model at epoch {epoch} (val_loss={val_loss:.4f})")

        # Save final model
        self.save_checkpoint(epoch, val_loss, "last.pt")
        torch.save(self.model.state_dict(), self.output_dir / "last_weights.pt")

        return history
