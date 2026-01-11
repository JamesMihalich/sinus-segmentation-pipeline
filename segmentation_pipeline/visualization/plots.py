"""
Training visualization utilities.

Provides functions for plotting training metrics and learning curves.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(
    metrics_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: tuple = (12, 8),
    style: str = "bmh",
) -> plt.Figure:
    """
    Plot training curves from metrics CSV file.

    Args:
        metrics_path: Path to metrics.csv from training.
        output_path: Optional path to save figure.
        show: Whether to display the plot.
        figsize: Figure size (width, height).
        style: Matplotlib style.

    Returns:
        Matplotlib Figure object.
    """
    metrics_path = Path(metrics_path)
    df = pd.read_csv(metrics_path)

    epoch = df["epoch"].tolist()
    train_loss = df["train_loss"].tolist()
    val_loss = df["val_loss"].tolist()
    val_dice = df["val_dice"].tolist()
    lr = df["lr"].tolist()

    plt.style.use(style)

    fig, axes = plt.subplots(3, 1, figsize=figsize, constrained_layout=True)
    ax_loss, ax_dice, ax_lr = axes

    # Color scheme
    train_kwargs = dict(color="#1f77b4", linewidth=2)
    val_kwargs = dict(color="#2ca02c", linewidth=2, alpha=0.9)

    # Loss plot
    ax_loss.plot(epoch, train_loss, label="Training Loss", **train_kwargs)
    ax_loss.plot(epoch, val_loss, label="Validation Loss", **val_kwargs)
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="best")
    ax_loss.grid(True, alpha=0.3)

    # Dice plot
    ax_dice.plot(
        epoch, val_dice, label="Validation Dice", color="#ff7f0e", linewidth=2
    )
    ax_dice.set_title("Validation Dice")
    ax_dice.set_xlabel("Epoch")
    ax_dice.set_ylabel("Dice")
    ax_dice.set_ylim(0, 1)
    ax_dice.legend(loc="best")
    ax_dice.grid(True, alpha=0.3)

    # Learning rate plot (log scale)
    ax_lr.plot(epoch, lr, label="Learning Rate", color="#9467bd", linewidth=2)
    ax_lr.set_title("Learning Rate")
    ax_lr.set_xlabel("Epoch")
    ax_lr.set_ylabel("LR")
    ax_lr.set_yscale("log")
    ax_lr.legend(loc="best")
    ax_lr.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_loss_comparison(
    metrics_paths: Dict[str, Path],
    output_path: Optional[Path] = None,
    show: bool = True,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Compare loss curves from multiple training runs.

    Args:
        metrics_paths: Dictionary mapping run names to metrics CSV paths.
        output_path: Optional path to save figure.
        show: Whether to display the plot.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for name, path in metrics_paths.items():
        df = pd.read_csv(path)
        ax.plot(df["epoch"], df["val_loss"], label=f"{name} (val)", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Training Comparison")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def print_training_summary(metrics_path: Union[str, Path]) -> Dict[str, float]:
    """
    Print summary statistics from training metrics.

    Args:
        metrics_path: Path to metrics.csv.

    Returns:
        Dictionary with best metrics.
    """
    df = pd.read_csv(metrics_path)

    best_val_loss = df["val_loss"].min()
    best_val_loss_epoch = df.loc[df["val_loss"].idxmin(), "epoch"]
    best_val_dice = df["val_dice"].max()
    best_val_dice_epoch = df.loc[df["val_dice"].idxmax(), "epoch"]
    final_lr = df["lr"].iloc[-1]

    print("\nTraining Summary")
    print("=" * 40)
    print(f"Best Val Loss: {best_val_loss:.4f} (epoch {best_val_loss_epoch})")
    print(f"Best Val Dice: {best_val_dice:.4f} (epoch {best_val_dice_epoch})")
    print(f"Final LR: {final_lr:.2e}")
    print(f"Total Epochs: {len(df)}")

    return {
        "best_val_loss": best_val_loss,
        "best_val_loss_epoch": best_val_loss_epoch,
        "best_val_dice": best_val_dice,
        "best_val_dice_epoch": best_val_dice_epoch,
    }
