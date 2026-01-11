"""
Training visualization utilities.

Provides functions for plotting training curves and metrics.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_training_curves(
    history: Union[Dict[str, List[float]], Path],
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 4),
) -> None:
    """
    Plot training and validation curves.

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'val_iou' keys,
                or path to CSV file with these columns.
        output_path: Path to save figure. If None, displays interactively.
        figsize: Figure size (width, height).
    """
    # Load from CSV if path provided
    if isinstance(history, (str, Path)):
        import pandas as pd
        df = pd.read_csv(history)
        history = {
            "epoch": df["epoch"].tolist(),
            "train_loss": df["train_loss"].tolist(),
            "val_loss": df["val_loss"].tolist(),
            "val_iou": df["val_iou"].tolist(),
        }

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Loss plot
    ax1 = axes[0]
    epochs = history.get("epoch", list(range(1, len(history["train_loss"]) + 1)))
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="blue")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # IoU plot
    ax2 = axes[1]
    ax2.plot(epochs, history["val_iou"], label="Val IoU", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    ax2.set_title("Validation IoU")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mark best IoU
    if history["val_iou"]:
        best_idx = np.argmax(history["val_iou"])
        best_epoch = epochs[best_idx]
        best_iou = history["val_iou"][best_idx]
        ax2.scatter([best_epoch], [best_iou], color="red", s=100, zorder=5)
        ax2.annotate(
            f"Best: {best_iou:.4f}",
            (best_epoch, best_iou),
            textcoords="offset points",
            xytext=(10, 10),
            fontsize=9,
        )

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_iou_distribution(
    ious: List[float],
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (8, 5),
) -> None:
    """
    Plot distribution of IoU values.

    Args:
        ious: List of IoU values.
        output_path: Path to save figure.
        figsize: Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(ious, bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(ious), color="red", linestyle="--", label=f"Mean: {np.mean(ious):.3f}")
    ax.axvline(np.median(ious), color="green", linestyle="--", label=f"Median: {np.median(ious):.3f}")

    ax.set_xlabel("IoU")
    ax.set_ylabel("Count")
    ax.set_title("IoU Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_learning_rate(
    history: Union[Dict[str, List[float]], Path],
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (8, 4),
) -> None:
    """
    Plot learning rate schedule.

    Args:
        history: Dictionary with 'lr' key or path to CSV.
        output_path: Path to save figure.
        figsize: Figure size.
    """
    if isinstance(history, (str, Path)):
        import pandas as pd
        df = pd.read_csv(history)
        history = {
            "epoch": df["epoch"].tolist(),
            "lr": df["lr"].tolist(),
        }

    fig, ax = plt.subplots(figsize=figsize)

    epochs = history.get("epoch", list(range(1, len(history["lr"]) + 1)))
    ax.plot(epochs, history["lr"], color="purple")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
