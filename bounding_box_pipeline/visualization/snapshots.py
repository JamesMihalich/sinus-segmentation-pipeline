"""
Prediction visualization utilities.

Provides functions for visualizing bounding box predictions on images.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def visualize_bbox_slice(
    image: np.ndarray,
    pred_bbox: np.ndarray,
    gt_bbox: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (8, 8),
) -> None:
    """
    Visualize bounding box on a 2D slice.

    Args:
        image: 3D volume (D, H, W).
        pred_bbox: Predicted bbox [z1, y1, x1, z2, y2, x2] (normalized or absolute).
        gt_bbox: Ground truth bbox (optional).
        slice_idx: Z-slice to visualize. If None, uses middle of bbox.
        output_path: Path to save figure.
        figsize: Figure size.
    """
    # Denormalize if needed (assuming normalized if max <= 1)
    if pred_bbox.max() <= 1.0:
        shape = np.array(list(image.shape) * 2)
        pred_bbox = (pred_bbox * shape).astype(int)
        if gt_bbox is not None and gt_bbox.max() <= 1.0:
            gt_bbox = (gt_bbox * shape).astype(int)

    # Determine slice index
    if slice_idx is None:
        # Use middle of predicted bbox
        slice_idx = int((pred_bbox[0] + pred_bbox[3]) / 2)

    # Clamp to valid range
    slice_idx = max(0, min(slice_idx, image.shape[0] - 1))

    fig, ax = plt.subplots(figsize=figsize)

    # Show slice
    ax.imshow(image[slice_idx], cmap="gray")

    # Draw predicted bbox (only if slice is within bbox)
    pred_in_slice = pred_bbox[0] <= slice_idx <= pred_bbox[3]
    if pred_in_slice:
        rect = patches.Rectangle(
            (pred_bbox[2], pred_bbox[1]),  # (x, y)
            pred_bbox[5] - pred_bbox[2],   # width
            pred_bbox[4] - pred_bbox[1],   # height
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            label="Prediction",
        )
        ax.add_patch(rect)
    else:
        ax.text(
            0.02, 0.98,
            f"Pred bbox: z=[{pred_bbox[0]}, {pred_bbox[3]}]",
            transform=ax.transAxes,
            fontsize=10,
            color="red",
            verticalalignment="top",
        )

    # Draw ground truth bbox
    if gt_bbox is not None:
        gt_in_slice = gt_bbox[0] <= slice_idx <= gt_bbox[3]
        if gt_in_slice:
            rect = patches.Rectangle(
                (gt_bbox[2], gt_bbox[1]),
                gt_bbox[5] - gt_bbox[2],
                gt_bbox[4] - gt_bbox[1],
                linewidth=2,
                edgecolor="lime",
                facecolor="none",
                linestyle="--",
                label="Ground Truth",
            )
            ax.add_patch(rect)

    ax.set_title(f"Slice {slice_idx}")
    ax.legend(loc="upper right")
    ax.axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


@torch.no_grad()
def save_prediction_snapshot(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    save_dir: Path,
    epoch: int,
) -> Path:
    """
    Save visualization of model predictions on validation batch.

    Args:
        model: Trained model.
        val_loader: Validation data loader.
        device: Device for inference.
        save_dir: Directory to save snapshot.
        epoch: Current epoch number.

    Returns:
        Path to saved snapshot.
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get first batch
    images, labels = next(iter(val_loader))
    images = images.to(device)

    # Predict
    predictions = model(images)

    # Get first sample
    image = images[0, 0].cpu().numpy()  # (D, H, W)
    pred_bbox = predictions[0].cpu().numpy()  # (6,)
    gt_bbox = labels[0].numpy()  # (6,)

    # Denormalize for visualization
    shape = np.array(list(image.shape) * 2)
    pred_abs = (pred_bbox * shape).astype(int)
    gt_abs = (gt_bbox * shape).astype(int)

    # Find slice with both boxes
    pred_center_z = int((pred_abs[0] + pred_abs[3]) / 2)
    gt_center_z = int((gt_abs[0] + gt_abs[3]) / 2)
    slice_idx = (pred_center_z + gt_center_z) // 2

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(image[slice_idx], cmap="gray")

    # Prediction box
    if pred_abs[0] <= slice_idx <= pred_abs[3]:
        rect = patches.Rectangle(
            (pred_abs[2], pred_abs[1]),
            pred_abs[5] - pred_abs[2],
            pred_abs[4] - pred_abs[1],
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            label="Prediction",
        )
        ax.add_patch(rect)

    # Ground truth box
    if gt_abs[0] <= slice_idx <= gt_abs[3]:
        rect = patches.Rectangle(
            (gt_abs[2], gt_abs[1]),
            gt_abs[5] - gt_abs[2],
            gt_abs[4] - gt_abs[1],
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
            linestyle="--",
            label="Ground Truth",
        )
        ax.add_patch(rect)

    # Compute IoU
    from ..evaluation.metrics import compute_iou
    iou = compute_iou(pred_bbox, gt_bbox)

    ax.set_title(f"Epoch {epoch} | Slice {slice_idx} | IoU: {iou:.4f}")
    ax.legend(loc="upper right")
    ax.axis("off")

    # Save
    output_path = save_dir / f"epoch_{epoch:03d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved snapshot: {output_path.name}")
    return output_path


def visualize_dataset_sample(
    npz_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Visualize a sample from the dataset.

    Args:
        npz_path: Path to NPZ file.
        output_path: Path to save figure.
    """
    npz_path = Path(npz_path)

    with np.load(npz_path) as data:
        image = data["image"]
        label = data["label"]

    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Label (bbox): {label}")

    # Denormalize bbox
    shape = np.array(list(image.shape) * 2)
    bbox_abs = (label * shape).astype(int)

    # Find middle slice
    slice_idx = int((bbox_abs[0] + bbox_abs[3]) / 2)

    visualize_bbox_slice(
        image,
        label,
        slice_idx=slice_idx,
        output_path=output_path,
    )
