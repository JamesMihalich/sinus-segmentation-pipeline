"""
Volume visualization utilities.

Provides functions for visualizing NPZ volumes and segmentation overlays.
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def visualize_npz(
    filepath: Union[str, Path],
    slice_idx: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    image_key: str = "image",
    label_key: str = "label",
    figsize: tuple = (12, 4),
) -> Optional[plt.Figure]:
    """
    Visualize a single slice from an NPZ file.

    Args:
        filepath: Path to NPZ file.
        slice_idx: Slice index to visualize. If None, uses slice with max mask area.
        output_path: Optional path to save figure.
        show: Whether to display the plot.
        image_key: Key for image array in NPZ.
        label_key: Key for label array in NPZ.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    filepath = Path(filepath)

    with np.load(filepath) as data:
        if image_key not in data or label_key not in data:
            print(f"Error: Keys '{image_key}' and '{label_key}' not found in NPZ.")
            print(f"Available keys: {data.files}")
            return None

        img = data[image_key]
        lbl = data[label_key]

    # Find best slice if not specified
    if slice_idx is None:
        mask_counts = np.sum(lbl, axis=(1, 2))
        if mask_counts.max() == 0:
            slice_idx = img.shape[0] // 2
        else:
            slice_idx = int(np.argmax(mask_counts))

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # CT Image
    axes[0].imshow(img[slice_idx], cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("CT Image (Windowed)")
    axes[0].axis("off")

    # Mask
    axes[1].imshow(lbl[slice_idx], cmap="gray")
    axes[1].set_title("Label Mask")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(img[slice_idx], cmap="gray", vmin=0, vmax=255)
    axes[2].imshow(lbl[slice_idx], cmap="jet", alpha=0.4)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.suptitle(f"{filepath.name} - Slice {slice_idx}")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def visualize_prediction_comparison(
    image_path: Union[str, Path],
    prediction_path: Union[str, Path],
    ground_truth_path: Optional[Union[str, Path]] = None,
    slice_idx: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: tuple = (16, 4),
) -> Optional[plt.Figure]:
    """
    Compare prediction with ground truth.

    Args:
        image_path: Path to image NPZ.
        prediction_path: Path to prediction NPZ.
        ground_truth_path: Optional path to ground truth NPZ.
        slice_idx: Slice to visualize.
        output_path: Optional output path.
        show: Whether to display.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    # Load image
    with np.load(image_path) as data:
        for key in ["image", "image.npy"]:
            if key in data:
                img = data[key]
                break
        else:
            img = data[data.files[0]]

    # Load prediction
    with np.load(prediction_path) as data:
        for key in ["mask", "prediction"]:
            if key in data:
                pred = data[key]
                break
        else:
            pred = data[data.files[0]]

    # Load ground truth if provided
    gt = None
    if ground_truth_path:
        with np.load(ground_truth_path) as data:
            for key in ["label", "mask"]:
                if key in data:
                    gt = data[key]
                    break
            else:
                gt = data[data.files[0]]

    # Find best slice
    if slice_idx is None:
        if gt is not None:
            mask_counts = np.sum(gt, axis=(1, 2))
        else:
            mask_counts = np.sum(pred, axis=(1, 2))

        if mask_counts.max() == 0:
            slice_idx = img.shape[0] // 2
        else:
            slice_idx = int(np.argmax(mask_counts))

    # Create figure
    n_cols = 4 if gt is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)

    # Image
    axes[0].imshow(img[slice_idx], cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Image")
    axes[0].axis("off")

    # Prediction overlay
    axes[1].imshow(img[slice_idx], cmap="gray", vmin=0, vmax=255)
    axes[1].imshow(pred[slice_idx], cmap="Reds", alpha=0.5)
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    if gt is not None:
        # Ground truth overlay
        axes[2].imshow(img[slice_idx], cmap="gray", vmin=0, vmax=255)
        axes[2].imshow(gt[slice_idx], cmap="Greens", alpha=0.5)
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

        # Comparison (overlap)
        axes[3].imshow(img[slice_idx], cmap="gray", vmin=0, vmax=255)
        # True positive: both pred and gt
        tp = np.logical_and(pred[slice_idx], gt[slice_idx])
        # False positive: pred but not gt
        fp = np.logical_and(pred[slice_idx], ~gt[slice_idx].astype(bool))
        # False negative: gt but not pred
        fn = np.logical_and(~pred[slice_idx].astype(bool), gt[slice_idx])

        overlay = np.zeros((*img[slice_idx].shape, 3))
        overlay[tp] = [0, 1, 0]  # Green: true positive
        overlay[fp] = [1, 0, 0]  # Red: false positive
        overlay[fn] = [0, 0, 1]  # Blue: false negative

        axes[3].imshow(overlay, alpha=0.5)
        axes[3].set_title("Comparison (G=TP, R=FP, B=FN)")
        axes[3].axis("off")
    else:
        # Just prediction mask
        axes[2].imshow(pred[slice_idx], cmap="gray")
        axes[2].set_title("Prediction Mask")
        axes[2].axis("off")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def analyze_npz_file(
    filepath: Union[str, Path],
    save_plot: bool = True,
) -> dict:
    """
    Analyze NPZ file and print statistics.

    Args:
        filepath: Path to NPZ file.
        save_plot: Whether to save visualization.

    Returns:
        Dictionary with file statistics.
    """
    filepath = Path(filepath)
    print(f"\n--- Analyzing: {filepath.name} ---")

    try:
        with np.load(filepath) as data:
            img = data.get("image", data[data.files[0]])
            lbl = data.get("label", None)

            stats = {
                "filename": filepath.name,
                "shape": img.shape,
                "image_dtype": str(img.dtype),
                "image_min": float(img.min()),
                "image_max": float(img.max()),
            }

            print(f"Shape: {img.shape}")
            print(f"Data Type: {img.dtype}")
            print(f"Image Range: min={img.min()}, max={img.max()}")

            if lbl is not None:
                unique_labels = np.unique(lbl)
                stats["label_dtype"] = str(lbl.dtype)
                stats["label_values"] = unique_labels.tolist()
                print(f"Label Values: {unique_labels}")

            if save_plot and lbl is not None:
                visualize_npz(filepath, show=False, output_path=filepath.with_suffix(".png"))

            return stats

    except Exception as e:
        print(f"Failed to read file: {e}")
        return {"error": str(e)}


def inspect_directory(
    dir_path: Union[str, Path],
    num_samples: int = 3,
    pattern: str = "*.npz",
) -> List[dict]:
    """
    Inspect random samples from a directory.

    Args:
        dir_path: Directory to inspect.
        num_samples: Number of samples to analyze.
        pattern: Glob pattern for files.

    Returns:
        List of statistics dictionaries.
    """
    p = Path(dir_path)
    files = list(p.glob(pattern))

    if not files:
        print(f"No files matching '{pattern}' found in {dir_path}")
        return []

    print(f"Found {len(files)} files. Inspecting {min(len(files), num_samples)} random samples...")

    samples = random.sample(files, min(len(files), num_samples))
    results = []

    for f in samples:
        stats = analyze_npz_file(f)
        results.append(stats)

    return results
