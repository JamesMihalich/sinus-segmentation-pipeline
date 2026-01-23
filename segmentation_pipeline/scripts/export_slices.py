#!/usr/bin/env python3
"""
Export NPZ volumes as 2D PNG slices.

Converts each slice to a 512x512 grayscale PNG with black padding to center the image.

Usage:
    python export_slices.py --input ./npz_data --output ./png_slices
    python export_slices.py --input ./npz_data --output ./png_slices --axis 2 --size 512
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def center_pad_slice(
    slice_2d: np.ndarray,
    target_size: int = 512,
) -> np.ndarray:
    """
    Center a 2D slice in a square image with black padding.

    If the slice is larger than target_size in either dimension,
    it will be resized to fit while preserving aspect ratio.

    Args:
        slice_2d: 2D numpy array (H, W).
        target_size: Output size (square).

    Returns:
        Centered and padded array of shape (target_size, target_size).
    """
    h, w = slice_2d.shape

    # Resize if needed to fit within target_size
    if h > target_size or w > target_size:
        scale = min(target_size / h, target_size / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        img = Image.fromarray(slice_2d)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        slice_2d = np.array(img)
        h, w = slice_2d.shape

    # Create black canvas
    canvas = np.zeros((target_size, target_size), dtype=slice_2d.dtype)

    # Calculate padding to center
    pad_top = (target_size - h) // 2
    pad_left = (target_size - w) // 2

    # Place slice in center
    canvas[pad_top:pad_top + h, pad_left:pad_left + w] = slice_2d

    return canvas


def export_volume_slices(
    npz_path: Path,
    output_dir: Path,
    axis: int = 2,
    target_size: int = 512,
    key: str = "image",
) -> int:
    """
    Export all slices from an NPZ volume as PNG images.

    Args:
        npz_path: Path to NPZ file.
        output_dir: Output directory for PNG files.
        axis: Axis to slice along (0=sagittal, 1=coronal, 2=axial).
        target_size: Output image size (square).
        key: Key to load from NPZ ("image" or "label").

    Returns:
        Number of slices exported.
    """
    # Extract patient ID from filename
    patient_id = npz_path.stem

    # Load volume
    try:
        data = np.load(npz_path)
    except Exception as e:
        logger.error(f"Failed to load {npz_path}: {e}")
        return 0

    if key not in data:
        logger.error(f"Key '{key}' not found in {npz_path}. Available: {data.files}")
        return 0

    volume = data[key]

    # Get number of slices along the specified axis
    num_slices = volume.shape[axis]

    # Calculate zero-padding width for slice numbers
    pad_width = len(str(num_slices))

    logger.info(f"Exporting {patient_id}: {volume.shape}, {num_slices} slices along axis {axis}")

    count = 0
    for i in range(num_slices):
        # Extract slice along specified axis
        if axis == 0:
            slice_2d = volume[i, :, :]
        elif axis == 1:
            slice_2d = volume[:, i, :]
        else:  # axis == 2
            slice_2d = volume[:, :, i]

        # Center and pad
        padded = center_pad_slice(slice_2d, target_size=target_size)

        # Ensure uint8
        if padded.dtype != np.uint8:
            padded = padded.astype(np.uint8)

        # Create filename: patientID_slice_XXX.png
        slice_str = str(i).zfill(pad_width)
        filename = f"{patient_id}_slice_{slice_str}.png"
        output_path = output_dir / filename

        # Save as grayscale PNG
        img = Image.fromarray(padded, mode='L')
        img.save(output_path)
        count += 1

    return count


def export_dataset_slices(
    input_dir: Path,
    output_dir: Path,
    axis: int = 2,
    target_size: int = 512,
    key: str = "image",
) -> Tuple[int, int]:
    """
    Export all NPZ files in a directory to PNG slices.

    Args:
        input_dir: Directory containing NPZ files.
        output_dir: Output directory for PNG files.
        axis: Axis to slice along.
        target_size: Output image size.
        key: Key to load from NPZ.

    Returns:
        Tuple of (number of volumes processed, total slices exported).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob("*.npz"))
    logger.info(f"Found {len(npz_files)} NPZ files")

    total_volumes = 0
    total_slices = 0

    for npz_path in npz_files:
        slices = export_volume_slices(
            npz_path,
            output_dir,
            axis=axis,
            target_size=target_size,
            key=key,
        )
        if slices > 0:
            total_volumes += 1
            total_slices += slices

    return total_volumes, total_slices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export NPZ volumes as 2D PNG slices"
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing NPZ files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for PNG slices",
    )
    parser.add_argument(
        "--axis",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Axis to slice along: 0=sagittal, 1=coronal, 2=axial (default: 2)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Output image size in pixels (default: 512)",
    )
    parser.add_argument(
        "--key",
        default="image",
        choices=["image", "label"],
        help="Which array to export from NPZ (default: image)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.exists():
        logger.error(f"Input directory not found: {args.input}")
        return

    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Axis: {args.axis}, Size: {args.size}x{args.size}")

    volumes, slices = export_dataset_slices(
        args.input,
        args.output,
        axis=args.axis,
        target_size=args.size,
        key=args.key,
    )

    logger.info(f"Exported {slices} slices from {volumes} volumes")


if __name__ == "__main__":
    main()
