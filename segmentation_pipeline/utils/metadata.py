"""
Dataset metadata extraction and summarization.

Provides tools for documenting NIfTI and NPZ datasets.
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


def extract_nifti_metadata(filepath: Union[str, Path]) -> Dict:
    """
    Extract metadata from a NIfTI file.

    Args:
        filepath: Path to NIfTI file.

    Returns:
        Dictionary with file metadata.
    """
    filepath = Path(filepath)

    try:
        nii = nib.load(filepath)
        header = nii.header

        # Extract patient ID from filename (assumes H#### pattern)
        patient_id = filepath.stem
        if "_" in patient_id:
            patient_id = patient_id.split("_")[0]

        zooms = header.get_zooms()[:3]

        return {
            "filename": filepath.name,
            "patient_id": patient_id,
            "dim_x": int(header.get_data_shape()[0]),
            "dim_y": int(header.get_data_shape()[1]),
            "dim_z": int(header.get_data_shape()[2]),
            "spacing_x": float(zooms[0]),
            "spacing_y": float(zooms[1]),
            "spacing_z": float(zooms[2]),
            "dtype": str(header.get_data_dtype()),
            "voxel_units": header.get_xyzt_units()[0],
            "file_path": str(filepath.absolute()),
        }

    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return {
            "filename": filepath.name,
            "error": str(e),
        }


def extract_npz_metadata(filepath: Union[str, Path]) -> Dict:
    """
    Extract metadata from an NPZ file.

    Args:
        filepath: Path to NPZ file.

    Returns:
        Dictionary with file metadata.
    """
    filepath = Path(filepath)

    try:
        with np.load(filepath) as data:
            keys = list(data.files)

            metadata = {
                "filename": filepath.name,
                "patient_id": filepath.stem.split("_")[0],
                "keys": keys,
                "file_size_mb": filepath.stat().st_size / (1024 * 1024),
            }

            for key in keys:
                arr = data[key]
                metadata[f"{key}_shape"] = arr.shape
                metadata[f"{key}_dtype"] = str(arr.dtype)
                metadata[f"{key}_min"] = float(arr.min())
                metadata[f"{key}_max"] = float(arr.max())

            return metadata

    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return {
            "filename": filepath.name,
            "error": str(e),
        }


def generate_nifti_metadata_csv(
    input_dir: Union[str, Path],
    output_path: Union[str, Path],
    pattern: str = "*.nii",
) -> Path:
    """
    Generate CSV with metadata for all NIfTI files in directory.

    Args:
        input_dir: Directory containing NIfTI files.
        output_path: Output CSV path.
        pattern: Glob pattern for files.

    Returns:
        Path to output CSV.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    files = sorted(input_dir.glob(pattern))
    logger.info(f"Found {len(files)} files matching '{pattern}'")

    if not files:
        logger.warning("No files found")
        return output_path

    # Extract metadata
    metadata_list = []
    for filepath in files:
        metadata = extract_nifti_metadata(filepath)
        metadata_list.append(metadata)

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filename",
        "patient_id",
        "dim_x",
        "dim_y",
        "dim_z",
        "spacing_x",
        "spacing_y",
        "spacing_z",
        "dtype",
        "voxel_units",
        "file_path",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metadata_list)

    logger.info(f"Saved metadata to: {output_path}")
    return output_path


def generate_npz_metadata_csv(
    input_dir: Union[str, Path],
    output_path: Union[str, Path],
    pattern: str = "*.npz",
) -> Path:
    """
    Generate CSV with metadata for all NPZ files in directory.

    Args:
        input_dir: Directory containing NPZ files.
        output_path: Output CSV path.
        pattern: Glob pattern for files.

    Returns:
        Path to output CSV.
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    files = sorted(input_dir.glob(pattern))
    logger.info(f"Found {len(files)} files matching '{pattern}'")

    if not files:
        logger.warning("No files found")
        return output_path

    # Extract metadata
    metadata_list = []
    for filepath in files:
        metadata = extract_npz_metadata(filepath)
        metadata_list.append(metadata)

    # Get all possible columns
    all_keys = set()
    for m in metadata_list:
        all_keys.update(m.keys())

    fieldnames = sorted(all_keys)
    # Put common fields first
    priority = ["filename", "patient_id", "keys", "file_size_mb"]
    fieldnames = [f for f in priority if f in fieldnames] + [
        f for f in fieldnames if f not in priority
    ]

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metadata_list)

    logger.info(f"Saved metadata to: {output_path}")
    return output_path


def summarize_dataset(input_dir: Union[str, Path], file_type: str = "auto") -> Dict:
    """
    Generate summary statistics for a dataset.

    Args:
        input_dir: Directory containing data files.
        file_type: "nifti", "npz", or "auto" to detect.

    Returns:
        Dictionary with dataset statistics.
    """
    input_dir = Path(input_dir)

    # Auto-detect file type
    if file_type == "auto":
        nii_files = list(input_dir.glob("*.nii")) + list(input_dir.glob("*.nii.gz"))
        npz_files = list(input_dir.glob("*.npz"))

        if len(nii_files) > len(npz_files):
            file_type = "nifti"
            files = nii_files
        else:
            file_type = "npz"
            files = npz_files
    elif file_type == "nifti":
        files = list(input_dir.glob("*.nii")) + list(input_dir.glob("*.nii.gz"))
    else:
        files = list(input_dir.glob("*.npz"))

    if not files:
        return {"error": "No files found"}

    summary = {
        "directory": str(input_dir),
        "file_type": file_type,
        "num_files": len(files),
        "total_size_mb": sum(f.stat().st_size for f in files) / (1024 * 1024),
    }

    # Sample a few files for shape statistics
    sample_size = min(5, len(files))
    shapes = []

    for filepath in files[:sample_size]:
        if file_type == "nifti":
            meta = extract_nifti_metadata(filepath)
            if "dim_x" in meta:
                shapes.append((meta["dim_x"], meta["dim_y"], meta["dim_z"]))
        else:
            meta = extract_npz_metadata(filepath)
            for key in ["image_shape", "label_shape"]:
                if key in meta:
                    shapes.append(meta[key])
                    break

    if shapes:
        summary["sample_shapes"] = shapes
        summary["shape_consistent"] = len(set(shapes)) == 1

    return summary
