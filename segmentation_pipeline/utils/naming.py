"""
Filename normalization and standardization utilities.

Provides tools for consistent naming across datasets.
"""

import logging
import re
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def extract_patient_id(
    filename: str,
    patterns: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Extract patient ID from filename using regex patterns.

    Args:
        filename: Filename to parse.
        patterns: List of regex patterns with a capture group for ID.
            Defaults to common patterns like H0001, patient_001, etc.

    Returns:
        Extracted patient ID or None if no match.
    """
    if patterns is None:
        patterns = [
            r"(H\d{4})",  # H0001
            r"patient[_-]?(\d+)",  # patient_001, patient001
            r"^(\d{4})",  # 0001 at start
            r"[_-](\d{3,4})[_-]",  # _001_ or -0001-
        ]

    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def normalize_filename(
    filename: str,
    prefix: str = "H",
    id_digits: int = 4,
    suffix: str = "",
    extension: str = ".nii",
) -> str:
    """
    Normalize filename to standard format.

    Args:
        filename: Original filename.
        prefix: Prefix for patient ID (e.g., "H").
        id_digits: Number of digits in ID (zero-padded).
        suffix: Suffix before extension (e.g., "_label").
        extension: File extension.

    Returns:
        Normalized filename.
    """
    patient_id = extract_patient_id(filename)

    if patient_id is None:
        logger.warning(f"Could not extract ID from: {filename}")
        return filename

    # Extract numeric part
    numeric = re.sub(r"\D", "", patient_id)

    if not numeric:
        logger.warning(f"No numeric ID found in: {filename}")
        return filename

    # Format new filename
    formatted_id = f"{prefix}{int(numeric):0{id_digits}d}"
    return f"{formatted_id}{suffix}{extension}"


def batch_rename(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    rename_func: Optional[Callable[[str], str]] = None,
    pattern: str = "*",
    dry_run: bool = True,
) -> List[Tuple[str, str]]:
    """
    Batch rename files in a directory.

    Args:
        input_dir: Source directory.
        output_dir: Destination directory. If None, renames in place.
        rename_func: Function to transform filenames. Defaults to normalize_filename.
        pattern: Glob pattern for files to rename.
        dry_run: If True, only print changes without executing.

    Returns:
        List of (old_name, new_name) tuples.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir

    if rename_func is None:
        rename_func = normalize_filename

    files = list(input_dir.glob(pattern))
    logger.info(f"Found {len(files)} files matching '{pattern}'")

    rename_map = []

    for filepath in files:
        old_name = filepath.name
        new_name = rename_func(old_name)

        if old_name != new_name:
            rename_map.append((old_name, new_name))

            if dry_run:
                logger.info(f"[DRY RUN] {old_name} -> {new_name}")
            else:
                new_path = output_dir / new_name
                if output_dir != input_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(filepath, new_path)
                else:
                    filepath.rename(new_path)
                logger.info(f"Renamed: {old_name} -> {new_name}")

    return rename_map


def create_naming_map(
    input_dir: Union[str, Path],
    pattern: str = "*.nii",
) -> Dict[str, str]:
    """
    Create a mapping of original filenames to normalized names.

    Args:
        input_dir: Directory containing files.
        pattern: Glob pattern.

    Returns:
        Dictionary mapping original names to normalized names.
    """
    input_dir = Path(input_dir)
    files = list(input_dir.glob(pattern))

    mapping = {}
    for filepath in files:
        old_name = filepath.name
        new_name = normalize_filename(old_name)
        mapping[old_name] = new_name

    return mapping


def validate_naming_consistency(
    image_dir: Union[str, Path],
    label_dir: Optional[Union[str, Path]] = None,
    image_pattern: str = "H*.nii",
    label_suffix: str = "_label",
) -> Dict[str, List[str]]:
    """
    Validate that images and labels have consistent naming.

    Args:
        image_dir: Directory with images.
        label_dir: Directory with labels (default: same as image_dir).
        image_pattern: Pattern for image files.
        label_suffix: Expected suffix for label files.

    Returns:
        Dictionary with 'matched', 'missing_label', 'orphan_label' lists.
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir) if label_dir else image_dir

    images = {f.stem: f for f in image_dir.glob(image_pattern) if label_suffix not in f.name}
    labels = {f.stem.replace(label_suffix, ""): f for f in label_dir.glob(f"*{label_suffix}.nii")}

    result = {
        "matched": [],
        "missing_label": [],
        "orphan_label": [],
    }

    for img_id, img_path in images.items():
        if img_id in labels:
            result["matched"].append(img_id)
        else:
            result["missing_label"].append(img_id)

    for lbl_id in labels:
        if lbl_id not in images:
            result["orphan_label"].append(lbl_id)

    logger.info(f"Matched: {len(result['matched'])}")
    logger.info(f"Missing labels: {len(result['missing_label'])}")
    logger.info(f"Orphan labels: {len(result['orphan_label'])}")

    return result
