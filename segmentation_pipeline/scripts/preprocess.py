#!/usr/bin/env python3
"""
Data preprocessing script.

Converts NIfTI files to training-ready NPZ format with cropping and windowing.

Usage:
    python preprocess.py --input ./nifti_data --output ./npz_data
    python preprocess.py --input ./nifti_data --output ./npz_data --window-level -300 --window-width 1000
"""

import argparse
import logging
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from segmentation_pipeline.data.preprocessing.cropper import process_dataset_to_npz
from segmentation_pipeline.data.preprocessing.resampler import (
    fix_header_mismatches,
    resample_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess NIfTI dataset")

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory with NIfTI files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for NPZ files",
    )
    parser.add_argument(
        "--window-level",
        type=float,
        default=-300,
        help="CT window level (HU)",
    )
    parser.add_argument(
        "--window-width",
        type=float,
        default=1000,
        help="CT window width (HU)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding around bounding box",
    )
    parser.add_argument(
        "--fix-headers",
        action="store_true",
        help="Fix header mismatches before processing",
    )
    parser.add_argument(
        "--resample",
        type=float,
        nargs=3,
        help="Resample to isotropic spacing (x, y, z) before processing",
    )
    parser.add_argument(
        "--image-pattern",
        default="H*.nii",
        help="Glob pattern for image files",
    )
    parser.add_argument(
        "--label-suffix",
        default="_label",
        help="Suffix for label files",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input
    if not args.input.exists():
        logger.error(f"Input directory not found: {args.input}")
        return

    working_dir = args.input

    # Fix headers if requested
    if args.fix_headers:
        logger.info("Fixing header mismatches...")
        fixed = fix_header_mismatches(
            working_dir,
            image_pattern=args.image_pattern,
            label_suffix=args.label_suffix,
        )
        logger.info(f"Fixed {fixed} header mismatches")

    # Resample if requested
    if args.resample:
        resample_dir = args.output.parent / f"{args.output.name}_resampled"
        logger.info(f"Resampling to {args.resample}...")

        resample_dataset(
            working_dir,
            resample_dir,
            target_spacing=tuple(args.resample),
            image_pattern=args.image_pattern,
            label_suffix=args.label_suffix,
        )

        working_dir = resample_dir

    # Process to NPZ
    logger.info("Processing dataset to NPZ...")
    results = process_dataset_to_npz(
        working_dir,
        args.output,
        window_level=args.window_level,
        window_width=args.window_width,
        padding=args.padding,
        image_pattern=args.image_pattern,
        label_suffix=args.label_suffix,
    )

    logger.info(f"\nProcessing complete!")
    logger.info(f"Created {len(results)} NPZ files in: {args.output}")


if __name__ == "__main__":
    main()
