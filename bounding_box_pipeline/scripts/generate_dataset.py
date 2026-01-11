#!/usr/bin/env python3
"""
Generate localization dataset from NIfTI files.

Usage:
    python generate_dataset.py --input /path/to/nifti --output /path/to/npz
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bounding_box_pipeline.data.preprocessing import create_localization_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate localization dataset from NIfTI files"
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing NIfTI image-label pairs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for NPZ files",
    )
    parser.add_argument(
        "--target-shape",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="Target volume shape (default: 128 128 128)",
    )
    parser.add_argument(
        "--window-level",
        type=float,
        default=600.0,
        help="CT window level (default: 600)",
    )
    parser.add_argument(
        "--window-width",
        type=float,
        default=1250.0,
        help="CT window width (default: 1250)",
    )
    parser.add_argument(
        "--label-suffix",
        default="_label",
        help="Suffix for label files (default: _label)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.exists():
        logger.error(f"Input directory not found: {args.input}")
        return 1

    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Target shape: {tuple(args.target_shape)}")
    logger.info(f"Window: level={args.window_level}, width={args.window_width}")

    results = create_localization_dataset(
        input_dir=args.input,
        output_dir=args.output,
        target_shape=tuple(args.target_shape),
        window_level=args.window_level,
        window_width=args.window_width,
        label_suffix=args.label_suffix,
    )

    logger.info(f"Created {len(results)} dataset samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
