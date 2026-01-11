#!/usr/bin/env python3
"""
Amira to NIfTI converter script.

Usage:
    python convert_amira.py --input ./amira_files --output ./nifti_files
    python convert_amira.py --input single_file.am --type label
"""

import argparse
import logging
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from segmentation_pipeline.data.converters.amira import AmiraConverter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Amira files to NIfTI")

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input .am file or directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (default: alongside input)",
    )
    parser.add_argument(
        "--type",
        choices=["label", "volume", "auto"],
        default="auto",
        help="File type: label (RLE masks), volume (raw), or auto-detect",
    )
    parser.add_argument(
        "--pattern",
        default="*.am",
        help="Glob pattern for directory input",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input
    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        return

    # Create converter
    converter = AmiraConverter(output_dir=args.output)

    if args.input.is_file():
        # Single file
        result = converter.convert(args.input, file_type=args.type)

        if result:
            logger.info(f"Converted: {args.input.name} -> {result.name}")
        else:
            logger.error("Conversion failed")
    else:
        # Directory
        results = converter.convert_directory(
            args.input,
            pattern=args.pattern,
            file_type=args.type,
        )

        logger.info(f"Converted {len(results)} files")


if __name__ == "__main__":
    main()
