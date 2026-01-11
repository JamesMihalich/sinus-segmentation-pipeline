#!/usr/bin/env python3
"""
Run bounding box prediction on new data.

Usage:
    python predict.py --input /path/to/data --checkpoint best_model.pth
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bounding_box_pipeline.inference import BBoxPredictor
from bounding_box_pipeline.models import BBoxRegressor3D

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run bounding box prediction")

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input NPZ file or directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="Model input size (default: 128 128 128)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run inference on",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return 1

    # Setup device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = "cpu"

    logger.info(f"Loading model from: {args.checkpoint}")

    # Create model
    model = BBoxRegressor3D(input_size=tuple(args.input_size))

    # Create predictor
    predictor = BBoxPredictor.from_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
        target_shape=tuple(args.input_size),
    )

    # Run prediction
    if args.input.is_file():
        # Single file
        result = predictor.predict_file(args.input, args.output)
        print(json.dumps(result, indent=2))
    else:
        # Directory
        if not args.output:
            args.output = args.input / "predictions"

        results = predictor.predict_batch(
            input_dir=args.input,
            output_dir=args.output,
        )

        logger.info(f"Processed {len(results)} files")
        logger.info(f"Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
