#!/usr/bin/env python3
"""
Inference script for 3D segmentation.

Usage:
    python predict.py --input /path/to/data --checkpoint best.pt --output ./predictions
    python predict.py --input single_file.npz --checkpoint best.pt
"""

import argparse
import logging
from pathlib import Path

import torch

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from segmentation_pipeline.models.unet import ResidualUnetSE3D
from segmentation_pipeline.inference.predictor import VolumePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on volumes")

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
        help="Output directory (default: alongside input)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=[224, 224, 256],
        help="Patch size for sliding window",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap between patches",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold",
    )
    parser.add_argument(
        "--skip-mode",
        choices=["concat", "additive"],
        default="additive",
        help="Model skip connection mode",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Disable post-processing",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate inputs
    if not args.input.exists():
        logger.error(f"Input not found: {args.input}")
        return

    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    logger.info("Loading model...")
    model = ResidualUnetSE3D(skip_mode=args.skip_mode)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    # Create predictor
    predictor = VolumePredictor(
        model=model,
        device=device,
        patch_size=tuple(args.patch_size),
        overlap=args.overlap,
        threshold=args.threshold,
        apply_postprocessing=not args.no_postprocess,
    )

    # Run inference
    if args.input.is_file():
        # Single file
        output_path = args.output if args.output else None
        result = predictor.predict_file(args.input, output_path=output_path)

        if result:
            logger.info(f"Saved prediction: {result}")
        else:
            logger.error("Prediction failed")
    else:
        # Directory
        output_dir = args.output if args.output else args.input / "predictions"
        results = predictor.predict_batch(
            args.input,
            output_dir=output_dir,
            show_progress=True,
        )

        logger.info(f"Completed {len(results)} predictions")
        logger.info(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
