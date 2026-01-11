#!/usr/bin/env python3
"""
Evaluation script for segmentation predictions.

Usage:
    python evaluate.py --predictions ./predictions --ground-truth ./data
    python evaluate.py --predictions ./predictions --ground-truth ./data --output metrics.csv
"""

import argparse
import logging
from pathlib import Path

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from segmentation_pipeline.evaluation.evaluator import SegmentationEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions")

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Directory containing prediction files",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        nargs="+",
        help="Directory or directories containing ground truth files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (default: predictions/evaluation_metrics.csv)",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Voxel spacing for distance metrics (x, y, z)",
    )
    parser.add_argument(
        "--pattern",
        default="*_prediction.npz",
        help="Glob pattern for prediction files",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate inputs
    if not args.predictions.exists():
        logger.error(f"Predictions directory not found: {args.predictions}")
        return

    for gt_dir in args.ground_truth:
        if not gt_dir.exists():
            logger.error(f"Ground truth directory not found: {gt_dir}")
            return

    # Create evaluator
    evaluator = SegmentationEvaluator(
        prediction_dir=args.predictions,
        ground_truth_dirs=args.ground_truth,
        spacing=tuple(args.spacing),
    )

    # Run evaluation
    logger.info("Running evaluation...")
    results = evaluator.evaluate_all(
        pattern=args.pattern,
        show_progress=True,
    )

    if len(results) == 0:
        logger.warning("No valid evaluations completed")
        return

    # Print summary
    summary = evaluator.compute_summary(results)
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"\nEvaluated: {len(results)} samples")
    print("\nMetric Statistics:")
    print(summary.to_string())

    # Save results
    output_path = args.output or (args.predictions / "evaluation_metrics.csv")
    evaluator.save_results(results, output_path, include_summary=False)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
