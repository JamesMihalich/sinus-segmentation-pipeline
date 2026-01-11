#!/usr/bin/env python3
"""
Evaluate bounding box predictions against ground truth.

Usage:
    python evaluate.py --predictions ./preds --ground-truth ./data
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bounding_box_pipeline.evaluation import BBoxEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate bounding box predictions")

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
        help="Directory containing ground truth files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./evaluation_results.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--pattern",
        default="*_prediction.npz",
        help="Glob pattern for prediction files",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.predictions.exists():
        logger.error(f"Predictions directory not found: {args.predictions}")
        return 1

    if not args.ground_truth.exists():
        logger.error(f"Ground truth directory not found: {args.ground_truth}")
        return 1

    logger.info(f"Predictions: {args.predictions}")
    logger.info(f"Ground truth: {args.ground_truth}")

    # Create evaluator
    evaluator = BBoxEvaluator(
        prediction_dir=args.predictions,
        ground_truth_dir=args.ground_truth,
    )

    # Evaluate
    results = evaluator.evaluate_all(pattern=args.pattern)

    if len(results) == 0:
        logger.error("No valid evaluations completed")
        return 1

    # Compute summary
    summary = evaluator.compute_summary(results)

    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"\nSamples evaluated: {len(results)}")
    print(f"\nMetrics:")
    print(summary.to_string())

    # Save results
    evaluator.save_results(results, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
