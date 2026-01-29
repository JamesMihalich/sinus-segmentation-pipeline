#!/usr/bin/env python
"""
Analyze class distribution in segmentation label volumes.

Computes the percentage of positive (airway) vs negative (background) voxels
in NPZ label volumes.

Usage:
    # Analyze a single file
    python analyze_class_distribution.py --input /path/to/file.npz

    # Analyze all NPZ files in a directory
    python analyze_class_distribution.py --input /path/to/npz_dir

    # Output results to CSV
    python analyze_class_distribution.py --input /path/to/npz_dir --output results.csv
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from segmentation_pipeline.utils.io import load_npz_label
from segmentation_pipeline.utils.volume_ops import compute_class_distribution


def analyze_file(path: Path) -> dict:
    """Analyze class distribution for a single NPZ file."""
    label = load_npz_label(path)
    stats = compute_class_distribution(label)
    stats["filename"] = path.name
    return stats


def analyze_directory(input_dir: Path) -> List[dict]:
    """Analyze class distribution for all NPZ files in a directory."""
    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        raise ValueError(f"No NPZ files found in {input_dir}")

    results = []
    for npz_path in npz_files:
        try:
            stats = analyze_file(npz_path)
            results.append(stats)
            print(
                f"{npz_path.name}: {stats['positive_percentage']:.4f}% positive, "
                f"{stats['negative_percentage']:.4f}% negative"
            )
        except Exception as e:
            print(f"Error processing {npz_path.name}: {e}")

    return results


def compute_summary(results: List[dict]) -> dict:
    """Compute summary statistics across all volumes."""
    if not results:
        return {}

    positive_pcts = [r["positive_percentage"] for r in results]
    negative_pcts = [r["negative_percentage"] for r in results]
    total_positive = sum(r["positive_voxels"] for r in results)
    total_negative = sum(r["negative_voxels"] for r in results)
    total_voxels = sum(r["total_voxels"] for r in results)

    return {
        "num_volumes": len(results),
        "mean_positive_pct": np.mean(positive_pcts),
        "std_positive_pct": np.std(positive_pcts),
        "min_positive_pct": np.min(positive_pcts),
        "max_positive_pct": np.max(positive_pcts),
        "median_positive_pct": np.median(positive_pcts),
        "overall_positive_pct": 100.0 * total_positive / total_voxels,
        "overall_class_ratio": total_positive / total_negative if total_negative > 0 else float("inf"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze class distribution in segmentation label volumes."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to NPZ file or directory containing NPZ files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save results as CSV.",
    )
    args = parser.parse_args()

    input_path = args.input

    if input_path.is_file():
        # Single file analysis
        stats = analyze_file(input_path)
        print(f"\nClass Distribution for {input_path.name}:")
        print(f"  Shape: {stats['shape']}")
        print(f"  Total voxels: {stats['total_voxels']:,}")
        print(f"  Positive (airway): {stats['positive_voxels']:,} ({stats['positive_percentage']:.4f}%)")
        print(f"  Negative (background): {stats['negative_voxels']:,} ({stats['negative_percentage']:.4f}%)")
        print(f"  Class ratio (pos/neg): {stats['class_ratio']:.6f}")

        if args.output:
            df = pd.DataFrame([stats])
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")

    elif input_path.is_dir():
        # Directory analysis
        print(f"Analyzing NPZ files in {input_path}...\n")
        results = analyze_directory(input_path)

        if results:
            summary = compute_summary(results)
            print(f"\n{'='*60}")
            print("Summary Statistics:")
            print(f"  Number of volumes: {summary['num_volumes']}")
            print(f"  Mean positive %: {summary['mean_positive_pct']:.4f}%")
            print(f"  Std positive %: {summary['std_positive_pct']:.4f}%")
            print(f"  Min positive %: {summary['min_positive_pct']:.4f}%")
            print(f"  Max positive %: {summary['max_positive_pct']:.4f}%")
            print(f"  Median positive %: {summary['median_positive_pct']:.4f}%")
            print(f"  Overall positive %: {summary['overall_positive_pct']:.4f}%")
            print(f"  Overall class ratio: {summary['overall_class_ratio']:.6f}")
            print(f"{'='*60}")

            if args.output:
                df = pd.DataFrame(results)
                df.to_csv(args.output, index=False)
                print(f"\nResults saved to {args.output}")
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()
