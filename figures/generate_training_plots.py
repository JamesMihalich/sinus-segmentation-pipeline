#!/usr/bin/env python3
"""
Generate publication-ready training plots for regression and segmentation models.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from pathlib import Path

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
})

# Colorblind-friendly palette (same as TikZ figures)
COLORS = {
    'train': '#377eb8',      # Blue
    'val': '#4daf4a',        # Green
    'metric': '#ff7f00',     # Orange
    'lr': '#984ea3',         # Purple
}


def plot_regression_training(csv_path: Path, output_dir: Path):
    """Generate training plots for bounding box regression model."""
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    # Plot 1: Loss curves
    ax1 = axes[0]
    ax1.plot(df['epoch'], df['train_loss'], color=COLORS['train'], label='Train')
    ax1.plot(df['epoch'], df['val_loss'], color=COLORS['val'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Loss')
    ax1.legend(frameon=False)
    ax1.set_xlim(1, df['epoch'].max())
    ax1.set_ylim(0, None)

    # Plot 2: Validation IoU
    ax2 = axes[1]
    ax2.plot(df['epoch'], df['val_iou'], color=COLORS['metric'], linewidth=1.5)
    ax2.axhline(y=df['val_iou'].max(), color=COLORS['metric'], linestyle='--',
                alpha=0.5, linewidth=1)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.set_title('Validation IoU')
    ax2.set_xlim(1, df['epoch'].max())
    ax2.set_ylim(0.4, 0.8)

    # Add best IoU annotation
    best_epoch = df.loc[df['val_iou'].idxmax(), 'epoch']
    best_iou = df['val_iou'].max()
    ax2.annotate(f'Best: {best_iou:.3f}',
                 xy=(best_epoch, best_iou),
                 xytext=(best_epoch + 5, best_iou - 0.05),
                 fontsize=8, color=COLORS['metric'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['metric'], lw=0.8))

    # Plot 3: Learning rate schedule
    ax3 = axes[2]
    ax3.plot(df['epoch'], df['lr'], color=COLORS['lr'], linewidth=1.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlim(1, df['epoch'].max())
    ax3.set_yscale('log')
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.0e}'))

    plt.tight_layout()

    # Save in multiple formats
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'regression_training_curves.pdf')
    fig.savefig(output_dir / 'regression_training_curves.png', dpi=300)
    plt.close(fig)

    print(f"Saved regression plots to {output_dir}")
    print(f"  Best validation IoU: {best_iou:.4f} at epoch {best_epoch}")


def plot_segmentation_training(csv_path: Path, output_dir: Path):
    """Generate training plots for segmentation model."""
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    # Plot 1: Loss curves
    ax1 = axes[0]
    ax1.plot(df['epoch'], df['train_loss'], color=COLORS['train'], label='Train')
    ax1.plot(df['epoch'], df['val_loss'], color=COLORS['val'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Dice Loss')
    ax1.set_title('Training Loss')
    ax1.legend(frameon=False)
    ax1.set_xlim(1, df['epoch'].max())
    ax1.set_ylim(0, None)

    # Plot 2: Validation Dice
    ax2 = axes[1]
    ax2.plot(df['epoch'], df['val_dice'], color=COLORS['metric'], linewidth=1.5)
    ax2.axhline(y=df['val_dice'].max(), color=COLORS['metric'], linestyle='--',
                alpha=0.5, linewidth=1)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Validation Dice')
    ax2.set_xlim(1, df['epoch'].max())
    ax2.set_ylim(0.5, 1.0)

    # Add best Dice annotation
    best_epoch = df.loc[df['val_dice'].idxmax(), 'epoch']
    best_dice = df['val_dice'].max()
    ax2.annotate(f'Best: {best_dice:.3f}',
                 xy=(best_epoch, best_dice),
                 xytext=(best_epoch - 20, best_dice - 0.03),
                 fontsize=8, color=COLORS['metric'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['metric'], lw=0.8))

    # Plot 3: Learning rate schedule
    ax3 = axes[2]
    ax3.plot(df['epoch'], df['lr'], color=COLORS['lr'], linewidth=1.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_xlim(1, df['epoch'].max())
    ax3.set_yscale('log')
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.0e}'))

    plt.tight_layout()

    # Save in multiple formats
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'segmentation_training_curves.pdf')
    fig.savefig(output_dir / 'segmentation_training_curves.png', dpi=300)
    plt.close(fig)

    print(f"Saved segmentation plots to {output_dir}")
    print(f"  Best validation Dice: {best_dice:.4f} at epoch {best_epoch}")


def plot_combined_figure(reg_csv: Path, seg_csv: Path, output_dir: Path):
    """Generate a combined 2-row figure with both models."""
    reg_df = pd.read_csv(reg_csv)
    seg_df = pd.read_csv(seg_csv)

    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5))

    # Row 1: Regression
    # Loss
    axes[0, 0].plot(reg_df['epoch'], reg_df['train_loss'], color=COLORS['train'], label='Train')
    axes[0, 0].plot(reg_df['epoch'], reg_df['val_loss'], color=COLORS['val'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('Bounding Box Regression')
    axes[0, 0].legend(frameon=False, loc='upper right')
    axes[0, 0].set_xlim(1, reg_df['epoch'].max())
    axes[0, 0].set_ylim(0, None)

    # IoU
    axes[0, 1].plot(reg_df['epoch'], reg_df['val_iou'], color=COLORS['metric'])
    best_iou = reg_df['val_iou'].max()
    axes[0, 1].axhline(y=best_iou, color=COLORS['metric'], linestyle='--', alpha=0.5, linewidth=1)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].set_title(f'Validation IoU (Best: {best_iou:.3f})')
    axes[0, 1].set_xlim(1, reg_df['epoch'].max())
    axes[0, 1].set_ylim(0.4, 0.8)

    # LR
    axes[0, 2].plot(reg_df['epoch'], reg_df['lr'], color=COLORS['lr'])
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].set_title('Learning Rate Schedule')
    axes[0, 2].set_xlim(1, reg_df['epoch'].max())
    axes[0, 2].set_yscale('log')
    axes[0, 2].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.0e}'))

    # Row 2: Segmentation
    # Loss
    axes[1, 0].plot(seg_df['epoch'], seg_df['train_loss'], color=COLORS['train'], label='Train')
    axes[1, 0].plot(seg_df['epoch'], seg_df['val_loss'], color=COLORS['val'], label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Loss')
    axes[1, 0].set_title('Segmentation')
    axes[1, 0].legend(frameon=False, loc='upper right')
    axes[1, 0].set_xlim(1, seg_df['epoch'].max())
    axes[1, 0].set_ylim(0, None)

    # Dice
    axes[1, 1].plot(seg_df['epoch'], seg_df['val_dice'], color=COLORS['metric'])
    best_dice = seg_df['val_dice'].max()
    axes[1, 1].axhline(y=best_dice, color=COLORS['metric'], linestyle='--', alpha=0.5, linewidth=1)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Dice Score')
    axes[1, 1].set_title(f'Validation Dice (Best: {best_dice:.3f})')
    axes[1, 1].set_xlim(1, seg_df['epoch'].max())
    axes[1, 1].set_ylim(0.5, 1.0)

    # LR
    axes[1, 2].plot(seg_df['epoch'], seg_df['lr'], color=COLORS['lr'])
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].set_title('Learning Rate Schedule')
    axes[1, 2].set_xlim(1, seg_df['epoch'].max())
    axes[1, 2].set_yscale('log')
    axes[1, 2].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:.0e}'))

    # Add row labels
    fig.text(0.02, 0.75, 'A', fontsize=14, fontweight='bold', va='center')
    fig.text(0.02, 0.28, 'B', fontsize=14, fontweight='bold', va='center')

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, hspace=0.35)

    # Save
    fig.savefig(output_dir / 'combined_training_curves.pdf')
    fig.savefig(output_dir / 'combined_training_curves.png', dpi=300)
    plt.close(fig)

    print(f"Saved combined plot to {output_dir}")


if __name__ == '__main__':
    base_dir = Path(__file__).parent

    reg_csv = base_dir / 'regression-paper-stats' / 'training_log.csv'
    seg_csv = base_dir / 'segmentation-paper-stats' / 'metrics.csv'

    # Generate individual plots
    plot_regression_training(reg_csv, base_dir)
    plot_segmentation_training(seg_csv, base_dir)

    # Generate combined figure
    plot_combined_figure(reg_csv, seg_csv, base_dir)

    print("\nAll plots generated successfully!")
