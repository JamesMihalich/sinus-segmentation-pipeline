# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical imaging ML toolkit with two parallel pipelines for 3D CT/MRI volumes:
- **Bounding Box Pipeline** (`bounding_box_pipeline/`): Localizes anatomical structures by predicting axis-aligned bounding boxes
- **Segmentation Pipeline** (`segmentation_pipeline/`): Full 3D semantic segmentation with pixel-level predictions

Both follow the same workflow: NIfTI → preprocessing → NPZ → training → inference → evaluation.

## Commands

### Bounding Box Pipeline

```bash
# Generate dataset from NIfTI pairs (image.nii + image_label.nii)
python bounding_box_pipeline/scripts/generate_dataset.py \
    --input /path/to/nifti --output /path/to/npz \
    --window-level 600 --window-width 1250

# Train bounding box regressor
python bounding_box_pipeline/scripts/train.py \
    --data-dir /path/to/npz --output-dir ./training_logs --epochs 50

# Run inference
python bounding_box_pipeline/scripts/predict.py \
    --input /path/to/test --checkpoint best_model.pth --output ./predictions

# Evaluate (IoU, center error, size error)
python bounding_box_pipeline/scripts/evaluate.py \
    --predictions ./predictions --ground-truth /path/to/npz
```

### Segmentation Pipeline

```bash
# Convert Amira .am files to NIfTI (if needed)
python segmentation_pipeline/scripts/convert_amira.py \
    --input /path/to/amira --output /path/to/nifti --type label

# Preprocess NIfTI to NPZ (resample, crop, window)
python segmentation_pipeline/scripts/preprocess.py \
    --input /path/to/nifti --output /path/to/npz \
    --resample 0.33 0.33 0.33 --window-level -300 --window-width 1000

# Train segmentation model
python segmentation_pipeline/scripts/train.py \
    --data-dir /path/to/npz --output-dir ./results --epochs 100

# Run inference with sliding window
python segmentation_pipeline/scripts/predict.py \
    --input /path/to/test --checkpoint best.pt --output ./predictions \
    --overlap 0.5 --threshold 0.5

# Evaluate (Dice, IoU, HD95, ASSD)
python segmentation_pipeline/scripts/evaluate.py \
    --predictions ./predictions --ground-truth /path/to/gt \
    --spacing 0.33 0.33 0.33
```

All scripts support `--config path/to/config.yaml` for YAML configuration.

## Architecture

### Data Flow
- **Input**: NIfTI files with naming convention `H0001.nii` (image) + `H0001_label.nii` (mask)
- **Intermediate**: NPZ files containing `image` (uint8) and `label` (binary mask or 6-element bbox)
- **Coordinates**: Bounding boxes normalized to [0,1] relative to original volume shape

### Models
- `BBoxRegressor3D`: 5 conv blocks (32→64→128→256→512) + FC head → 6 sigmoid outputs (~8.5M params)
- `ResidualUnetSE3D`: 3D Residual U-Net with Squeeze-and-Excitation blocks, configurable skip connections (`concat` or `additive`)

### Key Configuration
- **Bounding box**: Input 128³, window L=600/W=1250, batch size 4
- **Segmentation**: Patch 224×224×256, window L=-300/W=1000, batch size 1

### CT Windowing Reference
| Anatomy | Level | Width |
|---------|-------|-------|
| Soft tissue | 40 | 400 |
| Bone | 400 | 1800 |
| Lung | -600 | 1500 |

## Code Patterns

- YAML configs with dataclass structures in `configs/config.py`
- Factory pattern: `BBoxPredictor.from_checkpoint()`, `VolumePredictor.from_checkpoint()`
- Smart NPZ loading tries multiple key names (`image`, `image.npy`, etc.)
- Data splits use seed=42 for reproducibility
- Checkpoints save both `best.pt` (best validation) and `last.pt`

## Dependencies

Core: torch, numpy, nibabel (NIfTI), scipy, scikit-image, pandas, matplotlib, pyyaml
