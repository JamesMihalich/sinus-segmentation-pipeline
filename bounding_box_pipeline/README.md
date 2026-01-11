# Bounding Box Pipeline

A toolkit for 3D bounding box localization in medical imaging. Train deep learning models to predict axis-aligned bounding boxes around anatomical structures in CT/MRI volumes.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Overview](#pipeline-overview)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Module Reference](#module-reference)

---

## Installation

### Requirements

```bash
pip install torch torchvision
pip install numpy nibabel scipy
pip install pandas matplotlib tqdm pyyaml
```

### Setup

The package can be used directly from its location:

```python
import sys
sys.path.append('/path/to/nn_code')

from bounding_box_pipeline.models import BBoxRegressor3D
from bounding_box_pipeline.training import Trainer
```

---

## Quick Start

### 1. Generate dataset from NIfTI files

```bash
python scripts/generate_dataset.py \
    --input /path/to/nifti_files \
    --output /path/to/npz_dataset \
    --window-level 600 \
    --window-width 1250
```

### 2. Train the model

```bash
python scripts/train.py \
    --data-dir /path/to/npz_dataset \
    --output-dir ./training_logs \
    --epochs 50
```

### 3. Run inference

```bash
python scripts/predict.py \
    --input /path/to/test_data \
    --checkpoint ./training_logs/run-XXXXX/best_model.pth \
    --output ./predictions
```

### 4. Evaluate results

```bash
python scripts/evaluate.py \
    --predictions ./predictions \
    --ground-truth /path/to/npz_dataset
```

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA GENERATION                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  NIfTI Files (image.nii + image_label.nii)                      │
│       │                                                         │
│       ▼                                                         │
│  1. Load mask volume                                            │
│  2. Extract bounding box (normalized 0-1)                       │
│  3. Apply CT windowing to image                                 │
│  4. Resize to 128×128×128                                       │
│       │                                                         │
│       ▼                                                         │
│  NPZ Files                                                      │
│    - 'image': (128,128,128) uint8                               │
│    - 'label': (6,) float32 [z1,y1,x1,z2,y2,x2]                  │
│    - 'original_shape': metadata                                 │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                       TRAINING                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LocalizationDataset                                            │
│    - Load NPZ files                                             │
│    - Normalize image to [0,1]                                   │
│    - Return (image, bbox) tensors                               │
│       │                                                         │
│       ▼                                                         │
│  BBoxRegressor3D                                                │
│    - 5 Conv3d blocks with BatchNorm + MaxPool                   │
│    - FC layers with Dropout                                     │
│    - Sigmoid output → [0,1] normalized coordinates              │
│       │                                                         │
│       ▼                                                         │
│  SmoothL1Loss                                                   │
│    - Robust to outliers                                         │
│    - Good for coordinate regression                             │
│       │                                                         │
│       ▼                                                         │
│  Outputs:                                                       │
│    - best_model.pth (best IoU checkpoint)                       │
│    - training_log.csv (metrics per epoch)                       │
│    - training_curves.png                                        │
│    - snapshots/epoch_XXX.png                                    │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                       INFERENCE                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Volume → Preprocess → Model → Denormalize → Output       │
│                                                                 │
│  Normalized bbox [0,1] → Absolute pixel coordinates             │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                      EVALUATION                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Metrics computed:                                              │
│    - IoU (Intersection over Union)                              │
│    - Center error (Euclidean distance)                          │
│    - Size error (dimension differences)                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Preprocessing

### Input Data Format

The pipeline expects paired NIfTI files:
- **Image**: `H0001.nii` (CT/MRI volume)
- **Label**: `H0001_label.nii` (binary segmentation mask)

### Generating the Dataset

```bash
python scripts/generate_dataset.py \
    --input /path/to/nifti \
    --output /path/to/npz \
    --target-shape 128 128 128 \
    --window-level 600 \
    --window-width 1250
```

### How Bounding Boxes are Extracted

1. **Load mask volume** from NIfTI
2. **Find non-zero coordinates** (the segmented region)
3. **Compute min/max** for each dimension (Z, Y, X)
4. **Normalize to [0, 1]** relative to original volume shape
5. **Store as 6 floats**: `[z1, y1, x1, z2, y2, x2]`

**Key insight**: Bounding boxes are extracted BEFORE resizing, so normalized coordinates remain valid regardless of the target volume size.

### Programmatic Usage

```python
from bounding_box_pipeline.data.preprocessing import create_localization_dataset

create_localization_dataset(
    input_dir='/path/to/nifti',
    output_dir='/path/to/npz',
    target_shape=(128, 128, 128),
    window_level=600,
    window_width=1250,
)
```

### Output NPZ Format

Each NPZ file contains:
- `image`: (128, 128, 128) uint8 - Windowed and resized volume
- `label`: (6,) float32 - Normalized bbox `[z1, y1, x1, z2, y2, x2]`
- `original_shape`: (3,) - Original volume dimensions

---

## Training

### Command Line

```bash
# Basic training
python scripts/train.py \
    --data-dir /path/to/npz \
    --output-dir ./training_logs \
    --epochs 50

# With custom parameters
python scripts/train.py \
    --data-dir /path/to/npz \
    --epochs 100 \
    --batch-size 4 \
    --lr 1e-4

# Resume from checkpoint
python scripts/train.py \
    --data-dir /path/to/npz \
    --resume ./training_logs/run-XXXXX/best_model.pth
```

### Using Config File

```bash
python scripts/train.py --config configs/my_config.yaml
```

### Programmatic Usage

```python
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from bounding_box_pipeline.models import BBoxRegressor3D
from bounding_box_pipeline.data import LocalizationDataset, create_data_splits, get_dataset_files
from bounding_box_pipeline.training import Trainer

# Setup
device = torch.device('cuda')
data_dir = Path('/path/to/npz')
output_dir = Path('./training_logs/my_run')

# Load and split data
files = get_dataset_files(data_dir)
train_files, val_files = create_data_splits(files, train_ratio=0.8)

# Create datasets
train_ds = LocalizationDataset(train_files)
val_ds = LocalizationDataset(val_files)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4)

# Create model
model = BBoxRegressor3D(
    input_size=(128, 128, 128),
    base_channels=32,
    dropout=0.5,
)

# Train
trainer = Trainer(model, device, output_dir, learning_rate=1e-4)
history = trainer.fit(train_loader, val_loader, epochs=50)
```

### Training Outputs

```
training_logs/run-20240115_143022/
├── best_model.pth         # Best validation IoU checkpoint
├── last_model.pth         # Final epoch checkpoint
├── config.yaml            # Saved configuration
├── training_log.csv       # Per-epoch metrics
├── training_curves.png    # Loss and IoU plots
└── snapshots/
    ├── epoch_005.png      # Visual predictions
    ├── epoch_010.png
    └── ...
```

---

## Inference

### Command Line

```bash
# Single file
python scripts/predict.py \
    --input /path/to/volume.npz \
    --checkpoint best_model.pth

# Batch inference
python scripts/predict.py \
    --input /path/to/test_data \
    --checkpoint best_model.pth \
    --output ./predictions
```

### Programmatic Usage

```python
from bounding_box_pipeline.models import BBoxRegressor3D
from bounding_box_pipeline.inference import BBoxPredictor
import numpy as np

# Create predictor
model = BBoxRegressor3D()
predictor = BBoxPredictor.from_checkpoint(
    model=model,
    checkpoint_path='best_model.pth',
    device='cuda',
)

# Predict on volume
volume = np.load('test.npz')['image']
absolute_bbox = predictor.predict_single(volume)

print(f"Predicted bbox: {absolute_bbox}")
# Output: [z1, y1, x1, z2, y2, x2] in pixel coordinates
```

### Output Format

Predictions are returned as absolute pixel coordinates:
```
[z_min, y_min, x_min, z_max, y_max, x_max]
```

To get normalized coordinates:
```python
absolute, normalized = predictor.predict_single(volume, return_normalized=True)
```

---

## Evaluation

### Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **IoU** | Intersection over Union | > 0.7 |
| **Center Error** | Distance between bbox centers | < 10 voxels |
| **Size Error** | Absolute dimension differences | < 20 voxels |

### Command Line

```bash
python scripts/evaluate.py \
    --predictions ./predictions \
    --ground-truth /path/to/npz_dataset \
    --output evaluation_results.csv
```

### Programmatic Usage

```python
from bounding_box_pipeline.evaluation import compute_iou, compute_metrics, BBoxEvaluator

# Single pair
pred = np.array([10, 20, 30, 50, 60, 70])
gt = np.array([12, 18, 32, 48, 62, 68])

iou = compute_iou(pred, gt)
metrics = compute_metrics(pred, gt)

print(f"IoU: {iou:.4f}")
print(f"Center error: {metrics['center_error']:.2f}")

# Batch evaluation
evaluator = BBoxEvaluator(
    prediction_dir='./predictions',
    ground_truth_dir='./data',
)
results = evaluator.evaluate_all()
print(evaluator.compute_summary(results))
```

---

## Configuration

### YAML Config File

```yaml
# configs/default_config.yaml

data_root: /path/to/data
output_root: ./training_logs

model:
  architecture: BBoxRegressor3D
  input_size: [128, 128, 128]
  base_channels: 32
  dropout: 0.5

training:
  batch_size: 4
  learning_rate: 0.0001
  epochs: 50
  train_ratio: 0.8
  val_ratio: 0.2

preprocessing:
  target_shape: [128, 128, 128]
  window_level: 600.0
  window_width: 1250.0
```

### Loading Config

```python
from bounding_box_pipeline.configs import Config

config = Config.from_yaml('configs/my_config.yaml')
print(config.training.learning_rate)
print(config.model.input_size)
```

---

## Module Reference

### Models

```python
from bounding_box_pipeline.models import BBoxRegressor3D, create_regressor

# Standard model
model = BBoxRegressor3D(
    input_size=(128, 128, 128),
    base_channels=32,
    dropout=0.5,
)

# Using factory function
model = create_regressor('standard')  # Full model
model = create_regressor('lite')      # Lightweight variant
```

### Data

```python
from bounding_box_pipeline.data import (
    LocalizationDataset,
    create_localization_dataset,
    create_data_splits,
)

# Dataset
dataset = LocalizationDataset(file_paths)

# Data splits
train_files, val_files = create_data_splits(
    files, train_ratio=0.8, seed=42
)
```

### Utilities

```python
from bounding_box_pipeline.utils import (
    get_relative_bbox,
    normalize_bbox,
    denormalize_bbox,
    compute_iou,
)

# Extract bbox from mask
bbox = get_relative_bbox(mask_volume)  # Returns normalized [0,1]

# Convert between formats
absolute = denormalize_bbox(normalized, volume_shape)
normalized = normalize_bbox(absolute, volume_shape)
```

### Visualization

```python
from bounding_box_pipeline.visualization import (
    plot_training_curves,
    visualize_bbox_slice,
    visualize_dataset_sample,
)

# Plot training curves
plot_training_curves('training_log.csv', output_path='curves.png')

# Visualize prediction
visualize_bbox_slice(
    image=volume,
    pred_bbox=prediction,
    gt_bbox=ground_truth,
    slice_idx=64,
)

# Quick dataset inspection
visualize_dataset_sample('sample.npz')
```

---

## Model Architecture

The `BBoxRegressor3D` uses a simple but effective architecture:

```
Input: (B, 1, 128, 128, 128)
       │
       ▼
┌─────────────────────────┐
│ Conv3d(1→32) + BN + ReLU│  Block 1: 128→64
│ MaxPool3d(2)            │
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│ Conv3d(32→64) + BN + ReLU│  Block 2: 64→32
│ MaxPool3d(2)             │
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│ Conv3d(64→128) + BN + ReLU│  Block 3: 32→16
│ MaxPool3d(2)              │
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│ Conv3d(128→256) + BN + ReLU│  Block 4: 16→8
│ MaxPool3d(2)               │
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│ Conv3d(256→512) + BN + ReLU│  Block 5: 8→4
│ MaxPool3d(2)               │
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│ Flatten: 512×4×4×4 = 32K│
│ FC(32K→512) + ReLU + Drop│
│ FC(512→64) + ReLU        │
│ FC(64→6) + Sigmoid       │
└─────────────────────────┘
       │
       ▼
Output: (B, 6) in [0, 1]
```

**Total parameters**: ~8.5M

---

## Tips and Best Practices

### CT Windowing

Choose window parameters based on your target anatomy:

| Anatomy | Window Level | Window Width |
|---------|-------------|--------------|
| Soft tissue | 40 | 400 |
| Lung | -600 | 1500 |
| Bone | 400 | 1800 |
| Liver | 60 | 150 |
| Brain | 40 | 80 |

### Training Tips

1. **Start with default hyperparameters** - they work well for most cases
2. **Monitor IoU** - more meaningful than loss for localization
3. **Check snapshots** - visual validation catches issues early
4. **Use ReduceLROnPlateau** - automatically adapts learning rate

### Common Issues

**Low IoU after training:**
- Check CT window settings match your anatomy
- Verify labels are binary (0 and 1)
- Ensure bboxes are extracted before resizing

**NaN loss:**
- Reduce learning rate
- Check for empty masks in dataset
- Verify data normalization

**Memory errors:**
- Reduce batch size
- Use `BBoxRegressorLite` model variant

---

## File Structure

```
bounding_box_pipeline/
├── __init__.py
├── README.md
├── configs/
│   ├── __init__.py
│   ├── config.py
│   └── default_config.yaml
├── data/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   └── windowing.py
│   └── datasets/
│       ├── __init__.py
│       └── localization_dataset.py
├── models/
│   ├── __init__.py
│   └── bbox_regressor.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   └── losses.py
├── inference/
│   ├── __init__.py
│   └── predictor.py
├── evaluation/
│   ├── __init__.py
│   └── metrics.py
├── visualization/
│   ├── __init__.py
│   ├── plots.py
│   └── snapshots.py
├── utils/
│   ├── __init__.py
│   ├── io.py
│   └── bbox_utils.py
└── scripts/
    ├── generate_dataset.py
    ├── train.py
    ├── predict.py
    └── evaluate.py
```

---

## License

This code is provided for research and educational purposes.

