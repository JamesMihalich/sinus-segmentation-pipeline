# Segmentation Pipeline

A consolidated toolkit for 3D medical image segmentation using deep learning. This package provides end-to-end functionality from data preprocessing to model evaluation.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Overview](#pipeline-overview)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Model Architectures](#model-architectures)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Module Reference](#module-reference)

---

## Installation

### Requirements

```bash
pip install torch torchvision
pip install numpy nibabel scipy scikit-image
pip install pandas matplotlib tqdm pyyaml
pip install numpy-stl  # Optional: for STL mesh export
```

### Setup

The package can be used directly from its location. Add to your Python path:

```python
import sys
sys.path.append('/path/to/nn_code')

from segmentation_pipeline.models import ResidualUnetSE3D
from segmentation_pipeline.training import Trainer
```

---

## Quick Start

### 1. Preprocess your data

Convert NIfTI files to training-ready NPZ format:

```bash
python scripts/preprocess.py \
    --input /path/to/nifti_files \
    --output /path/to/npz_output \
    --window-level -300 \
    --window-width 1000
```

### 2. Train a model

```bash
python scripts/train.py \
    --data-dir /path/to/npz_output \
    --output-dir ./results \
    --epochs 100
```

### 3. Run inference

```bash
python scripts/predict.py \
    --input /path/to/test_data \
    --checkpoint ./results/run-XXXXXX/checkpoints/best.pt \
    --output ./predictions
```

### 4. Evaluate results

```bash
python scripts/evaluate.py \
    --predictions ./predictions \
    --ground-truth /path/to/ground_truth
```

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Amira (.am)  ──► convert_amira.py ──► NIfTI (.nii)            │
│                                            │                    │
│                                            ▼                    │
│                                    preprocess.py                │
│                                    - Resample to isotropic      │
│                                    - Crop to bounding box       │
│                                    - Apply CT windowing         │
│                                            │                    │
│                                            ▼                    │
│                                      NPZ files                  │
│                                   (image + label)               │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                       TRAINING PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  NPZ files ──► VolumeDataset ──► DataLoader ──► Trainer        │
│                     │                              │            │
│                     ▼                              ▼            │
│              VolumeAugmentation           ResidualUnetSE3D      │
│              - Random flips                    │                │
│              - Rotations                       ▼                │
│              - Scaling                   BCEDiceLoss            │
│              - Noise                           │                │
│                                                ▼                │
│                                         Checkpoints             │
│                                         metrics.csv             │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                      INFERENCE PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Test NPZ ──► VolumePredictor ──► Post-processing ──► Output   │
│                    │                    │                       │
│                    ▼                    ▼                       │
│            Sliding window       Keep largest component          │
│            with overlap         Threshold probabilities         │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                      EVALUATION PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Predictions + GT ──► SegmentationEvaluator ──► metrics.csv    │
│                              │                                  │
│                              ▼                                  │
│                    Dice, IoU, HD95, Precision, Recall           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Preprocessing

### Converting Amira Files

If your data is in Amira format (.am), convert to NIfTI first:

```bash
# Convert label masks (RLE compressed)
python scripts/convert_amira.py \
    --input /path/to/amira_files \
    --output /path/to/nifti_output \
    --type label

# Convert volume scans (raw binary)
python scripts/convert_amira.py \
    --input /path/to/amira_files \
    --type volume
```

### Preprocessing NIfTI to NPZ

The preprocessing script handles:
- **Header alignment**: Ensures image/mask headers match
- **Resampling**: Optional isotropic resampling (e.g., 0.33mm)
- **Cropping**: Crops to bounding box of segmentation mask
- **Windowing**: Applies CT window/level normalization

```bash
# Basic preprocessing
python scripts/preprocess.py \
    --input /path/to/nifti \
    --output /path/to/npz

# With resampling and custom windowing
python scripts/preprocess.py \
    --input /path/to/nifti \
    --output /path/to/npz \
    --resample 0.33 0.33 0.33 \
    --window-level -300 \
    --window-width 1000 \
    --padding 10 \
    --fix-headers

# Without cropping (keep full volume)
python scripts/preprocess.py \
    --input /path/to/nifti \
    --output /path/to/npz \
    --no-crop
```

### Preprocessing Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | Input directory with NIfTI files |
| `--output` | required | Output directory for NPZ files |
| `--window-level` | -300 | CT window level (HU) |
| `--window-width` | 1000 | CT window width (HU) |
| `--padding` | 10 | Padding around bounding box (ignored with `--no-crop`) |
| `--no-crop` | off | Disable cropping, keep full volume |
| `--resample` | none | Resample to isotropic spacing (x y z) |
| `--fix-headers` | off | Fix header mismatches before processing |
| `--image-pattern` | `H*.nii` | Glob pattern for image files |
| `--label-suffix` | `_label` | Suffix for label files |

### Expected File Naming

The preprocessing expects paired files:
- Images: `H0001.nii`, `H0002.nii`, ...
- Labels: `H0001_label.nii`, `H0002_label.nii`, ...

Output NPZ files contain:
- `image`: Windowed uint8 array (0-255)
- `label`: Binary uint8 mask (0 or 1)

### Exporting Slices as PNG

Convert NPZ volumes to 2D PNG slices for visualization or 2D model training:

```bash
# Export all slices as 512x512 grayscale PNGs
python scripts/export_slices.py \
    --input /path/to/npz \
    --output /path/to/png_slices

# Custom size and axis
python scripts/export_slices.py \
    --input /path/to/npz \
    --output /path/to/png_slices \
    --size 256 \
    --axis 0

# Export labels instead of images
python scripts/export_slices.py \
    --input /path/to/npz \
    --output /path/to/label_slices \
    --key label
```

Output files are named `{patient_id}_slice_{number}.png` (e.g., `P0001_slice_042.png`).

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | Input directory with NPZ files |
| `--output` | required | Output directory for PNG files |
| `--axis` | 2 | Slice axis: 0=sagittal, 1=coronal, 2=axial |
| `--size` | 512 | Output image size (square, with black padding) |
| `--key` | `image` | Array to export: `image` or `label` |

### Programmatic Usage

```python
from segmentation_pipeline.data.preprocessing import (
    process_dataset_to_npz,
    resample_dataset,
    fix_header_mismatches,
)

# Fix headers first
fix_header_mismatches('/path/to/nifti')

# Resample to isotropic
resample_dataset(
    input_dir='/path/to/nifti',
    output_dir='/path/to/resampled',
    target_spacing=(0.33, 0.33, 0.33),
)

# Convert to NPZ
process_dataset_to_npz(
    input_dir='/path/to/resampled',
    output_dir='/path/to/npz',
    window_level=-300,
    window_width=1000,
    padding=10,
)
```

---

## Training

### Command Line

```bash
# Basic training (standard model)
python scripts/train.py \
    --data-dir /path/to/npz \
    --output-dir ./results \
    --epochs 100

# Enhanced model with attention gates and deep supervision
python scripts/train.py \
    --data-dir /path/to/npz \
    --output-dir ./results \
    --model enhanced \
    --epochs 100

# Enhanced model with custom base channels
python scripts/train.py \
    --data-dir /path/to/npz \
    --output-dir ./results \
    --model enhanced \
    --base-channels 48

# Enhanced model without deep supervision
python scripts/train.py \
    --data-dir /path/to/npz \
    --output-dir ./results \
    --model enhanced \
    --no-deep-supervision

# With custom parameters
python scripts/train.py \
    --data-dir /path/to/npz \
    --output-dir ./results \
    --epochs 100 \
    --batch-size 1 \
    --lr 1e-4 \
    --skip-mode concat

# Resume from checkpoint
python scripts/train.py \
    --data-dir /path/to/npz \
    --resume ./results/run-XXXXX/checkpoints/best.pt
```

### Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | required | Directory containing NPZ files |
| `--output-dir` | `./results` | Output directory for checkpoints |
| `--model` | `standard` | Model architecture: `standard` or `enhanced` |
| `--base-channels` | 16/32 | Base feature channels (auto-set by model type) |
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 1 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--skip-mode` | `concat` | Skip connections: `concat` or `additive` |
| `--no-deep-supervision` | off | Disable deep supervision (enhanced model only) |
| `--resume` | none | Resume from checkpoint path |

### Using a Config File

```bash
python scripts/train.py --config configs/my_experiment.yaml
```

### Programmatic Usage

```python
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from segmentation_pipeline.models import ResidualUnetSE3D
from segmentation_pipeline.data import VolumeDataset, create_data_splits
from segmentation_pipeline.training import Trainer

# Setup
device = torch.device('cuda')
data_dir = Path('/path/to/npz')
output_dir = Path('./results/my_run')

# Find and split data
files = sorted(data_dir.glob('*.npz'))
train_files, val_files, test_files = create_data_splits(files)

# Create datasets
train_ds = VolumeDataset(
    train_files,
    patch_size=(224, 224, 256),
    augment=True,
    aug_params={
        'flip_prob': 0.5,
        'rotate_prob': 0.4,
        'rotate_range': (-10, 10),
    }
)
val_ds = VolumeDataset(val_files, patch_size=(224, 224, 256))

# Create loaders
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

# Create model
model = ResidualUnetSE3D(
    skip_mode='concat',      # or 'additive'
    se_reduction_ratio=8,    # SE block compression
)

# Create trainer
trainer = Trainer(
    model=model,
    device=device,
    output_dir=output_dir,
    learning_rate=1e-4,
    weight_decay=1e-4,
)

# Train
history = trainer.fit(
    train_loader,
    val_loader,
    epochs=100,
)
```

### Training Outputs

After training, the output directory contains:

```
results/run-20240115_143022/
├── checkpoints/
│   ├── best.pt          # Best validation loss checkpoint
│   └── last.pt          # Final epoch checkpoint
├── config.yaml          # Saved configuration
├── data_split.csv       # Train/val/test file assignments
├── metrics.csv          # Per-epoch metrics
└── last_weights.pt      # Final model weights only
```

### Visualizing Training

```python
from segmentation_pipeline.visualization import plot_training_curves

plot_training_curves(
    'results/run-XXXXX/metrics.csv',
    output_path='training_curves.png',
)
```

---

## Model Architectures

### Standard 3D U-Net (`ResidualUnetSE3D`)

The default model with residual blocks and Squeeze-and-Excitation attention.

| Property | Value |
|----------|-------|
| Base channels | 16 |
| Channel progression | 16 → 32 → 64 → 128 |
| Attention | SE (channel attention in blocks) |
| Skip connections | Concat or Additive |

```python
from segmentation_pipeline.models import ResidualUnetSE3D, create_unet

# Direct instantiation
model = ResidualUnetSE3D(
    in_channels=1,
    out_channels=1,
    base_channels=16,
    skip_mode='concat',
)

# Using presets
model = create_unet('standard')    # Concat skips, SE=8
model = create_unet('additive')    # Add skips, SE=2
model = create_unet('lightweight') # Smaller base channels
```

### Enhanced 3D U-Net (`ResidualUnetSE3DEnhanced`)

Improved model with attention gates on skip connections and deep supervision.

| Property | Value |
|----------|-------|
| Base channels | 32 |
| Channel progression | 32 → 64 → 128 → 256 |
| Attention | SE blocks + Attention Gates on skips |
| Deep supervision | Auxiliary losses at decoder stages |
| Dropout | 0.15 after bottleneck |
| Bottleneck | 2 residual blocks (deeper) |

```python
from segmentation_pipeline.models import ResidualUnetSE3DEnhanced, create_enhanced_unet

# Direct instantiation
model = ResidualUnetSE3DEnhanced(
    in_channels=1,
    out_channels=1,
    base_channels=32,
    deep_supervision=True,
    dropout_rate=0.15,
)

# Using presets
model = create_enhanced_unet('standard')    # 32 base channels
model = create_enhanced_unet('large')       # 48 base channels
model = create_enhanced_unet('lightweight') # 24 base channels
```

#### Deep Supervision

Deep supervision adds auxiliary loss functions at intermediate decoder stages, improving gradient flow and regularization.

```
Decoder Stage 3 → Auxiliary Loss (weight: 0.25)
       ↓
Decoder Stage 2 → Auxiliary Loss (weight: 0.5)
       ↓
Final Output   → Main Loss (weight: 1.0)
```

During training, the total loss is: `L = 1.0 * L_main + 0.5 * L_ds1 + 0.25 * L_ds2`

During inference, only the main output is used.

```python
from segmentation_pipeline.models import DeepSupervisionLoss

# Training with deep supervision
model = ResidualUnetSE3DEnhanced(deep_supervision=True)
criterion = DeepSupervisionLoss(
    base_loss=nn.BCEWithLogitsLoss(),
    weights=(1.0, 0.5, 0.25),
)

outputs = model(x)  # Returns {"out": ..., "ds1": ..., "ds2": ...}
loss = criterion(outputs, target)
```

#### Attention Gates

Attention gates learn to focus on relevant spatial regions in skip connections, helping the decoder attend to important encoder features.

```python
from segmentation_pipeline.models import AttentionGate3D

# Standalone usage
att_gate = AttentionGate3D(
    gate_channels=128,   # From decoder
    skip_channels=128,   # From encoder
)
attended_skip = att_gate(gate=decoder_features, skip=encoder_features)
```

### 2D Attention U-Net (`AttentionResUNet2D`)

A 2D model for slice-based segmentation with attention gates.

| Property | Value |
|----------|-------|
| Input size | 512 × 512 (single channel) |
| Base channels | 64 |
| Channel progression | 64 → 128 → 256 → 512 → 1024 |
| Attention | Attention Gates on skip connections |

```python
from segmentation_pipeline.models import AttentionResUNet2D

model = AttentionResUNet2D(
    in_channels=1,
    out_channels=1,
    base_features=64,
)

# Input: (B, 1, 512, 512)
# Output: (B, 1, 512, 512)
```

### Model Comparison

| Model | Parameters | Base Ch | Attention | Deep Supervision |
|-------|------------|---------|-----------|------------------|
| `ResidualUnetSE3D` | ~500K | 16 | SE blocks | No |
| `ResidualUnetSE3DEnhanced` | ~2M | 32 | SE + Attention Gates | Yes |
| `AttentionResUNet2D` | ~8M | 64 | Attention Gates | No |

---

## Inference

### Command Line

```bash
# Single file
python scripts/predict.py \
    --input /path/to/volume.npz \
    --checkpoint best.pt

# Batch inference on directory
python scripts/predict.py \
    --input /path/to/test_data \
    --checkpoint best.pt \
    --output ./predictions \
    --overlap 0.5 \
    --threshold 0.5
```

### Programmatic Usage

```python
from segmentation_pipeline.models import ResidualUnetSE3D
from segmentation_pipeline.inference import VolumePredictor

# Load model
model = ResidualUnetSE3D(skip_mode='additive')
predictor = VolumePredictor.from_checkpoint(
    model=model,
    checkpoint_path='best.pt',
    device='cuda',
    patch_size=(224, 224, 256),
    overlap=0.5,
    threshold=0.5,
)

# Predict single volume
import numpy as np
volume = np.load('test.npz')['image']
mask = predictor.predict_single(volume)

# Or with probabilities
mask, probabilities = predictor.predict_single(volume, return_probabilities=True)

# Batch predict directory
results = predictor.predict_batch(
    input_dir='/path/to/test_data',
    output_dir='./predictions',
)
```

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_size` | (224, 224, 256) | Size of sliding window patches |
| `overlap` | 0.5 | Overlap between patches (0.0-0.9) |
| `threshold` | 0.5 | Probability threshold for binarization |
| `apply_postprocessing` | True | Apply connected component analysis |
| `keep_largest_component` | True | Keep only largest connected region |

---

## Evaluation

### Command Line

```bash
python scripts/evaluate.py \
    --predictions ./predictions \
    --ground-truth /path/to/ground_truth \
    --output evaluation_results.csv \
    --spacing 0.33 0.33 0.33
```

### Programmatic Usage

```python
from segmentation_pipeline.evaluation import (
    compute_metrics,
    SegmentationEvaluator,
    evaluate_directory,
)

# Single pair
import numpy as np
pred = np.load('prediction.npz')['mask']
gt = np.load('ground_truth.npz')['label']

metrics = compute_metrics(pred, gt, spacing=(0.33, 0.33, 0.33))
print(f"Dice: {metrics['dice']:.4f}")
print(f"HD95: {metrics['hd95']:.2f} mm")

# Batch evaluation
evaluator = SegmentationEvaluator(
    prediction_dir='./predictions',
    ground_truth_dirs='/path/to/ground_truth',
    spacing=(0.33, 0.33, 0.33),
)

results_df = evaluator.evaluate_all()
summary = evaluator.compute_summary(results_df)
print(summary)

# Or use convenience function
results = evaluate_directory(
    prediction_dir='./predictions',
    ground_truth_dir='./ground_truth',
    output_path='metrics.csv',
)
```

### Available Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `dice` | Dice coefficient (F1 score) | 0-1 (higher = better) |
| `iou` | Intersection over Union | 0-1 (higher = better) |
| `precision` | Positive predictive value | 0-1 (higher = better) |
| `recall` | Sensitivity / True positive rate | 0-1 (higher = better) |
| `hd95` | 95th percentile Hausdorff distance | mm (lower = better) |
| `assd` | Average symmetric surface distance | mm (lower = better) |

---

## Configuration

### YAML Configuration

Copy and modify `configs/default_config.yaml`:

```yaml
# Paths
data_root: /path/to/data
output_root: ./results

# Model
model:
  architecture: ResidualUnetSE3D
  skip_mode: concat        # 'concat' or 'additive'
  se_reduction_ratio: 8

# Training
training:
  patch_size: [224, 224, 256]
  batch_size: 1
  learning_rate: 0.0001
  epochs: 100
  bce_weight: 0.5
  dice_weight: 0.5

# Augmentation
augmentation:
  flip_prob: 0.5
  rotate_prob: 0.4
  rotate_range: [-10, 10]
  noise_prob: 0.4

# Inference
inference:
  overlap: 0.5
  threshold: 0.5
  keep_largest_component: true
```

### Loading Config Programmatically

```python
from segmentation_pipeline.configs import Config

# Load from YAML
config = Config.from_yaml('configs/my_experiment.yaml')

# Access values
print(config.training.learning_rate)
print(config.model.skip_mode)

# Save config
config.to_yaml('configs/saved_config.yaml')
```

---

## Module Reference

### Models

```python
from segmentation_pipeline.models import (
    # 3D Models
    ResidualUnetSE3D,
    create_unet,
    ResidualUnetSE3DEnhanced,
    create_enhanced_unet,
    DeepSupervisionLoss,
    AttentionGate3D,
    # 2D Models
    AttentionResUNet2D,
)
from segmentation_pipeline.models.blocks import ChannelSELayer3D, ResNetBlockSE

# Standard 3D model
model = ResidualUnetSE3D(
    in_channels=1,
    out_channels=1,
    base_channels=16,
    skip_mode='concat',
    se_reduction_ratio=8,
)

# Enhanced 3D model with attention gates and deep supervision
model = ResidualUnetSE3DEnhanced(
    in_channels=1,
    out_channels=1,
    base_channels=32,
    deep_supervision=True,
    dropout_rate=0.15,
)

# 2D model for slice-based segmentation
model_2d = AttentionResUNet2D(
    in_channels=1,
    out_channels=1,
    base_features=64,
)

# Using presets
model = create_unet('standard')           # Standard 3D
model = create_enhanced_unet('standard')  # Enhanced 3D
model = create_enhanced_unet('large')     # Enhanced 3D with more capacity
```

### Data

```python
from segmentation_pipeline.data import VolumeDataset, VolumeAugmentation
from segmentation_pipeline.data.converters import AmiraConverter
from segmentation_pipeline.data.preprocessing import (
    apply_ct_window,
    crop_to_mask_bbox,
    resample_to_isotropic,
)

# Dataset
dataset = VolumeDataset(
    file_paths=list_of_paths,
    patch_size=(224, 224, 256),
    augment=True,
)

# Augmentation standalone
aug = VolumeAugmentation(flip_prob=0.5, rotate_prob=0.4)
augmented_vol, augmented_mask = aug(volume, mask)

# CT windowing
windowed = apply_ct_window(ct_data, window_level=-300, window_width=1000)
```

### Training

```python
from segmentation_pipeline.training import Trainer, DiceLoss, BCEDiceLoss

# Loss functions
dice_loss = DiceLoss()
combined_loss = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

# Trainer
trainer = Trainer(model, device, output_dir)
history = trainer.fit(train_loader, val_loader, epochs=100)
```

### Inference

```python
from segmentation_pipeline.inference import VolumePredictor, postprocess_prediction

# Predictor
predictor = VolumePredictor(model, device)
mask = predictor.predict_single(volume)

# Manual post-processing
from segmentation_pipeline.inference.postprocessing import (
    keep_largest_component,
    threshold_predictions,
    fill_holes,
)

mask = threshold_predictions(probs, threshold=0.5)
mask = keep_largest_component(mask)
```

### Evaluation

```python
from segmentation_pipeline.evaluation import (
    dice_score,
    iou_score,
    hausdorff_95,
    compute_metrics,
    SegmentationEvaluator,
)

# Individual metrics
dice = dice_score(pred, gt)
hd95 = hausdorff_95(pred, gt, spacing=(0.33, 0.33, 0.33))

# All metrics
metrics = compute_metrics(pred, gt, spacing=(0.33, 0.33, 0.33))
```

### Utilities

```python
from segmentation_pipeline.utils import (
    load_npz,
    save_npz,
    load_nifti,
    save_nifti,
    crop_to_nonzero,
    pad_to_shape,
)

# Load NPZ with smart key detection
image = load_npz('file.npz')  # Tries 'image', 'image.npy', etc.

# Save standardized NPZ
save_npz('output.npz', image=img_array, label=mask_array)

# Volume operations
cropped, slices = crop_to_nonzero(volume, padding=5, return_slices=True)
padded = pad_to_shape(volume, target_shape=(256, 256, 256))
```

### Visualization

```python
from segmentation_pipeline.visualization import (
    plot_training_curves,
    visualize_npz,
    visualize_prediction_comparison,
    create_stl_mesh,
)

# Training curves
plot_training_curves('metrics.csv', output_path='curves.png')

# Visualize NPZ file
visualize_npz('data.npz', output_path='preview.png')

# Compare prediction to ground truth
visualize_prediction_comparison(
    image_path='image.npz',
    prediction_path='prediction.npz',
    ground_truth_path='ground_truth.npz',
)

# Export to 3D mesh
create_stl_mesh(mask, 'output.stl', smooth_sigma=1.0)
```

---

## Common Workflows

### Complete Training Pipeline

```python
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from segmentation_pipeline.configs import Config
from segmentation_pipeline.data import VolumeDataset, create_data_splits
from segmentation_pipeline.models import ResidualUnetSE3D
from segmentation_pipeline.training import Trainer

# Load config
config = Config.from_yaml('configs/my_config.yaml')

# Prepare data
files = sorted(config.data_root.glob('*.npz'))
train_files, val_files, _ = create_data_splits(
    files,
    train_ratio=0.75,
    val_ratio=0.15,
)

# Create datasets and loaders
train_ds = VolumeDataset(
    train_files,
    patch_size=config.training.patch_size,
    augment=True,
    aug_params=config.augmentation.to_dict(),
)
val_ds = VolumeDataset(val_files, patch_size=config.training.patch_size)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

# Train
device = torch.device('cuda')
model = ResidualUnetSE3D(**config.model.to_dict())

trainer = Trainer(
    model=model,
    device=device,
    output_dir=config.output_root,
    learning_rate=config.training.learning_rate,
)

trainer.fit(train_loader, val_loader, epochs=config.training.epochs)
```

### Complete Inference Pipeline

```python
from segmentation_pipeline.models import ResidualUnetSE3D
from segmentation_pipeline.inference import VolumePredictor
from segmentation_pipeline.evaluation import evaluate_directory

# Load model
model = ResidualUnetSE3D(skip_mode='additive')
predictor = VolumePredictor.from_checkpoint(model, 'best.pt')

# Run inference
predictor.predict_batch(
    input_dir='./test_data',
    output_dir='./predictions',
)

# Evaluate
results = evaluate_directory(
    prediction_dir='./predictions',
    ground_truth_dir='./test_data',
    output_path='./evaluation_results.csv',
)

print(f"Mean Dice: {results['dice'].mean():.4f}")
print(f"Mean HD95: {results['hd95'].mean():.2f} mm")
```

---

## Troubleshooting

### CUDA Out of Memory

- Reduce `patch_size` (e.g., from 224×224×256 to 128×128×128)
- Reduce `batch_size` to 1
- Use gradient checkpointing (not implemented, but could be added)

### Poor Segmentation Results

- Check CT window settings match your anatomy
- Ensure labels are binary (0 and 1 only)
- Try different `skip_mode` ('concat' vs 'additive')
- Increase training epochs
- Adjust augmentation parameters

### Shape Mismatch Errors

- Use `use_interpolation_safeguard=True` in model
- Ensure patch size is divisible by 8 (for 3 levels of 2× downsampling)
- Check that image and label have matching shapes

---

## License

This code is provided for research and educational purposes.
