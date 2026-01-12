# 3D Bounding Box Regression Model Architecture

## Overview

A deep learning pipeline for localizing anatomical structures in 3D CT/MRI volumes by predicting axis-aligned bounding boxes.

| Property | Value |
|----------|-------|
| **Task** | 3D Bounding Box Regression |
| **Input** | 128 x 128 x 128 voxel volume |
| **Output** | 6 normalized coordinates [z1, y1, x1, z2, y2, x2] |
| **Coordinate Range** | [0, 1] (relative to volume dimensions) |

---

## Model Variants

### 1. Standard CNN (`BBoxRegressor3D`)

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: (1, 128, 128, 128)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  BLOCK 1: Conv3D(1→32) → BatchNorm → ReLU → MaxPool(2)      │
│  Output: (32, 64, 64, 64)                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  BLOCK 2: Conv3D(32→64) → BatchNorm → ReLU → MaxPool(2)     │
│  Output: (64, 32, 32, 32)                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  BLOCK 3: Conv3D(64→128) → BatchNorm → ReLU → MaxPool(2)    │
│  Output: (128, 16, 16, 16)                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  BLOCK 4: Conv3D(128→256) → BatchNorm → ReLU → MaxPool(2)   │
│  Output: (256, 8, 8, 8)                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  BLOCK 5: Conv3D(256→512) → BatchNorm → ReLU → MaxPool(2)   │
│  Output: (512, 4, 4, 4)                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      FLATTEN: 32,768                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  FC HEAD: Linear(32768→512) → ReLU → Dropout(0.3)           │
│           Linear(512→64) → ReLU                             │
│           Linear(64→6) → Sigmoid                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              OUTPUT: (6,) normalized bbox coords            │
└─────────────────────────────────────────────────────────────┘
```

**Parameters:** ~8.5M

---

### 2. Residual CNN (`BBoxRegressorResidual`) - Recommended

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: (1, 128, 128, 128)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  STEM: Conv3D(1→32) → BatchNorm → ReLU                      │
│  Output: (32, 128, 128, 128)                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  RESIDUAL BLOCK 1: (32→32, stride=2)                        │
│  Output: (32, 64, 64, 64)                            ─┐     │
│                                                       │     │
│  ┌──────────────────────────────────────────────┐    │     │
│  │ BN → ReLU → Conv3D → BN → ReLU → Conv3D      │────┼──►+ │
│  └──────────────────────────────────────────────┘    │     │
│                        skip connection ──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  RESIDUAL BLOCK 2: (32→64, stride=2)                        │
│  Output: (64, 32, 32, 32)                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  RESIDUAL BLOCK 3: (64→128, stride=2)                       │
│  Output: (128, 16, 16, 16)                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  RESIDUAL BLOCK 4: (128→256, stride=2)                      │
│  Output: (256, 8, 8, 8)                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  RESIDUAL BLOCK 5: (256→512, stride=2)                      │
│  Output: (512, 4, 4, 4)                                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│       FLATTEN: 512 × 4 × 4 × 4 = 32,768                     │
│       (Preserves spatial info for localization!)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  FC HEAD: Linear(32768→512) → ReLU → Dropout(0.3)           │
│           Linear(512→64) → ReLU                             │
│           Linear(64→6) → Sigmoid                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              OUTPUT: (6,) normalized bbox coords            │
└─────────────────────────────────────────────────────────────┘
```

**Parameters:** ~5.2M

---

## Residual Block Design

Uses **pre-activation** pattern (BN-ReLU-Conv) for improved gradient flow:

```
        Input
          │
          ├────────────────────┐
          │                    │
          ▼                    │
    ┌───────────┐              │
    │ BatchNorm │              │
    └───────────┘              │
          │                    │
          ▼                    │
    ┌───────────┐              │
    │   ReLU    │              │
    └───────────┘              │
          │                    │
          ▼                    │
    ┌───────────┐              │
    │  Conv3D   │              │ Skip Connection
    │ (stride)  │              │ (1x1 Conv if dims change)
    └───────────┘              │
          │                    │
          ▼                    │
    ┌───────────┐              │
    │ BatchNorm │              │
    └───────────┘              │
          │                    │
          ▼                    │
    ┌───────────┐              │
    │   ReLU    │              │
    └───────────┘              │
          │                    │
          ▼                    │
    ┌───────────┐              │
    │  Conv3D   │              │
    └───────────┘              │
          │                    │
          ▼                    │
    ┌───────────┐              │
    │     +     │◄─────────────┘
    └───────────┘
          │
          ▼
       Output
```

---

## Loss Function: Generalized IoU (GIoU)

### Why GIoU over L1/L2 Loss?

| Loss Type | Pros | Cons |
|-----------|------|------|
| **L1/L2** | Simple, stable gradients | Doesn't optimize IoU directly |
| **IoU** | Directly optimizes overlap | Zero gradient when boxes don't overlap |
| **GIoU** | Optimizes overlap + provides gradients for non-overlapping boxes | Slightly more complex |

### GIoU Formula

```
GIoU = IoU - (|C \ (A ∪ B)| / |C|)

Where:
  - A, B = predicted and ground truth boxes
  - C = smallest enclosing box containing both A and B
  - IoU = |A ∩ B| / |A ∪ B|
```

### Visual Explanation

```
┌─────────────────────────────────┐
│ C (enclosing box)               │
│   ┌───────────┐                 │
│   │     A     │    ┌─────────┐  │
│   │  (pred)   │    │    B    │  │
│   │           │    │  (gt)   │  │
│   └───────────┘    └─────────┘  │
│                                 │
│   Penalty area = C \ (A ∪ B)    │
└─────────────────────────────────┘

Loss = 1 - GIoU
```

---

## Data Augmentation Pipeline

Applied during training only:

```
┌──────────────────┐
│   Load Volume    │
│   (128³ uint8)   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Normalize to    │
│     [0, 1]       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────────────┐
│  Random Flip     │────►│ Flip bbox coords:           │
│  (p=0.5/axis)    │     │ new_min = 1 - old_max       │
│  Z, Y, X axes    │     │ new_max = 1 - old_min       │
└────────┬─────────┘     └─────────────────────────────┘
         │
         ▼
┌──────────────────┐
│ Intensity Shift  │
│   ±10% range     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Intensity Scale  │
│  0.9x - 1.1x     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Clamp [0, 1]    │
└────────┬─────────┘
         │
         ▼
      Training
```

---

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Optimizer** | Adam | β1=0.9, β2=0.999 |
| **Learning Rate** | 1e-4 | Initial |
| **LR Scheduler** | ReduceLROnPlateau | Factor=0.5, Patience=3 |
| **Min LR** | 1e-6 | Lower bound |
| **Weight Decay** | 1e-4 | L2 regularization |
| **Dropout** | 0.3 | In FC layers |
| **Batch Size** | 4 | Limited by 3D volume memory |
| **Epochs** | 100 | With early stopping via best checkpoint |

---

## CT Preprocessing

### Windowing (Hounsfield Units → Normalized)

```
         Raw HU Values
              │
              ▼
┌─────────────────────────────┐
│  Apply Window Level/Width   │
│                             │
│  min = level - width/2      │
│  max = level + width/2      │
│                             │
│  Clip to [min, max]         │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Normalize to [0, 255]      │
│  (uint8 for storage)        │
└─────────────────────────────┘
```

### Recommended Window Settings

| Anatomy | Window Level (HU) | Window Width (HU) |
|---------|-------------------|-------------------|
| Bone | 400 | 1800 |
| Sinus/Bone | 600 | 1250 |
| Soft Tissue | 40 | 400 |
| Lung | -600 | 1500 |

---

## Model Comparison

| Feature | Standard | Residual |
|---------|----------|----------|
| **Parameters** | ~8.5M | ~8.7M |
| **Skip Connections** | No | Yes |
| **Pooling** | MaxPool + Flatten | Strided Conv + Flatten |
| **Gradient Flow** | Standard | Improved |
| **Spatial Preservation** | 4x4x4 grid | 4x4x4 grid |

---

## Design Lesson: Why GAP Fails for Localization

### The Problem

Early versions used Global Average Pooling (GAP) in the residual model, which performed **10% worse** than the standard model (0.62 vs 0.72 IoU).

### Classification vs Localization

```
CLASSIFICATION: "Is there a cat?"
┌─────────────┐
│ Feature Map │ ──► GAP ──► "Yes, cat detected"
│  (spatial)  │         (where doesn't matter)
└─────────────┘

LOCALIZATION: "Where is the structure?"
┌─────────────┐
│ Feature Map │ ──► GAP ──► Lost spatial info!
│  (spatial)  │         (can't answer "where")
└─────────────┘
```

### The Fix

Preserve the 4×4×4 spatial grid before the FC layers:

```
WRONG for localization:
  Features (512, 4, 4, 4) → GAP → (512,) → FC

CORRECT for localization:
  Features (512, 4, 4, 4) → Flatten → (32768,) → FC
```

### Key Insight

> **For any task requiring spatial output (localization, detection, regression of coordinates), avoid Global Average Pooling.** The spatial structure of the final feature map encodes "where" information that FC layers need to predict positions.

---

## Output Interpretation

```
Model Output: [z1, y1, x1, z2, y2, x2]
              └─ min ─┘  └─ max ─┘

All values in [0, 1] (normalized coordinates)

To convert to voxel coordinates:
  voxel_coords = normalized_coords × original_shape

Example:
  Output: [0.2, 0.3, 0.25, 0.6, 0.7, 0.75]
  Shape:  [128, 128, 128]

  BBox corners:
    Min: (25.6, 38.4, 32.0)
    Max: (76.8, 89.6, 96.0)
```

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **IoU** | Intersection / Union | 0-1, higher is better |
| **Center Error** | ‖pred_center - gt_center‖ | Euclidean distance (voxels) |
| **Size Error** | ‖pred_size - gt_size‖ | Dimension difference (voxels) |

### IoU Calculation (3D)

```
Intersection Volume = Π (min(pred_max, gt_max) - max(pred_min, gt_min))
                      for each dimension, clamped to 0

Union Volume = pred_volume + gt_volume - intersection

IoU = Intersection Volume / Union Volume
```

---

## Key Takeaways

1. **Residual architecture** provides better gradient flow and training stability
2. **GIoU loss** directly optimizes the evaluation metric with gradients for all cases
3. **Data augmentation** with geometric transforms requires corresponding bbox adjustment
4. **Global average pooling** reduces parameters and prevents overfitting
5. **Normalized coordinates** enable resolution-independent predictions
