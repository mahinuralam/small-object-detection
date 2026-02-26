# Refactored Component Naming Convention

## Overview
This document describes the refactored naming convention for components in the small object detection framework. The new names reflect our modified architectures while maintaining clean, descriptive terminology.

## Component Mappings

### 1. Multi-Scale Feature Enhancer (MSFE)
**Previous**: DPA (Dual-Path Attention) Module
**New Module**: `msfe_module.py`
**New Class**: `MultiScaleFeatureEnhancer`

**Description**: Multi-scale feature enhancement using edge detection pathways with spatial and channel attention mechanisms.

**Key Features**:
- Multi-scale depthwise convolutions (3×3 and 5×5 kernels)
- Spatial attention for edge preservation  
- Channel attention (Squeeze-and-Excitation style)
- Memory-efficient design for real-time applications

**File**: `models/enhancements/msfe_module.py`

---

### 2. Feature Reconstructor (FR)
**Previous**: Reconstruction Head
**New Module**: `feature_reconstructor.py`
**New Class**: `FeatureReconstructor`

**Description**: Decoder network that reconstructs RGB images from low-level FPN features (P2).

**Key Features**:
- Progressive upsampling: 256ch → 128ch → 64ch → 3ch (RGB)
- ConvTranspose2d for 2× upsampling at each stage
- Double convolution blocks for feature refinement
- Sigmoid activation for [0,1] pixel value range

**File**: `models/enhancements/feature_reconstructor.py`

---

### 3. Difference-Guided Feature Fusion (DGFF)
**Previous**: DGFE (Difference Map Guided Feature Enhancement)
**New Module**: `dgff_module.py`
**New Class**: `DifferenceGuidedFeatureFusion`

**Description**: Uses reconstruction error maps as spatial priors to guide feature enhancement.

**Key Features**:
- Three-stage processing: Filtration → Spatial Guidance → Channel Reweighting
- Learnable threshold for difference map binarization
- CBAM-style attention (avg + max pooling)
- Selective enhancement of difficult regions

**File**: `models/enhancements/dgff_module.py`

---

### 4. Reconstruction-Guided Detector (RGD)
**Previous**: SR-TOD (Self-Reconstructed Tiny Object Detection)
**New Module**: `rgd_model.py`
**New Class**: `ReconstructionGuidedDetector`

**Description**: Complete detection framework using image reconstruction as guidance for small object detection.

**Architecture Flow**:
```
Input Image → Backbone+FPN → [P2, P3, P4, P5, P6]
                    ↓
                P2 → Feature Reconstructor → Reconstructed Image
                    ↓
            Difference Map = |Reconstructed - Original|
                    ↓
            Enhanced P2 = DGFF(P2, Difference Map)
                    ↓
            [Enhanced P2, P3, P4, P5, P6] → RPN + ROI → Detections
```

**Key Features**:
- Reconstruction-based spatial guidance
- MSE reconstruction loss
- Learnable difference threshold
- End-to-end trainable

**File**: `models/rgd_model.py`

---

### 5. Multi-Scale Enhanced Faster R-CNN
**Previous**: DPA-Enhanced Faster R-CNN  
**New Module**: `msfe_fasterrcnn.py`
**New Class**: `MSFEFasterRCNN`

**Description**: Faster R-CNN with multi-scale feature enhancement applied to selected FPN levels.

**Key Features**:
- Modular enhancement on configurable FPN levels (default: P3, P4)
- Preserves standard Faster R-CNN training pipeline
- Minimal overhead, efficient inference

**File**: `models/msfe_fasterrcnn.py`

---

## Usage Examples

### 1. Multi-Scale Enhanced Model
```python
from models.msfe_fasterrcnn import MSFEFasterRCNN

# Create model with enhancement on P3 and P4 levels
model = MSFEFasterRCNN(
    num_classes=11,
    enhance_levels=['0', '1'],  # P3, P4
    pretrained=True
)

# Training
losses = model(images, targets)

# Inference
model.eval()
detections = model(images)
```

### 2. Reconstruction-Guided Detector
```python
from models.rgd_model import ReconstructionGuidedDetector

# Create RGD model
model = ReconstructionGuidedDetector(
    num_classes=11,
    learnable_thresh=0.0156862,  # 4/255
    pretrained_backbone=True
)

# Training
losses = model(images, targets)
# losses includes: loss_reconstruction, loss_rpn_box_reg, loss_classifier, etc.

# Inference
model.eval()
detections = model(images)

# Get reconstruction outputs for visualization
recon_outputs = model.get_reconstruction_outputs(images)
# Returns: original, reconstructed, difference_map, binary_mask, threshold
```

### 3. Individual Enhancement Modules
```python
from models.enhancements import (
    MultiScaleFeatureEnhancer,
    FeatureReconstructor,
    DifferenceGuidedFeatureFusion
)

# Multi-scale enhancement
msfe = MultiScaleFeatureEnhancer(channels=256, reduction=16)
enhanced_features = msfe(features)

# Feature reconstruction
reconstructor = FeatureReconstructor(in_channels=256, out_channels=3)
reconstructed_image = reconstructor(p2_features)

# Difference-guided fusion
dgff = DifferenceGuidedFeatureFusion(feature_channels=256)
enhanced = dgff(features, difference_map, threshold)
```

---

## Training Scripts

### MSFE Model Training
**File**: `scripts/train/train_msfe.py` (to be created)
```bash
python scripts/train/train_msfe.py \
    --model msfe \
    --epochs 50 \
    --batch-size 4
```

### RGD Model Training  
**File**: `scripts/train/train_rgd.py` (to be created)
```bash
python scripts/train/train_rgd.py \
    --model rgd \
    --epochs 50 \
    --batch-size 4 \
    --learnable-thresh 0.0156862
```

---

## Evaluation Scripts

### MSFE Evaluation
**File**: `scripts/eval/evaluate_msfe.py` (to be created)
```bash
python scripts/eval/evaluate_msfe.py \
    --checkpoint results/outputs_msfe/best_model.pth
```

### RGD Evaluation
**File**: `scripts/eval/evaluate_rgd.py` (to be created)
```bash
python scripts/eval/evaluate_rgd.py \
    --checkpoint results/outputs_rgd/best_model.pth
```

---

## Directory Structure
```
models/
├── msfe_fasterrcnn.py          # Multi-Scale Enhanced Faster R-CNN
├── rgd_model.py                # Reconstruction-Guided Detector
├── enhancements/
│   ├── __init__.py
│   ├── msfe_module.py          # Multi-Scale Feature Enhancer
│   ├── feature_reconstructor.py # Feature Reconstructor  
│   ├── dgff_module.py          # Difference-Guided Feature Fusion
│   ├── dpa_module.py           # (Legacy - deprecated)
│   ├── reconstruction_head.py   # (Legacy - deprecated)
│   └── dgfe_module.py          # (Legacy - deprecated)

scripts/
├── train/
│   ├── train_msfe.py           # Train MSFE model
│   └── train_rgd.py            # Train RGD model
└── eval/
    ├── evaluate_msfe.py        # Evaluate MSFE
    └── evaluate_rgd.py         # Evaluate RGD
```

---

## Backward Compatibility

Legacy imports are maintained for backward compatibility:
- `SimplifiedDPAModule` → Use `MultiScaleFeatureEnhancer` instead
- `ReconstructionHead` → Use `FeatureReconstructor` instead
- `DGFE` → Use `DifferenceGuidedFeatureFusion` instead

**Note**: Legacy imports will be deprecated in future versions. Please migrate to new naming convention.

---

## Model Performance (VisDrone Dataset)

| Model | mAP@0.5 | mAP@0.75 | mAP (0.5:0.95) |
|-------|---------|----------|----------------|
| Baseline Faster R-CNN | 38.02% | - | - |
| MSFE (Multi-Scale Enhanced) | 43.44% | - | - |
| RGD (Reconstruction-Guided) | 38.83% | 21.73% | 22.01% |

---

## Citations & Acknowledgments

While we use descriptive names for our modified architectures, the underlying concepts are inspired by and build upon prior work. When publishing results, appropriate citations should be included in academic papers:

- Multi-scale attention mechanisms
- Feature reconstruction approaches  
- Difference map guidance strategies

Consult your research advisor for proper attribution guidelines.
