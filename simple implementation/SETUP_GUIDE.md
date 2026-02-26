# Quick Setup Guide for Small Object Detection Research

## Prerequisites
- NVIDIA GPU (RTX 3090 24GB or similar)
- CUDA 12.4 or compatible
- Linux OS (Ubuntu 20.04+ recommended)
- Python 3.11

## Method 1: Full Environment (Recommended)
```bash
# Install full research environment
pip install -r requirements_full.txt
```

## Method 2: Minimal Installation (Faster)
```bash
# Install only essential packages
pip install -r requirements_minimal.txt
```

## Method 3: Manual Installation (Step-by-step)

### Step 1: Install PyTorch with CUDA
```bash
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
```

### Step 2: Install Computer Vision Libraries
```bash
pip install torchmetrics pytorch-lightning
pip install opencv-python albumentations scikit-image pillow
```

### Step 3: Install Data Science Stack
```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn
```

### Step 4: Install Utilities
```bash
pip install tqdm pyyaml jupyter
```

## Verify Installation

### Check PyTorch + CUDA
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
PyTorch: 2.5.0
CUDA: True
GPU: NVIDIA GeForce RTX 3090
```

### Check Critical Packages
```bash
python -c "import torchvision, torchmetrics, cv2, albumentations, numpy, pandas; print('✓ All packages imported successfully')"
```

### Check GPU Memory
```bash
nvidia-smi
```

## Dataset Setup

### Download VisDrone-2018
```bash
cd dataset/
# Download from: https://github.com/VisDrone/VisDrone-Dataset
# Extract to: dataset/VisDrone-2018/
```

Expected structure:
```
dataset/VisDrone-2018/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── annotations/
├── VisDrone2019-DET-val/
│   ├── images/
│   └── annotations/
└── VisDrone2019-DET-test-dev/
    └── images/
```

## Run Training

### Train SimplifiedDPA (Best Model - 43.44% mAP)
```bash
cd "simple implementation/scripts/train"
python 8_train_frcnn_with_dpa.py
```

### Evaluate Model
```bash
cd "simple implementation/scripts/eval"
python 9_evaluate_frcnn_dpa.py
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch_size in training script (default: 4)
- Enable mixed precision training (FP16)
- Use gradient checkpointing

### Slow Training
- Check if using GPU: `nvidia-smi`
- Increase num_workers in DataLoader
- Use SSD for dataset storage

### Import Errors
- Check Python version: `python --version` (should be 3.11.x)
- Reinstall package: `pip install --force-reinstall <package>`

## Performance Benchmarks

On RTX 3090 24GB:
- SimplifiedDPA: ~20 mins/epoch, 43.44% mAP@0.5 ✅
- Baseline: ~18 mins/epoch, 38.02% mAP@0.5
- CD-DPA: ~20 mins/epoch, 35.29% mAP@0.5 (failed)

## Resources

- Complete environment: RESEARCH_ENVIRONMENT_SETUP.md
- Model comparison: FINAL_RESULTS_SUMMARY.md
- Failed experiment analysis: CDDPA_FAILURE_ANALYSIS.md
