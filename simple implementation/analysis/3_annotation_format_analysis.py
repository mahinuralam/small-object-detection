"""
Step 3: Deep Analysis of Annotation Format and Model Selection
This script provides detailed insights to choose the right model architecture
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

DATASET_ROOT = Path("../dataset/VisDrone-2018")

print("="*80)
print("ANNOTATION FORMAT & MODEL SELECTION GUIDE")
print("="*80)

# Find validation split
val_split = None
for split_dir in DATASET_ROOT.iterdir():
    if 'val' in split_dir.name.lower():
        img_dir = split_dir / 'images' if (split_dir / 'images').exists() else split_dir / split_dir.name / 'images'
        ann_dir = split_dir / 'annotations' if (split_dir / 'annotations').exists() else split_dir / split_dir.name / 'annotations'
        if img_dir.exists() and ann_dir.exists():
            val_split = {'images': img_dir, 'annotations': ann_dir}
            break

if not val_split:
    print("Error: Could not find validation split")
    exit(1)

print("\n1. ANNOTATION FORMAT DETAILS")
print("-" * 80)
print("""
VisDrone uses a SIMPLE TEXT FORMAT (one file per image):
  
Format: <x>, <y>, <w>, <h>, <score>, <class>, <truncation>, <occlusion>

Field Details:
  • x, y:        Top-left corner coordinates (in pixels)
  • w, h:        Width and height (in pixels)  
  • score:       0 (ignored), 1 (uncertain), 2 (certain)
  • class:       0 (ignored), 1-10 (valid classes), 11 (others)
  • truncation:  0 (none), 1 (partial), 2 (heavy)
  • occlusion:   0 (none), 1 (partial), 2 (heavy)

Key Characteristics:
  ✓ Box format: (x, y, w, h) - NOT (x1, y1, x2, y2)!
  ✓ Coordinates: Absolute pixels, NOT normalized [0-1]
  ✓ Multiple objects per image (avg ~70)
  ✓ Small objects common (drone/aerial view)
""")

# Analyze dataset characteristics
print("\n2. DATASET CHARACTERISTICS ANALYSIS")
print("-" * 80)

all_boxes = []
all_areas = []
all_aspect_ratios = []
image_sizes = []
objects_per_image = []

# Sample analysis on validation set
ann_files = list(val_split['annotations'].glob('*.txt'))
sample_size = min(200, len(ann_files))

print(f"Analyzing {sample_size} validation images...")

for ann_file in ann_files[:sample_size]:
    boxes = []
    with open(ann_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                x, y, w, h = map(float, parts[:4])
                score, cls_id = int(parts[4]), int(parts[5])
                
                # Filter valid objects
                if score > 0 and 1 <= cls_id <= 10 and w >= 5 and h >= 5:
                    area = w * h
                    aspect = w / h if h > 0 else 1.0
                    
                    all_boxes.append([x, y, w, h])
                    all_areas.append(area)
                    all_aspect_ratios.append(aspect)
                    boxes.append([x, y, w, h])
    
    objects_per_image.append(len(boxes))

all_areas = np.array(all_areas)
all_aspect_ratios = np.array(all_aspect_ratios)
objects_per_image = np.array(objects_per_image)

print(f"\nAnalyzed {len(all_boxes)} valid objects")
print()

# Object size analysis
print("Object Size Distribution:")
tiny = np.sum(all_areas < 32**2)
small = np.sum((all_areas >= 32**2) & (all_areas < 64**2))
medium = np.sum((all_areas >= 64**2) & (all_areas < 96**2))
large = np.sum(all_areas >= 96**2)

total = len(all_areas)
print(f"  Tiny   (< 32²):     {tiny:5d} ({100*tiny/total:5.1f}%)")
print(f"  Small  (32² - 64²): {small:5d} ({100*small/total:5.1f}%)")
print(f"  Medium (64² - 96²): {medium:5d} ({100*medium/total:5.1f}%)")
print(f"  Large  (≥ 96²):     {large:5d} ({100*large/total:5.1f}%)")

print(f"\nObject Statistics:")
print(f"  Area: min={all_areas.min():.0f}, max={all_areas.max():.0f}, mean={all_areas.mean():.1f}")
print(f"  Width: mean={np.mean([b[2] for b in all_boxes]):.1f}")
print(f"  Height: mean={np.mean([b[3] for b in all_boxes]):.1f}")
print(f"  Aspect ratio: mean={all_aspect_ratios.mean():.2f}")

print(f"\nObjects per image:")
print(f"  Mean: {objects_per_image.mean():.1f}")
print(f"  Median: {np.median(objects_per_image):.0f}")
print(f"  Max: {objects_per_image.max()}")

print("\n3. ANNOTATION FORMAT COMPATIBILITY")
print("-" * 80)
print("""
VisDrone Format: (x, y, w, h) with (x,y) = top-left corner

Common Model Formats:
┌────────────────────┬─────────────────────┬──────────────────┐
│ Model              │ Expected Format     │ Compatible?      │
├────────────────────┼─────────────────────┼──────────────────┤
│ YOLO (v5/v8/v11)   │ (x_center, y_center │ ⚠️  Need         │
│                    │  w, h) normalized   │    conversion    │
├────────────────────┼─────────────────────┼──────────────────┤
│ Faster R-CNN       │ (x1, y1, x2, y2)    │ ⚠️  Need         │
│                    │ absolute pixels     │    conversion    │
├────────────────────┼─────────────────────┼──────────────────┤
│ RetinaNet          │ (x1, y1, x2, y2)    │ ⚠️  Need         │
│                    │ absolute pixels     │    conversion    │
├────────────────────┼─────────────────────┼──────────────────┤
│ SSD                │ (x_center, y_center │ ⚠️  Need         │
│                    │  w, h) normalized   │    conversion    │
├────────────────────┼─────────────────────┼──────────────────┤
│ DETR               │ (x_center, y_center │ ⚠️  Need         │
│                    │  w, h) normalized   │    conversion    │
└────────────────────┴─────────────────────┴──────────────────┘

Conversion Required:
  VisDrone (x, y, w, h) → Model Format
  
  For YOLO (normalized center):
    x_center = (x + w/2) / image_width
    y_center = (y + h/2) / image_height
    w_norm = w / image_width
    h_norm = h / image_height
  
  For Faster R-CNN (corner format):
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
""")

print("\n4. RECOMMENDED MODEL ARCHITECTURES")
print("-" * 80)
print("""
Based on dataset characteristics:
  • Many small objects (60% < 64²)
  • High object density (avg 70 per image)
  • Drone/aerial imagery
  • 10 object classes

RECOMMENDED OPTIONS:

1. ✅ YOLOv8/v11 (BEST CHOICE)
   Pros:
   • Fast inference (real-time capable)
   • Good small object detection with proper anchors
   • Easy to train and deploy
   • Built-in data augmentation
   • Native PyTorch
   
   Cons:
   • Needs format conversion (easy)
   • May struggle with very tiny objects
   
   Best for: Real-time applications, edge deployment

2. ✅ Faster R-CNN with FPN
   Pros:
   • Excellent accuracy
   • Multi-scale feature pyramid
   • Good for small objects
   • Well-tested architecture
   
   Cons:
   • Slower inference
   • More complex training
   • Higher memory usage
   
   Best for: High accuracy requirements, offline processing

3. ⚠️  RetinaNet
   Pros:
   • Focal loss helps with class imbalance
   • Good accuracy
   • Single-stage detector
   
   Cons:
   • Slower than YOLO
   • May overfit on small objects
   
   Best for: Balanced speed/accuracy

4. ❌ SSD (NOT RECOMMENDED)
   Cons:
   • Poor small object detection
   • Struggles with high density
   
5. ❌ DETR (NOT RECOMMENDED for this task)
   Cons:
   • Requires lots of training data
   • Slow convergence
   • Computationally expensive

FINAL RECOMMENDATION:
  → Start with YOLOv11 (or YOLOv8)
  → Use pretrained COCO weights
  → Fine-tune on VisDrone
  → Add FPN/PANet for multi-scale features
  → Consider attention modules for small objects
""")

print("\n5. TRAINING STRATEGY")
print("-" * 80)
print("""
Recommended Approach:

Step 1: Data Preparation
  • Convert (x, y, w, h) → YOLO format (x_center, y_center, w, h)
  • Normalize coordinates to [0, 1]
  • Filter invalid annotations (score=0, class=0/11)
  • Split: 6471 train, 548 val

Step 2: Model Configuration
  • Input size: 640×640 (or 416×416 for speed)
  • Batch size: 8-16 (depends on GPU)
  • Anchors: Auto-calculate from dataset
  • Classes: 10 (VisDrone classes)

Step 3: Training
  • Load pretrained weights (COCO)
  • Learning rate: 0.01 → 0.0001 (cosine)
  • Epochs: 50-100 with early stopping
  • Augmentation: mosaic, mixup, rotate, flip
  • Loss: box + objectness (+ classification if needed)

Step 4: Evaluation
  • Metrics: mAP@0.5, mAP@0.5:0.95
  • Focus on small object AP
  • Check per-class performance
""")

print("\n6. KEY TAKEAWAYS")
print("-" * 80)
print("""
✓ VisDrone uses (x, y, w, h) format - easy to understand
✓ Small objects dominate (60% < 64² pixels)
✓ High object density (avg 70 per image)
✓ YOLOv11/v8 is the best choice for this dataset
✓ Simple conversion needed: (x,y,w,h) → (cx,cy,w,h) normalized
✓ Start with pretrained COCO weights
✓ Focus on multi-scale detection
""")

print("\n" + "="*80)
print("Analysis complete! Ready to build the model.")
print("="*80)
