"""
Step 1: Understand the VisDrone Dataset Format
This script analyzes the dataset structure and annotation format
"""

import os
from pathlib import Path
import numpy as np

# Dataset path
DATASET_ROOT = Path("../dataset/VisDrone-2018")

print("="*80)
print("VISDRONE DATASET STRUCTURE ANALYSIS")
print("="*80)

# Check dataset structure
print("\n1. DATASET DIRECTORY STRUCTURE:")
print("-" * 80)

if DATASET_ROOT.exists():
    print(f"✓ Dataset found at: {DATASET_ROOT.absolute()}")
    print("\nSubdirectories:")
    for item in sorted(DATASET_ROOT.iterdir()):
        if item.is_dir():
            print(f"  📁 {item.name}/")
            # Check subdirectories
            for subitem in sorted(item.iterdir())[:5]:  # Show first 5 items
                if subitem.is_dir():
                    # Count files in subdirectories
                    if 'images' in subitem.name:
                        img_count = len(list(subitem.glob('*.jpg')))
                        print(f"      📁 {subitem.name}/ ({img_count} images)")
                    elif 'annotations' in subitem.name:
                        ann_count = len(list(subitem.glob('*.txt')))
                        print(f"      📁 {subitem.name}/ ({ann_count} annotations)")
                    else:
                        print(f"      📁 {subitem.name}/")
else:
    print(f"✗ Dataset not found at: {DATASET_ROOT.absolute()}")
    exit(1)

# Find train and val splits
print("\n2. AVAILABLE SPLITS:")
print("-" * 80)

splits = {}
for split_dir in DATASET_ROOT.iterdir():
    if split_dir.is_dir():
        # Check for images directory
        img_dir = split_dir / 'images' if (split_dir / 'images').exists() else split_dir / split_dir.name / 'images'
        ann_dir = split_dir / 'annotations' if (split_dir / 'annotations').exists() else split_dir / split_dir.name / 'annotations'
        
        if img_dir.exists() and ann_dir.exists():
            images = list(img_dir.glob('*.jpg'))
            annotations = list(ann_dir.glob('*.txt'))
            splits[split_dir.name] = {
                'images': img_dir,
                'annotations': ann_dir,
                'num_images': len(images),
                'num_annotations': len(annotations)
            }
            print(f"  ✓ {split_dir.name}:")
            print(f"      Images: {len(images)}")
            print(f"      Annotations: {len(annotations)}")

# Analyze annotation format
print("\n3. ANNOTATION FORMAT ANALYSIS:")
print("-" * 80)
print("\nVisDrone annotation format (from documentation):")
print("  Each line: <bbox_left>, <bbox_top>, <bbox_width>, <bbox_height>, <score>,")
print("             <object_category>, <truncation>, <occlusion>")
print()
print("  Where:")
print("    - bbox: top-left corner (x, y) + width and height")
print("    - score: confidence (0=ignored region, 1=uncertain, 2=certain)")
print("    - object_category: class ID (0=ignored, 1-10=valid classes, 11=other)")
print("    - truncation: 0=none, 1=partial, 2=heavy")
print("    - occlusion: 0=none, 1=partial, 2=heavy")

# Load and analyze sample annotation
if splits:
    split_name = list(splits.keys())[0]
    ann_dir = splits[split_name]['annotations']
    sample_ann = list(ann_dir.glob('*.txt'))[0]
    
    print(f"\n4. SAMPLE ANNOTATION FILE: {sample_ann.name}")
    print("-" * 80)
    
    with open(sample_ann, 'r') as f:
        lines = f.readlines()
    
    print(f"Total objects in this image: {len(lines)}")
    print(f"\nFirst 5 annotations:")
    print()
    
    for i, line in enumerate(lines[:5]):
        parts = line.strip().split(',')
        if len(parts) >= 8:
            x, y, w, h = map(float, parts[:4])
            score, cls_id, trunc, occl = map(int, parts[4:8])
            
            print(f"  Object {i+1}:")
            print(f"    Box: x={x:.1f}, y={y:.1f}, w={w:.1f}, h={h:.1f}")
            print(f"    Category: {cls_id}, Score: {score}, Truncation: {trunc}, Occlusion: {occl}")

# Class statistics
print("\n5. CLASS DISTRIBUTION (from sample):")
print("-" * 80)

if splits:
    split_name = list(splits.keys())[0]
    ann_dir = splits[split_name]['annotations']
    
    # Analyze first 100 annotations
    class_counts = {}
    total_objects = 0
    
    for ann_file in list(ann_dir.glob('*.txt'))[:100]:
        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    cls_id = int(parts[5])
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                    total_objects += 1
    
    # Class names
    class_names = {
        0: "ignored",
        1: "pedestrian",
        2: "people",
        3: "bicycle",
        4: "car",
        5: "van",
        6: "truck",
        7: "tricycle",
        8: "awning-tricycle",
        9: "bus",
        10: "motor",
        11: "others"
    }
    
    print(f"Analyzed {total_objects} objects from 100 images:")
    print()
    for cls_id in sorted(class_counts.keys()):
        count = class_counts[cls_id]
        percentage = 100 * count / total_objects
        cls_name = class_names.get(cls_id, f"unknown-{cls_id}")
        print(f"  Class {cls_id:2d} ({cls_name:15s}): {count:5d} ({percentage:5.1f}%)")

print("\n6. KEY INSIGHTS:")
print("-" * 80)
print("  ✓ Dataset uses (x, y, w, h) format where (x,y) is TOP-LEFT corner")
print("  ✓ Coordinates are in PIXELS (absolute values)")
print("  ✓ Need to filter: score=0, class_id=0, class_id=11")
print("  ✓ 10 valid object classes (1-10)")
print("  ✓ Many small objects (common in aerial/drone imagery)")

print("\n" + "="*80)
print("Analysis complete! Next step: Visualize annotations on images")
print("="*80)
