"""
Step 2: Visualize VisDrone Dataset with Annotations
This script loads images and draws bounding boxes to understand the data
"""

import cv2
import numpy as np
from pathlib import Path
import random

# Dataset path
DATASET_ROOT = Path("../dataset/VisDrone-2018")

# Class names and colors
CLASS_NAMES = {
    1: "pedestrian",
    2: "people", 
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor"
}

# Assign different colors to each class
COLORS = {
    1: (255, 100, 100),   # pedestrian - light red
    2: (100, 255, 100),   # people - light green
    3: (100, 100, 255),   # bicycle - light blue
    4: (255, 255, 100),   # car - yellow
    5: (255, 100, 255),   # van - magenta
    6: (100, 255, 255),   # truck - cyan
    7: (200, 150, 100),   # tricycle - brown
    8: (150, 100, 200),   # awning-tricycle - purple
    9: (255, 200, 100),   # bus - orange
    10: (150, 255, 150),  # motor - mint
}

def load_annotations(ann_path):
    """Load annotations from file"""
    boxes = []
    labels = []
    
    if not ann_path.exists():
        return boxes, labels
    
    with open(ann_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            
            x, y, w, h = map(float, parts[:4])
            score = int(parts[4])
            class_id = int(parts[5])
            
            # Filter ignored regions and invalid classes
            if score == 0 or class_id == 0 or class_id == 11:
                continue
            
            # Filter very small boxes
            if w < 5 or h < 5:
                continue
            
            boxes.append([x, y, w, h])
            labels.append(class_id)
    
    return boxes, labels

def visualize_image(img_path, ann_path, output_path=None):
    """Visualize one image with annotations"""
    
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to load image: {img_path}")
        return None
    
    h, w = img.shape[:2]
    
    # Load annotations
    boxes, labels = load_annotations(ann_path)
    
    # Draw each box
    for box, label in zip(boxes, labels):
        x, y, bw, bh = box
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + bw), int(y + bh)
        
        # Get color for this class
        color = COLORS.get(label, (255, 255, 255))
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label_text = CLASS_NAMES.get(label, f"class-{label}")
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(img, label_text, (x1 + 2, y1 - 2), font, font_scale, (0, 0, 0), thickness)
    
    # Add statistics overlay
    stats_text = f"Objects: {len(boxes)} | Image size: {w}x{h}"
    cv2.rectangle(img, (10, 10), (400, 40), (0, 0, 0), -1)
    cv2.putText(img, stats_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(str(output_path), img)
        print(f"✓ Saved visualization: {output_path.name}")
    
    return img, len(boxes)

def main():
    print("="*80)
    print("VISUALIZING VISDRONE DATASET")
    print("="*80)
    
    # Find validation split
    val_split = None
    for split_dir in DATASET_ROOT.iterdir():
        if 'val' in split_dir.name.lower():
            img_dir = split_dir / 'images' if (split_dir / 'images').exists() else split_dir / split_dir.name / 'images'
            ann_dir = split_dir / 'annotations' if (split_dir / 'annotations').exists() else split_dir / split_dir.name / 'annotations'
            
            if img_dir.exists() and ann_dir.exists():
                val_split = {'images': img_dir, 'annotations': ann_dir}
                print(f"\n✓ Found validation split: {split_dir.name}")
                break
    
    if not val_split:
        print("✗ Could not find validation split")
        return
    
    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Get all images
    image_files = sorted(list(val_split['images'].glob('*.jpg')))
    print(f"✓ Found {len(image_files)} images")
    
    # Visualize random sample
    num_samples = 12
    print(f"\nVisualizing {num_samples} random samples...")
    print("-" * 80)
    
    # Select random images
    if len(image_files) > num_samples:
        sample_images = random.sample(image_files, num_samples)
    else:
        sample_images = image_files[:num_samples]
    
    stats = []
    
    for i, img_path in enumerate(sample_images):
        ann_path = val_split['annotations'] / f"{img_path.stem}.txt"
        output_path = output_dir / f"sample_{i+1:02d}_{img_path.name}"
        
        img, num_objects = visualize_image(img_path, ann_path, output_path)
        if img is not None:
            stats.append(num_objects)
    
    print("\n" + "="*80)
    print("VISUALIZATION SUMMARY")
    print("="*80)
    print(f"Saved {len(stats)} visualizations to: {output_dir.absolute()}")
    print(f"\nObject statistics:")
    print(f"  Total objects: {sum(stats)}")
    print(f"  Average per image: {np.mean(stats):.1f}")
    print(f"  Min: {min(stats)}, Max: {max(stats)}")
    print(f"  Median: {np.median(stats):.1f}")
    
    print("\n" + "="*80)
    print("✓ Visualization complete!")
    print(f"  View images in: {output_dir.absolute()}/")
    print("="*80)

if __name__ == '__main__':
    main()
