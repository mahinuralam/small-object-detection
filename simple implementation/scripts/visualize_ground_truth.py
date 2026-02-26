"""
Visualize VisDrone Ground Truth Annotations
"""
import cv2
import numpy as np
from pathlib import Path
import argparse


# VisDrone class names
VISDRONE_CLASSES = {
    0: 'ignored',
    1: 'pedestrian',
    2: 'people',
    3: 'bicycle',
    4: 'car',
    5: 'van',
    6: 'truck',
    7: 'tricycle',
    8: 'awning-tricycle',
    9: 'bus',
    10: 'motor',
    11: 'others'
}

# Colors for each class (BGR)
CLASS_COLORS = {
    0: (128, 128, 128),  # gray - ignored
    1: (0, 255, 0),      # green - pedestrian
    2: (0, 255, 255),    # yellow - people
    3: (255, 0, 0),      # blue - bicycle
    4: (0, 0, 255),      # red - car
    5: (255, 0, 255),    # magenta - van
    6: (255, 255, 0),    # cyan - truck
    7: (128, 0, 255),    # purple - tricycle
    8: (255, 128, 0),    # orange - awning-tricycle
    9: (0, 128, 255),    # light blue - bus
    10: (255, 255, 255), # white - motor
    11: (64, 64, 64)     # dark gray - others
}


def parse_visdrone_annotation(annotation_file):
    """
    Parse VisDrone annotation file
    
    Format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
    
    Returns:
        List of dicts with 'box', 'class_id', 'score', 'truncation', 'occlusion'
    """
    annotations = []
    
    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) < 6:
                continue
            
            x, y, w, h = map(int, parts[:4])
            score = int(parts[4])
            class_id = int(parts[5])
            truncation = int(parts[6]) if len(parts) > 6 else 0
            occlusion = int(parts[7]) if len(parts) > 7 else 0
            
            # Skip ignored regions (class 0 or score 0)
            if class_id == 0 or score == 0:
                continue
            
            annotations.append({
                'box': [x, y, x + w, y + h],  # Convert to [x1, y1, x2, y2]
                'class_id': class_id,
                'score': score,
                'truncation': truncation,
                'occlusion': occlusion
            })
    
    return annotations


def visualize_annotations(image_path, annotation_file, output_path):
    """
    Visualize ground truth annotations on image
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Parse annotations
    annotations = parse_visdrone_annotation(annotation_file)
    
    print(f"Found {len(annotations)} objects in {annotation_file.name}")
    
    # Draw annotations
    for ann in annotations:
        x1, y1, x2, y2 = ann['box']
        class_id = ann['class_id']
        
        # Get color and class name
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        class_name = VISDRONE_CLASSES.get(class_id, 'unknown')
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        label = f"{class_name}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw label background
        cv2.rectangle(
            image,
            (x1, y1 - label_height - baseline - 5),
            (x1 + label_width + 5, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            image,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
    
    # Add title
    title = "Ground Truth Annotations"
    cv2.putText(
        image,
        title,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )
    
    # Save visualization
    cv2.imwrite(str(output_path), image)
    print(f"✓ Saved ground truth visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize VisDrone ground truth annotations')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--annotation', type=str, required=True, help='Path to annotation file')
    parser.add_argument('--output', type=str, required=True, help='Output path')
    
    args = parser.parse_args()
    
    visualize_annotations(
        Path(args.image),
        Path(args.annotation),
        Path(args.output)
    )


if __name__ == "__main__":
    main()
