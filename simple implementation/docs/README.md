# Simple Implementation - VisDrone Object Detection

This folder contains a step-by-step implementation for understanding and working with the VisDrone dataset.

## Steps

### Step 1: Understand the Dataset Format
**Script:** `1_understand_dataset.py`

This script analyzes:
- Dataset directory structure
- Available splits (train/val/test)
- Annotation format
- Class distribution
- Key characteristics

**Run:**
```bash
cd "simple implementation"
python 1_understand_dataset.py
```

### Step 2: Visualize Dataset with Annotations
**Script:** `2_visualize_dataset.py`

This script:
- Loads images and annotations
- Draws bounding boxes with class labels
- Saves visualizations to `visualizations/` folder
- Provides statistics

**Run:**
```bash
cd "simple implementation"
python 2_visualize_dataset.py
```

**Output:** Check the `visualizations/` folder for annotated images.

## Dataset Format Summary

**VisDrone Annotation Format:**
```
<bbox_left>, <bbox_top>, <bbox_width>, <bbox_height>, <score>, <object_category>, <truncation>, <occlusion>
```

**Key Points:**
- Box format: `(x, y, w, h)` where `(x, y)` is **TOP-LEFT** corner
- Coordinates are in **pixels** (absolute values)
- 10 valid classes (IDs 1-10)
- Filter out: `score=0`, `class_id=0`, `class_id=11`

**Classes:**
1. pedestrian
2. people
3. bicycle
4. car
5. van
6. truck
7. tricycle
8. awning-tricycle
9. bus
10. motor

## Next Steps
After understanding the dataset format, you can:
1. Build a simple dataloader
2. Implement a basic detection model
3. Train and evaluate
