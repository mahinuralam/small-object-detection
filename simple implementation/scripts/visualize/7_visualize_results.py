"""
Step 7: Visualize Detection Results
Shows side-by-side comparison of ground truth and predictions
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import sys

# Import modules
import importlib.util

# Import dataset
spec = importlib.util.spec_from_file_location("visdrone_dataset", Path(__file__).parent / "4_visdrone_dataset.py")
visdrone_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visdrone_module)
VisDroneDataset = visdrone_module.VisDroneDataset

# Import training utilities
spec = importlib.util.spec_from_file_location("train_frcnn", Path(__file__).parent / "5_train_frcnn.py")
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)
get_model = train_module.get_model


# Color palette for classes (consistent with dataset visualization)
COLORS = [
    (255, 0, 0),      # 1: pedestrian - red
    (0, 255, 0),      # 2: people - green
    (0, 0, 255),      # 3: bicycle - blue
    (255, 255, 0),    # 4: car - yellow
    (255, 0, 255),    # 5: van - magenta
    (0, 255, 255),    # 6: truck - cyan
    (255, 128, 0),    # 7: tricycle - orange
    (128, 0, 255),    # 8: awning-tricycle - purple
    (0, 128, 255),    # 9: bus - sky blue
    (255, 128, 128),  # 10: motor - pink
]


def draw_boxes_on_image(image, boxes, labels, scores=None, title=""):
    """
    Draw bounding boxes on image
    
    Args:
        image: PIL Image or numpy array
        boxes: (N, 4) array of [x1, y1, x2, y2]
        labels: (N,) array of class labels
        scores: (N,) array of confidence scores (optional)
        title: Title text for the image
    
    Returns:
        PIL Image with boxes drawn
    """
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
    else:
        image = image.copy()
    
    draw = ImageDraw.Draw(image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Draw title
    if title:
        draw.text((10, 10), title, fill=(255, 255, 255), font=title_font)
    
    # Draw boxes
    class_names = VisDroneDataset.CLASS_NAMES
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        # Get color for this class
        color = COLORS[(label - 1) % len(COLORS)] if label > 0 else (128, 128, 128)
        
        # Draw box
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw label
        label_text = class_names[label] if label < len(class_names) else f"class_{label}"
        if scores is not None:
            label_text = f"{label_text}: {scores[i]:.2f}"
        
        # Draw label background
        text_bbox = draw.textbbox((x1, y1 - 20), label_text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 20), label_text, fill=(255, 255, 255), font=font)
    
    return image


def visualize_predictions(model, dataset, device, num_samples=12, score_threshold=0.3, output_dir='visualizations_results'):
    """
    Create side-by-side visualizations of ground truth and predictions
    
    Args:
        model: Trained model
        dataset: Dataset to visualize
        device: Device to run on
        num_samples: Number of samples to visualize
        score_threshold: Minimum confidence to show predictions
        output_dir: Directory to save visualizations
    """
    model.eval()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Select random samples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    print(f"Generating {len(indices)} visualizations...")
    print(f"Score threshold: {score_threshold}")
    
    stats = {
        'total_gt': 0,
        'total_pred': 0,
        'total_pred_filtered': 0
    }
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(indices, 1):
            # Get image and target
            image, target = dataset[sample_idx]
            
            # Get ground truth
            gt_boxes = target['boxes'].cpu().numpy()
            gt_labels = target['labels'].cpu().numpy()
            
            stats['total_gt'] += len(gt_boxes)
            
            # Get predictions
            image_tensor = image.to(device)
            predictions = model([image_tensor])[0]
            
            # Filter predictions by score
            pred_boxes = predictions['boxes'].cpu().numpy()
            pred_labels = predictions['labels'].cpu().numpy()
            pred_scores = predictions['scores'].cpu().numpy()
            
            stats['total_pred'] += len(pred_boxes)
            
            # Apply score threshold
            keep = pred_scores >= score_threshold
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]
            
            stats['total_pred_filtered'] += len(pred_boxes)
            
            # Convert image tensor to PIL
            image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
            
            # Create ground truth visualization
            gt_image = draw_boxes_on_image(
                image_pil.copy(),
                gt_boxes,
                gt_labels,
                title=f"Ground Truth ({len(gt_boxes)} objects)"
            )
            
            # Create prediction visualization
            pred_image = draw_boxes_on_image(
                image_pil.copy(),
                pred_boxes,
                pred_labels,
                pred_scores,
                title=f"Predictions ({len(pred_boxes)} detections)"
            )
            
            # Combine side by side
            combined_width = gt_image.width + pred_image.width + 20
            combined_height = max(gt_image.height, pred_image.height)
            combined = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))
            
            combined.paste(gt_image, (0, 0))
            combined.paste(pred_image, (gt_image.width + 20, 0))
            
            # Save
            output_path = output_dir / f"comparison_{idx:03d}.jpg"
            combined.save(output_path, quality=95)
            
            print(f"✓ Saved {output_path.name}: GT={len(gt_boxes)}, Pred={len(pred_boxes)} (filtered from {keep.sum()})")
    
    # Print statistics
    print("\n" + "="*80)
    print("VISUALIZATION SUMMARY")
    print("="*80)
    print(f"Samples visualized: {len(indices)}")
    print(f"Total ground truth boxes: {stats['total_gt']}")
    print(f"Total predictions (all): {stats['total_pred']}")
    print(f"Total predictions (score >= {score_threshold}): {stats['total_pred_filtered']}")
    print(f"Average predictions per image: {stats['total_pred_filtered'] / len(indices):.1f}")
    print(f"\n✓ Visualizations saved to: {output_dir}")


def main():
    print("="*80)
    print("FASTER R-CNN VISUALIZATION")
    print("="*80)
    
    # Configuration
    config = {
        'dataset_root': '../dataset/VisDrone-2018',
        'num_classes': 10,
        'model_path': 'outputs/best_model.pth',  # Using best trained model
        'split': 'val',
        'num_samples': 12,  # Generate 12 visualizations
        'score_threshold': 0.3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load dataset
    print("\n" + "-"*80)
    print("Loading dataset...")
    
    dataset = VisDroneDataset(
        root_dir=config['dataset_root'],
        split=config['split'],
        min_size=5
    )
    
    print(f"✓ Loaded {len(dataset)} images")
    
    # Load model
    print("\n" + "-"*80)
    print("Loading model...")
    
    model_path = Path(config['model_path'])
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("\nAvailable models:")
        outputs_dir = Path('outputs')
        if outputs_dir.exists():
            for model_file in outputs_dir.glob('*.pth'):
                print(f"  - {model_file}")
        else:
            print("  No models found. Train a model first.")
        return
    
    model = get_model(num_classes=config['num_classes'], pretrained=False)
    
    checkpoint = torch.load(model_path, map_location=config['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config['device'])
    model.eval()
    
    print(f"✓ Loaded model from {model_path}")
    print(f"  Trained for {checkpoint['epoch']} epochs")
    
    # Visualize
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    visualize_predictions(
        model,
        dataset,
        config['device'],
        num_samples=config['num_samples'],
        score_threshold=config['score_threshold']
    )
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE ✓")
    print("="*80)


if __name__ == '__main__':
    main()
