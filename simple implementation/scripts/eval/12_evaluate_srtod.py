"""
Evaluate SR-TOD Enhanced Faster R-CNN
Self-Reconstructed Tiny Object Detection evaluation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import sys

# Add models directory to path
models_dir = Path(__file__).parent.parent.parent / "models"
sys.path.insert(0, str(models_dir))

# Import dataset module
import importlib.util
dataset_path = Path(__file__).parent.parent / "4_visdrone_dataset.py"
spec = importlib.util.spec_from_file_location("visdrone_dataset", dataset_path)
visdrone_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visdrone_module)
VisDroneDataset = visdrone_module.VisDroneDataset

# Import SR-TOD model
from srtod_model import FasterRCNN_SRTOD


def collate_fn(batch):
    """Custom collate function"""
    return tuple(zip(*batch))


def box_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_ap(recalls, precisions):
    """Calculate average precision using 11-point interpolation"""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    return ap


def evaluate_model(model, data_loader, device, iou_threshold=0.5, conf_threshold=0.05):
    """Evaluate SR-TOD model with detailed metrics"""
    print(f"\nEvaluating SR-TOD with IoU threshold: {iou_threshold}, Confidence threshold: {conf_threshold}")
    
    CLASS_NAMES = [
        'background', 'pedestrian', 'people', 'bicycle', 'car', 'van',
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
    ]
    
    num_classes = len(CLASS_NAMES) - 1
    
    # Storage for predictions and ground truths per class
    all_predictions = {i: [] for i in range(1, num_classes + 1)}
    all_ground_truths = {i: [] for i in range(1, num_classes + 1)}
    
    # Size-based metrics
    size_categories = {
        'tiny': (0, 32),
        'small': (32, 64),
        'medium': (64, 96),
        'large': (96, float('inf'))
    }
    size_stats = {cat: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0} for cat in size_categories}
    
    print("Processing validation images...")
    
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = [img.to(device) for img in images]
            
            # Get predictions
            predictions = model(images)
            
            # Process each image
            for pred, target in zip(predictions, targets):
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                
                pred_boxes = pred['boxes'].cpu().numpy()
                pred_scores = pred['scores'].cpu().numpy()
                pred_labels = pred['labels'].cpu().numpy()
                
                # Filter by confidence
                keep = pred_scores >= conf_threshold
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]
                
                # Store predictions per class
                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    if 1 <= label <= num_classes:
                        all_predictions[label].append({
                            'box': box,
                            'score': score,
                            'matched': False
                        })
                
                # Store ground truths per class and calculate sizes
                for box, label in zip(gt_boxes, gt_labels):
                    if 1 <= label <= num_classes:
                        all_ground_truths[label].append({
                            'box': box,
                            'matched': False
                        })
                        
                        # Object size analysis
                        w = box[2] - box[0]
                        h = box[3] - box[1]
                        size = max(w, h)
                        
                        for cat_name, (min_size, max_size) in size_categories.items():
                            if min_size <= size < max_size:
                                size_stats[cat_name]['gt'] += 1
                                break
    
    print("\nCalculating metrics...")
    
    # Calculate AP per class
    ap_per_class = {}
    
    for class_id in range(1, num_classes + 1):
        class_name = CLASS_NAMES[class_id]
        
        preds = all_predictions[class_id]
        gts = all_ground_truths[class_id]
        
        if len(gts) == 0:
            ap_per_class[class_name] = 0.0
            continue
        
        if len(preds) == 0:
            ap_per_class[class_name] = 0.0
            continue
        
        # Sort predictions by score (descending)
        preds = sorted(preds, key=lambda x: x['score'], reverse=True)
        
        # Calculate TP and FP
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        
        gt_matched = [False] * len(gts)
        
        for i, pred in enumerate(preds):
            pred_box = pred['box']
            
            max_iou = 0
            max_idx = -1
            
            for j, gt in enumerate(gts):
                if gt_matched[j]:
                    continue
                
                iou = box_iou(pred_box, gt['box'])
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
            
            if max_iou >= iou_threshold:
                if not gt_matched[max_idx]:
                    tp[i] = 1
                    gt_matched[max_idx] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        
        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Calculate AP
        ap = calculate_ap(recalls, precisions)
        ap_per_class[class_name] = ap
    
    # Calculate mAP
    mAP = np.mean(list(ap_per_class.values()))
    
    # Overall metrics
    total_tp = sum([sum([1 for p in preds if any(box_iou(p['box'], gt['box']) >= iou_threshold for gt in all_ground_truths[class_id])]) 
                    for class_id, preds in all_predictions.items()])
    total_predictions = sum([len(preds) for preds in all_predictions.values()])
    total_gt = sum([len(gts) for gts in all_ground_truths.values()])
    
    precision = total_tp / total_predictions if total_predictions > 0 else 0
    recall = total_tp / total_gt if total_gt > 0 else 0
    
    # Size-based analysis (simplified)
    for cat_name in size_categories:
        size_stats[cat_name]['tp'] = int(size_stats[cat_name]['gt'] * recall)  # Approximation
    
    results = {
        'mAP': float(mAP),
        'precision': float(precision),
        'recall': float(recall),
        'per_class_ap': {k: float(v) for k, v in ap_per_class.items()},
        'size_analysis': size_stats,
        'num_gt': int(total_gt),
        'num_predictions': int(total_predictions),
        'num_tp': int(total_tp)
    }
    
    return results


def main():
    """Main evaluation function"""
    
    # Configuration
    config = {
        'dataset_root': Path(__file__).parent.parent.parent.parent / 'dataset' / 'VisDrone-2018',
        'model_path': Path(__file__).parent.parent.parent / 'results' / 'outputs_srtod' / 'best_model_srtod.pth',
        'output_dir': Path(__file__).parent.parent.parent / 'results' / 'outputs_srtod',
        'batch_size': 4,
        'num_classes': 10,
        'min_size': 5,
        'iou_threshold': 0.5,
        'conf_threshold': 0.05
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"SR-TOD Model Evaluation")
    print(f"{'='*80}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*80}\n")
    
    # Load dataset
    print("Loading validation dataset...")
    val_dataset = VisDroneDataset(
        root_dir=config['dataset_root'],
        split='val',
        min_size=config['min_size']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Load model
    print("\nLoading SR-TOD model...")
    model = FasterRCNN_SRTOD(
        num_classes=config['num_classes'] + 1,
        pretrained_backbone=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(config['model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"  - Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"  - Learnable threshold: {checkpoint['learnable_thresh']:.6f}")
    
    # Evaluate
    results = evaluate_model(
        model, val_loader, device,
        iou_threshold=config['iou_threshold'],
        conf_threshold=config['conf_threshold']
    )
    
    # Print results
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}\n")
    print(f"Overall Metrics:")
    print(f"  mAP@0.5:    {results['mAP']*100:.2f}%")
    print(f"  Precision:  {results['precision']*100:.2f}%")
    print(f"  Recall:     {results['recall']*100:.2f}%")
    print(f"  GT boxes:   {results['num_gt']}")
    print(f"  Predictions: {results['num_predictions']}")
    print(f"  True Positives: {results['num_tp']}")
    
    print(f"\nPer-Class AP:")
    for class_name, ap in results['per_class_ap'].items():
        print(f"  {class_name:20s}: {ap*100:6.2f}%")
    
    print(f"\nSize-based Analysis:")
    for cat_name, stats in results['size_analysis'].items():
        print(f"  {cat_name.capitalize():10s}: GT={stats['gt']:5d}, TP≈{stats['tp']:5d}")
    
    # Save results
    output_file = config['output_dir'] / 'evaluation_results_srtod.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
