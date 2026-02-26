"""
Step 6: Evaluate Faster R-CNN Model
Calculate mAP, precision, recall on VisDrone test set
"""

import torch
import torchvision
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys
import json
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

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


def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes
    box format: [x1, y1, x2, y2]
    """
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    
    if x2_min < x1_max or y2_min < y1_max:
        return 0.0
    
    intersection = (x2_min - x1_max) * (y2_min - y1_max)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_ap(recalls, precisions):
    """Compute Average Precision (AP)"""
    # Add sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Compute precision envelope
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    
    # Calculate area under curve
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap


def evaluate_model(model, data_loader, device, iou_threshold=0.5, score_threshold=0.05, num_classes=10):
    """
    Evaluate model and calculate mAP
    
    Args:
        model: Trained model
        data_loader: DataLoader for test set
        device: Device to run on
        iou_threshold: IoU threshold for positive match
        score_threshold: Minimum score to consider detection
        num_classes: Number of classes
    
    Returns:
        results: Dictionary with metrics
    """
    model.eval()
    
    # Store all predictions and ground truths
    all_predictions = defaultdict(list)  # {class_id: [(image_id, score, box), ...]}
    all_ground_truths = defaultdict(list)  # {class_id: [(image_id, box), ...]}
    num_gt_per_class = defaultdict(int)
    
    print("\nCollecting predictions and ground truths...")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(data_loader)):
            images = [img.to(device) for img in images]
            
            # Get predictions
            predictions = model(images)
            
            # Process each image in batch
            for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
                image_id = batch_idx * len(images) + img_idx
                
                # Store ground truths
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                
                for box, label in zip(gt_boxes, gt_labels):
                    all_ground_truths[label].append((image_id, box))
                    num_gt_per_class[label] += 1
                
                # Store predictions
                pred_boxes = pred['boxes'].cpu().numpy()
                pred_scores = pred['scores'].cpu().numpy()
                pred_labels = pred['labels'].cpu().numpy()
                
                # Filter by score threshold
                keep = pred_scores >= score_threshold
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]
                
                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    all_predictions[label].append((image_id, score, box))
    
    print(f"\nTotal ground truth boxes: {sum(num_gt_per_class.values())}")
    print(f"Total predictions (score >= {score_threshold}): {sum(len(v) for v in all_predictions.values())}")
    
    # Calculate AP for each class
    print("\nCalculating Average Precision for each class...")
    ap_per_class = {}
    
    for class_id in range(1, num_classes + 1):
        # Get predictions and ground truths for this class
        preds = all_predictions[class_id]
        gts = all_ground_truths[class_id]
        num_gt = num_gt_per_class[class_id]
        
        if num_gt == 0:
            ap_per_class[class_id] = 0.0
            continue
        
        if len(preds) == 0:
            ap_per_class[class_id] = 0.0
            continue
        
        # Sort predictions by score (descending)
        preds.sort(key=lambda x: x[1], reverse=True)
        
        # Track which ground truths have been matched
        gt_matched = defaultdict(lambda: [False] * len([g for g in gts if g[0] == img_id]))
        
        # Build ground truth index
        gt_by_image = defaultdict(list)
        for gt_idx, (img_id, box) in enumerate(gts):
            gt_by_image[img_id].append((gt_idx, box))
        
        # Match predictions to ground truths
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        
        for pred_idx, (img_id, score, pred_box) in enumerate(preds):
            # Get ground truths for this image
            img_gts = gt_by_image[img_id]
            
            if len(img_gts) == 0:
                fp[pred_idx] = 1
                continue
            
            # Find best matching ground truth
            max_iou = 0.0
            max_gt_idx = -1
            
            for gt_idx, gt_box in img_gts:
                iou = calculate_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            # Check if match is good enough and not already matched
            if max_iou >= iou_threshold:
                # Find position in matched array
                gt_position = sum(1 for g_idx, _ in gt_by_image[img_id] if g_idx < max_gt_idx)
                
                if not gt_matched[img_id][gt_position]:
                    tp[pred_idx] = 1
                    gt_matched[img_id][gt_position] = True
                else:
                    fp[pred_idx] = 1
            else:
                fp[pred_idx] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP
        ap = compute_ap(recalls, precisions)
        ap_per_class[class_id] = ap
    
    # Calculate mAP
    valid_aps = [ap for ap in ap_per_class.values() if ap > 0]
    mAP = np.mean(valid_aps) if len(valid_aps) > 0 else 0.0
    
    # Overall precision and recall
    total_tp = sum(len([p for p in all_predictions[c] if any(
        calculate_iou(p[2], gt[1]) >= iou_threshold 
        for gt in all_ground_truths[c] if gt[0] == p[0]
    )]) for c in range(1, num_classes + 1))
    
    total_predictions = sum(len(all_predictions[c]) for c in range(1, num_classes + 1))
    total_gt = sum(num_gt_per_class.values())
    
    precision = total_tp / total_predictions if total_predictions > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    
    results = {
        'mAP': mAP,
        'ap_per_class': ap_per_class,
        'precision': precision,
        'recall': recall,
        'num_gt': total_gt,
        'num_predictions': total_predictions,
        'num_tp': total_tp
    }
    
    return results


def main():
    print("="*80)
    print("FASTER R-CNN EVALUATION ON VISDRONE")
    print("="*80)
    
    # Configuration
    config = {
        'dataset_root': '../dataset/VisDrone-2018',
        'num_classes': 10,
        'batch_size': 4,
        'num_workers': 4,
        'model_path': 'outputs/best_model.pth',
        'iou_threshold': 0.5,
        'score_threshold': 0.05,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load test dataset
    print("\n" + "-"*80)
    print("Loading test dataset...")
    
    test_dataset = VisDroneDataset(
        root_dir=config['dataset_root'],
        split='val',  # Use val for testing
        min_size=5
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=VisDroneDataset.collate_fn,
        pin_memory=True
    )
    
    print(f"✓ Test samples: {len(test_dataset)}")
    print(f"✓ Test batches: {len(test_loader)}")
    
    # Load model
    print("\n" + "-"*80)
    print("Loading model...")
    
    model = get_model(num_classes=config['num_classes'], pretrained=False)
    
    checkpoint = torch.load(config['model_path'], map_location=config['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config['device'])
    
    print(f"✓ Loaded model from {config['model_path']}")
    print(f"  Trained for {checkpoint['epoch']} epochs")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION START")
    print("="*80)
    
    results = evaluate_model(
        model,
        test_loader,
        config['device'],
        iou_threshold=config['iou_threshold'],
        score_threshold=config['score_threshold'],
        num_classes=config['num_classes']
    )
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nOverall Metrics:")
    print(f"  mAP@{config['iou_threshold']}: {results['mAP']:.4f} ({results['mAP']*100:.2f}%)")
    print(f"  Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"  Recall: {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"  True Positives: {results['num_tp']}")
    print(f"  Total Predictions: {results['num_predictions']}")
    print(f"  Total Ground Truths: {results['num_gt']}")
    
    print(f"\nPer-Class AP:")
    class_names = VisDroneDataset.CLASS_NAMES
    for class_id in sorted(results['ap_per_class'].keys()):
        ap = results['ap_per_class'][class_id]
        print(f"  {class_names[class_id]:20s}: {ap:.4f} ({ap*100:.2f}%)")
    
    # Save results
    output_dir = Path('outputs')
    results_file = output_dir / 'evaluation_results.json'
    
    # Convert to JSON-serializable format
    results_json = {
        'mAP': float(results['mAP']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'num_gt': int(results['num_gt']),
        'num_predictions': int(results['num_predictions']),
        'num_tp': int(results['num_tp']),
        'ap_per_class': {class_names[k]: float(v) for k, v in results['ap_per_class'].items()}
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")


if __name__ == '__main__':
    main()
