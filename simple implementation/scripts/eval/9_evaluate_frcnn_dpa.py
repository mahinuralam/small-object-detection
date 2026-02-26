"""
Evaluate Faster R-CNN with DPA Enhancement
Compare baseline vs DPA-enhanced performance
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import sys

# Import dataset and DPA modules
import importlib.util
spec = importlib.util.spec_from_file_location("visdrone_dataset", Path(__file__).parent / "4_visdrone_dataset.py")
visdrone_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visdrone_module)
VisDroneDataset = visdrone_module.VisDroneDataset

class SimplifiedDPAModule(nn.Module):
    """Simplified Dual-Path Attention - same as in training script"""
    def __init__(self, channels):
        super().__init__()
        
        self.edge_conv3 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.edge_conv5 = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels)
        
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        identity = x
        edge3 = self.edge_conv3(x)
        edge5 = self.edge_conv5(x)
        edge_features = edge3 + edge5
        spatial_weight = self.spatial_att(edge_features)
        edge_features = edge_features * spatial_weight
        channel_weight = self.channel_att(x)
        semantic_features = x * channel_weight
        combined = torch.cat([edge_features, semantic_features], dim=1)
        out = self.fusion(combined)
        return out + identity


class FasterRCNN_with_DPA(nn.Module):
    """Same as in training script"""
    def __init__(self, base_model, fpn_channels=256):
        super().__init__()
        self.base_model = base_model
        
        # Apply enhancement to P3 and P4
        self.enhancers = nn.ModuleDict({
            '0': SimplifiedDPAModule(fpn_channels),
            '1': SimplifiedDPAModule(fpn_channels),
        })
        
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        # Let the base model handle image preprocessing
        original_images = images
        images, targets = self.base_model.transform(images, targets)
        
        # Get features from backbone
        features = self.base_model.backbone(images.tensors)
        
        # Apply enhancement to selected levels
        enhanced_features = {}
        for name, feat in features.items():
            if name in self.enhancers:
                enhanced_features[name] = self.enhancers[name](feat)
            else:
                enhanced_features[name] = feat
        
        # Continue with RPN
        proposals, proposal_losses = self.base_model.rpn(images, enhanced_features, targets)
        
        # ROI heads
        detections, detector_losses = self.base_model.roi_heads(
            enhanced_features, proposals, images.image_sizes, targets
        )
        
        # Transform detections back
        detections = self.base_model.transform.postprocess(
            detections, images.image_sizes, original_images.image_sizes if hasattr(original_images, 'image_sizes') else [(img.shape[-2], img.shape[-1]) for img in original_images]
        )
        
        if self.training:
            losses = {}
            losses.update(proposal_losses)
            losses.update(detector_losses)
            return losses
        
        return detections


def load_model_with_dpa(checkpoint_path, num_classes, device):
    """Load DPA-enhanced model from checkpoint"""
    # Create base model
    base_model = fasterrcnn_resnet50_fpn(weights=None, trainable_backbone_layers=5)
    in_features = base_model.roi_heads.box_predictor.cls_score.in_features
    base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    
    # Wrap with DPA
    model = FasterRCNN_with_DPA(base_model, fpn_channels=256)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
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
    """
    Evaluate model with detailed metrics including object size analysis
    """
    print(f"\nEvaluating with IoU threshold: {iou_threshold}, Confidence threshold: {conf_threshold}")
    
    CLASS_NAMES = [
        'background', 'pedestrian', 'people', 'bicycle', 'car', 'van',
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
    ]
    
    num_classes = len(CLASS_NAMES) - 1  # Exclude background
    
    # Storage for all predictions and ground truths per class
    all_predictions = {i: [] for i in range(1, num_classes + 1)}
    all_ground_truths = {i: [] for i in range(1, num_classes + 1)}
    
    # Size-based metrics
    size_categories = {
        'tiny': (0, 32),      # < 32x32
        'small': (32, 64),    # 32-64
        'medium': (64, 96),   # 64-96
        'large': (96, float('inf'))  # > 96
    }
    size_stats = {cat: {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0} for cat in size_categories}
    
    print("Processing test images...")
    
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
                
                # Store ground truths per class
                for box, label in zip(gt_boxes, gt_labels):
                    if 1 <= label <= num_classes:
                        all_ground_truths[label].append({
                            'box': box,
                            'matched': False
                        })
                        
                        # Calculate object size for size-based analysis
                        w = box[2] - box[0]
                        h = box[3] - box[1]
                        size = max(w, h)
                        
                        for cat_name, (min_size, max_size) in size_categories.items():
                            if min_size <= size < max_size:
                                size_stats[cat_name]['gt'] += 1
                                break
                
                # Match predictions to ground truths per class
                for label in range(1, num_classes + 1):
                    # Get predictions and GTs for this class
                    class_preds = [p for p in all_predictions[label] if not p['matched']]
                    class_gts = [g for g in all_ground_truths[label] if not g['matched']]
                    
                    # Sort predictions by confidence
                    class_preds.sort(key=lambda x: x['score'], reverse=True)
                    
                    # Match predictions to GTs
                    for pred in class_preds:
                        best_iou = 0
                        best_gt_idx = -1
                        
                        for gt_idx, gt in enumerate(class_gts):
                            if gt['matched']:
                                continue
                            iou = calculate_iou(pred['box'], gt['box'])
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_idx
                        
                        if best_iou >= iou_threshold and best_gt_idx >= 0:
                            pred['matched'] = True
                            class_gts[best_gt_idx]['matched'] = True
                            
                            # Size-based TP
                            box = class_gts[best_gt_idx]['box']
                            w = box[2] - box[0]
                            h = box[3] - box[1]
                            size = max(w, h)
                            
                            for cat_name, (min_size, max_size) in size_categories.items():
                                if min_size <= size < max_size:
                                    size_stats[cat_name]['tp'] += 1
                                    break
    
    # Calculate metrics per class
    print("\n" + "="*80)
    print("PER-CLASS RESULTS")
    print("="*80)
    
    aps = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for label in range(1, num_classes + 1):
        preds = all_predictions[label]
        gts = all_ground_truths[label]
        
        if len(gts) == 0:
            print(f"\n{CLASS_NAMES[label]}: No ground truth samples")
            continue
        
        # Sort predictions by score
        preds.sort(key=lambda x: x['score'], reverse=True)
        
        # Calculate precision-recall curve
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        
        for i, pred in enumerate(preds):
            if pred['matched']:
                tp[i] = 1
            else:
                fp[i] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        
        # Calculate AP
        ap = calculate_ap(recalls, precisions)
        aps.append(ap)
        
        # Count TPs, FPs, FNs
        class_tp = int(tp.sum())
        class_fp = int(fp.sum())
        class_fn = len(gts) - class_tp
        
        total_tp += class_tp
        total_fp += class_fp
        total_fn += class_fn
        
        print(f"\n{CLASS_NAMES[label]}:")
        print(f"  AP@{iou_threshold}: {ap*100:.2f}%")
        print(f"  Precision: {precisions[-1]*100:.2f}%")
        print(f"  Recall: {recalls[-1]*100:.2f}%")
        print(f"  TP: {class_tp}, FP: {class_fp}, FN: {class_fn}")
        print(f"  Ground truths: {len(gts)}, Predictions: {len(preds)}")
    
    # Calculate mAP
    mAP = np.mean(aps) if aps else 0.0
    
    # Overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    
    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)
    print(f"mAP@{iou_threshold}: {mAP*100:.2f}%")
    print(f"Overall Precision: {overall_precision*100:.2f}%")
    print(f"Overall Recall: {overall_recall*100:.2f}%")
    print(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    
    # Size-based analysis
    print("\n" + "="*80)
    print("OBJECT SIZE ANALYSIS")
    print("="*80)
    
    for cat_name, (min_size, max_size) in size_categories.items():
        stats = size_stats[cat_name]
        if stats['gt'] > 0:
            recall = stats['tp'] / stats['gt'] * 100
            size_range = f"{min_size}-{max_size}px" if max_size != float('inf') else f">{min_size}px"
            print(f"\n{cat_name.capitalize()} objects ({size_range}):")
            print(f"  Ground truths: {stats['gt']}")
            print(f"  True Positives: {stats['tp']}")
            print(f"  Recall: {recall:.2f}%")
            print(f"  Detection rate: {stats['tp']}/{stats['gt']}")
    
    results = {
        'mAP': mAP,
        'precision': overall_precision,
        'recall': overall_recall,
        'per_class_ap': {CLASS_NAMES[i+1]: aps[i] for i in range(len(aps))},
        'size_analysis': size_stats
    }
    
    return results


def main():
    print("="*80)
    print("FASTER R-CNN WITH DPA - MODEL EVALUATION")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = VisDroneDataset(
        root_dir='../dataset/VisDrone-2018',
        split='val',  # Use validation set as test
        min_size=5
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=VisDroneDataset.collate_fn
    )
    
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # Load DPA-enhanced model
    print("\nLoading DPA-enhanced model...")
    dpa_model_path = Path('outputs_dpa/best_model_dpa.pth')
    
    if not dpa_model_path.exists():
        print(f"❌ DPA model not found at: {dpa_model_path}")
        print("Please train the DPA model first using 8_train_frcnn_with_dpa.py")
        return
    
    model_dpa = load_model_with_dpa(dpa_model_path, num_classes=10, device=device)
    print(f"✓ Loaded DPA model from: {dpa_model_path}")
    
    # Evaluate DPA model
    print("\n" + "="*80)
    print("EVALUATING DPA-ENHANCED MODEL")
    print("="*80)
    
    results_dpa = evaluate_model(model_dpa, test_loader, device, iou_threshold=0.5)
    
    # Save results
    output_file = Path('outputs_dpa/evaluation_results_dpa.json')
    with open(output_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        results_json = {
            'mAP': float(results_dpa['mAP']),
            'precision': float(results_dpa['precision']),
            'recall': float(results_dpa['recall']),
            'per_class_ap': {k: float(v) for k, v in results_dpa['per_class_ap'].items()},
            'size_analysis': {
                cat: {k: int(v) if isinstance(v, (np.integer, int)) else float(v) 
                      for k, v in stats.items()}
                for cat, stats in results_dpa['size_analysis'].items()
            }
        }
        json.dump(results_json, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Load and compare with baseline if available
    baseline_results_path = Path('outputs/evaluation_results.json')
    if baseline_results_path.exists():
        print("\n" + "="*80)
        print("COMPARISON: BASELINE vs DPA")
        print("="*80)
        
        with open(baseline_results_path, 'r') as f:
            results_baseline = json.load(f)
        
        mAP_baseline = results_baseline['mAP']
        mAP_dpa = results_dpa['mAP']
        improvement = (mAP_dpa - mAP_baseline) * 100
        
        print(f"\nmAP Comparison:")
        print(f"  Baseline:     {mAP_baseline*100:.2f}%")
        print(f"  DPA-Enhanced: {mAP_dpa*100:.2f}%")
        print(f"  Improvement:  {improvement:+.2f} percentage points")
        
        if improvement > 0:
            print(f"\n✓ DPA enhancement achieved {improvement:.2f}% mAP improvement!")
        else:
            print(f"\n⚠ DPA did not improve performance in this run")


if __name__ == '__main__':
    main()
