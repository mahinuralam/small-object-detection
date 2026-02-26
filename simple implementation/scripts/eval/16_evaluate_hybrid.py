"""
Evaluate Hybrid Detector (MSFE + RGD)
Fast evaluation using TorchMetrics
"""

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import sys
from pathlib import Path
import json

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

# Import hybrid model
from hybrid_detector import HybridDetector


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))


def evaluate_hybrid():
    """Evaluate trained hybrid detector"""
    
    # Configuration
    config = {
        'dataset_root': '/home/mahin/Documents/notebook/small-object-detection/dataset/VisDrone-2018',
        'checkpoint_path': Path('../../results/outputs_hybrid/best_model_hybrid.pth'),
        'output_path': Path('../../results/outputs_hybrid/evaluation_results_hybrid.json'),
        'num_classes': 10,
        'batch_size': 4
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*80}")
    print("HYBRID DETECTOR EVALUATION (MSFE + RGD)")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Checkpoint: {config['checkpoint_path']}")
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataset = VisDroneDataset(
        root_dir=config['dataset_root'],
        split='val',
        transforms=None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"✓ Loaded {len(val_dataset)} validation images")
    
    # Load model
    print("\nLoading hybrid model...")
    model = HybridDetector(
        num_classes=config['num_classes'] + 1,
        learnable_thresh=0.0156862,
        reconstruction_weight=0.2,
        pretrained_backbone=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(config['checkpoint_path'], map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  - Val detection loss: {checkpoint['val_detection_loss']:.4f}")
    print(f"  - Val total loss: {checkpoint['val_total_loss']:.4f}")
    print(f"  - Learnable threshold: {checkpoint['learnable_thresh']:.6f}")
    
    # Initialize metric
    metric = MeanAveragePrecision(
        box_format='xyxy',
        iou_type='bbox',
        iou_thresholds=None,  # Default: 0.5:0.95:0.05
        class_metrics=True
    ).to(device)
    
    # Evaluation
    print(f"\n{'='*80}")
    print("Evaluating on validation set...")
    print(f"{'='*80}\n")
    
    num_predictions = 0
    num_gt = 0
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            outputs = model(images)
            
            # Prepare for metric (convert to CPU)
            preds = []
            for output in outputs:
                pred = {
                    'boxes': output['boxes'].cpu(),
                    'scores': output['scores'].cpu(),
                    'labels': output['labels'].cpu()
                }
                preds.append(pred)
                num_predictions += len(output['boxes'])
            
            gts = []
            for target in targets:
                gt = {
                    'boxes': target['boxes'].cpu(),
                    'labels': target['labels'].cpu()
                }
                gts.append(gt)
                num_gt += len(target['boxes'])
            
            # Update metric
            metric.update(preds, gts)
    
    # Compute metrics
    print("\nComputing metrics...")
    results = metric.compute()
    
    # Extract results
    map_50 = results['map_50'].item()
    map_75 = results['map_75'].item()
    map_overall = results['map'].item()
    
    # Per-class AP (at IoU=0.50)
    per_class_ap = {}
    if 'map_per_class' in results:
        class_names = VisDroneDataset.CLASS_NAMES[1:11]  # Exclude 'ignored' and 'others'
        for i, ap in enumerate(results['map_per_class']):
            if i < len(class_names):
                per_class_ap[class_names[i]] = ap.item()
    
    # Print results
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"mAP (IoU=0.50:0.95): {map_overall*100:.2f}%")
    print(f"mAP@0.50:            {map_50*100:.2f}%")
    print(f"mAP@0.75:            {map_75*100:.2f}%")
    print(f"\nPredictions: {num_predictions:,}")
    print(f"Ground Truth: {num_gt:,}")
    
    print(f"\nPer-Class AP@0.50:")
    for class_name, ap in per_class_ap.items():
        print(f"  {class_name:20s}: {ap*100:.2f}%")
    
    # Save results
    results_dict = {
        'mAP': map_overall,
        'mAP_50': map_50,
        'mAP_75': map_75,
        'per_class_ap': per_class_ap,
        'num_gt': num_gt,
        'num_predictions': num_predictions,
        'model_info': {
            'epoch': checkpoint['epoch'],
            'val_detection_loss': checkpoint['val_detection_loss'],
            'val_total_loss': checkpoint['val_total_loss'],
            'learnable_thresh': checkpoint['learnable_thresh']
        }
    }
    
    config['output_path'].parent.mkdir(parents=True, exist_ok=True)
    with open(config['output_path'], 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✓ Results saved to: {config['output_path']}")
    print(f"{'='*80}\n")
    
    return results_dict


if __name__ == "__main__":
    evaluate_hybrid()
