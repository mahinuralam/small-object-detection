"""
Fast SR-TOD Evaluation using TorchMetrics
Optimized for large-scale evaluation on VisDrone dataset
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
from tqdm import tqdm
import json
from torchmetrics.detection.mean_ap import MeanAveragePrecision

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


def main():
    """Fast evaluation using TorchMetrics"""
    
    # Configuration
    config = {
        'dataset_root': Path(__file__).parent.parent.parent.parent / 'dataset' / 'VisDrone-2018',
        'model_path': Path(__file__).parent.parent.parent / 'results' / 'outputs_srtod' / 'best_model_srtod.pth',
        'output_dir': Path(__file__).parent.parent.parent / 'results' / 'outputs_srtod',
        'batch_size': 4,
        'num_classes': 10,
        'min_size': 5,
        'conf_threshold': 0.05
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"SR-TOD Fast Evaluation (TorchMetrics)")
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
    
    checkpoint = torch.load(config['model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"  - Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"  - Learnable threshold: {checkpoint['learnable_thresh']:.6f}")
    
    # Initialize TorchMetrics mAP calculator
    print("\nInitializing metrics calculator...")
    metric = MeanAveragePrecision(
        box_format='xyxy',
        iou_type='bbox',
        class_metrics=True
    )
    
    # Class names
    CLASS_NAMES = [
        'background', 'pedestrian', 'people', 'bicycle', 'car', 'van',
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
    ]
    
    # Evaluate
    print("\nRunning evaluation...")
    print("Processing batches (predictions + metrics)...")
    
    total_predictions = 0
    total_gt = 0
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            
            # Get predictions
            predictions = model(images)
            
            # Prepare for torchmetrics format
            preds = []
            targets_tm = []
            
            for pred, target in zip(predictions, targets):
                # Filter by confidence
                keep = pred['scores'] >= config['conf_threshold']
                
                preds.append({
                    'boxes': pred['boxes'][keep].cpu(),
                    'scores': pred['scores'][keep].cpu(),
                    'labels': pred['labels'][keep].cpu()
                })
                
                targets_tm.append({
                    'boxes': target['boxes'].cpu(),
                    'labels': target['labels'].cpu()
                })
                
                total_predictions += len(pred['boxes'][keep])
                total_gt += len(target['boxes'])
            
            # Update metrics
            metric.update(preds, targets_tm)
    
    # Compute final metrics
    print("\nComputing final metrics...")
    results = metric.compute()
    
    # Extract results
    mAP = results['map'].item()
    mAP_50 = results['map_50'].item()
    mAP_75 = results['map_75'].item()
    
    # Per-class AP at IoU=0.5
    per_class_ap = {}
    if 'map_per_class' in results and results['map_per_class'] is not None:
        for i, ap in enumerate(results['map_per_class']):
            if i < len(CLASS_NAMES) - 1:  # Exclude background
                per_class_ap[CLASS_NAMES[i + 1]] = ap.item()
    
    # Calculate precision and recall (approximation)
    precision = results.get('mar_100', torch.tensor(0.0)).item() if 'mar_100' in results else 0.0
    recall = results.get('mar_100', torch.tensor(0.0)).item() if 'mar_100' in results else 0.0
    
    # Print results
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS (SR-TOD)")
    print(f"{'='*80}\n")
    print(f"Overall Metrics:")
    print(f"  mAP (IoU 0.5:0.95): {mAP*100:.2f}%")
    print(f"  mAP@0.5:            {mAP_50*100:.2f}%")
    print(f"  mAP@0.75:           {mAP_75*100:.2f}%")
    print(f"  Total GT boxes:     {total_gt}")
    print(f"  Total predictions:  {total_predictions}")
    
    if per_class_ap:
        print(f"\nPer-Class AP@0.5:")
        for class_name, ap in per_class_ap.items():
            print(f"  {class_name:20s}: {ap*100:6.2f}%")
    
    # Save results
    output = {
        'mAP': float(mAP),
        'mAP_50': float(mAP_50),
        'mAP_75': float(mAP_75),
        'per_class_ap': {k: float(v) for k, v in per_class_ap.items()},
        'num_gt': int(total_gt),
        'num_predictions': int(total_predictions),
        'model_info': {
            'epoch': int(checkpoint['epoch']),
            'val_loss': float(checkpoint['val_loss']),
            'learnable_thresh': float(checkpoint['learnable_thresh'])
        }
    }
    
    output_file = config['output_dir'] / 'evaluation_results_srtod_fast.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    # Comparison with other models
    print("COMPARISON WITH OTHER MODELS:")
    print(f"  Baseline:  mAP@0.5 = 38.02%")
    print(f"  DPA:       mAP@0.5 = 43.44% (+5.42%)")
    print(f"  SR-TOD:    mAP@0.5 = {mAP_50*100:.2f}% ({mAP_50*100 - 38.02:+.2f}% vs baseline)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
