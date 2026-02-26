"""
Evaluate Faster R-CNN with Cascaded Deformable Dual-Path Attention (CD-DPA)

Evaluates the SOTA CD-DPA model on VisDrone validation set.

Expected Performance: 48-50% mAP@0.5
Comparison:
  - Baseline: 38.02% mAP
  - SimplifiedDPA: 43.44% mAP
  - CD-DPA: 48-50% mAP (target)

Usage:
    cd scripts/eval
    python 18_evaluate_cddpa.py
"""

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
import sys
from pathlib import Path
from tqdm import tqdm
import json

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.cddpa_model import FasterRCNN_CDDPA

# Import dataset module dynamically
import importlib.util
spec = importlib.util.spec_from_file_location("visdrone_dataset", Path(__file__).parent.parent / "4_visdrone_dataset.py")
visdrone_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visdrone_module)
VisDroneDataset = visdrone_module.VisDroneDataset


def collate_fn(batch):
    """Custom collate function"""
    return tuple(zip(*batch))


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """
    Evaluate model using TorchMetrics
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    # Initialize metric
    metric = MeanAveragePrecision(iou_type='bbox')
    
    print("\nEvaluating on validation set...")
    for images, targets in tqdm(dataloader, desc='Evaluating'):
        # Move to device
        images = [img.to(device) for img in images]
        
        # Inference
        predictions = model(images)
        
        # Move predictions and targets to CPU for metric computation
        preds = []
        targs = []
        for pred, target in zip(predictions, targets):
            preds.append({
                'boxes': pred['boxes'].cpu(),
                'scores': pred['scores'].cpu(),
                'labels': pred['labels'].cpu()
            })
            targs.append({
                'boxes': target['boxes'].cpu(),
                'labels': target['labels'].cpu()
            })
        
        # Update metric
        metric.update(preds, targs)
    
    # Compute final metrics
    print("\nComputing metrics...")
    results = metric.compute()
    
    return results


def main():
    # Configuration
    config = {
        'checkpoint_path': Path('../../results/outputs_cddpa/best_model_cddpa.pth'),
        'data_root': Path('../../../dataset/VisDrone-2018'),
        'batch_size': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_classes': 11
    }
    
    print("=" * 80)
    print("CASCADED DEFORMABLE DUAL-PATH ATTENTION (CD-DPA) EVALUATION")
    print("=" * 80)
    print(f"Device: {config['device']}")
    print(f"Checkpoint: {config['checkpoint_path']}")
    
    # Check GPU
    if config['device'] == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataset = VisDroneDataset(
        root_dir=config['data_root'],
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
    print("\nLoading model...")
    model = FasterRCNN_CDDPA(
        num_classes=config['num_classes'],
        enhance_levels=['0', '1', '2'],
        use_checkpoint=False,  # No checkpointing for inference
        pretrained=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(config['checkpoint_path'], map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config['device'])
    model.eval()
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Training val loss: {checkpoint['val_loss']:.4f}")
    
    # Evaluate
    print("=" * 80)
    results = evaluate_model(model, val_loader, config['device'])
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"mAP (IoU=0.50:0.95): {results['map'].item() * 100:.2f}%")
    print(f"mAP@0.50:            {results['map_50'].item() * 100:.2f}%")
    print(f"mAP@0.75:            {results['map_75'].item() * 100:.2f}%")
    
    # Per-class AP
    if 'map_per_class' in results:
        print(f"\nPer-Class AP@0.50:")
        class_names = [
            'pedestrian', 'people', 'bicycle', 'car', 'van',
            'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
        ]
        for i, (name, ap) in enumerate(zip(class_names, results['map_per_class'])):
            print(f"  {name:20s}: {ap.item() * 100:5.2f}%")
    
    # Comparison
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    baseline_map = 38.02
    simplified_dpa_map = 43.44
    cddpa_map = results['map_50'].item() * 100
    
    print(f"Baseline Faster R-CNN:     {baseline_map:.2f}% mAP@0.5")
    print(f"SimplifiedDPA:             {simplified_dpa_map:.2f}% mAP@0.5 (+{simplified_dpa_map - baseline_map:.2f}%)")
    print(f"CD-DPA (SOTA):             {cddpa_map:.2f}% mAP@0.5 (+{cddpa_map - baseline_map:.2f}%)")
    print(f"\nImprovement over SimplifiedDPA: {cddpa_map - simplified_dpa_map:+.2f}%")
    
    # Achievement check
    print("\n" + "=" * 80)
    if cddpa_map >= 48.0:
        print("🎉 TARGET ACHIEVED! CD-DPA reaches SOTA performance!")
    elif cddpa_map >= 46.0:
        print("✓ Strong performance! Close to SOTA target.")
    elif cddpa_map >= 44.0:
        print("✓ Good improvement over SimplifiedDPA.")
    else:
        print("⚠️  Below expected performance. Consider:")
        print("   - Longer training (more epochs)")
        print("   - Different learning rate")
        print("   - Fine-tuning cascade structure")
    print("=" * 80)
    
    # Save results
    results_dict = {
        'mAP': results['map'].item(),
        'mAP_50': results['map_50'].item(),
        'mAP_75': results['map_75'].item(),
        'comparison': {
            'baseline': baseline_map / 100,
            'simplified_dpa': simplified_dpa_map / 100,
            'cddpa': cddpa_map / 100,
            'improvement_vs_baseline': (cddpa_map - baseline_map) / 100,
            'improvement_vs_simplified_dpa': (cddpa_map - simplified_dpa_map) / 100
        }
    }
    
    output_path = config['checkpoint_path'].parent / 'evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
