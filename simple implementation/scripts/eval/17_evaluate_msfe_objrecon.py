"""
Evaluate MSFE with Object-Aware Reconstruction
"""
import torch
import sys
from pathlib import Path
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

# Import model
from msfe_with_object_reconstruction import MSFEWithObjectReconstruction


def collate_fn(batch):
    return tuple(zip(*batch))


def evaluate_model():
    # Configuration
    config = {
        'dataset_root': '/home/mahin/Documents/notebook/small-object-detection/dataset/VisDrone-2018',
        'num_classes': 10,
        'checkpoint_path': Path('../../results/outputs_msfe_objrecon/best_model_msfe_objrecon.pth'),
        'output_path': Path('../../results/outputs_msfe_objrecon/evaluation_results.json'),
        'batch_size': 4
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("MSFE WITH OBJECT-AWARE RECONSTRUCTION EVALUATION")
    print("="*80)
    print(f"Device: {device}")
    print(f"Checkpoint: {config['checkpoint_path']}\n")
    
    # Load validation dataset
    print("Loading validation dataset...")
    val_dataset = VisDroneDataset(
        root_dir=config['dataset_root'],
        split='val',
        transforms=None
    )
    print(f"✓ Loaded {len(val_dataset)} validation images\n")
    
    # Create data loader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Initialize model
    print("Loading model...")
    model = MSFEWithObjectReconstruction(
        num_classes=config['num_classes'] + 1,
        pretrained_backbone=False
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(config['checkpoint_path'], map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  - Val detection loss: {checkpoint['val_detection_loss']:.4f}")
    print(f"  - Val total loss: {checkpoint['val_total_loss']:.4f}")
    print(f"  - Learnable threshold: {checkpoint['learnable_thresh']:.6f}\n")
    
    # Evaluation
    print("="*80)
    print("Evaluating on validation set...")
    print("="*80 + "\n")
    
    metric = MeanAveragePrecision(
        iou_type="bbox",
        iou_thresholds=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        class_metrics=True
    )
    metric = metric.to(device)
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            
            # Forward pass
            detections = model(images)
            
            # Prepare for metric
            preds = []
            gts = []
            
            for det, target in zip(detections, targets):
                pred_dict = {
                    'boxes': det['boxes'].to(device),
                    'scores': det['scores'].to(device),
                    'labels': det['labels'].to(device)
                }
                
                gt_dict = {
                    'boxes': target['boxes'].to(device),
                    'labels': target['labels'].to(device)
                }
                
                preds.append(pred_dict)
                gts.append(gt_dict)
            
            metric.update(preds, gts)
    
    # Compute metrics
    print("\nComputing metrics...")
    results = metric.compute()
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"mAP (IoU=0.50:0.95): {results['map'].item()*100:.2f}%")
    print(f"mAP@0.50:            {results['map_50'].item()*100:.2f}%")
    print(f"mAP@0.75:            {results['map_75'].item()*100:.2f}%")
    
    # Per-class results
    print("\nPer-Class AP@0.50:")
    class_names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 
                   'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    
    per_class_ap = {}
    for i, name in enumerate(class_names):
        ap = results['map_per_class'][i].item()
        per_class_ap[name] = ap
        print(f"  {name:18s}: {ap*100:5.2f}%")
    
    # Save results
    results_dict = {
        'mAP': results['map'].item(),
        'mAP_50': results['map_50'].item(),
        'mAP_75': results['map_75'].item(),
        'per_class_ap': per_class_ap,
        'checkpoint_epoch': int(checkpoint['epoch']),
        'val_detection_loss': float(checkpoint['val_detection_loss']),
        'learnable_threshold': float(checkpoint['learnable_thresh'])
    }
    
    with open(config['output_path'], 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✓ Results saved to: {config['output_path']}")
    print("="*80)


if __name__ == '__main__':
    evaluate_model()
