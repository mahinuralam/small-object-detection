"""
Hybrid Detector Training Script
Combines MSFE (P3, P4) + RGD (P2) for maximum performance on VisDrone dataset

Key Improvements:
- Reduced reconstruction weight: 0.2 (vs 1.0 in RGD-only)
- Cosine annealing LR schedule with warmup
- Separate tracking of detection-only metrics
- Hybrid enhancement: P2 (RGD) + P3,P4 (MSFE)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import time
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
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


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train for one epoch with hybrid detector"""
    model.train()
    
    total_loss = 0.0
    total_loss_reconstruction = 0.0
    total_loss_classifier = 0.0
    total_loss_box_reg = 0.0
    total_loss_objectness = 0.0
    total_loss_rpn_box_reg = 0.0
    total_detection_loss = 0.0  # Track detection-only
    num_batches = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        
        # Total loss
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Track losses
        total_loss += losses.item()
        total_loss_reconstruction += loss_dict['loss_reconstruction'].item()
        total_loss_classifier += loss_dict['loss_classifier'].item()
        total_loss_box_reg += loss_dict['loss_box_reg'].item()
        total_loss_objectness += loss_dict['loss_objectness'].item()
        total_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
        
        # Detection-only loss (excluding reconstruction)
        detection_loss = (
            loss_dict['loss_classifier'].item() +
            loss_dict['loss_box_reg'].item() +
            loss_dict['loss_objectness'].item() +
            loss_dict['loss_rpn_box_reg'].item()
        )
        total_detection_loss += detection_loss
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses.item():.4f}",
            'det': f"{detection_loss:.4f}",
            'recon': f"{loss_dict['loss_reconstruction'].item():.4f}",
            'cls': f"{loss_dict['loss_classifier'].item():.4f}"
        })
    
    # Average losses
    return {
        'total_loss': total_loss / num_batches,
        'detection_loss': total_detection_loss / num_batches,
        'loss_reconstruction': total_loss_reconstruction / num_batches,
        'loss_classifier': total_loss_classifier / num_batches,
        'loss_box_reg': total_loss_box_reg / num_batches,
        'loss_objectness': total_loss_objectness / num_batches,
        'loss_rpn_box_reg': total_loss_rpn_box_reg / num_batches
    }


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    
    total_loss = 0.0
    total_loss_reconstruction = 0.0
    total_loss_classifier = 0.0
    total_loss_box_reg = 0.0
    total_loss_objectness = 0.0
    total_loss_rpn_box_reg = 0.0
    total_detection_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(data_loader, desc="Validation")
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Temporarily set to train mode to get losses
        model.train()
        with torch.no_grad():
            loss_dict = model(images, targets)
        model.eval()
        
        # Track losses
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        total_loss_reconstruction += loss_dict['loss_reconstruction'].item()
        total_loss_classifier += loss_dict['loss_classifier'].item()
        total_loss_box_reg += loss_dict['loss_box_reg'].item()
        total_loss_objectness += loss_dict['loss_objectness'].item()
        total_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
        
        detection_loss = (
            loss_dict['loss_classifier'].item() +
            loss_dict['loss_box_reg'].item() +
            loss_dict['loss_objectness'].item() +
            loss_dict['loss_rpn_box_reg'].item()
        )
        total_detection_loss += detection_loss
        num_batches += 1
        
        pbar.set_postfix({'val_loss': f"{losses.item():.4f}"})
    
    # Average losses
    return {
        'total_loss': total_loss / num_batches,
        'detection_loss': total_detection_loss / num_batches,
        'loss_reconstruction': total_loss_reconstruction / num_batches,
        'loss_classifier': total_loss_classifier / num_batches,
        'loss_box_reg': total_loss_box_reg / num_batches,
        'loss_objectness': total_loss_objectness / num_batches,
        'loss_rpn_box_reg': total_loss_rpn_box_reg / num_batches
    }


def main():
    # Configuration
    config = {
        'dataset_root': '/home/mahin/Documents/notebook/small-object-detection/dataset/VisDrone-2018',
        'num_classes': 10,
        'batch_size': 2,  # Reduced due to higher memory usage
        'num_epochs': 25,
        'learning_rate': 0.005,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'reconstruction_weight': 0.2,  # Key improvement
        'patience': 10,
        'T_0': 10,  # Cosine annealing restart period
        'T_mult': 2,
        'eta_min': 1e-6,
        'output_dir': Path('../../results/outputs_hybrid'),
        'log_file': Path('../../logs/train_hybrid.log')
    }
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    config['log_file'].parent.mkdir(parents=True, exist_ok=True)
    
    # Clear log file
    with open(config['log_file'], 'w') as f:
        f.write(f"{'='*80}\n")
        f.write("HYBRID DETECTOR TRAINING (MSFE + RGD)\n")
        f.write(f"{'='*80}\n\n")
        f.write("Configuration:\n")
        for key, value in config.items():
            if key not in ['output_dir', 'log_file']:
                f.write(f"  {key}: {value}\n")
        f.write("\nEnhancements:\n")
        f.write("  P2: RGD (Reconstruction + DGFF) for very tiny objects\n")
        f.write("  P3: MSFE (Multi-scale attention) for tiny objects\n")
        f.write("  P4: MSFE (Multi-scale attention) for small objects\n")
        f.write(f"\nReconstruction weight: {config['reconstruction_weight']} (reduced from 1.0)\n")
        f.write(f"LR Schedule: Cosine annealing with restarts (T_0={config['T_0']})\n")
        f.write(f"\n{'='*80}\n\n")
    
    print(f"{'='*80}")
    print("HYBRID DETECTOR TRAINING")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  Dataset: {config['dataset_root']}")
    print(f"  Classes: {config['num_classes']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Reconstruction weight: {config['reconstruction_weight']}")
    print(f"  Device: {device}")
    
    # Datasets
    print("\nLoading datasets...")
    train_dataset = VisDroneDataset(
        root_dir=config['dataset_root'],
        split='train',
        transforms=None
    )
    
    val_dataset = VisDroneDataset(
        root_dir=config['dataset_root'],
        split='val',
        transforms=None
    )
    
    print(f"✓ Train: {len(train_dataset)} images")
    print(f"✓ Val: {len(val_dataset)} images")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Model
    print("\nInitializing Hybrid Detector...")
    model = HybridDetector(
        num_classes=config['num_classes'] + 1,  # +1 for background
        learnable_thresh=0.0156862,
        reconstruction_weight=config['reconstruction_weight'],
        pretrained_backbone=True
    )
    model.to(device)
    
    print(f"✓ Hybrid model initialized")
    print(f"  - P2: RGD (Reconstruction + DGFF)")
    print(f"  - P3: MSFE (Multi-scale + Attention)")
    print(f"  - P4: MSFE (Multi-scale + Attention)")
    print(f"  - Reconstruction weight: {config['reconstruction_weight']}")
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    # Cosine Annealing LR Scheduler with Warm Restarts
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['T_0'],
        T_mult=config['T_mult'],
        eta_min=config['eta_min']
    )
    
    print(f"\n✓ Optimizer: SGD (lr={config['learning_rate']}, momentum={config['momentum']})")
    print(f"✓ LR Scheduler: CosineAnnealingWarmRestarts (T_0={config['T_0']}, T_mult={config['T_mult']})")
    
    # Training loop
    print(f"\n{'='*80}")
    print("Starting Training")
    print(f"{'='*80}\n")
    
    best_val_detection_loss = float('inf')  # Track detection-only loss
    best_val_total_loss = float('inf')
    epochs_without_improvement = 0
    training_log = []
    
    start_time = time.time()
    
    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device)
        
        # Update LR
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Log
        log_str = (
            f"Epoch {epoch}/{config['num_epochs']} | "
            f"Time: {epoch_time/60:.2f}min | "
            f"LR: {current_lr:.6f}\n"
            f"  Train - Total: {train_metrics['total_loss']:.4f} | "
            f"Det: {train_metrics['detection_loss']:.4f} | "
            f"Recon: {train_metrics['loss_reconstruction']:.4f}\n"
            f"    Cls: {train_metrics['loss_classifier']:.4f} | "
            f"Box: {train_metrics['loss_box_reg']:.4f} | "
            f"RPN Obj: {train_metrics['loss_objectness']:.4f} | "
            f"RPN Box: {train_metrics['loss_rpn_box_reg']:.4f}\n"
            f"  Val   - Total: {val_metrics['total_loss']:.4f} | "
            f"Det: {val_metrics['detection_loss']:.4f} | "
            f"Recon: {val_metrics['loss_reconstruction']:.4f}\n"
            f"    Cls: {val_metrics['loss_classifier']:.4f} | "
            f"Box: {val_metrics['loss_box_reg']:.4f} | "
            f"RPN Obj: {val_metrics['loss_objectness']:.4f} | "
            f"RPN Box: {val_metrics['loss_rpn_box_reg']:.4f}"
        )
        print(log_str)
        print()
        
        # Save to log file
        with open(config['log_file'], 'a') as f:
            f.write(log_str + '\n\n')
        
        # Track training history
        training_log.append({
            'epoch': epoch,
            'train_total_loss': train_metrics['total_loss'],
            'train_detection_loss': train_metrics['detection_loss'],
            'val_total_loss': val_metrics['total_loss'],
            'val_detection_loss': val_metrics['detection_loss'],
            'lr': current_lr,
            'time': epoch_time
        })
        
        # Save best model based on detection loss (not total loss)
        if val_metrics['detection_loss'] < best_val_detection_loss:
            best_val_detection_loss = val_metrics['detection_loss']
            best_val_total_loss = val_metrics['total_loss']
            epochs_without_improvement = 0
            
            # Save model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'val_total_loss': best_val_total_loss,
                'val_detection_loss': best_val_detection_loss,
                'learnable_thresh': model.learnable_thresh.item(),
                'config': config
            }
            torch.save(checkpoint, config['output_dir'] / 'best_model_hybrid.pth')
            print(f"✓ Saved best model (det_loss: {best_val_detection_loss:.4f}, "
                  f"total_loss: {best_val_total_loss:.4f}, "
                  f"thresh: {model.learnable_thresh.item():.6f})\n")
        else:
            epochs_without_improvement += 1
            print(f"⚠ No improvement for {epochs_without_improvement} epoch(s)\n")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'val_total_loss': val_metrics['total_loss'],
                'val_detection_loss': val_metrics['detection_loss'],
                'learnable_thresh': model.learnable_thresh.item(),
                'config': config
            }
            torch.save(checkpoint, config['output_dir'] / f'checkpoint_epoch_{epoch}.pth')
        
        # Early stopping
        if epochs_without_improvement >= config['patience']:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation detection loss: {best_val_detection_loss:.4f}")
            break
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Best validation detection loss: {best_val_detection_loss:.4f}")
    print(f"Best validation total loss: {best_val_total_loss:.4f}")
    print(f"Model saved to: {config['output_dir'] / 'best_model_hybrid.pth'}")
    print(f"{'='*80}\n")
    
    # Save training log
    log_path = config['output_dir'] / 'training_log.json'
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"✓ Training log saved to: {log_path}")


if __name__ == "__main__":
    main()
