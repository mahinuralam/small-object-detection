"""
Train Faster R-CNN with Cascaded Deformable Dual-Path Attention (CD-DPA)

SOTA Architecture for Small Object Detection on VisDrone

Novel Contributions:
1. CD-DPA: Cascaded Deformable Dual-Path Attention
   - Deformable convolutions for adaptive receptive fields
   - Dual-path attention (edge + semantic)
   - Cascade refinement for iterative enhancement

2. Memory Optimizations (fits 24GB):
   - Mixed precision training (FP16)
   - Gradient checkpointing
   - Gradient accumulation

Target Performance: 48-50% mAP@0.5 (baseline: 38.02%, current best: 43.44%)

Usage:
    cd scripts/train
    python 14_train_cddpa.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time
import sys
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np

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
    """Custom collate function for Faster R-CNN"""
    return tuple(zip(*batch))


def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch, accumulation_steps=1):
    """
    Train for one epoch with mixed precision and gradient accumulation
    
    Args:
        model: FasterRCNN_CDDPA model
        dataloader: Training dataloader
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
        device: cuda device
        epoch: Current epoch number
        accumulation_steps: Steps to accumulate gradients
    """
    model.train()
    
    total_loss = 0
    loss_components = {
        'classifier': 0,
        'box_reg': 0,
        'objectness': 0,
        'rpn_box_reg': 0
    }
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, targets) in enumerate(pbar):
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass with mixed precision
        with autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Scale loss for gradient accumulation
            losses = losses / accumulation_steps
        
        # Backward pass with gradient scaling
        scaler.scale(losses).backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Unscale gradients and clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Accumulate losses (unscaled for logging)
        total_loss += losses.item() * accumulation_steps
        for key in loss_components:
            if f'loss_{key}' in loss_dict:
                loss_components[key] += loss_dict[f'loss_{key}'].item()
        
        # Update progress bar
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    # Average losses
    num_batches = len(dataloader)
    avg_total = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    
    return avg_total, avg_components


@torch.no_grad()
def validate(model, dataloader, device):
    """Validation with mixed precision"""
    model.eval()
    
    total_loss = 0
    loss_components = {
        'classifier': 0,
        'box_reg': 0,
        'objectness': 0,
        'rpn_box_reg': 0
    }
    
    pbar = tqdm(dataloader, desc='Validating')
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Use autocast for inference too
        with autocast():
            model.train()  # Need train mode to get losses
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        total_loss += losses.item()
        for key in loss_components:
            if f'loss_{key}' in loss_dict:
                loss_components[key] += loss_dict[f'loss_{key}'].item()
    
    model.eval()
    
    num_batches = len(dataloader)
    avg_total = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    
    return avg_total, avg_components


def main():
    # Configuration
    config = {
        # Model
        'num_classes': 11,  # 10 classes + background
        'enhance_levels': ['0', '1', '2'],  # P2, P3, P4
        'use_checkpoint': True,  # Gradient checkpointing
        'pretrained_backbone': True,
        
        # Memory optimizations
        'mixed_precision': True,  # FP16 training
        'accumulation_steps': 4,  # Effective batch_size = 4 * 4 = 16
        
        # Training
        'batch_size': 4,  # Per GPU
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'warmup_epochs': 3,
        'early_stopping_patience': 15,
        
        # Paths
        'data_root': Path('../../../dataset/VisDrone-2018'),
        'output_dir': Path('../../results/outputs_cddpa'),
        'log_file': Path('../../logs/train_cddpa.log'),
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create output directory
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    config['log_file'].parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("CASCADED DEFORMABLE DUAL-PATH ATTENTION (CD-DPA) TRAINING")
    print("=" * 80)
    print(f"\n📊 Configuration:")
    print(f"  Model: FasterRCNN with CD-DPA (SOTA)")
    print(f"  Enhanced levels: P2, P3, P4")
    print(f"  Classes: {config['num_classes']}")
    print(f"\n💾 Memory Optimizations:")
    print(f"  Mixed Precision: {config['mixed_precision']} (FP16)")
    print(f"  Gradient Checkpointing: {config['use_checkpoint']}")
    print(f"  Gradient Accumulation: {config['accumulation_steps']} steps")
    print(f"  Effective Batch Size: {config['batch_size'] * config['accumulation_steps']}")
    print(f"\n⚙️ Training:")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Warmup Epochs: {config['warmup_epochs']}")
    print(f"  Early Stopping: {config['early_stopping_patience']} epochs")
    print(f"\n🎯 Target Performance: 48-50% mAP@0.5")
    print(f"  Baseline: 38.02% mAP")
    print(f"  Current Best (SimplifiedDPA): 43.44% mAP")
    print(f"  Expected Improvement: +4-6% mAP")
    print(f"\n📂 Output: {config['output_dir']}")
    print(f"📝 Log: {config['log_file']}")
    print("=" * 80)
    
    # Check GPU
    device = torch.device(config['device'])
    if device.type == 'cuda':
        print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
    
    # Load datasets
    print("\n📁 Loading datasets...")
    train_dataset = VisDroneDataset(
        root_dir=config['data_root'],
        split='train',
        transforms=None  # No data augmentation per your requirement
    )
    val_dataset = VisDroneDataset(
        root_dir=config['data_root'],
        split='val',
        transforms=None
    )
    
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
    
    print(f"✓ Train: {len(train_dataset)} images ({len(train_loader)} batches)")
    print(f"✓ Val: {len(val_dataset)} images ({len(val_loader)} batches)")
    
    # Initialize model
    print("\n🏗️  Initializing CD-DPA model...")
    model = FasterRCNN_CDDPA(
        num_classes=config['num_classes'],
        enhance_levels=config['enhance_levels'],
        use_checkpoint=config['use_checkpoint'],
        pretrained=config['pretrained_backbone']
    )
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'] - config['warmup_epochs'],
        eta_min=1e-6
    )
    
    # Warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=config['warmup_epochs']
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config['mixed_precision'] else None
    
    # Training loop
    print("\n🚀 Starting training...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    patience_counter = 0
    training_log = []
    
    start_time = time.time()
    
    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_components = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            epoch, config['accumulation_steps']
        )
        
        # Validate
        val_loss, val_components = validate(model, val_loader, device)
        
        # Update learning rate
        if epoch <= config['warmup_epochs']:
            warmup_scheduler.step()
        else:
            lr_scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = (time.time() - epoch_start) / 60
        
        # Log
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_components': train_components,
            'val_components': val_components,
            'lr': current_lr,
            'time_minutes': epoch_time
        }
        training_log.append(log_entry)
        
        # Print progress
        print(f"\nEpoch {epoch}/{config['num_epochs']} ({epoch_time:.1f}min)")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }
            torch.save(checkpoint, config['output_dir'] / 'best_model_cddpa.pth')
            print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config['early_stopping_patience']})")
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
            break
        
        # Save training log
        with open(config['output_dir'] / 'training_log.json', 'w') as f:
            json.dump(training_log, f, indent=2)
    
    total_time = (time.time() - start_time) / 3600
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time:.2f} hours")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved: {config['output_dir'] / 'best_model_cddpa.pth'}")
    print(f"Training log: {config['output_dir'] / 'training_log.json'}")
    print("\n🎯 Next Steps:")
    print("  1. Run evaluation: scripts/eval/18_evaluate_cddpa.py")
    print("  2. Expected mAP: 48-50%")
    print("  3. Compare with baseline (38.02%) and SimplifiedDPA (43.44%)")
    print("=" * 80)


if __name__ == '__main__':
    main()
