"""
SR-TOD Enhanced Faster R-CNN Training Script
Self-Reconstructed Tiny Object Detection for VisDrone dataset
Based on: https://github.com/Hiyuur/SR-TOD (ECCV 2024)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

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
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train for one epoch with SR-TOD"""
    model.train()
    
    total_loss = 0.0
    total_loss_reconstruction = 0.0
    total_loss_classifier = 0.0
    total_loss_box_reg = 0.0
    total_loss_objectness = 0.0
    total_loss_rpn_box_reg = 0.0
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
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses.item():.4f}",
            'recon': f"{loss_dict['loss_reconstruction'].item():.4f}",
            'cls': f"{loss_dict['loss_classifier'].item():.4f}",
            'box': f"{loss_dict['loss_box_reg'].item():.4f}"
        })
    
    # Average losses
    avg_loss = total_loss / num_batches
    avg_loss_reconstruction = total_loss_reconstruction / num_batches
    avg_loss_classifier = total_loss_classifier / num_batches
    avg_loss_box_reg = total_loss_box_reg / num_batches
    avg_loss_objectness = total_loss_objectness / num_batches
    avg_loss_rpn_box_reg = total_loss_rpn_box_reg / num_batches
    
    return {
        'total_loss': avg_loss,
        'loss_reconstruction': avg_loss_reconstruction,
        'loss_classifier': avg_loss_classifier,
        'loss_box_reg': avg_loss_box_reg,
        'loss_objectness': avg_loss_objectness,
        'loss_rpn_box_reg': avg_loss_rpn_box_reg
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
    num_batches = 0
    
    pbar = tqdm(data_loader, desc="Validation")
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Temporarily set to train mode to get losses (FasterRCNN only returns losses in train mode)
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
        num_batches += 1
        
        pbar.set_postfix({'val_loss': f"{losses.item():.4f}"})
    
    # Average losses
    avg_loss = total_loss / num_batches
    avg_loss_reconstruction = total_loss_reconstruction / num_batches
    avg_loss_classifier = total_loss_classifier / num_batches
    avg_loss_box_reg = total_loss_box_reg / num_batches
    avg_loss_objectness = total_loss_objectness / num_batches
    avg_loss_rpn_box_reg = total_loss_rpn_box_reg / num_batches
    
    return {
        'total_loss': avg_loss,
        'loss_reconstruction': avg_loss_reconstruction,
        'loss_classifier': avg_loss_classifier,
        'loss_box_reg': avg_loss_box_reg,
        'loss_objectness': avg_loss_objectness,
        'loss_rpn_box_reg': avg_loss_rpn_box_reg
    }


def main():
    """Main training function"""
    
    # Configuration
    config = {
        'dataset_root': Path(__file__).parent.parent.parent.parent / 'dataset' / 'VisDrone-2018',
        'batch_size': 4,
        'num_epochs': 50,
        'learning_rate': 0.005,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'step_size': 10,
        'gamma': 0.1,
        'num_classes': 10,  # VisDrone has 10 classes
        'learnable_thresh': 0.0156862,  # 4/255
        'patience': 10,  # Early stopping patience
        'output_dir': Path(__file__).parent.parent.parent / 'results' / 'outputs_srtod',
        'log_file': Path(__file__).parent.parent.parent / 'logs' / 'train_srtod.log',
        'min_size': 5  # Minimum box size to keep
    }
    
    # Create output directories
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    config['log_file'].parent.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"SR-TOD Enhanced Faster R-CNN Training")
    print(f"{'='*80}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*80}\n")
    
    # Dataset
    print("Loading datasets...")
    train_dataset = VisDroneDataset(root_dir=config['dataset_root'], split='train', min_size=config['min_size'])
    val_dataset = VisDroneDataset(root_dir=config['dataset_root'], split='val', min_size=config['min_size'])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
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
    print("\nInitializing SR-TOD model...")
    model = FasterRCNN_SRTOD(
        num_classes=config['num_classes'] + 1,  # +1 for background
        learnable_thresh=config['learnable_thresh'],
        pretrained_backbone=True
    )
    model.to(device)
    
    print(f"✓ Model initialized")
    print(f"  - Reconstruction Head: P2 (256ch) → RGB (3ch)")
    print(f"  - DGFE: Channel attention + spatial guidance")
    print(f"  - Learnable threshold: {config['learnable_thresh']:.6f} ({config['learnable_thresh'] * 255:.2f}/255)")
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    # LR Scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['step_size'],
        gamma=config['gamma']
    )
    
    # Training loop
    print(f"\n{'='*80}")
    print("Starting Training")
    print(f"{'='*80}\n")
    
    best_val_loss = float('inf')
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
            f"  Train - Loss: {train_metrics['total_loss']:.4f} | "
            f"Recon: {train_metrics['loss_reconstruction']:.4f} | "
            f"Cls: {train_metrics['loss_classifier']:.4f} | "
            f"Box: {train_metrics['loss_box_reg']:.4f} | "
            f"RPN Obj: {train_metrics['loss_objectness']:.4f} | "
            f"RPN Box: {train_metrics['loss_rpn_box_reg']:.4f}\n"
            f"  Val   - Loss: {val_metrics['total_loss']:.4f} | "
            f"Recon: {val_metrics['loss_reconstruction']:.4f} | "
            f"Cls: {val_metrics['loss_classifier']:.4f} | "
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
            'train_loss': train_metrics['total_loss'],
            'val_loss': val_metrics['total_loss'],
            'lr': current_lr,
            'time': epoch_time
        })
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            epochs_without_improvement = 0
            
            # Save model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'val_loss': best_val_loss,
                'learnable_thresh': model.learnable_thresh.item(),
                'config': config
            }
            torch.save(checkpoint, config['output_dir'] / 'best_model_srtod.pth')
            print(f"✓ Saved best model (val_loss: {best_val_loss:.4f}, thresh: {model.learnable_thresh.item():.6f})\n")
        else:
            epochs_without_improvement += 1
            print(f"⚠ No improvement for {epochs_without_improvement} epoch(s)\n")
        
        # Early stopping
        if epochs_without_improvement >= config['patience']:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
    
    # Training complete
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {config['output_dir'] / 'best_model_srtod.pth'}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
