"""
Train MSFE with Object-Aware Reconstruction
Combines proven MSFE with smart reconstruction that focuses on objects
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

# Import model
from msfe_with_object_reconstruction import MSFEWithObjectReconstruction


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_loss_reconstruction = 0.0
    total_detection_loss = 0.0
    num_loss_components = {'classifier': 0, 'box_reg': 0, 'objectness': 0, 'rpn_box_reg': 0}
    total_loss_components = {'classifier': 0.0, 'box_reg': 0.0, 'objectness': 0.0, 'rpn_box_reg': 0.0}
    
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        
        # Compute total loss
        losses = sum(loss for loss in loss_dict.values())
        
        # Track reconstruction loss separately
        recon_loss = loss_dict.get('loss_reconstruction', torch.tensor(0.0))
        detection_losses = losses - recon_loss
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track losses
        total_loss += losses.item()
        total_loss_reconstruction += recon_loss.item()
        total_detection_loss += detection_losses.item()
        
        # Track individual components
        for key in ['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']:
            if key in loss_dict:
                clean_key = key.replace('loss_', '')
                total_loss_components[clean_key] += loss_dict[key].item()
                num_loss_components[clean_key] += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.item():.4f}',
            'det': f'{detection_losses.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'cls': f'{loss_dict.get("loss_classifier", torch.tensor(0.0)).item():.4f}'
        })
    
    n = len(data_loader)
    avg_losses = {
        'total': total_loss / n,
        'detection': total_detection_loss / n,
        'reconstruction': total_loss_reconstruction / n
    }
    
    # Average component losses
    for key in total_loss_components:
        if num_loss_components[key] > 0:
            avg_losses[key] = total_loss_components[key] / num_loss_components[key]
    
    return avg_losses


@torch.no_grad()
def validate(model, data_loader, device):
    """Validate the model"""
    model.train()  # Keep in train mode to get losses
    
    total_loss = 0.0
    total_loss_reconstruction = 0.0
    total_detection_loss = 0.0
    total_loss_components = {'classifier': 0.0, 'box_reg': 0.0, 'objectness': 0.0, 'rpn_box_reg': 0.0}
    num_loss_components = {'classifier': 0, 'box_reg': 0, 'objectness': 0, 'rpn_box_reg': 0}
    
    pbar = tqdm(data_loader, desc='Validation')
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        
        # Compute total loss
        losses = sum(loss for loss in loss_dict.values())
        
        # Track losses
        recon_loss = loss_dict.get('loss_reconstruction', torch.tensor(0.0))
        detection_losses = losses - recon_loss
        
        total_loss += losses.item()
        total_loss_reconstruction += recon_loss.item()
        total_detection_loss += detection_losses.item()
        
        # Track components
        for key in ['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']:
            if key in loss_dict:
                clean_key = key.replace('loss_', '')
                total_loss_components[clean_key] += loss_dict[key].item()
                num_loss_components[clean_key] += 1
        
        pbar.set_postfix({'val_loss': f'{losses.item():.4f}'})
    
    n = len(data_loader)
    avg_losses = {
        'total': total_loss / n,
        'detection': total_detection_loss / n,
        'reconstruction': total_loss_reconstruction / n
    }
    
    # Average component losses
    for key in total_loss_components:
        if num_loss_components[key] > 0:
            avg_losses[key] = total_loss_components[key] / num_loss_components[key]
    
    return avg_losses


def main():
    # Configuration
    config = {
        'dataset_root': '/home/mahin/Documents/notebook/small-object-detection/dataset/VisDrone-2018',
        'num_classes': 10,
        'batch_size': 2,
        'num_epochs': 50,
        'learning_rate': 0.005,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'reconstruction_weight': 0.3,  # Object-aware, so can be higher
        'object_weight': 10.0,          # Objects 10x more important
        'small_object_boost': 2.0,      # Small objects 2x additional boost
        'patience': 15,
        'T_0': 10,
        'T_mult': 2,
        'eta_min': 1e-6,
        'output_dir': Path('../../results/outputs_msfe_objrecon'),
        'log_file': Path('../../logs/train_msfe_objrecon.log')
    }
    
    # Setup
    config['output_dir'].mkdir(parents=True, exist_ok=True)
    config['log_file'].parent.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Print configuration
    print("="*80)
    print("MSFE WITH OBJECT-AWARE RECONSTRUCTION TRAINING")
    print("="*80)
    print("\nConfiguration:")
    print(f"  Dataset: {config['dataset_root']}")
    print(f"  Classes: {config['num_classes']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Reconstruction weight: {config['reconstruction_weight']}")
    print(f"  Object weight: {config['object_weight']}x")
    print(f"  Small object boost: {config['small_object_boost']}x")
    print(f"  Device: {device}\n")
    
    # Load datasets
    print("Loading datasets...")
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
    print(f"✓ Val: {len(val_dataset)} images\n")
    
    # Create data loaders
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
    
    # Initialize model
    print("Initializing MSFE with Object-Aware Reconstruction...")
    model = MSFEWithObjectReconstruction(
        num_classes=config['num_classes'] + 1,  # +1 for background
        reconstruction_weight=config['reconstruction_weight'],
        object_weight=config['object_weight'],
        small_object_boost=config['small_object_boost'],
        pretrained_backbone=True
    ).to(device)
    
    print("✓ Model initialized")
    print(f"  - MSFE on P2, P3, P4")
    print(f"  - Object-aware reconstruction (objects {config['object_weight']}x, small objects {config['object_weight']*config['small_object_boost']}x)")
    print(f"  - DGFF with object-weighted difference maps\n")
    
    # Optimizer and scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['T_0'],
        T_mult=config['T_mult'],
        eta_min=config['eta_min']
    )
    
    print(f"✓ Optimizer: SGD (lr={config['learning_rate']}, momentum={config['momentum']})")
    print(f"✓ LR Scheduler: CosineAnnealingWarmRestarts (T_0={config['T_0']}, T_mult={config['T_mult']})\n")
    
    # Training loop
    print("="*80)
    print("Starting Training")
    print("="*80 + "\n")
    
    best_val_detection_loss = float('inf')
    epochs_without_improvement = 0
    training_log = []
    
    start_time = time.time()
    
    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start = time.time()
        
        # Train
        train_losses = train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        # Validate
        val_losses = validate(model, val_loader, device)
        
        # Learning rate step
        scheduler.step()
        
        epoch_time = (time.time() - epoch_start) / 60
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{config['num_epochs']} | Time: {epoch_time:.2f}min | LR: {current_lr:.6f}")
        print(f"  Train - Total: {train_losses['total']:.4f} | Det: {train_losses['detection']:.4f} | Recon: {train_losses['reconstruction']:.4f}")
        print(f"    Cls: {train_losses.get('classifier', 0):.4f} | Box: {train_losses.get('box_reg', 0):.4f} | "
              f"RPN Obj: {train_losses.get('objectness', 0):.4f} | RPN Box: {train_losses.get('rpn_box_reg', 0):.4f}")
        print(f"  Val   - Total: {val_losses['total']:.4f} | Det: {val_losses['detection']:.4f} | Recon: {val_losses['reconstruction']:.4f}")
        print(f"    Cls: {val_losses.get('classifier', 0):.4f} | Box: {val_losses.get('box_reg', 0):.4f} | "
              f"RPN Obj: {val_losses.get('objectness', 0):.4f} | RPN Box: {val_losses.get('rpn_box_reg', 0):.4f}")
        
        # Save best model based on detection loss (not total loss)
        if val_losses['detection'] < best_val_detection_loss:
            best_val_detection_loss = val_losses['detection']
            epochs_without_improvement = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_detection_loss': val_losses['detection'],
                'val_total_loss': val_losses['total'],
                'val_reconstruction_loss': val_losses['reconstruction'],
                'learnable_thresh': model.learnable_thresh.item()
            }
            
            torch.save(checkpoint, config['output_dir'] / 'best_model_msfe_objrecon.pth')
            print(f"\n✓ Saved best model (det_loss: {val_losses['detection']:.4f}, "
                  f"total_loss: {val_losses['total']:.4f}, thresh: {model.learnable_thresh.item():.6f})")
        else:
            epochs_without_improvement += 1
            print(f"\n⚠ No improvement for {epochs_without_improvement} epoch(s)")
        
        # Log to training log
        training_log.append({
            'epoch': epoch,
            'train_losses': {k: float(v) for k, v in train_losses.items()},
            'val_losses': {k: float(v) for k, v in val_losses.items()},
            'lr': current_lr,
            'time_minutes': epoch_time
        })
        
        # Early stopping
        if epochs_without_improvement >= config['patience']:
            print(f"\n\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation detection loss: {best_val_detection_loss:.4f}")
            break
        
        # Save periodic checkpoint
        if epoch % 5 == 0:
            torch.save(checkpoint, config['output_dir'] / f'checkpoint_epoch_{epoch}.pth')
    
    total_time = (time.time() - start_time) / 3600
    
    # Final summary
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Total time: {total_time:.2f} hours")
    print(f"Best validation detection loss: {best_val_detection_loss:.4f}")
    print(f"Model saved to: {config['output_dir'] / 'best_model_msfe_objrecon.pth'}")
    print("="*80)
    
    # Save training log
    with open(config['output_dir'] / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"\n✓ Training log saved to: {config['output_dir'] / 'training_log.json'}")


if __name__ == '__main__':
    main()
