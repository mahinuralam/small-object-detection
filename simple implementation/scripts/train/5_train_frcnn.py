"""
Step 5: Faster R-CNN with FPN Training Script
Uses pretrained COCO weights and fine-tunes on VisDrone
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import dataset module
import importlib.util
spec = importlib.util.spec_from_file_location("visdrone_dataset", Path(__file__).parent / "4_visdrone_dataset.py")
visdrone_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visdrone_module)
VisDroneDataset = visdrone_module.VisDroneDataset


def get_model(num_classes: int, pretrained: bool = True):
    """
    Load Faster R-CNN ResNet50 FPN model
    
    Args:
        num_classes: Number of classes (10 for VisDrone + background)
        pretrained: Load COCO pretrained weights
    
    Returns:
        model: Faster R-CNN model
    """
    # Load pretrained model on COCO (91 classes)
    if pretrained:
        print("Loading Faster R-CNN ResNet50 FPN with COCO pretrained weights...")
        model = fasterrcnn_resnet50_fpn(
            weights='DEFAULT',  # COCO pretrained
            trainable_backbone_layers=3  # Fine-tune last 3 backbone layers
        )
        print("✓ Loaded COCO pretrained weights")
    else:
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            trainable_backbone_layers=5
        )
    
    # Replace the classifier head for VisDrone classes
    # num_classes = 10 classes + 1 background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    
    print(f"✓ Modified classifier for {num_classes} classes (+ background)")
    
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    """Train for one epoch"""
    model.train()
    
    # Metrics
    total_loss = 0.0
    total_loss_classifier = 0.0
    total_loss_box_reg = 0.0
    total_loss_objectness = 0.0
    total_loss_rpn_box_reg = 0.0
    num_batches = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for images, targets in pbar:
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        
        # Calculate total loss
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += losses.item()
        total_loss_classifier += loss_dict['loss_classifier'].item()
        total_loss_box_reg += loss_dict['loss_box_reg'].item()
        total_loss_objectness += loss_dict['loss_objectness'].item()
        total_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses.item():.4f}",
            'cls': f"{loss_dict['loss_classifier'].item():.4f}",
            'box': f"{loss_dict['loss_box_reg'].item():.4f}"
        })
    
    # Average losses
    avg_losses = {
        'total': total_loss / num_batches,
        'classifier': total_loss_classifier / num_batches,
        'box_reg': total_loss_box_reg / num_batches,
        'objectness': total_loss_objectness / num_batches,
        'rpn_box_reg': total_loss_rpn_box_reg / num_batches
    }
    
    return avg_losses


@torch.no_grad()
def evaluate(model, data_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(data_loader, desc="Validation")
    
    for images, targets in pbar:
        # Move to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass (with targets for loss calculation)
        model.train()  # Temporarily set to train mode to get losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        model.eval()
        
        total_loss += losses.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f"{losses.item():.4f}"})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    print("="*80)
    print("FASTER R-CNN WITH FPN TRAINING ON VISDRONE")
    print("="*80)
    
    # Configuration
    config = {
        'dataset_root': '../dataset/VisDrone-2018',
        'num_classes': 10,  # VisDrone has 10 valid classes
        'batch_size': 4,    # Adjust based on GPU memory
        'num_workers': 4,
        'num_epochs': 50,
        'learning_rate': 0.005,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'lr_step_size': 10,
        'lr_gamma': 0.1,
        'save_interval': 5,
        'patience': 10,  # Early stopping patience
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create output directory
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Load datasets
    print("\n" + "-"*80)
    print("Loading datasets...")
    
    train_dataset = VisDroneDataset(
        root_dir=config['dataset_root'],
        split='train',
        min_size=5
    )
    
    val_dataset = VisDroneDataset(
        root_dir=config['dataset_root'],
        split='val',
        min_size=5
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=VisDroneDataset.collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=VisDroneDataset.collate_fn,
        pin_memory=True
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Create model
    print("\n" + "-"*80)
    model = get_model(num_classes=config['num_classes'], pretrained=True)
    model.to(config['device'])
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['lr_step_size'],
        gamma=config['lr_gamma']
    )
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING START")
    print("="*80)
    
    best_val_loss = float('inf')
    train_history = []
    val_history = []
    patience_counter = 0
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 80)
        
        # Train
        epoch_start = time.time()
        train_losses = train_one_epoch(
            model, optimizer, train_loader, config['device'], epoch
        )
        
        # Validate
        val_loss = evaluate(model, val_loader, config['device'])
        
        # Update learning rate
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_losses['total']:.4f}")
        print(f"    - Classifier: {train_losses['classifier']:.4f}")
        print(f"    - Box Reg: {train_losses['box_reg']:.4f}")
        print(f"    - Objectness: {train_losses['objectness']:.4f}")
        print(f"    - RPN Box: {train_losses['rpn_box_reg']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save history
        train_history.append(train_losses)
        val_history.append(val_loss)
        
        # Save best model and check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, output_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs (patience: {config['patience']})")
            
            # Early stopping
            if patience_counter >= config['patience']:
                print(f"\n⚠ Early stopping triggered after {epoch} epochs")
                print(f"  Best val loss: {best_val_loss:.4f} at epoch {epoch - patience_counter}")
                break
        
        # Save checkpoint
        if epoch % config['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, output_dir / f'checkpoint_epoch_{epoch}.pth')
            print(f"  ✓ Saved checkpoint")
    
    # Save final model
    torch.save({
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': config
    }, output_dir / 'final_model.pth')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")


if __name__ == '__main__':
    main()
