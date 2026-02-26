"""
Memory-efficient DPA-like enhancement for Faster R-CNN
Simplified version without full attention to reduce memory usage
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Import dataset module
import importlib.util
spec = importlib.util.spec_from_file_location("visdrone_dataset", Path(__file__).parent / "4_visdrone_dataset.py")
visdrone_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visdrone_module)
VisDroneDataset = visdrone_module.VisDroneDataset


class SimplifiedDPAModule(nn.Module):
    """
    Simplified Dual-Path Attention without full cross-attention
    Memory efficient version focusing on:
    1. Multi-scale processing
    2. Spatial attention (edge preservation)
    3. Channel attention (semantic propagation)
    """
    def __init__(self, channels):
        super().__init__()
        
        # Multi-scale depthwise convolutions (edge branch)
        self.edge_conv3 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.edge_conv5 = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels)
        
        # Spatial attention for edge preservation
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        
        # Channel attention for semantic (SE block)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        identity = x
        
        # Edge branch: multi-scale
        edge3 = self.edge_conv3(x)
        edge5 = self.edge_conv5(x)
        edge_features = edge3 + edge5
        
        # Apply spatial attention
        spatial_weight = self.spatial_att(edge_features)
        edge_features = edge_features * spatial_weight
        
        # Semantic branch: channel attention
        channel_weight = self.channel_att(x)
        semantic_features = x * channel_weight
        
        # Fuse both branches
        combined = torch.cat([edge_features, semantic_features], dim=1)
        out = self.fusion(combined)
        
        return out + identity


class FasterRCNN_with_SimpleDPA(nn.Module):
    """Faster R-CNN with memory-efficient enhancement"""
    def __init__(self, base_model, fpn_channels=256):
        super().__init__()
        self.base_model = base_model
        
        # Apply enhancement only to P3 and P4 (critical for small objects)
        self.enhancers = nn.ModuleDict({
            '0': SimplifiedDPAModule(fpn_channels),  # P3
            '1': SimplifiedDPAModule(fpn_channels),  # P4
        })
        
        print(f"✓ Integrated simplified DPA enhancement on P3, P4")
        print(f"  (Memory optimized: No full attention, multi-scale + spatial/channel attention)")
        
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        # Let base model handle preprocessing
        original_images = images
        images, targets = self.base_model.transform(images, targets)
        
        # Get backbone features
        features = self.base_model.backbone(images.tensors)
        
        # Apply enhancement to selected levels
        enhanced_features = {}
        for name, feat in features.items():
            if name in self.enhancers:
                enhanced_features[name] = self.enhancers[name](feat)
            else:
                enhanced_features[name] = feat
        
        # RPN
        proposals, proposal_losses = self.base_model.rpn(images, enhanced_features, targets)
        
        # ROI heads
        detections, detector_losses = self.base_model.roi_heads(
            enhanced_features, proposals, images.image_sizes, targets
        )
        
        # Postprocess
        detections = self.base_model.transform.postprocess(
            detections, images.image_sizes, 
            original_images.image_sizes if hasattr(original_images, 'image_sizes') 
            else [(img.shape[-2], img.shape[-1]) for img in original_images]
        )
        
        if self.training:
            losses = {}
            losses.update(proposal_losses)
            losses.update(detector_losses)
            return losses
        
        return detections


def get_model_with_enhancement(num_classes: int, pretrained: bool = True):
    """Load Faster R-CNN with enhancement"""
    if pretrained:
        print("Loading Faster R-CNN ResNet50 FPN with COCO pretrained weights...")
        base_model = fasterrcnn_resnet50_fpn(
            weights='DEFAULT',
            trainable_backbone_layers=3
        )
        print("✓ Loaded COCO pretrained weights")
    else:
        base_model = fasterrcnn_resnet50_fpn(
            weights=None,
            trainable_backbone_layers=5
        )
    
    # Modify classifier
    in_features = base_model.roi_heads.box_predictor.cls_score.in_features
    base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    print(f"✓ Modified classifier for {num_classes} classes (+ background)")
    
    # Wrap with enhancement
    model = FasterRCNN_with_SimpleDPA(base_model, fpn_channels=256)
    
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    total_loss_classifier = 0.0
    total_loss_box_reg = 0.0
    total_loss_objectness = 0.0
    total_loss_rpn_box_reg = 0.0
    num_batches = 0
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        total_loss_classifier += loss_dict['loss_classifier'].item()
        total_loss_box_reg += loss_dict['loss_box_reg'].item()
        total_loss_objectness += loss_dict['loss_objectness'].item()
        total_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{losses.item():.4f}",
            'cls': f"{loss_dict['loss_classifier'].item():.4f}",
            'box': f"{loss_dict['loss_box_reg'].item():.4f}"
        })
    
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
    """Evaluate on validation set"""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(data_loader, desc="Validation")
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        model.train()  # Temporarily for losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        model.eval()
        
        total_loss += losses.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f"{losses.item():.4f}"})
    
    return total_loss / num_batches


def main():
    print("="*80)
    print("FASTER R-CNN WITH SIMPLIFIED DPA ENHANCEMENT")
    print("="*80)
    
    config = {
        'dataset_root': '../dataset/VisDrone-2018',
        'num_classes': 10,
        'batch_size': 4,    # Can use larger batch size now
        'num_workers': 4,
        'num_epochs': 50,
        'learning_rate': 0.005,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'lr_step_size': 10,
        'lr_gamma': 0.1,
        'save_interval': 5,
        'patience': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nEnhancements:")
    print("  ✓ Multi-scale edge detection (3x3, 5x5 kernels)")
    print("  ✓ Spatial attention for edge preservation")
    print("  ✓ Channel attention (SE block) for semantics")
    print("  ✓ Memory efficient (no full attention)")
    
    output_dir = Path('outputs_dpa')
    output_dir.mkdir(exist_ok=True)
    
    # Load datasets
    print("\n" + "-"*80)
    print("Loading datasets...")
    
    train_dataset = VisDroneDataset(root_dir=config['dataset_root'], split='train', min_size=5)
    val_dataset = VisDroneDataset(root_dir=config['dataset_root'], split='val', min_size=5)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], collate_fn=VisDroneDataset.collate_fn,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], collate_fn=VisDroneDataset.collate_fn,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    print(f"✓ Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"✓ Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # Create model
    print("\n" + "-"*80)
    model = get_model_with_enhancement(num_classes=config['num_classes'], pretrained=True)
    model.to(config['device'])
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=config['learning_rate'],
        momentum=config['momentum'], weight_decay=config['weight_decay']
    )
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config['lr_step_size'], gamma=config['lr_gamma']
    )
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING START")
    print("="*80)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 80)
        
        epoch_start = time.time()
        train_losses = train_one_epoch(model, optimizer, train_loader, config['device'], epoch)
        val_loss = evaluate(model, val_loader, config['device'])
        lr_scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_losses['total']:.4f}")
        print(f"    - Classifier: {train_losses['classifier']:.4f}")
        print(f"    - Box Reg: {train_losses['box_reg']:.4f}")
        print(f"    - Objectness: {train_losses['objectness']:.4f}")
        print(f"    - RPN Box: {train_losses['rpn_box_reg']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, output_dir / 'best_model_dpa.pth')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
            
            if patience_counter >= config['patience']:
                print(f"\n⚠ Early stopping at epoch {epoch}")
                break
        
        if epoch % config['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, output_dir / f'checkpoint_epoch_{epoch}_dpa.pth')
            print(f"  ✓ Saved checkpoint")
    
    # Save final
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': config
    }, output_dir / 'final_model_dpa.pth')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")


if __name__ == '__main__':
    main()
