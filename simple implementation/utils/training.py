"""
Training utilities
"""
import torch
from tqdm import tqdm


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Train model for one epoch
    
    Args:
        model: Model to train
        optimizer: Optimizer
        data_loader: Training data loader
        device: Device to train on
        epoch: Current epoch number
    
    Returns:
        dict: Average losses for the epoch
    """
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
def validate(model, data_loader, device):
    """
    Validate model on validation set
    
    Args:
        model: Model to validate
        data_loader: Validation data loader
        device: Device to run on
    
    Returns:
        float: Average validation loss
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(data_loader, desc="Validation")
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        model.train()  # Temporarily set to train mode to get losses
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        model.eval()
        
        total_loss += losses.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f"{losses.item():.4f}"})
    
    avg_loss = total_loss / num_batches
    return avg_loss
