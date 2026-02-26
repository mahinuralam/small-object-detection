"""
Train Lightweight Reconstructor for SAHI Pipeline
Self-supervised training on images (no labels needed)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from models.enhancements.lightweight_reconstructor import LightweightReconstructor


class ImageDataset(Dataset):
    """Simple image dataset for reconstruction"""
    
    def __init__(self, image_dir, img_size=(640, 640), augment=True):
        self.image_dir = Path(image_dir)
        self.img_size = img_size
        self.augment = augment
        
        # Find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_paths.extend(list(self.image_dir.glob(ext)))
            self.image_paths.extend(list(self.image_dir.glob(ext.upper())))
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Resize
        img = img.resize(self.img_size, Image.BILINEAR)
        
        # To tensor [0, 1]
        img = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        
        # Simple augmentation
        if self.augment:
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                img = torch.flip(img, dims=[2])
        
        return img


def train_reconstructor(args):
    """Train reconstructor"""
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset
    print("Loading dataset...")
    dataset = ImageDataset(
        args.data_dir,
        img_size=(args.img_size, args.img_size),
        augment=True
    )
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = LightweightReconstructor()
    model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Loss
    criterion = nn.L1Loss()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for images in pbar:
            images = images.to(device)
            
            # Forward
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]")
            for images in pbar:
                images = images.to(device)
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                val_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        val_loss /= len(val_loader)
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Log
        print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }
            torch.save(checkpoint, save_dir / 'best_reconstructor.pth')
            print(f"✓ Saved best model (val_loss={val_loss:.6f})")
        
        # Periodic save
        if epoch % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }
            torch.save(checkpoint, save_dir / f'reconstructor_epoch{epoch}.pth')
    
    print(f"\n✓ Training completed! Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to: {save_dir / 'best_reconstructor.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Reconstructor')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory with training images')
    parser.add_argument('--save_dir', type=str, default='models/reconstructor',
                       help='Directory to save models')
    parser.add_argument('--img_size', type=int, default=640,
                       help='Image size')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    train_reconstructor(args)
