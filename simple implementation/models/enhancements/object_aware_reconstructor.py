"""
Object-Aware Feature Reconstructor
Reconstruction that actually focuses on small objects by weighting loss based on object presence
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectAwareReconstructor(nn.Module):
    """
    Object-Aware Feature Reconstructor
    
    Key improvements over naive reconstruction:
    1. Uses ground truth boxes to create object importance maps
    2. Weights reconstruction loss by object presence (higher weight on objects)
    3. Small objects get even higher weight (inversely proportional to size)
    
    This ensures reconstruction actually learns about objects, not just background.
    """
    
    def __init__(self, in_channels=256, out_channels=3):
        super().__init__()
        
        # Decoder to upsample from P2 (H/4, W/4) to full resolution
        self.decoder = nn.Sequential(
            # 256 -> 128, upsample 2x
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 128 -> 64, upsample 2x
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 64 -> 3 (RGB)
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        """
        Reconstruct image from features
        
        Args:
            features: P2 features (B, C, H/4, W/4)
            
        Returns:
            reconstructed: RGB image (B, 3, H, W)
        """
        return self.decoder(features)
    
    def compute_object_aware_loss(self, reconstructed, original, targets, 
                                   object_weight=10.0, small_object_boost=2.0):
        """
        Compute reconstruction loss weighted by object importance
        
        Args:
            reconstructed: Reconstructed image (B, 3, H, W)
            original: Original image (B, 3, H, W)
            targets: List of target dicts with 'boxes' (list of tensors)
            object_weight: Weight multiplier for object regions (default: 10x)
            small_object_boost: Additional boost for small objects (default: 2x)
            
        Returns:
            loss: Weighted reconstruction loss
            importance_map: Visualization of what's being weighted
        """
        B, C, H, W = reconstructed.shape
        device = reconstructed.device
        
        # Create importance map: 1.0 for background, object_weight for objects
        importance_maps = []
        
        for i in range(B):
            importance_map = torch.ones(H, W, device=device)
            
            if targets is not None and len(targets) > i:
                boxes = targets[i]['boxes']  # [N, 4] in (x1, y1, x2, y2) format
                
                for box in boxes:
                    x1, y1, x2, y2 = box
                    # Clamp to image bounds
                    x1 = int(max(0, min(W-1, x1)))
                    y1 = int(max(0, min(H-1, y1)))
                    x2 = int(max(0, min(W, x2)))
                    y2 = int(max(0, min(H, y2)))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Compute box area to identify small objects
                    box_area = (x2 - x1) * (y2 - y1)
                    is_small = box_area < (32 * 32)  # Objects < 32x32 pixels
                    
                    # Weight: base object_weight, boosted for small objects
                    weight = object_weight
                    if is_small:
                        weight *= small_object_boost
                    
                    # Apply weight to box region
                    importance_map[y1:y2, x1:x2] = weight
            
            importance_maps.append(importance_map)
        
        importance_map_batch = torch.stack(importance_maps).unsqueeze(1)  # (B, 1, H, W)
        
        # Compute pixel-wise reconstruction error
        pixel_loss = (reconstructed - original) ** 2  # (B, 3, H, W)
        
        # Weight by importance map (broadcast across channels)
        weighted_loss = pixel_loss * importance_map_batch
        
        # Average over all pixels and channels
        loss = weighted_loss.mean()
        
        return loss, importance_map_batch
    
    def create_difference_map(self, reconstructed, original):
        """
        Create difference map for DGFF guidance
        
        Args:
            reconstructed: Reconstructed image (B, 3, H, W)
            original: Original image (B, 3, H, W)
            
        Returns:
            difference_map: (B, 1, H, W)
        """
        diff = torch.abs(reconstructed - original)
        diff = torch.mean(diff, dim=1, keepdim=True)
        return diff


def test_object_aware_reconstructor():
    """Test object-aware reconstruction"""
    print("Testing ObjectAwareReconstructor...")
    
    # Create sample data
    B, C, H, W = 2, 256, 160, 160
    features = torch.randn(B, C, H, W)
    original_images = torch.rand(B, 3, 640, 640)
    
    # Create sample targets with boxes
    targets = [
        {'boxes': torch.tensor([[100, 100, 150, 150],  # Large object
                                [400, 400, 420, 420]])},  # Small object (20x20)
        {'boxes': torch.tensor([[200, 200, 215, 215],   # Small object (15x15)
                                [300, 300, 380, 380]])}  # Medium object
    ]
    
    # Initialize reconstructor
    reconstructor = ObjectAwareReconstructor()
    
    # Forward pass
    reconstructed = reconstructor(features)
    print(f"✓ Reconstruction shape: {reconstructed.shape}")
    assert reconstructed.shape == (B, 3, 640, 640), "Output shape mismatch"
    
    # Compute object-aware loss
    loss, importance_map = reconstructor.compute_object_aware_loss(
        reconstructed, original_images, targets
    )
    print(f"✓ Object-aware loss: {loss.item():.6f}")
    print(f"✓ Importance map shape: {importance_map.shape}")
    
    # Check that importance map has higher values at object locations
    print(f"✓ Background weight: {importance_map[0, 0, 0, 0].item():.2f}")
    print(f"✓ Object region weight (sample): {importance_map[0, 0, 110, 110].item():.2f}")
    
    # Create difference map
    diff_map = reconstructor.create_difference_map(reconstructed, original_images)
    print(f"✓ Difference map shape: {diff_map.shape}")
    
    print("\n✓ ObjectAwareReconstructor test passed!")


if __name__ == '__main__':
    test_object_aware_reconstructor()
