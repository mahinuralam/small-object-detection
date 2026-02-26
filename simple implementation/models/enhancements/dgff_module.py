"""
Difference-Guided Feature Fusion (DGFF) Module
Uses reconstruction difference maps to guide feature enhancement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """Flatten layer for MLP"""
    def forward(self, x):
        return x.view(x.size(0), -1)


class DifferenceGuidedFeatureFusion(nn.Module):
    """
    Difference-Guided Feature Fusion Module
    
    Uses reconstruction error maps as spatial priors to enhance features.
    The key insight: regions with high reconstruction error (difficult to reconstruct)
    typically correspond to small or occluded objects that need feature enhancement.
    
    Three-stage processing:
    1. Filtration: Apply learnable threshold to difference map to create binary mask
    2. Spatial Guidance: Resize mask to feature map resolution  
    3. Channel Reweighting: Apply CBAM-style attention with guidance from difference map
    
    Args:
        feature_channels (int): Number of feature map channels. Default: 256 (FPN)
        reduction (int): Channel reduction ratio for attention. Default: 16
    
    Example:
        >>> dgff = DifferenceGuidedFeatureFusion(feature_channels=256)
        >>> features = torch.randn(2, 256, 160, 160)  # P2 features
        >>> diff_map = torch.randn(2, 1, 640, 640)    # Reconstruction difference
        >>> threshold = torch.tensor(0.015)            # Learnable threshold
        >>> enhanced = dgff(features, diff_map, threshold)
        >>> print(enhanced.shape)  # torch.Size([2, 256, 160, 160])
    """
    def __init__(self, feature_channels=256, reduction=16):
        super().__init__()
        
        self.feature_channels = feature_channels
        self.reduction = reduction
        
        # Channel attention pathway (CBAM-style)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for channel attention
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(feature_channels, feature_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(feature_channels // reduction, feature_channels),
            nn.Sigmoid()
        )
        
    def forward(self, features, difference_map, learnable_thresh):
        """
        Forward pass with difference-guided enhancement
        
        Args:
            features: Feature map to enhance (B, C, H, W)
            difference_map: Reconstruction difference map (B, 1, H_img, W_img)
            learnable_thresh: Threshold for filtering (scalar or tensor)
            
        Returns:
            Enhanced features (B, C, H, W)
        """
        B, C, H, W = features.shape
        
        # Stage 1: Filtration - Create binary mask from difference map
        # Higher difference = more difficult to reconstruct = likely small object
        binary_mask = (torch.sign(difference_map - learnable_thresh) + 1) * 0.5
        
        # Stage 2: Spatial Guidance - Resize mask to feature resolution
        spatial_mask = F.interpolate(
            binary_mask, 
            size=(H, W), 
            mode='nearest'
        )
        
        # Stage 3: Channel Reweighting - Compute channel attention
        # Use both average and max pooling for richer representation
        avg_out = self.mlp(self.avg_pool(features))
        max_out = self.mlp(self.max_pool(features))
        
        # Combine attention scores
        channel_attention = (avg_out + max_out).view(B, C, 1, 1)
        
        # Apply spatial and channel guidance
        # Features in masked regions get enhanced by channel attention
        enhanced = features * (1 + spatial_mask * channel_attention)
        
        return enhanced
