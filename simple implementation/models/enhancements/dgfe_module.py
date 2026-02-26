"""
Difference Map Guided Feature Enhancement (DGFE) module for SR-TOD
Enhances FPN features using difference maps from reconstruction
Based on: https://github.com/Hiyuur/SR-TOD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """Flatten tensor for MLP processing"""
    
    def forward(self, x):
        return x.view(x.size(0), -1)


class DGFE(nn.Module):
    """
    Difference Map Guided Feature Enhancement (DGFE)
    
    Three-stage process to enhance features using difference maps:
    1. Filtration: Threshold difference map using learnable threshold
    2. Spatial Guidance: Resize difference mask to feature map size
    3. Channel Reweighting: Apply channel attention (CBAM-style)
    
    The difference map highlights regions where reconstruction failed (tiny objects),
    and this module uses it to emphasize those regions in the feature maps.
    
    Args:
        gate_channels (int): Number of channels in input features. Default: 256
        reduction_ratio (int): Channel reduction ratio for MLP. Default: 16
        pool_types (list): Pooling types for channel attention. Default: ['avg', 'max']
    
    Input:
        x: FPN features (N, C, H, W)
        difference_map: Grayscale difference map (N, 1, H_img, W_img)
        learnable_thresh: Learnable threshold parameter (scalar tensor)
    
    Output:
        x_out: Enhanced features (N, C, H, W)
    
    Example:
        >>> dgfe = DGFE(gate_channels=256)
        >>> features = torch.randn(4, 256, 160, 160)
        >>> diff_map = torch.randn(4, 1, 640, 640)
        >>> thresh = torch.tensor(0.0156862)
        >>> enhanced = dgfe(features, diff_map, thresh)
        >>> print(enhanced.shape)  # torch.Size([4, 256, 160, 160])
    """
    
    def __init__(self, gate_channels=256, reduction_ratio=16, pool_types=['avg', 'max']):
        super(DGFE, self).__init__()
        
        # Channel attention MLP
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x, difference_map, learnable_thresh):
        """
        Args:
            x: FPN features (N, C, H, W) - e.g., P2 features
            difference_map: Difference between reconstructed and original image (N, 1, H_img, W_img)
            learnable_thresh: Learnable threshold (scalar tensor, typically ~0.0156862 or 4/255)
        
        Returns:
            x_out: Enhanced features (N, C, H, W)
        """
        
        # Stage 1: Filtration - Binary mask from thresholded difference map
        # sign(diff - thresh) gives -1 or +1, then (+1)*0.5 maps to [0, 1]
        difference_map_mask = (torch.sign(difference_map - learnable_thresh) + 1) * 0.5
        
        # Stage 2: Spatial Guidance - Resize mask to feature map size
        feat_difference_map = F.interpolate(
            difference_map_mask, 
            size=(x.shape[2], x.shape[3]),
            mode='nearest'
        )
        
        # Stage 3: Channel Reweighting - CBAM-style channel attention
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                # Global average pooling
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                # Global max pooling
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        
        # Apply sigmoid and expand to feature dimensions
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        
        # Expand spatial mask to all channels
        feat_diff_mat = feat_difference_map.repeat(1, x.shape[1], 1, 1)
        
        # Apply both channel attention and spatial guidance
        x_out = x * scale
        x_out = torch.mul(x_out, feat_diff_mat) + x_out
        
        return x_out


if __name__ == "__main__":
    # Test the DGFE module
    print("Testing DGFE...")
    
    dgfe = DGFE(gate_channels=256)
    
    # Test with typical sizes
    # P2 features: 640x640 input → 160x160 feature map
    features = torch.randn(2, 256, 160, 160)
    difference_map = torch.randn(2, 1, 640, 640).abs()  # Abs to simulate difference
    learnable_thresh = torch.tensor(0.0156862)  # 4/255
    
    enhanced = dgfe(features, difference_map, learnable_thresh)
    
    print(f"Input features shape: {features.shape}")
    print(f"Difference map shape: {difference_map.shape}")
    print(f"Learnable threshold: {learnable_thresh.item():.6f} ({learnable_thresh.item() * 255:.2f}/255)")
    print(f"Output enhanced shape: {enhanced.shape}")
    
    assert enhanced.shape == features.shape, "Output shape must match input!"
    
    print("✓ DGFE test passed!")
