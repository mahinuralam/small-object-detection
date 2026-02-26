"""
Feature Reconstructor Module
Reconstructs input images from low-level feature maps
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Double convolution block
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UpsampleBlock(nn.Module):
    """
    Upsampling block with convolution transpose
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class OutputBlock(nn.Module):
    """
    Final output block to produce RGB image
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels (typically 3 for RGB)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class FeatureReconstructor(nn.Module):
    """
    Feature Reconstructor: Reconstructs RGB images from FPN P2 features
    
    Uses a decoder architecture to upsample low-resolution features (H/4, W/4) 
    back to original resolution (H, W). The reconstruction quality serves as 
    a proxy for feature representation quality.
    
    Architecture:
        Input: P2 features (256 channels, H/4, W/4)
        → Upsample Block 1: 256 → 128 channels, ×2 resolution
        → Upsample Block 2: 128 → 64 channels, ×2 resolution  
        → Output Block: 64 → 3 channels (RGB), Sigmoid activation
        Output: Reconstructed image (3 channels, H, W)
    
    Args:
        in_channels (int): Number of input feature channels. Default: 256 (FPN P2)
        out_channels (int): Number of output channels. Default: 3 (RGB)
    
    Example:
        >>> reconstructor = FeatureReconstructor(in_channels=256, out_channels=3)
        >>> p2_features = torch.randn(2, 256, 160, 160)  # Batch of P2 features
        >>> reconstructed = reconstructor(p2_features)
        >>> print(reconstructed.shape)  # torch.Size([2, 3, 640, 640])
    """
    def __init__(self, in_channels=256, out_channels=3):
        super().__init__()
        
        # Progressive upsampling from P2 resolution to original resolution
        # P2: 256ch @ H/4×W/4 → 128ch @ H/2×W/2
        self.up1 = UpsampleBlock(in_channels, 128)
        
        # 128ch @ H/2×W/2 → 64ch @ H×W
        self.up2 = UpsampleBlock(128, 64)
        
        # 64ch @ H×W → 3ch @ H×W (RGB)
        self.out = OutputBlock(64, out_channels)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: P2 features (B, in_channels, H/4, W/4)
            
        Returns:
            Reconstructed RGB image (B, out_channels, H, W)
        """
        x = self.up1(x)  # (B, 128, H/2, W/2)
        x = self.up2(x)  # (B, 64, H, W)
        x = self.out(x)  # (B, 3, H, W)
        return x
