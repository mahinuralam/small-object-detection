"""
Reconstruction Head (RH) module for SR-TOD
Reconstructs input image from P2 FPN features to generate difference maps
Based on: https://github.com/Hiyuur/SR-TOD
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [ReLU]) * 2
    Two sequential 3x3 convolutions with ReLU activation
    """
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up_direct(nn.Module):
    """Upscaling then double conv
    Uses ConvTranspose2d for 2x upsampling followed by DoubleConv
    """
    
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Default: ConvTranspose2d
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
            self.conv = DoubleConv(in_channels // 2, out_channels)

    def forward(self, x1):
        x = self.up(x1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    """Output convolution with Sigmoid activation
    Maps to RGB channels (3) with values in [0, 1]
    """
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.conv(x)


class ReconstructionHead(nn.Module):
    """
    Reconstruction Head (RH) for SR-TOD
    
    Reconstructs input image from P2 FPN features (256 channels, H/4, W/4).
    Architecture:
        P2 (256ch, H/4, W/4) 
        → Up1 (128ch, H/2, W/2)
        → Up2 (64ch, H, W)
        → OutConv (3ch RGB, H, W) with Sigmoid
    
    The reconstructed image is compared with original to generate difference maps
    that highlight tiny objects (which are harder to reconstruct accurately).
    
    Args:
        in_channels (int): Input channels from P2 features. Default: 256
        out_channels (int): Output RGB channels. Default: 3
        bilinear (bool): Use bilinear upsampling instead of ConvTranspose2d. Default: False
    
    Input:
        x: P2 features with shape (N, 256, H/4, W/4)
    
    Output:
        r_img: Reconstructed image with shape (N, 3, H, W), values in [0, 1]
    
    Example:
        >>> rh = ReconstructionHead(in_channels=256, out_channels=3)
        >>> p2_features = torch.randn(4, 256, 160, 160)  # Batch 4, 640x640 input
        >>> reconstructed = rh(p2_features)
        >>> print(reconstructed.shape)  # torch.Size([4, 3, 640, 640])
    """
    
    def __init__(self, in_channels=256, out_channels=3, bilinear=False):
        super(ReconstructionHead, self).__init__()
        self.up1 = Up_direct(in_channels, 128, bilinear=bilinear)
        self.up2 = Up_direct(128, 64, bilinear=bilinear)
        self.out_conv = OutConv(64, out_channels)
    
    def forward(self, x):
        """
        Args:
            x: P2 features (N, 256, H/4, W/4)
        
        Returns:
            r_img: Reconstructed image (N, 3, H, W) with values in [0, 1]
        """
        P0 = self.up1(x)     # (N, 128, H/2, W/2)
        P0 = self.up2(P0)    # (N, 64, H, W)
        r_img = self.out_conv(P0)  # (N, 3, H, W)
        return r_img


if __name__ == "__main__":
    # Test the ReconstructionHead
    print("Testing ReconstructionHead...")
    
    rh = ReconstructionHead(in_channels=256, out_channels=3)
    
    # Test with typical P2 feature size (640x640 input → 160x160 P2)
    p2_features = torch.randn(2, 256, 160, 160)
    
    reconstructed = rh(p2_features)
    
    print(f"Input P2 shape: {p2_features.shape}")
    print(f"Output reconstructed shape: {reconstructed.shape}")
    print(f"Output value range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
    
    assert reconstructed.shape == (2, 3, 640, 640), "Output shape mismatch!"
    assert reconstructed.min() >= 0 and reconstructed.max() <= 1, "Output values should be in [0, 1]!"
    
    print("✓ ReconstructionHead test passed!")
