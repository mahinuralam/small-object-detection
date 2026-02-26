"""
Lightweight Reconstructor for Self-Supervised Learning
Small UNet architecture for fast RGB reconstruction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightReconstructor(nn.Module):
    """
    Lightweight UNet for image reconstruction
    
    Architecture:
        - Encoder: 3 scales with MaxPool downsampling
        - Decoder: 3 scales with TransConv upsampling + skip connections
        - Output: Sigmoid activation for RGB [0,1]
    
    Training:
        - Self-supervised on normal images
        - Loss: L1(I, I_hat)
        - No labels required
    
    Fast inference: ~8-10ms on GPU for 640×640 images
    """
    
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder (downsampling path)
        self.enc1 = self._conv_block(in_channels, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(128, 256)
        
        # Decoder (upsampling path)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(256, 128)  # 256 = 128 (upconv) + 128 (skip)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)   # 128 = 64 (upconv) + 64 (skip)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(64, 32)    # 64 = 32 (upconv) + 32 (skip)
        
        # Output layer
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def _conv_block(self, in_channels, out_channels):
        """
        Basic conv block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image (B, 3, H, W) in [0, 1]
            
        Returns:
            Reconstructed image (B, 3, H, W) in [0, 1]
        """
        # Encoder with skip connections
        enc1 = self.enc1(x)          # (B, 32, H, W)
        x = self.pool(enc1)          # (B, 32, H/2, W/2)
        
        enc2 = self.enc2(x)          # (B, 64, H/2, W/2)
        x = self.pool(enc2)          # (B, 64, H/4, W/4)
        
        enc3 = self.enc3(x)          # (B, 128, H/4, W/4)
        x = self.pool(enc3)          # (B, 128, H/8, W/8)
        
        # Bottleneck
        x = self.bottleneck(x)       # (B, 256, H/8, W/8)
        
        # Decoder with skip connections
        x = self.upconv3(x)          # (B, 128, H/4, W/4)
        # Match spatial dimensions before concatenation
        if x.shape[2:] != enc3.shape[2:]:
            x = F.interpolate(x, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, enc3], dim=1)  # (B, 256, H/4, W/4)
        x = self.dec3(x)             # (B, 128, H/4, W/4)
        
        x = self.upconv2(x)          # (B, 64, H/2, W/2)
        # Match spatial dimensions before concatenation
        if x.shape[2:] != enc2.shape[2:]:
            x = F.interpolate(x, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, enc2], dim=1)  # (B, 128, H/2, W/2)
        x = self.dec2(x)             # (B, 64, H/2, W/2)
        
        x = self.upconv1(x)          # (B, 32, H, W)
        # Match spatial dimensions before concatenation
        if x.shape[2:] != enc1.shape[2:]:
            x = F.interpolate(x, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, enc1], dim=1)  # (B, 64, H, W)
        x = self.dec1(x)             # (B, 32, H, W)
        
        # Output
        x = self.out_conv(x)         # (B, 3, H, W)
        
        return x
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = LightweightReconstructor()
    print(f"Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 640, 640)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert y.shape == x.shape, "Output shape mismatch"
    assert y.min() >= 0 and y.max() <= 1, "Output not in [0,1]"
    print("✓ Model test passed!")
