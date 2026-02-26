"""
Multi-Scale Feature Enhancer (MSFE) Module
Enhances features using multi-scale processing with spatial and channel attention mechanisms
"""
import torch
import torch.nn as nn


class MultiScaleFeatureEnhancer(nn.Module):
    """
    Multi-Scale Feature Enhancement Module
    
    Processes features through multiple pathways to enhance small object detection:
    1. Multi-scale edge detection (using different kernel sizes)
    2. Spatial attention mechanism for edge preservation
    3. Channel-wise attention for semantic feature propagation
    
    This is a memory-efficient design suitable for real-time applications.
    
    Args:
        channels (int): Number of input/output channels
        reduction (int): Channel reduction ratio for attention. Default: 16
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        # Multi-scale depthwise convolutions for edge detection
        self.edge_conv3 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.edge_conv5 = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels)
        
        # Spatial attention for edge preservation
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        
        # Channel attention (Squeeze-and-Excitation style)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input features (B, C, H, W)
            
        Returns:
            Enhanced features (B, C, H, W)
        """
        # Multi-scale edge detection
        edge3 = self.edge_conv3(x)
        edge5 = self.edge_conv5(x)
        edge_features = edge3 + edge5
        
        # Spatial attention on edge features
        spatial_att = self.spatial_att(edge_features)
        edge_enhanced = edge_features * spatial_att
        
        # Channel attention on original features
        channel_att = self.channel_att(x)
        semantic_enhanced = x * channel_att
        
        # Fuse both pathways
        combined = torch.cat([edge_enhanced, semantic_enhanced], dim=1)
        output = self.fusion(combined)
        
        # Residual connection
        output = output + x
        
        return output
