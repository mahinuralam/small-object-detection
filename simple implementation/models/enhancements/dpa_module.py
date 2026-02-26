"""
Simplified Dual-Path Attention Module for Small Object Detection
"""
import torch
import torch.nn as nn


class SimplifiedDPAModule(nn.Module):
    """
    Simplified Dual-Path Attention without full cross-attention
    Memory efficient version focusing on:
    1. Multi-scale processing (edge detection)
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
