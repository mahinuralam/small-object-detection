"""
Hybrid Detector: MSFE + RGD
Combines Multi-Scale Feature Enhancement with Reconstruction-Guided Detection
for comprehensive coverage of all object sizes in VisDrone dataset.

Architecture:
- RGD on P2: Reconstruction-guided enhancement for very tiny objects (<16px)
- MSFE on P3, P4: Multi-scale attention for tiny/small objects (16-64px)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator

from enhancements.msfe_module import MultiScaleFeatureEnhancer
from enhancements.feature_reconstructor import FeatureReconstructor
from enhancements.dgff_module import DifferenceGuidedFeatureFusion


class HybridDetector(nn.Module):
    """
    Hybrid detection framework combining:
    1. RGD (Reconstruction-Guided Detection) on P2
    2. MSFE (Multi-Scale Feature Enhancement) on P3, P4
    
    This provides complementary enhancement mechanisms:
    - P2 (H/4): Explicit spatial priors from reconstruction (~21% of objects)
    - P3 (H/8): Multi-scale attention for tiny objects (~52% of objects)
    - P4 (H/16): Multi-scale attention for small objects (~33% of objects)
    
    Total coverage: All object sizes in VisDrone (0-64+ pixels)
    """
    
    def __init__(
        self,
        num_classes=11,
        learnable_thresh=0.0156862,  # 4/255
        reconstruction_weight=0.2,   # Reduced from 1.0 based on analysis
        pretrained_backbone=True,
        feature_channels=256,
        reduction=16
    ):
        super().__init__()
        
        # Base Faster R-CNN with ResNet50-FPN
        backbone = resnet_fpn_backbone(
            'resnet50',
            pretrained=pretrained_backbone,
            trainable_layers=5
        )
        
        # Anchor generator (same as baseline)
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        
        # Initialize base Faster R-CNN
        self.base_model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            min_size=640,
            max_size=640
        )
        
        # ===== RGD Components (P2) =====
        self.feature_reconstructor = FeatureReconstructor(
            in_channels=feature_channels,
            out_channels=3  # RGB
        )
        
        self.dgff_p2 = DifferenceGuidedFeatureFusion(
            feature_channels=feature_channels,
            reduction=reduction
        )
        
        # Learnable threshold for difference map binarization
        self.learnable_thresh = nn.Parameter(
            torch.tensor(learnable_thresh, dtype=torch.float32)
        )
        
        # ===== MSFE Components (P3, P4) =====
        self.msfe_p3 = MultiScaleFeatureEnhancer(
            channels=feature_channels,
            reduction=reduction
        )
        
        self.msfe_p4 = MultiScaleFeatureEnhancer(
            channels=feature_channels,
            reduction=reduction
        )
        
        # Loss function for reconstruction
        self.reconstruction_loss_fn = nn.MSELoss()
        self.reconstruction_weight = reconstruction_weight
    
    def forward(self, images, targets=None):
        """
        Forward pass with hybrid enhancement.
        
        Args:
            images: List of images (Tensor[C, H, W])
            targets: List of target dicts (for training)
        
        Returns:
            If training: Dict of losses
            If inference: List of detection dicts
        """
        # In training mode, we need to handle the model's internal forward differently
        if self.training:
            # Store original backbone forward
            original_backbone_forward = self.base_model.backbone.forward
            
            # Create enhanced backbone forward
            def enhanced_backbone_forward(x):
                """Enhanced backbone that applies RGD and MSFE"""
                features = original_backbone_forward(x)
                
                # ===== RGD Enhancement on P2 =====
                p2_features = features['0']
                reconstructed_image = self.feature_reconstructor(p2_features)
                
                # Ensure reconstructed matches input size
                if reconstructed_image.shape[-2:] != x.shape[-2:]:
                    reconstructed_image = F.interpolate(
                        reconstructed_image,
                        size=x.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Store for loss computation
                self._reconstructed_image = reconstructed_image
                self._original_image = x
                
                # Compute difference map
                difference_map = torch.abs(reconstructed_image - x)
                difference_map = torch.mean(difference_map, dim=1, keepdim=True)
                
                # Enhance P2 features
                enhanced_p2 = self.dgff_p2(p2_features, difference_map, self.learnable_thresh)
                features['0'] = enhanced_p2
                
                # ===== MSFE Enhancement on P3, P4 =====
                features['1'] = self.msfe_p3(features['1'])
                features['2'] = self.msfe_p4(features['2'])
                
                return features
            
            # Replace backbone forward temporarily
            self.base_model.backbone.forward = enhanced_backbone_forward
            
            # Forward through base model (handles image preprocessing, batching, etc.)
            loss_dict = self.base_model(images, targets)
            
            # Compute reconstruction loss
            reconstruction_loss = self.reconstruction_loss_fn(
                self._reconstructed_image,
                self._original_image
            ) * self.reconstruction_weight
            
            # Add to loss dict
            loss_dict['loss_reconstruction'] = reconstruction_loss
            
            # Restore original forward
            self.base_model.backbone.forward = original_backbone_forward
            
            # Clean up temporary tensors
            del self._reconstructed_image
            del self._original_image
            
            return loss_dict
        
        else:
            # Inference mode - use base model's transform and forward
            # The base_model handles image preprocessing and wrapping
            # We just need to intercept and enhance the features
            
            # Store original forward method
            original_backbone_forward = self.base_model.backbone.forward
            
            # Create enhanced backbone forward that applies our enhancements
            def enhanced_backbone_forward(x):
                features = original_backbone_forward(x)
                
                # Apply RGD to P2
                p2_features = features['0']
                reconstructed_image = self.feature_reconstructor(p2_features)
                
                if reconstructed_image.shape[-2:] != x.shape[-2:]:
                    reconstructed_image = F.interpolate(
                        reconstructed_image,
                        size=x.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                difference_map = torch.abs(reconstructed_image - x)
                difference_map = torch.mean(difference_map, dim=1, keepdim=True)
                
                enhanced_p2 = self.dgff_p2(p2_features, difference_map, self.learnable_thresh)
                features['0'] = enhanced_p2
                
                # Apply MSFE to P3, P4
                features['1'] = self.msfe_p3(features['1'])
                features['2'] = self.msfe_p4(features['2'])
                
                return features
            
            # Temporarily replace backbone forward
            self.base_model.backbone.forward = enhanced_backbone_forward
            
            # Use base model's forward (handles all preprocessing)
            detections = self.base_model(images)
            
            # Restore original forward
            self.base_model.backbone.forward = original_backbone_forward
            
            return detections
    
    def get_reconstruction_outputs(self, images):
        """
        Get reconstruction outputs for visualization.
        
        Args:
            images: Batch of images (Tensor[B, C, H, W])
        
        Returns:
            Dict with reconstruction outputs
        """
        self.eval()
        with torch.no_grad():
            # Extract P2 features
            features = self.base_model.backbone(images)
            p2_features = features['0']
            
            # Reconstruct
            reconstructed_image = self.feature_reconstructor(p2_features)
            
            if reconstructed_image.shape[-2:] != images.shape[-2:]:
                reconstructed_image = F.interpolate(
                    reconstructed_image,
                    size=images.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Difference map
            difference_map = torch.abs(reconstructed_image - images)
            difference_map = torch.mean(difference_map, dim=1, keepdim=True)
            
            # Binary mask
            binary_mask = (torch.sign(difference_map - self.learnable_thresh) + 1) * 0.5
            
            return {
                'reconstructed': reconstructed_image,
                'difference_map': difference_map,
                'binary_mask': binary_mask,
                'learnable_thresh': self.learnable_thresh.item()
            }


def build_hybrid_detector(num_classes=11, pretrained=True, reconstruction_weight=0.2):
    """
    Build Hybrid Detector (MSFE + RGD).
    
    Args:
        num_classes: Number of object classes (excluding background)
        pretrained: Use COCO pretrained backbone
        reconstruction_weight: Weight for reconstruction loss (default: 0.2)
    
    Returns:
        HybridDetector model
    """
    model = HybridDetector(
        num_classes=num_classes + 1,  # +1 for background
        learnable_thresh=0.0156862,    # 4/255
        reconstruction_weight=reconstruction_weight,
        pretrained_backbone=pretrained
    )
    return model


if __name__ == "__main__":
    # Test model instantiation
    print("Testing Hybrid Detector...")
    
    model = build_hybrid_detector(num_classes=10, pretrained=False)
    model.eval()
    
    # Test forward pass
    dummy_images = torch.randn(2, 3, 640, 640)
    
    print("\nForward pass test:")
    with torch.no_grad():
        outputs = model([img for img in dummy_images])
    
    print(f"✓ Model instantiated successfully")
    print(f"✓ Forward pass successful")
    print(f"✓ Output: {len(outputs)} detections")
    
    # Test reconstruction outputs
    print("\nReconstruction outputs test:")
    recon_outputs = model.get_reconstruction_outputs(dummy_images)
    print(f"✓ Reconstructed shape: {recon_outputs['reconstructed'].shape}")
    print(f"✓ Difference map shape: {recon_outputs['difference_map'].shape}")
    print(f"✓ Binary mask shape: {recon_outputs['binary_mask'].shape}")
    print(f"✓ Learnable threshold: {recon_outputs['learnable_thresh']:.6f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
