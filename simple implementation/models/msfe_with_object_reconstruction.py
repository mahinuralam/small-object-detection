"""
MSFE with Object-Aware Reconstruction
Combines the proven MSFE enhancement with smart reconstruction that actually focuses on objects
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator

from enhancements.msfe_module import MultiScaleFeatureEnhancer
from enhancements.object_aware_reconstructor import ObjectAwareReconstructor
from enhancements.dgff_module import DifferenceGuidedFeatureFusion


class MSFEWithObjectReconstruction(nn.Module):
    """
    MSFE enhanced with object-aware reconstruction
    
    Key improvements:
    1. MSFE on P2, P3, P4 (consistent enhancement)
    2. Object-aware reconstruction from P2 (focused on objects, not background)
    3. DGFF uses object-weighted difference maps
    
    This addresses the fundamental issues:
    - Reconstruction loss weighted by object importance (10x for objects)
    - Small objects get additional 2x boost (20x total vs background)
    - Gradient flow prioritizes object features
    - Difference maps more meaningful for object detection
    """
    
    def __init__(
        self,
        num_classes=11,
        reconstruction_weight=0.3,  # Slightly higher than hybrid since it's object-aware
        object_weight=10.0,         # Objects 10x more important than background
        small_object_boost=2.0,     # Small objects get 2x additional boost
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
        
        # Anchor generator
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
        
        # ===== MSFE Components (P2, P3, P4) =====
        self.msfe_p2 = MultiScaleFeatureEnhancer(
            channels=feature_channels,
            reduction=reduction
        )
        
        self.msfe_p3 = MultiScaleFeatureEnhancer(
            channels=feature_channels,
            reduction=reduction
        )
        
        self.msfe_p4 = MultiScaleFeatureEnhancer(
            channels=feature_channels,
            reduction=reduction
        )
        
        # ===== Object-Aware Reconstruction =====
        self.reconstructor = ObjectAwareReconstructor(
            in_channels=feature_channels,
            out_channels=3
        )
        
        # ===== DGFF for P2 (uses object-aware difference maps) =====
        self.dgff_p2 = DifferenceGuidedFeatureFusion(
            feature_channels=feature_channels,
            reduction=reduction
        )
        
        # Learnable threshold for DGFF
        self.learnable_thresh = nn.Parameter(
            torch.tensor(0.0156862, dtype=torch.float32)  # 4/255
        )
        
        # Loss weights
        self.reconstruction_weight = reconstruction_weight
        self.object_weight = object_weight
        self.small_object_boost = small_object_boost
    
    def forward(self, images, targets=None):
        """
        Forward pass with MSFE + object-aware reconstruction
        
        Args:
            images: List of images (Tensor[C, H, W])
            targets: List of target dicts (for training)
        
        Returns:
            If training: Dict of losses
            If inference: List of detection dicts
        """
        if self.training:
            # Store original backbone forward
            original_backbone_forward = self.base_model.backbone.forward
            
            # Create enhanced backbone forward
            def enhanced_backbone_forward(x):
                """Enhanced backbone with MSFE + object-aware reconstruction"""
                features = original_backbone_forward(x)
                
                # Apply MSFE to P2 first (for reconstruction)
                p2_msfe_enhanced = self.msfe_p2(features['0'])
                
                # Object-aware reconstruction from P2
                reconstructed_image = self.reconstructor(p2_msfe_enhanced)
                
                # Ensure size match
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
                
                # Create difference map
                difference_map = self.reconstructor.create_difference_map(
                    reconstructed_image, x
                )
                
                # Apply DGFF to P2 using difference guidance
                p2_final = self.dgff_p2(
                    p2_msfe_enhanced, difference_map, self.learnable_thresh
                )
                features['0'] = p2_final
                
                # Apply MSFE to P3, P4
                features['1'] = self.msfe_p3(features['1'])
                features['2'] = self.msfe_p4(features['2'])
                
                return features
            
            # Replace backbone forward temporarily
            self.base_model.backbone.forward = enhanced_backbone_forward
            
            # Forward through base model
            loss_dict = self.base_model(images, targets)
            
            # Compute object-aware reconstruction loss
            recon_loss, importance_map = self.reconstructor.compute_object_aware_loss(
                self._reconstructed_image,
                self._original_image,
                targets,
                object_weight=self.object_weight,
                small_object_boost=self.small_object_boost
            )
            
            loss_dict['loss_reconstruction'] = recon_loss * self.reconstruction_weight
            
            # Restore original forward
            self.base_model.backbone.forward = original_backbone_forward
            
            # Clean up
            del self._reconstructed_image
            del self._original_image
            
            return loss_dict
        
        else:
            # Inference mode
            original_backbone_forward = self.base_model.backbone.forward
            
            def enhanced_backbone_forward(x):
                features = original_backbone_forward(x)
                
                # Apply MSFE + DGFF to P2
                p2_msfe_enhanced = self.msfe_p2(features['0'])
                reconstructed_image = self.reconstructor(p2_msfe_enhanced)
                
                if reconstructed_image.shape[-2:] != x.shape[-2:]:
                    reconstructed_image = F.interpolate(
                        reconstructed_image,
                        size=x.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                difference_map = self.reconstructor.create_difference_map(
                    reconstructed_image, x
                )
                
                p2_final = self.dgff_p2(
                    p2_msfe_enhanced, difference_map, self.learnable_thresh
                )
                features['0'] = p2_final
                
                # Apply MSFE to P3, P4
                features['1'] = self.msfe_p3(features['1'])
                features['2'] = self.msfe_p4(features['2'])
                
                return features
            
            # Temporarily replace backbone forward
            self.base_model.backbone.forward = enhanced_backbone_forward
            
            # Use base model's forward
            detections = self.base_model(images)
            
            # Restore original forward
            self.base_model.backbone.forward = original_backbone_forward
            
            return detections


def test_model():
    """Test MSFE with object-aware reconstruction"""
    print("Testing MSFEWithObjectReconstruction...")
    
    # Create sample data
    images = [torch.rand(3, 640, 640) for _ in range(2)]
    targets = [
        {
            'boxes': torch.tensor([[100, 100, 150, 150], [400, 400, 420, 420]], dtype=torch.float32),
            'labels': torch.tensor([1, 2], dtype=torch.int64)
        },
        {
            'boxes': torch.tensor([[200, 200, 215, 215]], dtype=torch.float32),
            'labels': torch.tensor([3], dtype=torch.int64)
        }
    ]
    
    # Initialize model
    model = MSFEWithObjectReconstruction(
        num_classes=11,
        pretrained_backbone=False
    )
    model.train()
    
    # Training forward pass
    print("\n✓ Testing training mode...")
    loss_dict = model(images, targets)
    print(f"✓ Loss dict keys: {loss_dict.keys()}")
    print(f"✓ Reconstruction loss: {loss_dict['loss_reconstruction'].item():.6f}")
    
    # Inference mode
    print("\n✓ Testing inference mode...")
    model.eval()
    with torch.no_grad():
        detections = model(images)
    print(f"✓ Got {len(detections)} detection results")
    
    print("\n✓ MSFEWithObjectReconstruction test passed!")


if __name__ == '__main__':
    test_model()
