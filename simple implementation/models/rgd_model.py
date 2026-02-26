"""
Reconstruction-Guided Detector (RGD)
Uses image reconstruction as guidance for enhancing small object detection
"""

import torch
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from enhancements.feature_reconstructor import FeatureReconstructor
    from enhancements.dgff_module import DifferenceGuidedFeatureFusion
except ImportError:
    from .enhancements.feature_reconstructor import FeatureReconstructor
    from .enhancements.dgff_module import DifferenceGuidedFeatureFusion


class ReconstructionGuidedDetector(nn.Module):
    """
    Reconstruction-Guided Detector for Small Object Detection
    
    A novel detection framework that uses image reconstruction as a proxy for 
    identifying challenging regions (typically containing small objects). The reconstruction
    error serves as spatial guidance for feature enhancement.
    
    Key Components:
    1. Feature Reconstructor (FR): Reconstructs input image from low-level FPN features (P2)
    2. Difference Map: Measures reconstruction error |reconstructed - original|
    3. DGFF: Uses difference map to guide feature enhancement on P2 level
    
    Intuition:
    - Small objects are harder to reconstruct from low-resolution features
    - High reconstruction error → likely contains small/difficult objects
    - Use this as spatial prior to enhance features in those regions
    
    Architecture Flow:
        Input Image (H×W)
        ↓
        Backbone + FPN → [P2, P3, P4, P5, P6]
        ↓
        P2 (256ch, H/4×W/4) → Feature Reconstructor → Reconstructed Image (3ch, H×W)
        ↓
        Difference Map = |Reconstructed - Original| / 3
        ↓
        Enhanced P2 = DGFF(P2, Difference Map, learnable_threshold)
        ↓
        [Enhanced P2, P3, P4, P5, P6] → RPN + ROI Head → Detections
    
    Args:
        num_classes (int): Number of object classes (including background)
        learnable_thresh (float): Initial threshold for DGFF. Default: 0.0156862 (4/255)
        pretrained_backbone (bool): Use ImageNet pretrained ResNet50. Default: True
    
    Training Loss:
        total_loss = reconstruction_loss + detection_loss
        where reconstruction_loss = MSE(reconstructed, original)
    
    Example:
        >>> model = ReconstructionGuidedDetector(num_classes=11)
        >>> images = torch.randn(2, 3, 640, 640)
        >>> targets = [{'boxes': ..., 'labels': ...}, ...]
        >>> model.train()
        >>> losses = model(images, targets)
        >>> model.eval()
        >>> predictions = model(images)
    """
    
    def __init__(self, num_classes=11, learnable_thresh=0.0156862, pretrained_backbone=True):
        super(ReconstructionGuidedDetector, self).__init__()
        
        # Base Faster R-CNN with ResNet50-FPN
        backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained_backbone)
        self.base_model = FasterRCNN(backbone, num_classes=91)  # Start with COCO classes
        
        # Replace box predictor for custom number of classes
        in_features = self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Reconstruction-guided enhancement components
        # Feature Reconstructor: P2 features → RGB image
        self.feature_reconstructor = FeatureReconstructor(in_channels=256, out_channels=3)
        
        # Difference-Guided Feature Fusion: Uses reconstruction error to enhance P2
        self.dgff = DifferenceGuidedFeatureFusion(feature_channels=256, reduction=16)
        
        # Learnable threshold for difference map filtering
        self.learnable_thresh = nn.Parameter(torch.tensor(learnable_thresh))
        
        # Loss function for reconstruction
        self.reconstruction_loss_fn = nn.MSELoss()
        
    def forward(self, images, targets=None):
        """
        Forward pass with reconstruction-guided enhancement
        
        Args:
            images: Input images - List of tensors or batch tensor (B, 3, H, W)
            targets: Target annotations (training only) - List of dicts with 'boxes' and 'labels'
        
        Returns:
            Training mode: Dictionary of losses including reconstruction loss
            Inference mode: List of detection dictionaries with 'boxes', 'labels', 'scores'
        """
        # Handle both list and batch tensor inputs
        if isinstance(images, torch.Tensor):
            original_images = images
            image_list = [images[i] for i in range(images.shape[0])]
        else:
            image_list = images
            original_images = torch.stack(images) if len(images) > 1 else images[0].unsqueeze(0)
        
        # Extract backbone features
        transformed_images, transformed_targets = self.base_model.transform(image_list, targets)
        features = self.base_model.backbone(transformed_images.tensors)
        
        # Get P2 features (lowest FPN level, key='0')
        p2_features = features['0']
        
        # Reconstruct image from P2 features
        reconstructed_image = self.feature_reconstructor(p2_features)
        
        # Resize reconstructed to match original if needed
        if reconstructed_image.shape[2:] != transformed_images.tensors.shape[2:]:
            reconstructed_image = torch.nn.functional.interpolate(
                reconstructed_image,
                size=transformed_images.tensors.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Compute difference map (reconstruction error)
        difference_map = torch.sum(
            torch.abs(reconstructed_image - transformed_images.tensors),
            dim=1,
            keepdim=True
        ) / 3.0
        
        # Apply DGFF to enhance P2 features using difference map
        enhanced_p2 = self.dgff(p2_features, difference_map, self.learnable_thresh)
        
        # Update features dict with enhanced P2
        enhanced_features = features.copy()
        enhanced_features['0'] = enhanced_p2
        
        # Continue with standard Faster R-CNN pipeline
        if self.training and targets is not None:
            # Training mode: return losses
            # Get detection losses (RPN + ROI head)
            proposals, proposal_losses = self.base_model.rpn(
                transformed_images, enhanced_features, transformed_targets
            )
            detections, detector_losses = self.base_model.roi_heads(
                enhanced_features, proposals, transformed_images.image_sizes, transformed_targets
            )
            
            # Compute reconstruction loss
            reconstruction_loss = self.reconstruction_loss_fn(
                reconstructed_image,
                transformed_images.tensors
            )
            
            # Combine all losses
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses['loss_reconstruction'] = reconstruction_loss
            
            return losses
        else:
            # Inference mode: return detections
            proposals, _ = self.base_model.rpn(
                transformed_images, enhanced_features, None
            )
            detections, _ = self.base_model.roi_heads(
                enhanced_features, proposals, transformed_images.image_sizes, None
            )
            
            # Post-process detections
            detections = self.base_model.transform.postprocess(
                detections,
                transformed_images.image_sizes,
                [img.shape[-2:] for img in image_list]
            )
            
            return detections
    
    def get_reconstruction_outputs(self, images):
        """
        Get reconstruction intermediate outputs for visualization
        
        Args:
            images: Input images (B, 3, H, W) or list of tensors
        
        Returns:
            Dictionary containing:
                - original: Original images
                - reconstructed: Reconstructed images
                - difference_map: Reconstruction difference map
                - binary_mask: Binary mask after thresholding
                - threshold: Current threshold value
        """
        self.eval()
        with torch.no_grad():
            if isinstance(images, list):
                images = torch.stack(images)
            
            # Get features
            transformed_images, _ = self.base_model.transform(
                [images[i] for i in range(images.shape[0])], None
            )
            features = self.base_model.backbone(transformed_images.tensors)
            p2_features = features['0']
            
            # Reconstruct
            reconstructed = self.feature_reconstructor(p2_features)
            
            # Resize if needed
            if reconstructed.shape[2:] != transformed_images.tensors.shape[2:]:
                reconstructed = torch.nn.functional.interpolate(
                    reconstructed,
                    size=transformed_images.tensors.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Compute difference
            difference_map = torch.sum(
                torch.abs(reconstructed - transformed_images.tensors),
                dim=1,
                keepdim=True
            ) / 3.0
            
            # Binary mask
            binary_mask = (torch.sign(difference_map - self.learnable_thresh) + 1) * 0.5
            
            return {
                'original': transformed_images.tensors,
                'reconstructed': reconstructed,
                'difference_map': difference_map,
                'binary_mask': binary_mask,
                'threshold': self.learnable_thresh.item()
            }
