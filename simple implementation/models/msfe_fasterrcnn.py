"""
Multi-Scale Enhanced Faster R-CNN
Applies Multi-Scale Feature Enhancer to improve small object detection
"""
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from enhancements.msfe_module import MultiScaleFeatureEnhancer
except ImportError:
    from .enhancements.msfe_module import MultiScaleFeatureEnhancer


class MSFEFasterRCNN(nn.Module):
    """
    Multi-Scale Feature Enhanced Faster R-CNN
    
    Enhances Faster R-CNN by applying multi-scale feature enhancement modules
    to selected FPN levels, improving detection of small objects.
    
    Args:
        num_classes (int): Number of object classes (including background)
        fpn_channels (int): Number of FPN feature channels. Default: 256
        enhance_levels (list): FPN levels to enhance. Default: ['0', '1'] (P3, P4)
        pretrained (bool): Use COCO pretrained weights. Default: True
        trainable_backbone_layers (int): Number of trainable backbone layers. Default: 3
    
    Example:
        >>> model = MSFEFasterRCNN(num_classes=11, enhance_levels=['0', '1'])
        >>> images = [torch.randn(3, 640, 640) for _ in range(2)]
        >>> targets = [{'boxes': ..., 'labels': ...} for _ in range(2)]
        >>> losses = model(images, targets)  # Training
        >>> model.eval()
        >>> predictions = model(images)      # Inference
    """
    def __init__(self, num_classes, fpn_channels=256, enhance_levels=['0', '1'], 
                 pretrained=True, trainable_backbone_layers=3):
        super().__init__()
        
        # Load base Faster R-CNN
        if pretrained:
            base_model = fasterrcnn_resnet50_fpn(
                weights='DEFAULT',
                trainable_backbone_layers=trainable_backbone_layers
            )
        else:
            base_model = fasterrcnn_resnet50_fpn(
                weights=None,
                trainable_backbone_layers=5
            )
        
        # Modify classifier for custom number of classes
        in_features = base_model.roi_heads.box_predictor.cls_score.in_features
        base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        
        self.base_model = base_model
        
        # Create enhancement modules for specified FPN levels
        self.enhancers = nn.ModuleDict({
            level: MultiScaleFeatureEnhancer(fpn_channels) 
            for level in enhance_levels
        })
        
        self.enhance_levels = enhance_levels
        
    def forward(self, images, targets=None):
        """
        Forward pass with multi-scale feature enhancement
        
        Args:
            images: List of input images or batch tensor
            targets: List of target dictionaries (training only)
        
        Returns:
            losses (dict) in training mode or detections (list) in inference mode
        """
        # Convert single image to list if needed
        if isinstance(images, torch.Tensor):
            images = [images[i] for i in range(images.shape[0])]
        
        # Extract features through backbone
        original_images, targets = self.base_model.transform(images, targets)
        features = self.base_model.backbone(original_images.tensors)
        
        # Apply enhancement to selected FPN levels
        enhanced_features = {}
        for name, feature_map in features.items():
            if name in self.enhance_levels:
                enhanced_features[name] = self.enhancers[name](feature_map)
            else:
                enhanced_features[name] = feature_map
        
        # Continue through detection pipeline
        if self.training and targets is not None:
            proposals, proposal_losses = self.base_model.rpn(
                original_images, enhanced_features, targets
            )
            detections, detector_losses = self.base_model.roi_heads(
                enhanced_features, proposals, original_images.image_sizes, targets
            )
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        else:
            # Inference mode
            proposals, _ = self.base_model.rpn(
                original_images, enhanced_features, targets
            )
            detections, _ = self.base_model.roi_heads(
                enhanced_features, proposals, original_images.image_sizes, targets
            )
            detections = self.base_model.transform.postprocess(
                detections, original_images.image_sizes, 
                [img.shape[-2:] for img in images]
            )
            return detections
