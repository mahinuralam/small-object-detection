"""
Faster R-CNN with DPA Enhancement
"""
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .enhancements import SimplifiedDPAModule


class FasterRCNN_DPA(nn.Module):
    """
    Faster R-CNN enhanced with Simplified Dual-Path Attention
    
    Args:
        num_classes: Number of object classes
        fpn_channels: Number of FPN feature channels (default: 256)
        enhance_levels: List of FPN levels to enhance (default: ['0', '1'] for P3, P4)
        pretrained: Whether to use COCO pretrained weights
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
        
        # Modify classifier
        in_features = base_model.roi_heads.box_predictor.cls_score.in_features
        base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        
        self.base_model = base_model
        
        # Create DPA enhancers for specified FPN levels
        self.enhancers = nn.ModuleDict({
            level: SimplifiedDPAModule(fpn_channels) 
            for level in enhance_levels
        })
        
        self.enhance_levels = enhance_levels
        
    def forward(self, images, targets=None):
        """
        Forward pass with DPA enhancement
        
        Args:
            images: List of input images
            targets: List of target dictionaries (training only)
        
        Returns:
            losses (training) or detections (inference)
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        # Let base model handle preprocessing
        original_images = images
        images, targets = self.base_model.transform(images, targets)
        
        # Extract backbone features
        features = self.base_model.backbone(images.tensors)
        
        # Apply DPA enhancement to specified levels
        enhanced_features = {}
        for name, feat in features.items():
            if name in self.enhancers:
                enhanced_features[name] = self.enhancers[name](feat)
            else:
                enhanced_features[name] = feat
        
        # RPN
        proposals, proposal_losses = self.base_model.rpn(images, enhanced_features, targets)
        
        # ROI head
        detections, detector_losses = self.base_model.roi_heads(
            enhanced_features, proposals, images.image_sizes, targets
        )
        
        # Postprocess detections
        detections = self.base_model.transform.postprocess(
            detections, images.image_sizes,
            original_images.image_sizes if hasattr(original_images, 'image_sizes')
            else [(img.shape[-2], img.shape[-1]) for img in original_images]
        )
        
        if self.training:
            losses = {}
            losses.update(proposal_losses)
            losses.update(detector_losses)
            return losses
        
        return detections


def get_dpa_model(num_classes, fpn_channels=256, enhance_levels=['0', '1'],
                  pretrained=True, trainable_backbone_layers=3):
    """
    Factory function to create DPA-enhanced Faster R-CNN
    
    Args:
        num_classes: Number of object classes
        fpn_channels: FPN feature channels
        enhance_levels: FPN levels to enhance
        pretrained: Use COCO pretrained weights
        trainable_backbone_layers: Number of trainable backbone layers
    
    Returns:
        model: DPA-enhanced Faster R-CNN
    """
    return FasterRCNN_DPA(
        num_classes=num_classes,
        fpn_channels=fpn_channels,
        enhance_levels=enhance_levels,
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers
    )
