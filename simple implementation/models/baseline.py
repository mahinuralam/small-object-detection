"""
Faster R-CNN model definitions
"""
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_baseline_model(num_classes, pretrained=True, trainable_backbone_layers=3):
    """
    Load standard Faster R-CNN with FPN
    
    Args:
        num_classes: Number of object classes (excluding background)
        pretrained: Whether to load COCO pretrained weights
        trainable_backbone_layers: Number of trainable backbone layers
    
    Returns:
        model: Faster R-CNN model
    """
    if pretrained:
        model = fasterrcnn_resnet50_fpn(
            weights='DEFAULT',
            trainable_backbone_layers=trainable_backbone_layers
        )
    else:
        model = fasterrcnn_resnet50_fpn(
            weights=None,
            trainable_backbone_layers=5
        )
    
    # Replace classifier head for custom number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    
    return model
