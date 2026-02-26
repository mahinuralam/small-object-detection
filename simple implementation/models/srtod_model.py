"""
SR-TOD enhanced Faster R-CNN model
Applies Self-Reconstructed Tiny Object Detection to P2 FPN features
Based on: https://github.com/Hiyuur/SR-TOD (ECCV 2024)
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

from enhancements.reconstruction_head import ReconstructionHead
from enhancements.dgfe_module import DGFE


class FasterRCNN_SRTOD(nn.Module):
    """
    SR-TOD Enhanced Faster R-CNN
    
    Applies SR-TOD (Self-Reconstructed Tiny Object Detection) enhancement to P2 FPN features:
    1. Reconstruction Head (RH): Reconstructs input image from P2 features
    2. Difference Map: Computes |reconstructed - original| to highlight reconstruction errors
    3. DGFE: Uses difference map to guide feature enhancement on P2
    
    The key insight: Tiny objects are harder to reconstruct, so reconstruction errors
    serve as spatial priors to emphasize tiny object regions in feature maps.
    
    Architecture:
        Input Image (640x640)
        ↓
        Backbone + FPN → [P2, P3, P4, P5, P6]
        ↓
        P2 (256ch, 160x160) → RH → Reconstructed Image (3ch, 640x640)
        ↓
        Difference Map = |Reconstructed - Original| / 3
        ↓
        Enhanced P2 = DGFE(P2, Difference Map, learnable_thresh)
        ↓
        [Enhanced P2, P3, P4, P5, P6] → RPN + ROI Head → Detections
    
    Args:
        num_classes (int): Number of object classes (including background)
        learnable_thresh (float): Initial threshold for DGFE filtration. Default: 0.0156862 (4/255)
        pretrained_backbone (bool): Use ImageNet pretrained ResNet50. Default: True
    
    Training:
        loss = loss_reconstruction + loss_rpn + loss_roi
        where loss_reconstruction = MSE(reconstructed_image, original_image)
    
    Example:
        >>> model = FasterRCNN_SRTOD(num_classes=11, learnable_thresh=0.0156862)
        >>> images = torch.randn(2, 3, 640, 640)
        >>> # Training mode
        >>> model.train()
        >>> targets = [{'boxes': ..., 'labels': ...}, ...]
        >>> losses = model(images, targets)
        >>> # Inference mode
        >>> model.eval()
        >>> with torch.no_grad():
        >>>     predictions = model(images)
    """
    
    def __init__(self, num_classes=11, learnable_thresh=0.0156862, pretrained_backbone=True):
        super(FasterRCNN_SRTOD, self).__init__()
        
        # Base Faster R-CNN with ResNet50-FPN
        backbone = resnet_fpn_backbone('resnet50', pretrained=pretrained_backbone)
        self.base_model = FasterRCNN(backbone, num_classes=91)  # Start with COCO classes
        
        # Replace box predictor for custom number of classes
        in_features = self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # SR-TOD components
        self.rh = ReconstructionHead(in_channels=256, out_channels=3)
        self.dgfe = DGFE(gate_channels=256, reduction_ratio=16)
        
        # Learnable threshold for DGFE (initialized to 4/255 ≈ 0.0156862)
        self.learnable_thresh = nn.Parameter(torch.tensor(learnable_thresh), requires_grad=True)
        
        # MSE loss for reconstruction
        self.reconstruction_loss = nn.MSELoss(reduction='mean')
        
        self.num_classes = num_classes
    
    def extract_features(self, images):
        """
        Extract FPN features from backbone
        
        Args:
            images: Preprocessed images (ImageList)
        
        Returns:
            features: OrderedDict with keys '0', '1', '2', '3', '4' for P2-P6
        """
        # Extract backbone features
        features = self.base_model.backbone(images.tensors)
        return features
    
    def forward(self, images, targets=None):
        """
        Forward pass with SR-TOD enhancement
        
        Args:
            images (list[Tensor] or Tensor): Input images
                - List of tensors with shape (3, H, W) where H, W can vary
                - Or batched tensor (N, 3, H, W) if all same size
            targets (list[dict], optional): Ground truth for training
                Each dict contains:
                - 'boxes': FloatTensor[N, 4] in (x1, y1, x2, y2) format
                - 'labels': Int64Tensor[N] class labels
        
        Returns:
            If training (targets provided):
                losses (dict): Dictionary with loss components
                    - 'loss_reconstruction': MSE reconstruction loss
                    - 'loss_objectness': RPN objectness loss
                    - 'loss_rpn_box_reg': RPN box regression loss
                    - 'loss_classifier': ROI classifier loss
                    - 'loss_box_reg': ROI box regression loss
            
            If inference (targets=None):
                predictions (list[dict]): One dict per image
                    - 'boxes': FloatTensor[N, 4] detected boxes
                    - 'labels': Int64Tensor[N] predicted classes
                    - 'scores': FloatTensor[N] confidence scores
        """
        
        # Store original images for reconstruction loss
        # Need to handle both list and tensor inputs
        if isinstance(images, list):
            # Convert list of tensors to batch (pad if needed)
            original_images = images
        else:
            # Already a tensor
            original_images = images
        
        # Preprocess images through transform
        images_transformed, targets_processed = self.base_model.transform(images, targets)
        
        # Extract FPN features
        features = self.extract_features(images_transformed)
        
        # SR-TOD: Image reconstruction from P2 features
        # P2 is features['0'] in FPN output
        p2_features = features['0'].clone()
        reconstructed_image = self.rh(p2_features)
        
        # Compute difference map: |reconstructed - original| / 3
        # Normalize by 3 channels to get grayscale difference
        difference_map = torch.sum(
            torch.abs(reconstructed_image - images_transformed.tensors), 
            dim=1, 
            keepdim=True
        ) / 3
        
        # SR-TOD: Apply DGFE to enhance P2 features
        enhanced_p2 = self.dgfe(features['0'], difference_map, self.learnable_thresh)
        
        # Replace P2 with enhanced version
        features['0'] = enhanced_p2
        
        # Training mode
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be provided")
            
            # Get detection losses from base model
            # RPN losses
            proposals, proposal_losses = self.base_model.rpn(images_transformed, features, targets_processed)
            
            # ROI head losses
            detections, detector_losses = self.base_model.roi_heads(features, proposals, images_transformed.image_sizes, targets_processed)
            
            # Combine all losses
            losses = {}
            losses.update(proposal_losses)
            losses.update(detector_losses)
            
            # Add reconstruction loss
            loss_reconstruction = self.reconstruction_loss(reconstructed_image, images_transformed.tensors)
            losses['loss_reconstruction'] = loss_reconstruction
            
            return losses
        
        # Inference mode
        else:
            # Get proposals from RPN
            proposals, _ = self.base_model.rpn(images_transformed, features, None)
            
            # Get detections from ROI head
            detections, _ = self.base_model.roi_heads(features, proposals, images_transformed.image_sizes, None)
            
            # Post-process detections
            detections = self.base_model.transform.postprocess(detections, images_transformed.image_sizes, 
                                                                [(img.shape[1], img.shape[2]) for img in original_images] 
                                                                if isinstance(original_images, list) 
                                                                else [(original_images.shape[2], original_images.shape[3])] * original_images.shape[0])
            
            return detections


if __name__ == "__main__":
    # Test the SR-TOD model
    print("Testing FasterRCNN_SRTOD...")
    
    # Create model
    model = FasterRCNN_SRTOD(num_classes=11, learnable_thresh=0.0156862, pretrained_backbone=False)
    model.eval()
    
    # Test inference
    print("\n1. Testing inference mode...")
    images = [torch.randn(3, 640, 640) for _ in range(2)]
    
    with torch.no_grad():
        predictions = model(images)
    
    print(f"   Number of predictions: {len(predictions)}")
    for i, pred in enumerate(predictions):
        print(f"   Image {i}: {len(pred['boxes'])} detections")
    
    # Test training
    print("\n2. Testing training mode...")
    model.train()
    
    targets = [
        {'boxes': torch.tensor([[10, 20, 50, 60], [100, 150, 200, 250]], dtype=torch.float32),
         'labels': torch.tensor([1, 2], dtype=torch.int64)},
        {'boxes': torch.tensor([[30, 40, 80, 90]], dtype=torch.float32),
         'labels': torch.tensor([3], dtype=torch.int64)}
    ]
    
    losses = model(images, targets)
    
    print(f"   Losses: {list(losses.keys())}")
    print(f"   Total loss: {sum(losses.values()):.4f}")
    print(f"   - Reconstruction loss: {losses['loss_reconstruction']:.4f}")
    print(f"   - RPN losses: objectness={losses['loss_objectness']:.4f}, box_reg={losses['loss_rpn_box_reg']:.4f}")
    print(f"   - ROI losses: classifier={losses['loss_classifier']:.4f}, box_reg={losses['loss_box_reg']:.4f}")
    
    print(f"\n3. Learnable threshold: {model.learnable_thresh.item():.6f} ({model.learnable_thresh.item() * 255:.2f}/255)")
    
    print("\n✓ FasterRCNN_SRTOD test passed!")
