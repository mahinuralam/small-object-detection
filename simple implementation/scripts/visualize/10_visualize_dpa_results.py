"""
Visualize DPA-Enhanced Model Results
Side-by-side comparison of ground truth vs DPA predictions
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
from pathlib import Path
import random
import importlib.util

# Import dataset module
spec = importlib.util.spec_from_file_location("visdrone_dataset", Path(__file__).parent / "4_visdrone_dataset.py")
visdrone_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visdrone_module)
VisDroneDataset = visdrone_module.VisDroneDataset


class SimplifiedDPAModule(nn.Module):
    """Simplified Dual-Path Attention - same as in training script"""
    def __init__(self, channels):
        super().__init__()
        
        self.edge_conv3 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.edge_conv5 = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels)
        
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        identity = x
        edge3 = self.edge_conv3(x)
        edge5 = self.edge_conv5(x)
        edge_features = edge3 + edge5
        spatial_weight = self.spatial_att(edge_features)
        edge_features = edge_features * spatial_weight
        channel_weight = self.channel_att(x)
        semantic_features = x * channel_weight
        combined = torch.cat([edge_features, semantic_features], dim=1)
        out = self.fusion(combined)
        return out + identity


class FasterRCNN_with_SimpleDPA(nn.Module):
    """Faster R-CNN with memory-efficient enhancement"""
    def __init__(self, base_model, fpn_channels=256):
        super().__init__()
        self.base_model = base_model
        
        self.enhancers = nn.ModuleDict({
            '0': SimplifiedDPAModule(fpn_channels),
            '1': SimplifiedDPAModule(fpn_channels),
        })
        
    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        original_images = images
        images, targets = self.base_model.transform(images, targets)
        
        features = self.base_model.backbone(images.tensors)
        
        enhanced_features = {}
        for name, feat in features.items():
            if name in self.enhancers:
                enhanced_features[name] = self.enhancers[name](feat)
            else:
                enhanced_features[name] = feat
        
        proposals, proposal_losses = self.base_model.rpn(images, enhanced_features, targets)
        detections, detector_losses = self.base_model.roi_heads(
            enhanced_features, proposals, images.image_sizes, targets
        )
        
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


def load_model_with_dpa(checkpoint_path, num_classes, device):
    """Load DPA-enhanced model"""
    base_model = fasterrcnn_resnet50_fpn(weights=None, trainable_backbone_layers=5)
    in_features = base_model.roi_heads.box_predictor.cls_score.in_features
    base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    
    model = FasterRCNN_with_SimpleDPA(base_model, fpn_channels=256)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


# Class names and colors
CLASS_NAMES = ['background', 'pedestrian', 'people', 'bicycle', 'car', 'van',
               'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']

CLASS_COLORS = {
    1: (255, 0, 0),      # pedestrian - Red
    2: (0, 255, 0),      # people - Green
    3: (0, 0, 255),      # bicycle - Blue
    4: (255, 255, 0),    # car - Cyan
    5: (255, 0, 255),    # van - Magenta
    6: (0, 255, 255),    # truck - Yellow
    7: (128, 0, 128),    # tricycle - Purple
    8: (255, 128, 0),    # awning-tricycle - Orange
    9: (0, 128, 255),    # bus - Sky blue
    10: (128, 255, 0)    # motor - Lime
}


def draw_boxes(image, boxes, labels, scores=None, title="", color_map=CLASS_COLORS):
    """Draw bounding boxes on image"""
    img = image.copy()
    
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = map(int, box)
        label_int = int(label)
        
        # Get color
        color = color_map.get(label_int, (255, 255, 255))
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Create label text
        label_text = CLASS_NAMES[label_int]
        if scores is not None and idx < len(scores):
            label_text += f" {scores[idx]:.2f}"
        
        # Draw label background
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(img, (x1, y1 - text_height - 4), 
                     (x1 + text_width, y1), color, -1)
        
        # Draw label text
        cv2.putText(img, label_text, (x1, y1 - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add title
    if title:
        cv2.putText(img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 255, 0), 2)
    
    return img


def visualize_comparison(image, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores,
                        output_path, conf_threshold=0.3):
    """Create side-by-side comparison"""
    
    # Filter predictions by confidence
    keep = pred_scores >= conf_threshold
    pred_boxes_filtered = pred_boxes[keep]
    pred_labels_filtered = pred_labels[keep]
    pred_scores_filtered = pred_scores[keep]
    
    # Draw ground truth
    gt_image = draw_boxes(
        image, gt_boxes, gt_labels,
        title=f"Ground Truth ({len(gt_boxes)} objects)"
    )
    
    # Draw predictions
    pred_image = draw_boxes(
        image, pred_boxes_filtered, pred_labels_filtered, pred_scores_filtered,
        title=f"DPA Predictions ({len(pred_boxes_filtered)} detections, conf>={conf_threshold})"
    )
    
    # Combine side by side
    combined = np.hstack([gt_image, pred_image])
    
    # Save
    cv2.imwrite(str(output_path), combined)
    
    return len(gt_boxes), len(pred_boxes_filtered)


def main():
    print("="*80)
    print("DPA-ENHANCED MODEL VISUALIZATION")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Configuration
    dataset_root = '../dataset/VisDrone-2018'
    model_path = 'outputs_dpa/best_model_dpa.pth'
    output_dir = Path('visualizations_dpa')
    output_dir.mkdir(exist_ok=True)
    
    num_samples = 12
    conf_threshold = 0.3
    
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Samples: {num_samples}")
    print(f"Confidence threshold: {conf_threshold}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = VisDroneDataset(
        root_dir=dataset_root,
        split='val',
        min_size=5
    )
    print(f"✓ Loaded {len(dataset)} images")
    
    # Load DPA model
    print("\nLoading DPA-enhanced model...")
    model = load_model_with_dpa(model_path, num_classes=10, device=device)
    print(f"✓ Model loaded")
    
    # Select random samples
    random.seed(42)
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    print(f"\nGenerating visualizations...")
    print("-" * 80)
    
    total_gt = 0
    total_pred = 0
    
    for i, idx in enumerate(sample_indices, 1):
        # Get data
        image_tensor, target = dataset[idx]
        image = image_tensor.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8).copy()
        
        # Get ground truth
        gt_boxes = target['boxes'].numpy()
        gt_labels = target['labels'].numpy()
        
        # Get predictions
        with torch.no_grad():
            image_input = image_tensor.to(device)
            predictions = model([image_input])
        
        pred = predictions[0]
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        
        # Create visualization
        output_path = output_dir / f'comparison_{i:02d}.jpg'
        num_gt, num_pred = visualize_comparison(
            image, gt_boxes, gt_labels,
            pred_boxes, pred_labels, pred_scores,
            output_path, conf_threshold
        )
        
        total_gt += num_gt
        total_pred += num_pred
        
        print(f"[{i}/{num_samples}] Sample {idx}: {num_gt} GT, {num_pred} predictions → {output_path.name}")
    
    print("-" * 80)
    print(f"\n✓ Generated {num_samples} visualizations in {output_dir}/")
    print(f"\nStatistics:")
    print(f"  Average GT per image: {total_gt/num_samples:.1f}")
    print(f"  Average predictions per image: {total_pred/num_samples:.1f}")
    print(f"  Detection coverage: {total_pred/total_gt*100:.1f}%")
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
