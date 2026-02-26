"""
SR-TOD Visualization: Reconstruction Process and Detection Comparison
Shows how SR-TOD reconstructs images to enhance small object detection
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

# Add models directory to path
models_dir = Path(__file__).parent.parent.parent / "models"
sys.path.insert(0, str(models_dir))
sys.path.insert(0, str(models_dir / "enhancements"))

# Import dataset module
import importlib.util
dataset_path = Path(__file__).parent.parent / "4_visdrone_dataset.py"
spec = importlib.util.spec_from_file_location("visdrone_dataset", dataset_path)
visdrone_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visdrone_module)
VisDroneDataset = visdrone_module.VisDroneDataset

# Import enhancements modules
from dpa_module import SimplifiedDPAModule
from reconstruction_head import ReconstructionHead
from dgfe_module import DGFE

# Import models
from srtod_model import FasterRCNN_SRTOD
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Manually load DPA model to avoid import issues
def load_dpa_model_class():
    """Load DPA model class dynamically"""
    import torch.nn as nn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    
    class FasterRCNN_DPA(nn.Module):
        def __init__(self, num_classes=11, pretrained_backbone=True):
            super().__init__()
            self.fasterrcnn = fasterrcnn_resnet50_fpn(
                pretrained=False,
                pretrained_backbone=pretrained_backbone,
                num_classes=num_classes,
                min_size=800,
                max_size=1333
            )
            
            # Add DPA module after C5 features
            self.dpa = SimplifiedDPAModule(channels=2048)
            self._hook_handle = None
            self._dpa_features = None
            
        def _dpa_hook(self, module, input, output):
            self._dpa_features = self.dpa(output)
            return self._dpa_features
            
        def forward(self, images, targets=None):
            if self._hook_handle is None:
                layer4 = self.fasterrcnn.backbone.body.layer4
                self._hook_handle = layer4.register_forward_hook(self._dpa_hook)
            
            if self.training and targets is not None:
                loss_dict = self.fasterrcnn(images, targets)
                return loss_dict
            else:
                self.fasterrcnn.eval()
                detections = self.fasterrcnn(images)
                return detections
    
    return FasterRCNN_DPA

FasterRCNN_DPA = load_dpa_model_class()


def collate_fn(batch):
    """Custom collate function"""
    return tuple(zip(*batch))


def denormalize_image(img_tensor):
    """Convert tensor to displayable numpy image"""
    img = img_tensor.cpu().numpy()
    if img.shape[0] == 3:  # CHW to HWC
        img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def get_srtod_reconstruction(model, image_tensor, device):
    """Get reconstruction outputs from SR-TOD model"""
    model.eval()
    with torch.no_grad():
        image = image_tensor.unsqueeze(0).to(device)
        
        # Get backbone features
        features = model.backbone(image)
        
        # Get P2 features (lowest FPN level)
        p2_features = features['0']
        
        # Reconstruct image
        reconstructed = model.rh(p2_features)
        
        # Calculate difference map
        difference_map = torch.sum(
            torch.abs(reconstructed - image), 
            dim=1, 
            keepdim=True
        ) / 3.0
        
        # Get threshold
        threshold = model.learnable_thresh.item()
        
        # Create binary mask
        mask = ((torch.sign(difference_map - threshold) + 1) * 0.5)
        
        return {
            'original': image.squeeze(0),
            'reconstructed': reconstructed.squeeze(0),
            'difference_map': difference_map.squeeze(0),
            'binary_mask': mask.squeeze(0),
            'threshold': threshold
        }


def visualize_reconstruction_process(recon_data, save_path):
    """Visualize the SR-TOD reconstruction process"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SR-TOD Reconstruction Process for Small Object Detection', 
                 fontsize=16, fontweight='bold')
    
    # Original image
    original = denormalize_image(recon_data['original'])
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('(a) Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Reconstructed image
    reconstructed = denormalize_image(recon_data['reconstructed'])
    axes[0, 1].imshow(reconstructed)
    axes[0, 1].set_title('(b) Reconstructed from P2 Features', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Difference map (grayscale)
    diff_map = recon_data['difference_map'].squeeze().cpu().numpy()
    im1 = axes[0, 2].imshow(diff_map, cmap='hot', vmin=0, vmax=diff_map.max())
    axes[0, 2].set_title(f'(c) Difference Map (max={diff_map.max():.3f})', 
                         fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046)
    
    # Binary mask (after threshold)
    binary_mask = recon_data['binary_mask'].squeeze().cpu().numpy()
    axes[1, 0].imshow(binary_mask, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f'(d) Binary Mask (thresh={recon_data["threshold"]:.4f})', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Overlay: Original + Difference Map
    overlay1 = original.copy()
    diff_colored = cv2.applyColorMap((diff_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)
    overlay1 = cv2.addWeighted(overlay1, 0.6, diff_colored, 0.4, 0)
    axes[1, 1].imshow(overlay1)
    axes[1, 1].set_title('(e) Original + Difference Overlay', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Overlay: Original + Binary Mask
    overlay2 = original.copy()
    mask_colored = np.zeros_like(overlay2)
    mask_colored[:, :, 0] = binary_mask * 255  # Red channel
    overlay2 = cv2.addWeighted(overlay2, 0.7, mask_colored, 0.3, 0)
    axes[1, 2].imshow(overlay2)
    axes[1, 2].set_title('(f) Original + Binary Mask (Red)', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved reconstruction process to: {save_path}")


def draw_boxes_on_image(image_np, boxes, labels, scores, class_names, score_thresh=0.5):
    """Draw bounding boxes on image"""
    img = image_np.copy()
    h, w = img.shape[:2]
    
    # Colors for different classes
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
            
        x1, y1, x2, y2 = box
        color = (colors[label % 10][:3] * 255).astype(int)
        color = tuple([int(c) for c in color])
        
        # Draw box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw label
        text = f"{class_names[label]}: {score:.2f}"
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                               font_scale, thickness)
        
        # Background for text
        cv2.rectangle(img, (int(x1), int(y1) - text_h - 4), 
                     (int(x1) + text_w, int(y1)), color, -1)
        cv2.putText(img, text, (int(x1), int(y1) - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    return img


def visualize_side_by_side_detections(image_tensor, predictions_dict, gt_boxes, gt_labels, 
                                      class_names, save_path, score_thresh=0.5):
    """Visualize detection results side-by-side"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle('Detection Comparison: Baseline vs DPA vs SR-TOD', 
                 fontsize=16, fontweight='bold')
    
    original = denormalize_image(image_tensor)
    
    # Original with GT
    img_gt = original.copy()
    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box
        cv2.rectangle(img_gt, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img_gt, class_names[label], (int(x1), int(y1) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    axes[0, 0].imshow(img_gt)
    axes[0, 0].set_title(f'Ground Truth ({len(gt_boxes)} objects)', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Baseline
    if 'baseline' in predictions_dict:
        pred = predictions_dict['baseline']
        img_baseline = draw_boxes_on_image(
            original, pred['boxes'], pred['labels'], pred['scores'],
            class_names, score_thresh
        )
        count = (pred['scores'] >= score_thresh).sum()
        axes[0, 1].imshow(img_baseline)
        axes[0, 1].set_title(f'Baseline Faster R-CNN ({count} detections)', 
                            fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
    
    # DPA
    if 'dpa' in predictions_dict:
        pred = predictions_dict['dpa']
        img_dpa = draw_boxes_on_image(
            original, pred['boxes'], pred['labels'], pred['scores'],
            class_names, score_thresh
        )
        count = (pred['scores'] >= score_thresh).sum()
        axes[1, 0].imshow(img_dpa)
        axes[1, 0].set_title(f'DPA-Enhanced ({count} detections)', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
    
    # SR-TOD
    if 'srtod' in predictions_dict:
        pred = predictions_dict['srtod']
        img_srtod = draw_boxes_on_image(
            original, pred['boxes'], pred['labels'], pred['scores'],
            class_names, score_thresh
        )
        count = (pred['scores'] >= score_thresh).sum()
        axes[1, 1].imshow(img_srtod)
        axes[1, 1].set_title(f'SR-TOD ({count} detections)', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved detection comparison to: {save_path}")


def main():
    """Main visualization function"""
    
    # Configuration
    config = {
        'dataset_root': Path(__file__).parent.parent.parent.parent / 'dataset' / 'VisDrone-2018',
        'baseline_path': Path(__file__).parent.parent.parent / 'results' / 'outputs' / 'best_model.pth',
        'dpa_path': Path(__file__).parent.parent.parent / 'results' / 'outputs_dpa' / 'best_model_dpa.pth',
        'srtod_path': Path(__file__).parent.parent.parent / 'results' / 'outputs_srtod' / 'best_model_srtod.pth',
        'output_dir': Path(__file__).parent.parent.parent / 'results' / 'visualizations_srtod',
        'num_samples': 10,
        'min_size': 5,
        'score_thresh': 0.5
    }
    
    config['output_dir'].mkdir(exist_ok=True, parents=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"SR-TOD Visualization")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Output directory: {config['output_dir']}")
    print(f"{'='*80}\n")
    
    # Class names
    CLASS_NAMES = [
        'background', 'pedestrian', 'people', 'bicycle', 'car', 'van',
        'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
    ]
    
    # Load dataset
    print("Loading validation dataset...")
    val_dataset = VisDroneDataset(
        root_dir=config['dataset_root'],
        split='val',
        min_size=config['min_size']
    )
    print(f"✓ Loaded {len(val_dataset)} validation images\n")
    
    # Load models
    print("Loading models...")
    
    # Baseline
    baseline_model = fasterrcnn_resnet50_fpn(
        pretrained=False,
        num_classes=11,
        min_size=800,
        max_size=1333
    )
    checkpoint = torch.load(config['baseline_path'], map_location=device)
    baseline_model.load_state_dict(checkpoint['model_state_dict'])
    baseline_model.to(device)
    baseline_model.eval()
    print(f"✓ Loaded Baseline (epoch {checkpoint['epoch']}, val_loss {checkpoint['val_loss']:.4f})")
    
    # DPA
    dpa_model = FasterRCNN_DPA(num_classes=11, pretrained_backbone=False)
    checkpoint = torch.load(config['dpa_path'], map_location=device)
    dpa_model.load_state_dict(checkpoint['model_state_dict'])
    dpa_model.to(device)
    dpa_model.eval()
    print(f"✓ Loaded DPA (epoch {checkpoint['epoch']}, val_loss {checkpoint['val_loss']:.4f})")
    
    # SR-TOD
    srtod_model = FasterRCNN_SRTOD(num_classes=11, pretrained_backbone=False)
    checkpoint = torch.load(config['srtod_path'], map_location=device)
    srtod_model.load_state_dict(checkpoint['model_state_dict'])
    srtod_model.to(device)
    srtod_model.eval()
    print(f"✓ Loaded SR-TOD (epoch {checkpoint['epoch']}, val_loss {checkpoint['val_loss']:.4f})")
    print(f"  - Learnable threshold: {checkpoint['learnable_thresh']:.6f}\n")
    
    # Select samples with many small objects
    print("Selecting samples with small objects...")
    samples_with_small = []
    for idx in range(len(val_dataset)):
        img, target = val_dataset[idx]
        boxes = target['boxes']
        
        # Calculate box sizes
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        
        # Count small objects (area < 32x32)
        small_count = (areas < 32*32).sum().item()
        
        if small_count >= 5:  # At least 5 small objects
            samples_with_small.append((idx, small_count, len(boxes)))
    
    # Sort by number of small objects
    samples_with_small.sort(key=lambda x: x[1], reverse=True)
    selected_indices = [x[0] for x in samples_with_small[:config['num_samples']]]
    
    print(f"✓ Selected {len(selected_indices)} samples with many small objects\n")
    
    # Process samples
    print("Generating visualizations...")
    for i, idx in enumerate(tqdm(selected_indices, desc="Processing")):
        image, target = val_dataset[idx]
        image_tensor = image.clone()
        image = image.to(device)
        
        # Get SR-TOD reconstruction
        recon_data = get_srtod_reconstruction(srtod_model, image, device)
        
        # Save reconstruction visualization
        recon_path = config['output_dir'] / f'sample_{i+1:02d}_reconstruction.png'
        visualize_reconstruction_process(recon_data, recon_path)
        
        # Get predictions from all models
        with torch.no_grad():
            baseline_pred = baseline_model([image])[0]
            dpa_pred = dpa_model([image])[0]
            srtod_pred = srtod_model([image])[0]
        
        predictions = {
            'baseline': {
                'boxes': baseline_pred['boxes'].cpu().numpy(),
                'labels': baseline_pred['labels'].cpu().numpy(),
                'scores': baseline_pred['scores'].cpu().numpy()
            },
            'dpa': {
                'boxes': dpa_pred['boxes'].cpu().numpy(),
                'labels': dpa_pred['labels'].cpu().numpy(),
                'scores': dpa_pred['scores'].cpu().numpy()
            },
            'srtod': {
                'boxes': srtod_pred['boxes'].cpu().numpy(),
                'labels': srtod_pred['labels'].cpu().numpy(),
                'scores': srtod_pred['scores'].cpu().numpy()
            }
        }
        
        # Save detection comparison
        det_path = config['output_dir'] / f'sample_{i+1:02d}_detection_comparison.png'
        visualize_side_by_side_detections(
            image_tensor,
            predictions,
            target['boxes'].numpy(),
            target['labels'].numpy(),
            CLASS_NAMES,
            det_path,
            config['score_thresh']
        )
    
    print(f"\n{'='*80}")
    print(f"✓ Generated {len(selected_indices) * 2} visualizations")
    print(f"✓ Saved to: {config['output_dir']}")
    print(f"{'='*80}\n")
    
    # Create summary
    summary_path = config['output_dir'] / 'VISUALIZATION_SUMMARY.txt'
    with open(summary_path, 'w') as f:
        f.write("SR-TOD VISUALIZATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write("Generated Visualizations:\n")
        f.write(f"  - Reconstruction process: {len(selected_indices)} images\n")
        f.write(f"  - Detection comparisons: {len(selected_indices)} images\n\n")
        f.write("Files:\n")
        for i in range(len(selected_indices)):
            f.write(f"  Sample {i+1}:\n")
            f.write(f"    - sample_{i+1:02d}_reconstruction.png\n")
            f.write(f"    - sample_{i+1:02d}_detection_comparison.png\n")
        f.write("\nReconstruction Process Visualization:\n")
        f.write("  (a) Original Image\n")
        f.write("  (b) Reconstructed from P2 Features\n")
        f.write("  (c) Difference Map (highlights discrepancies)\n")
        f.write("  (d) Binary Mask (after learnable threshold)\n")
        f.write("  (e) Original + Difference Overlay\n")
        f.write("  (f) Original + Binary Mask\n\n")
        f.write("Detection Comparison:\n")
        f.write("  - Ground Truth\n")
        f.write("  - Baseline Faster R-CNN\n")
        f.write("  - DPA-Enhanced\n")
        f.write("  - SR-TOD\n\n")
        f.write("Note: Red highlights in binary mask indicate regions where\n")
        f.write("      the reconstruction differs significantly from original,\n")
        f.write("      suggesting the presence of small/difficult objects.\n")
    
    print(f"✓ Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
