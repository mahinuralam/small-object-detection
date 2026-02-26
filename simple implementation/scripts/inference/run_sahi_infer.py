"""
Run SAHI Pipeline on Single Image
"""
import torch
import numpy as np
from pathlib import Path
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs.sahi_config import SAHIPipelineConfig, get_preset_config
from models.sahi_pipeline import SAHIPipeline


def load_image(image_path):
    """Load image as numpy array"""
    img = Image.open(image_path).convert('RGB')
    return np.array(img)


def visualize_detections(image, detections, save_path=None, class_names=None):
    """Visualize detections on image"""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    boxes = detections['boxes'].cpu().numpy()
    scores = detections['scores'].cpu().numpy()
    labels = detections['labels'].cpu().numpy()
    
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # Draw box
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Label
        label_text = f"{int(label)}: {score:.2f}"
        if class_names and int(label) < len(class_names):
            label_text = f"{class_names[int(label)]}: {score:.2f}"
        
        ax.text(
            x1, y1 - 5,
            label_text,
            color='white', fontsize=8,
            bbox=dict(facecolor='red', alpha=0.8, edgecolor='none', pad=2)
        )
    
    ax.axis('off')
    ax.set_title(f"Detections: {len(boxes)}")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_detections_json(detections, metadata, save_path):
    """Save detections to JSON"""
    output = {
        'detections': [
            {
                'box': [float(x) for x in box],
                'score': float(score),
                'label': int(label)
            }
            for box, score, label in zip(
                detections['boxes'].cpu().numpy(),
                detections['scores'].cpu().numpy(),
                detections['labels'].cpu().numpy()
            )
        ],
        'metadata': {
            'uncertainty': float(metadata['U_t']),
            'sahi_triggered': bool(metadata['triggered']),
            'num_tiles': int(metadata['num_tiles']),
            'num_base_detections': int(metadata['num_base_dets']),
            'num_sahi_detections': int(metadata['num_sahi_dets']),
            'num_final_detections': int(metadata['num_final_dets']),
            'latency_ms': float(metadata['latency_ms']),
            'timings': {k: float(v) for k, v in metadata['timings'].items()}
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Saved detections to {save_path}")


def main(args):
    """Main inference function"""
    
    # Load config
    if args.config:
        # Load from file (TODO: implement YAML loading)
        print(f"Loading config from {args.config}")
        config = SAHIPipelineConfig()
    elif args.preset:
        print(f"Using preset: {args.preset}")
        config = get_preset_config(args.preset)
    else:
        print("Using default config")
        config = SAHIPipelineConfig()
    
    # Override with command-line args
    if args.detector_checkpoint:
        config.detector_checkpoint = args.detector_checkpoint
    if args.reconstructor_checkpoint:
        config.reconstructor_checkpoint = args.reconstructor_checkpoint
    if args.theta is not None:
        config.theta = args.theta
    if args.debug:
        config.debug = True
        config.debug_dir = args.debug_dir
    
    config.device = args.device
    
    print("\n" + "="*60)
    print("SAHI Pipeline Configuration")
    print("="*60)
    print(config)
    print("="*60 + "\n")
    
    # Initialize pipeline
    pipeline = SAHIPipeline(config)
    
    # Load image
    print(f"Loading image: {args.image}")
    image = load_image(args.image)
    print(f"Image shape: {image.shape}")
    
    # Run inference
    print("\nRunning inference...")
    detections, metadata = pipeline.process_image(image)
    
    # Print results
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    print(f"Uncertainty (U_t): {metadata['U_t']:.4f}")
    print(f"SAHI Triggered: {metadata['triggered']}")
    if metadata['triggered']:
        print(f"Number of Tiles: {metadata['num_tiles']}")
        print(f"Base Detections: {metadata['num_base_dets']}")
        print(f"SAHI Detections: {metadata['num_sahi_dets']}")
    print(f"Final Detections: {metadata['num_final_dets']}")
    print(f"Total Latency: {metadata['latency_ms']:.1f} ms")
    print("\nTiming Breakdown:")
    for component, time_ms in metadata['timings'].items():
        print(f"  {component:20s}: {time_ms:6.1f} ms")
    print("="*60 + "\n")
    
    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = output_dir / f"{Path(args.image).stem}_detections.json"
    save_detections_json(detections, metadata, json_path)
    
    # Visualize
    if args.visualize:
        vis_path = output_dir / f"{Path(args.image).stem}_visualization.png"
        class_names = [
            'pedestrian', 'people', 'bicycle', 'car', 'van',
            'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
        ] if config.num_classes == 11 else None
        visualize_detections(image, detections, vis_path, class_names)
    
    print(f"✓ All outputs saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SAHI Pipeline on Image')
    
    # Input/Output
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='results/sahi_inference',
                       help='Output directory')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization')
    
    # Config
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (YAML)')
    parser.add_argument('--preset', type=str, choices=['fast', 'balanced', 'accurate'],
                       help='Use preset config')
    
    # Model Checkpoints
    parser.add_argument('--detector_checkpoint', type=str, default=None,
                       help='Path to detector checkpoint')
    parser.add_argument('--reconstructor_checkpoint', type=str, default=None,
                       help='Path to reconstructor checkpoint')
    
    # Override Parameters
    parser.add_argument('--theta', type=float, default=None,
                       help='Uncertainty threshold')
    
    # Debug
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--debug_dir', type=str, default='results/debug',
                       help='Debug output directory')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    main(args)
