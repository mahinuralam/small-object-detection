"""
Run SAHI Pipeline on Video
"""
import torch
import numpy as np
from pathlib import Path
import argparse
import json
import cv2
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs.sahi_config import SAHIPipelineConfig, get_preset_config
from models.sahi_pipeline import SAHIPipeline


def process_video(pipeline, video_path, output_json=None, save_video=None, 
                  show_progress=True):
    """Process video frame by frame"""
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}×{height}, {fps:.1f} FPS, {total_frames} frames")
    
    # Video writer (if saving)
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_video, fourcc, fps, (width, height))
    
    # Results storage
    all_results = []
    
    # Statistics
    total_latency = 0.0
    num_triggered = 0
    
    # Process frames
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Processing video") if show_progress else None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        detections, metadata = pipeline.process_image(frame_rgb)
        
        # Statistics
        total_latency += metadata['latency_ms']
        if metadata['triggered']:
            num_triggered += 1
        
        # Store results
        frame_result = {
            'frame_id': frame_idx,
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
                'num_detections': int(metadata['num_final_dets']),
                'latency_ms': float(metadata['latency_ms'])
            }
        }
        all_results.append(frame_result)
        
        # Visualize on frame
        if writer:
            # Draw detections
            for box, score, label in zip(
                detections['boxes'].cpu().numpy(),
                detections['scores'].cpu().numpy(),
                detections['labels'].cpu().numpy()
            ):
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0) if not metadata['triggered'] else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label_text = f"{int(label)}: {score:.2f}"
                cv2.putText(frame, label_text, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add info text
            info_text = f"Frame {frame_idx} | U_t={metadata['U_t']:.3f} | " \
                       f"{'SAHI' if metadata['triggered'] else 'BASE'} | " \
                       f"{metadata['latency_ms']:.0f}ms"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            writer.write(frame)
        
        frame_idx += 1
        if pbar:
            pbar.update(1)
            pbar.set_postfix({
                'U_t': f"{metadata['U_t']:.3f}",
                'SAHI': '✓' if metadata['triggered'] else '✗',
                'dets': metadata['num_final_dets']
            })
    
    cap.release()
    if writer:
        writer.release()
    if pbar:
        pbar.close()
    
    # Summary statistics
    avg_latency = total_latency / frame_idx if frame_idx > 0 else 0
    trigger_rate = num_triggered / frame_idx if frame_idx > 0 else 0
    
    summary = {
        'video_path': str(video_path),
        'total_frames': frame_idx,
        'avg_latency_ms': avg_latency,
        'avg_fps': 1000.0 / avg_latency if avg_latency > 0 else 0,
        'sahi_trigger_rate': trigger_rate,
        'num_sahi_triggered': num_triggered
    }
    
    # Save results
    if output_json:
        output = {
            'summary': summary,
            'frames': all_results
        }
        with open(output_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n✓ Saved results to {output_json}")
    
    return summary


def main(args):
    """Main function"""
    
    # Load config
    if args.config:
        config = SAHIPipelineConfig()
    elif args.preset:
        print(f"Using preset: {args.preset}")
        config = get_preset_config(args.preset)
    else:
        config = SAHIPipelineConfig()
    
    # Override
    if args.detector_checkpoint:
        config.detector_checkpoint = args.detector_checkpoint
    if args.reconstructor_checkpoint:
        config.reconstructor_checkpoint = args.reconstructor_checkpoint
    if args.theta is not None:
        config.theta = args.theta
    
    config.device = args.device
    config.debug = False  # Disable debug for video
    
    print("\n" + "="*60)
    print("SAHI Pipeline Configuration")
    print("="*60)
    print(config)
    print("="*60 + "\n")
    
    # Initialize pipeline
    pipeline = SAHIPipeline(config)
    
    # Create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare paths
    video_name = Path(args.video).stem
    output_json = Path(args.output_dir) / f"{video_name}_detections.json" if args.output_dir else None
    save_video = Path(args.output_dir) / f"{video_name}_annotated.mp4" if args.save_video and args.output_dir else None
    
    # Process video
    summary = process_video(
        pipeline,
        args.video,
        output_json=output_json,
        save_video=save_video,
        show_progress=True
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Video Processing Summary")
    print("="*60)
    print(f"Total Frames: {summary['total_frames']}")
    print(f"Avg Latency: {summary['avg_latency_ms']:.1f} ms")
    print(f"Avg FPS: {summary['avg_fps']:.1f}")
    print(f"SAHI Trigger Rate: {summary['sahi_trigger_rate']*100:.1f}%")
    print(f"Frames with SAHI: {summary['num_sahi_triggered']}/{summary['total_frames']}")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SAHI Pipeline on Video')
    
    # Input/Output
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video')
    parser.add_argument('--output_dir', type=str, default='results/sahi_video',
                       help='Output directory')
    parser.add_argument('--save_video', action='store_true',
                       help='Save annotated video')
    
    # Config
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--preset', type=str, choices=['fast', 'balanced', 'accurate'],
                       help='Use preset config')
    
    # Model Checkpoints
    parser.add_argument('--detector_checkpoint', type=str, default=None,
                       help='Path to detector checkpoint')
    parser.add_argument('--reconstructor_checkpoint', type=str, default=None,
                       help='Path to reconstructor checkpoint')
    
    # Parameters
    parser.add_argument('--theta', type=float, default=None,
                       help='Uncertainty threshold')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    main(args)
