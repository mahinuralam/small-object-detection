"""Evaluate CD-DPA on VisDrone (optionally with SAHI Stage-1 flow).

By default, this script evaluates plain CD-DPA on full images.
With ``--use-sahi``, it runs Stage 1 CD-DPA+SAHI and also computes
Stage 2 weak-tile statistics (bottom-K by confidence score).

Usage:
    python scripts/eval/18_evaluate_cddpa.py
    python scripts/eval/18_evaluate_cddpa.py --use-sahi --batch-size 1
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.cddpa_model import FasterRCNN_CDDPA
from models.sahi_pipeline import CDDPADetector, ConfidenceScoringTiler, SAHIInferenceRunner

# Import dataset module dynamically
spec = importlib.util.spec_from_file_location(
    "visdrone_dataset", Path(__file__).parent.parent / "4_visdrone_dataset.py"
)
visdrone_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visdrone_module)
VisDroneDataset = visdrone_module.VisDroneDataset


def collate_fn(batch):
    """Custom collate function"""
    return tuple(zip(*batch))


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """
    Evaluate model using TorchMetrics
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    # Initialize metric
    metric = MeanAveragePrecision(iou_type='bbox', class_metrics=True)
    
    print("\nEvaluating on validation set...")
    for images, targets in tqdm(dataloader, desc='Evaluating'):
        # Move to device
        images = [img.to(device) for img in images]
        
        # Inference
        predictions = model(images)
        
        # Move predictions and targets to CPU for metric computation
        preds = []
        targs = []
        for pred, target in zip(predictions, targets):
            preds.append({
                'boxes': pred['boxes'].cpu(),
                'scores': pred['scores'].cpu(),
                'labels': pred['labels'].cpu()
            })
            targs.append({
                'boxes': target['boxes'].cpu(),
                'labels': target['labels'].cpu()
            })
        
        # Update metric
        metric.update(preds, targs)
    
    # Compute final metrics
    print("\nComputing metrics...")
    results = metric.compute()
    
    return results


@torch.no_grad()
def evaluate_model_with_sahi(detector, dataloader, cfg):
    """Evaluate CD-DPA+SAHI Stage 1 and collect weak-tile stats for Stage 2."""
    metric = MeanAveragePrecision(iou_type='bbox', class_metrics=True)

    tiler = ConfidenceScoringTiler(
        tile_size=(cfg.tile_width, cfg.tile_height),
        overlap_width_ratio=cfg.overlap,
        overlap_height_ratio=cfg.overlap,
    )
    sahi_runner = SAHIInferenceRunner(
        detector=detector,
        merge_iou_thresh=cfg.tile_merge_iou,
        postprocess_type=cfg.postprocess,
        postprocess_match_metric=cfg.match_metric,
        postprocess_match_threshold=cfg.match_threshold,
    )

    weak_tile_means = []
    all_tile_means = []
    total_tiles = 0
    total_weak_tiles = 0

    print("\nEvaluating CD-DPA + SAHI (Stage 1) and weak-tile scoring (Stage 2)...")
    for images, targets in tqdm(dataloader, desc='Evaluating SAHI'):
        for image, target in zip(images, targets):
            image = image.to(cfg.device)
            _, h, w = image.shape

            # Stage 1a: full-image CD-DPA
            d_full = detector.predict(image)

            # Stage 1b/1c: tile CD-DPA and per-tile confidence
            tile_grid = tiler.generate_grid((h, w))
            tile_dets = []
            tile_confidences = []

            for x0, y0, x1, y1 in tile_grid:
                tile_img = image[:, y0:y1, x0:x1]
                tile_pred = detector.predict(tile_img)

                if len(tile_pred['boxes']) > 0:
                    remapped = {
                        'boxes': tile_pred['boxes'].clone(),
                        'scores': tile_pred['scores'].clone(),
                        'labels': tile_pred['labels'].clone(),
                    }
                    remapped['boxes'][:, [0, 2]] += x0
                    remapped['boxes'][:, [1, 3]] += y0
                    mean_conf = float(tile_pred['scores'].mean().item())
                else:
                    remapped = tile_pred
                    mean_conf = 0.0

                tile_dets.append(remapped)
                tile_confidences.append(mean_conf)

            # Stage 1d: merge full-image + tile detections
            all_boxes = [d_full['boxes']]
            all_scores = [d_full['scores']]
            all_labels = [d_full['labels']]

            for td in tile_dets:
                if len(td['boxes']) > 0:
                    all_boxes.append(td['boxes'])
                    all_scores.append(td['scores'])
                    all_labels.append(td['labels'])

            if len(all_boxes) == 1 and len(all_boxes[0]) == 0:
                pred = {
                    'boxes': torch.empty(0, 4),
                    'scores': torch.empty(0),
                    'labels': torch.empty(0, dtype=torch.long),
                }
            else:
                merged_boxes = torch.cat(all_boxes, dim=0)
                merged_scores = torch.cat(all_scores, dim=0)
                merged_labels = torch.cat(all_labels, dim=0)

                if cfg.postprocess == 'GREEDYNMM':
                    pred = sahi_runner._greedy_nmm(
                        boxes=merged_boxes,
                        scores=merged_scores,
                        labels=merged_labels,
                        metric=cfg.match_metric,
                        threshold=cfg.match_threshold,
                    )
                else:
                    pred = sahi_runner._class_wise_nms(
                        boxes=merged_boxes,
                        scores=merged_scores,
                        labels=merged_labels,
                        iou_thresh=cfg.tile_merge_iou,
                    )

            # Stage 2: select weak tiles from Stage-1 confidence outputs
            weak_tiles, _, tile_scores = tiler.select_weak(
                tile_grid=tile_grid,
                tile_confidences=tile_confidences,
                tile_detections=tile_dets,
                K=cfg.weak_k,
                min_expected=cfg.min_expected_dets,
            )

            if tile_scores:
                all_tile_means.append(float(sum(tile_scores) / len(tile_scores)))
            if weak_tiles:
                weak_scores = sorted(tile_scores)[:len(weak_tiles)]
                weak_tile_means.append(float(sum(weak_scores) / len(weak_scores)))

            total_tiles += len(tile_grid)
            total_weak_tiles += len(weak_tiles)

            metric.update(
                [{'boxes': pred['boxes'], 'scores': pred['scores'], 'labels': pred['labels']}],
                [{'boxes': target['boxes'].cpu(), 'labels': target['labels'].cpu()}],
            )

    results = metric.compute()
    extra = {
        'avg_total_tiles_per_image': (total_tiles / len(dataloader.dataset)) if len(dataloader.dataset) > 0 else 0.0,
        'avg_selected_weak_tiles_per_image': (total_weak_tiles / len(dataloader.dataset)) if len(dataloader.dataset) > 0 else 0.0,
        'avg_tile_score': (sum(all_tile_means) / len(all_tile_means)) if all_tile_means else 0.0,
        'avg_weak_tile_score': (sum(weak_tile_means) / len(weak_tile_means)) if weak_tile_means else 0.0,
    }
    return results, extra


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CD-DPA on VisDrone.')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=str(PROJECT_ROOT / 'results' / 'outputs_cddpa' / 'best_model_cddpa.pth'),
        help='Path to CD-DPA checkpoint',
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default=str(PROJECT_ROOT.parent / 'dataset' / 'VisDrone-2018'),
        help='VisDrone dataset root',
    )
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use-sahi', action='store_true', help='Evaluate using CD-DPA + SAHI Stage 1 flow')
    parser.add_argument('--tile-width', type=int, default=512)
    parser.add_argument('--tile-height', type=int, default=512)
    parser.add_argument('--overlap', type=float, default=0.35)
    parser.add_argument('--weak-k', type=int, default=15)
    parser.add_argument('--min-expected-dets', type=int, default=2)
    parser.add_argument('--score-thresh', type=float, default=0.05,
                        help='Detection score threshold used by SAHI detector wrapper')
    parser.add_argument('--postprocess', type=str, default='GREEDYNMM', choices=['GREEDYNMM', 'NMS'])
    parser.add_argument('--match-metric', type=str, default='IOS', choices=['IOS', 'IOU'])
    parser.add_argument('--match-threshold', type=float, default=0.5)
    parser.add_argument('--tile-merge-iou', type=float, default=0.6)
    return parser.parse_args()


def main():
    args = parse_args()

    # Configuration
    config = {
        'checkpoint_path': Path(args.checkpoint),
        'data_root': Path(args.data_root),
        'batch_size': args.batch_size,
        'device': args.device,
        'num_classes': 11,
    }
    
    print("=" * 80)
    mode = "CD-DPA + SAHI (Stage 1)" if args.use_sahi else "CD-DPA (full-image only)"
    print(f"CASCADED DEFORMABLE DUAL-PATH ATTENTION EVALUATION | {mode}")
    print("=" * 80)
    print(f"Device: {config['device']}")
    print(f"Checkpoint: {config['checkpoint_path']}")
    
    # Check GPU
    if config['device'] == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataset = VisDroneDataset(
        root_dir=config['data_root'],
        split='val',
        transforms=None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1 if args.use_sahi else config['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"✓ Loaded {len(val_dataset)} validation images")
    
    # Load model
    print("\nLoading model...")
    model = FasterRCNN_CDDPA(
        num_classes=config['num_classes'],
        enhance_levels=['0', '1', '2'],
        use_checkpoint=False,  # No checkpointing for inference
        pretrained=False
    )
    
    # Load checkpoint
    checkpoint = torch.load(config['checkpoint_path'], map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config['device'])
    model.eval()
    
    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Training val loss: {checkpoint['val_loss']:.4f}")
    
    # Evaluate
    print("=" * 80)
    if args.use_sahi:
        detector = CDDPADetector(
            checkpoint_path=str(config['checkpoint_path']),
            num_classes=config['num_classes'],
            device=config['device'],
            score_thresh=args.score_thresh,
        )
        results, sahi_stats = evaluate_model_with_sahi(detector, val_loader, args)
    else:
        results = evaluate_model(model, val_loader, config['device'])
        sahi_stats = None
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"mAP (IoU=0.50:0.95): {results['map'].item() * 100:.2f}%")
    print(f"mAP@0.50:            {results['map_50'].item() * 100:.2f}%")
    print(f"mAP@0.75:            {results['map_75'].item() * 100:.2f}%")
    
    # Per-class AP
    if 'map_per_class' in results:
        per_class = results['map_per_class']
        # TorchMetrics may return a scalar when class-wise stats are unavailable.
        if isinstance(per_class, torch.Tensor) and per_class.ndim > 0 and per_class.numel() > 0:
            print(f"\nPer-Class AP@0.50:")
            class_names = [
                'pedestrian', 'people', 'bicycle', 'car', 'van',
                'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
            ]
            class_ids = results.get('classes', None)
            for i, ap in enumerate(per_class):
                if class_ids is not None and i < len(class_ids):
                    cls_id = int(class_ids[i].item())
                    name = class_names[cls_id - 1] if 1 <= cls_id <= len(class_names) else f'class_{cls_id}'
                else:
                    name = class_names[i] if i < len(class_names) else f'class_{i + 1}'
                print(f"  {name:20s}: {ap.item() * 100:5.2f}%")
        else:
            print("\nPer-Class AP@0.50: unavailable in this run")
    
    # Comparison
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    baseline_map = 38.02
    simplified_dpa_map = 43.44
    cddpa_map = results['map_50'].item() * 100
    
    print(f"Baseline Faster R-CNN:     {baseline_map:.2f}% mAP@0.5")
    print(f"SimplifiedDPA:             {simplified_dpa_map:.2f}% mAP@0.5 (+{simplified_dpa_map - baseline_map:.2f}%)")
    print(f"CD-DPA (SOTA):             {cddpa_map:.2f}% mAP@0.5 (+{cddpa_map - baseline_map:.2f}%)")
    print(f"\nImprovement over SimplifiedDPA: {cddpa_map - simplified_dpa_map:+.2f}%")
    
    # Achievement check
    print("\n" + "=" * 80)
    if cddpa_map >= 48.0:
        print("🎉 TARGET ACHIEVED! CD-DPA reaches SOTA performance!")
    elif cddpa_map >= 46.0:
        print("✓ Strong performance! Close to SOTA target.")
    elif cddpa_map >= 44.0:
        print("✓ Good improvement over SimplifiedDPA.")
    else:
        print("⚠️  Below expected performance. Consider:")
        print("   - Longer training (more epochs)")
        print("   - Different learning rate")
        print("   - Fine-tuning cascade structure")
    print("=" * 80)
    
    # Save results
    results_dict = {
        'mAP': results['map'].item(),
        'mAP_50': results['map_50'].item(),
        'mAP_75': results['map_75'].item(),
        'mode': 'cddpa_sahi_stage1' if args.use_sahi else 'cddpa_full_image',
        'comparison': {
            'baseline': baseline_map / 100,
            'simplified_dpa': simplified_dpa_map / 100,
            'cddpa': cddpa_map / 100,
            'improvement_vs_baseline': (cddpa_map - baseline_map) / 100,
            'improvement_vs_simplified_dpa': (cddpa_map - simplified_dpa_map) / 100
        }
    }

    if sahi_stats is not None:
        print("\nSAHI/Weak-Tile Stats:")
        print(f"  avg_total_tiles_per_image:       {sahi_stats['avg_total_tiles_per_image']:.2f}")
        print(f"  avg_selected_weak_tiles_per_image: {sahi_stats['avg_selected_weak_tiles_per_image']:.2f}")
        print(f"  avg_tile_score:                  {sahi_stats['avg_tile_score']:.4f}")
        print(f"  avg_weak_tile_score:             {sahi_stats['avg_weak_tile_score']:.4f}")
        results_dict['sahi_stage1_stats'] = sahi_stats
    
    out_name = 'evaluation_results_sahi_stage1.json' if args.use_sahi else 'evaluation_results.json'
    output_path = config['checkpoint_path'].parent / out_name
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
