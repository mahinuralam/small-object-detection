"""
Evaluate FasterRCNN_FullFramework (poster Fig. 3) on VisDrone val split.

Pipeline evaluated: ResNet50 → IFPN → CD-DPA(P2-P4) → Recon+RGR → Det head

Usage (from repo root):
    cd "simple implementation"
    python scripts/eval/19_evaluate_full_framework.py
    python scripts/eval/19_evaluate_full_framework.py --checkpoint results/outputs_full_framework/best_model.pth
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.full_framework import FasterRCNN_FullFramework

# Dynamic dataset import (same pattern as other eval scripts)
spec = importlib.util.spec_from_file_location(
    "visdrone_dataset", Path(__file__).parent.parent / "4_visdrone_dataset.py"
)
visdrone_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visdrone_module)
VisDroneDataset = visdrone_module.VisDroneDataset

VISDRONE_CLASS_NAMES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor',
]


def collate_fn(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metric = MeanAveragePrecision(iou_type='bbox', class_metrics=True)

    for images, targets in tqdm(loader, desc='Evaluating'):
        images = [img.to(device) for img in images]
        preds = model(images)

        metric.update(
            [{k: v.cpu() for k, v in p.items()} for p in preds],
            [{'boxes': t['boxes'].cpu(), 'labels': t['labels'].cpu()} for t in targets],
        )

    return metric.compute()


def parse_args():
    default_ckpt = str(
        PROJECT_ROOT / 'results' / 'outputs_full_framework' / 'best_model.pth'
    )
    default_data = str(PROJECT_ROOT.parent / 'dataset' / 'VisDrone-2018')

    parser = argparse.ArgumentParser(
        description='Evaluate FasterRCNN_FullFramework on VisDrone.'
    )
    parser.add_argument('--checkpoint', type=str, default=default_ckpt)
    parser.add_argument('--data-root',  type=str, default=default_data)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
    )
    parser.add_argument(
        '--split', type=str, default='val', choices=['val', 'test'],
        help='Dataset split to evaluate on.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    device = torch.device(args.device)

    print("=" * 70)
    print("FULL FRAMEWORK EVALUATION  (poster Fig. 3)")
    print("  ResNet50 → IFPN → CD-DPA(P2-P4) → Recon+RGR → Det")
    print("=" * 70)
    print(f"Device    : {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Split     : {args.split}")

    if device.type == 'cuda':
        print(f"GPU       : {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    # Dataset
    print("\nLoading dataset...")
    dataset = VisDroneDataset(
        root_dir=args.data_root, split=args.split, transforms=None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
    )
    print(f"  {len(dataset)} images loaded")

    # Model
    print("\nLoading model...")
    model = FasterRCNN_FullFramework(
        num_classes=11,
        fpn_channels=256,
        lambda_rec=0.1,
        pretrained_backbone=False,
        use_checkpoint=False,
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Train first with: python scripts/train/train_full_framework.py"
        )

    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)

    epoch_saved = ckpt.get('epoch', '?')
    val_loss_saved = ckpt.get('val_loss', float('nan'))
    print(f"  Loaded epoch {epoch_saved}  (saved val_loss={val_loss_saved:.4f})")

    # Evaluate
    print("\nRunning evaluation...")
    results = evaluate(model, loader, device)

    map_50_95 = results['map'].item() * 100
    map_50    = results['map_50'].item() * 100
    map_75    = results['map_75'].item() * 100

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  mAP (IoU=0.50:0.95) : {map_50_95:6.2f}%")
    print(f"  mAP@0.50            : {map_50:6.2f}%")
    print(f"  mAP@0.75            : {map_75:6.2f}%")

    per_class = results.get('map_per_class')
    if isinstance(per_class, torch.Tensor) and per_class.ndim > 0 and per_class.numel() > 0:
        print(f"\n  Per-class AP@0.50:")
        class_ids = results.get('classes')
        for i, ap in enumerate(per_class):
            if class_ids is not None and i < len(class_ids):
                cls_id = int(class_ids[i].item())
                name = VISDRONE_CLASS_NAMES[cls_id - 1] if 1 <= cls_id <= 10 else f'cls_{cls_id}'
            else:
                name = VISDRONE_CLASS_NAMES[i] if i < len(VISDRONE_CLASS_NAMES) else f'cls_{i+1}'
            print(f"    {name:20s}: {ap.item() * 100:5.2f}%")

    # Comparison against earlier ablation stages
    print("\n" + "=" * 70)
    print("ABLATION COMPARISON (mAP@0.50)")
    print("=" * 70)
    baselines = {
        'Baseline Faster R-CNN': 38.02,
        'CD-DPA only':           43.44,
    }
    for name, val in baselines.items():
        delta = map_50 - val
        print(f"  {name:30s}: {val:5.2f}%  (full-fw Δ = {delta:+.2f}%)")
    print(f"  {'Full Framework (this run)':30s}: {map_50:5.2f}%")

    # Save
    out = {
        'split': args.split,
        'checkpoint': str(ckpt_path),
        'epoch': epoch_saved,
        'mAP_50_95': map_50_95 / 100,
        'mAP_50':    map_50    / 100,
        'mAP_75':    map_75    / 100,
    }
    if isinstance(per_class, torch.Tensor) and per_class.ndim > 0:
        class_ids = results.get('classes')
        per_class_dict = {}
        for i, ap in enumerate(per_class):
            if class_ids is not None and i < len(class_ids):
                cls_id = int(class_ids[i].item())
                name = VISDRONE_CLASS_NAMES[cls_id - 1] if 1 <= cls_id <= 10 else f'cls_{cls_id}'
            else:
                name = VISDRONE_CLASS_NAMES[i] if i < len(VISDRONE_CLASS_NAMES) else f'cls_{i+1}'
            per_class_dict[name] = round(ap.item(), 6)
        out['per_class_AP_50'] = per_class_dict

    out_path = ckpt_path.parent / f'eval_results_{args.split}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
