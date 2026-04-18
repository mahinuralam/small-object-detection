"""
Train Faster R-CNN with Cascaded Deformable Dual-Path Attention (CD-DPA)

SOTA Architecture for Small Object Detection on VisDrone

Usage:
    cd "scripts/train"
    conda activate cooolenv
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u 14_train_cddpa.py 2>&1 | tee ../../results/outputs_cddpa/train.log
"""

import os
import sys
import time
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from tqdm import tqdm

# ──────────────────────── CUDA ENFORCEMENT ────────────────────────
assert torch.cuda.is_available(), (
    "CUDA is NOT available! This script requires a GPU.\n"
    "  • Activate the right env: conda activate cooolenv\n"
    "  • Verify: python -c 'import torch; print(torch.cuda.is_available())'\n"
    "  • Check GPU: nvidia-smi"
)
DEVICE = torch.device('cuda')
print(f"[CUDA] GPU: {torch.cuda.get_device_name(0)} "
    f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")

# ──────────────────────── PATH SETUP ──────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # "simple implementation/"
sys.path.insert(0, str(PROJECT_ROOT))

from models.cddpa_model import FasterRCNN_CDDPA

# Import dataset
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "visdrone_dataset", SCRIPT_DIR.parent / "4_visdrone_dataset.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
VisDroneDataset = _mod.VisDroneDataset


# ═══════════════════════ CONFIGURATION ════════════════════════════
CONFIG = {
    # Model
    'num_classes': 11,                  # 10 VisDrone classes + background
    'enhance_levels': ['0', '1', '2'],  # P2, P3, P4
    'use_checkpoint': False,
    'pretrained_backbone': True,

    # Training
    'batch_size': 4,
    'accumulation_steps': 4,            # effective batch = 4 × 4 = 16
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'backbone_lr': 1e-5,
    'weight_decay': 1e-4,
    'warmup_epochs': 3,
    'freeze_backbone_epochs': 5,
    'early_stopping_patience': 15,
    'max_grad_norm': 0.5,
    'augment_train': True,

    # DataLoader
    'train_workers': 6,
    'val_workers': 4,

    # Paths
    'data_root': PROJECT_ROOT.parent / 'dataset' / 'VisDrone-2018',
    'output_dir': PROJECT_ROOT / 'results' / 'outputs_cddpa',
    'tensorboard_dir': PROJECT_ROOT / 'results' / 'outputs_cddpa' / 'tensorboard',
}


# ═══════════════════════ AUGMENTATION ═════════════════════════════
def collate_fn(batch):
    return tuple(zip(*batch))


class TrainAugmentation:
    """Box-aware augmentation: hflip, color jitter, scale jitter, degenerate filter."""
    def __init__(self):
        self.color_jitter = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
        )

    def __call__(self, image, target):
        # 1. Random horizontal flip
        if random.random() > 0.5:
            w = image.width
            image = TF.hflip(image)
            if len(target['boxes']) > 0:
                boxes = target['boxes'].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target['boxes'] = boxes

        # 2. Color jitter
        image = self.color_jitter(image)

        # 3. Scale jitter [0.8, 1.2]
        if random.random() > 0.5:
            scale = random.uniform(0.8, 1.2)
            orig_w, orig_h = image.size
            new_w = max(int(orig_w * scale), 32)
            new_h = max(int(orig_h * scale), 32)
            image = TF.resize(image, [new_h, new_w])
            if len(target['boxes']) > 0:
                target['boxes'] = target['boxes'] * scale

        # 4. Filter degenerate boxes (w or h < 1 pixel)
        if len(target['boxes']) > 0:
            boxes = target['boxes']
            ws = boxes[:, 2] - boxes[:, 0]
            hs = boxes[:, 3] - boxes[:, 1]
            keep = (ws >= 1.0) & (hs >= 1.0)
            if keep.any():
                target['boxes'] = boxes[keep]
                target['labels'] = target['labels'][keep]
                if 'area' in target:
                    target['area'] = target['area'][keep]
                if 'iscrowd' in target:
                    target['iscrowd'] = target['iscrowd'][keep]

        # 5. To tensor [0, 1]
        image = TF.to_tensor(image)
        return image, target


# ═══════════════════════ TRAIN ONE EPOCH ══════════════════════════
def train_one_epoch(model, loader, optimizer, scaler, epoch, cfg,
                    writer=None, global_step=0):
    model.train()
    total_loss = 0.0
    valid_batches = 0
    nan_count = 0
    components = {'classifier': 0, 'box_reg': 0, 'objectness': 0, 'rpn_box_reg': 0}
    accum = cfg['accumulation_steps']

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f'Epoch {epoch}', dynamic_ncols=True)

    for batch_idx, (images, targets) in enumerate(pbar):
        try:
            images = [img.to(DEVICE, non_blocking=True) for img in images]
            targets = [{k: v.to(DEVICE, non_blocking=True) for k, v in t.items()}
                       for t in targets]

            # Skip empty-box batches
            if all(t['boxes'].shape[0] == 0 for t in targets):
                continue

            with torch.amp.autocast('cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values()) / accum

            # NaN guard
            if not torch.isfinite(losses):
                nan_count += 1
                if nan_count <= 5 or nan_count % 50 == 0:
                    print(f"  ⚠ NaN/Inf at batch {batch_idx} (total: {nan_count})")
                optimizer.zero_grad(set_to_none=True)
                scaler.update(scaler.get_scale() * 0.5)
                continue

            scaler.scale(losses).backward()

            # Optimizer step every `accum` batches
            if (batch_idx + 1) % accum == 0:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg['max_grad_norm']
                )
                if not torch.isfinite(grad_norm):
                    nan_count += 1
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Logging
            batch_loss = losses.item() * accum
            if np.isfinite(batch_loss):
                total_loss += batch_loss
                valid_batches += 1
                for key in components:
                    if f'loss_{key}' in loss_dict:
                        components[key] += loss_dict[f'loss_{key}'].item()
                if writer and valid_batches % 50 == 0:
                    writer.add_scalar('Batch/train_loss', batch_loss,
                                      global_step + valid_batches)

            if (batch_idx + 1) % 10 == 0:
                avg = total_loss / max(valid_batches, 1)
                pbar.set_postfix(loss=f'{avg:.4f}', nan=nan_count)

        except torch.cuda.OutOfMemoryError:
            print(f"  ⚠ OOM at batch {batch_idx}")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue

    if nan_count:
        print(f"  ⚠ NaN batches: {nan_count}/{len(loader)}")

    avg_loss = total_loss / max(valid_batches, 1)
    avg_comp = {k: v / max(valid_batches, 1) for k, v in components.items()}
    return avg_loss, avg_comp, valid_batches


# ═══════════════════════ VALIDATE ═════════════════════════════════
@torch.no_grad()
def validate(model, loader):
    model.train()  # need train mode for loss computation
    total_loss = 0.0
    valid_batches = 0
    components = {'classifier': 0, 'box_reg': 0, 'objectness': 0, 'rpn_box_reg': 0}

    pbar = tqdm(loader, desc='Validating', dynamic_ncols=True)
    for images, targets in pbar:
        try:
            images = [img.to(DEVICE, non_blocking=True) for img in images]
            targets = [{k: v.to(DEVICE, non_blocking=True) for k, v in t.items()}
                       for t in targets]

            with torch.amp.autocast('cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())

            if not torch.isfinite(losses):
                continue
            total_loss += losses.item()
            valid_batches += 1
            for key in components:
                if f'loss_{key}' in loss_dict:
                    components[key] += loss_dict[f'loss_{key}'].item()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

    model.eval()
    avg_loss = total_loss / max(valid_batches, 1)
    avg_comp = {k: v / max(valid_batches, 1) for k, v in components.items()}
    return avg_loss, avg_comp


# ═══════════════════════ MAIN ═════════════════════════════════════
def main():
    cfg = CONFIG.copy()

    # Parse optional --resume flag
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    args, _ = parser.parse_known_args()

    # Directories
    cfg['output_dir'].mkdir(parents=True, exist_ok=True)
    cfg['tensorboard_dir'].mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir=str(cfg['tensorboard_dir']))

    # ── Banner ──
    print("=" * 80)
    print("CD-DPA TRAINING  |  Faster R-CNN + Cascaded Deformable Dual-Path Attention")
    print("=" * 80)
    print(f"  GPU:              {torch.cuda.get_device_name(0)}")
    print(f"  Batch size:       {cfg['batch_size']}  "
          f"(effective {cfg['batch_size'] * cfg['accumulation_steps']})")
    print(f"  Epochs:           {cfg['num_epochs']}")
    print(f"  LR (head/bb):     {cfg['learning_rate']}/{cfg['backbone_lr']}")
    print(f"  Output:           {cfg['output_dir']}")
    print(f"  TensorBoard:      tensorboard --logdir {cfg['tensorboard_dir'].resolve()}")
    print("=" * 80)

    # ── Datasets ──
    print("\nLoading datasets...")
    train_aug = TrainAugmentation() if cfg['augment_train'] else None
    train_ds = VisDroneDataset(root_dir=cfg['data_root'], split='train',
                               transforms=train_aug)
    val_ds = VisDroneDataset(root_dir=cfg['data_root'], split='val',
                             transforms=None)

    train_loader = DataLoader(
        train_ds, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=cfg['train_workers'], collate_fn=collate_fn,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg['val_workers'], collate_fn=collate_fn,
        pin_memory=True, persistent_workers=True,
    )
    print(f"  Train: {len(train_ds)} images  ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_ds)} images  ({len(val_loader)} batches)")

    # ── Model ──
    print("\nInitializing CD-DPA model...")
    model = FasterRCNN_CDDPA(
        num_classes=cfg['num_classes'],
        enhance_levels=cfg['enhance_levels'],
        use_checkpoint=cfg['use_checkpoint'],
        pretrained=cfg['pretrained_backbone'],
    ).to(DEVICE)

    # ── Optimizer with differential LR ──
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if 'backbone' in name else head_params).append(p)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': cfg['backbone_lr']},
        {'params': head_params,     'lr': cfg['learning_rate']},
    ], weight_decay=cfg['weight_decay'])

    # ── Schedulers ──
    T_0 = max(10, (cfg['num_epochs'] - cfg['warmup_epochs']) // 3)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=1, eta_min=1e-7
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=cfg['warmup_epochs']
    )

    # ── Mixed precision scaler (modern API) ──
    scaler = torch.amp.GradScaler('cuda')

    # ── Resume ──
    start_epoch = 1
    best_val_loss = float('inf')
    patience_counter = 0
    training_log = []

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 1) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        patience_counter = ckpt.get('patience_counter', 0)
        training_log = ckpt.get('training_log', [])
        print(f"  Resumed from epoch {start_epoch}, "
              f"best_val_loss={best_val_loss:.4f}")

    # ── Helpers ──
    def _freeze_backbone():
        for n, p in model.named_parameters():
            if 'backbone' in n:
                p.requires_grad_(False)
        print("  ❄  Backbone FROZEN")

    def _unfreeze_backbone():
        for n, p in model.named_parameters():
            if 'backbone' in n:
                p.requires_grad_(True)
        print("  🔥 Backbone UNFROZEN")

    # ═══════════════════ TRAINING LOOP ════════════════════════════
    print("\n🚀 Training started")
    print("=" * 80)
    t0 = time.time()
    global_step = 0

    for epoch in range(start_epoch, cfg['num_epochs'] + 1):
        ep_start = time.time()

        # Backbone freeze/unfreeze
        if epoch == 1 and cfg['freeze_backbone_epochs'] > 0:
            _freeze_backbone()
        if epoch == cfg['freeze_backbone_epochs'] + 1:
            _unfreeze_backbone()

        # Train
        train_loss, train_comp, n_valid = train_one_epoch(
            model, train_loader, optimizer, scaler, epoch, cfg,
            writer=writer, global_step=global_step,
        )
        global_step += n_valid

        # Validate
        val_loss, val_comp = validate(model, val_loader)

        # LR step
        if epoch <= cfg['warmup_epochs']:
            warmup_scheduler.step()
        else:
            lr_scheduler.step()

        bb_lr = optimizer.param_groups[0]['lr']
        head_lr = optimizer.param_groups[1]['lr']
        ep_min = (time.time() - ep_start) / 60

        # Console
        print(f"\nEpoch {epoch}/{cfg['num_epochs']}  ({ep_min:.1f} min)")
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"lr_bb={bb_lr:.2e}  lr_head={head_lr:.2e}")

        # TensorBoard epoch-level
        writer.add_scalars('Loss/epoch',
                           {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalar('LR/backbone', bb_lr, epoch)
        writer.add_scalar('LR/head', head_lr, epoch)
        for k, v in train_comp.items():
            writer.add_scalar(f'TrainComponents/{k}', v, epoch)
        for k, v in val_comp.items():
            writer.add_scalar(f'ValComponents/{k}', v, epoch)
        writer.add_scalar('Misc/epoch_minutes', ep_min, epoch)
        writer.flush()

        # JSON log
        training_log.append({
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'train_components': train_comp, 'val_components': val_comp,
            'lr_backbone': bb_lr, 'lr_head': head_lr, 'time_minutes': ep_min,
        })
        with open(cfg['output_dir'] / 'training_log.json', 'w') as f:
            json.dump(training_log, f, indent=2)

        # Checkpoint (last)
        ckpt_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'val_loss': val_loss, 'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'training_log': training_log, 'config': cfg,
        }
        torch.save(ckpt_data, cfg['output_dir'] / 'last_checkpoint.pth')

        # Best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({**ckpt_data, 'best_val_loss': best_val_loss},
                       cfg['output_dir'] / 'best_model_cddpa.pth')
            print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement "
                  f"({patience_counter}/{cfg['early_stopping_patience']})")

        # Early stopping
        if patience_counter >= cfg['early_stopping_patience']:
            print(f"\n⚠  Early stopping at epoch {epoch}")
            break

    # ── Done ──
    hrs = (time.time() - t0) / 3600
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE — {hrs:.2f}h — best val_loss={best_val_loss:.4f}")
    print(f"  Best model: {cfg['output_dir'] / 'best_model_cddpa.pth'}")
    print("=" * 80)
    writer.close()


if __name__ == '__main__':
    main()
