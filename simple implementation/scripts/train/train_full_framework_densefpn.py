"""
Train the Full Framework with DenseFPN neck (poster Fig. 3, DenseFPN variant).

Pipeline: ResNet50 → Standard FPN → DenseFPN → CD-DPA(P2-P4) → [Recon + RGR on P2] → Det

Loss: L_det  +  λ_rec × L1(r_img, img_inputs)    (λ_rec = 0.1)

Usage (from repo root):
    cd "simple implementation"
    conda activate cooolenv
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        python -u scripts/train/train_full_framework_densefpn.py 2>&1 | \\
        tee results/outputs_full_framework_densefpn/train.log

For the 3-5 epoch sanity check:
    python scripts/train/train_full_framework_densefpn.py --sanity
"""

import os
import sys
import time
import json
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# ── CUDA ──────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), (
    "CUDA not available. Activate cooolenv and verify GPU with nvidia-smi."
)
DEVICE = torch.device('cuda')
print(f"[CUDA] GPU: {torch.cuda.get_device_name(0)} "
      f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent          # "simple implementation/"
sys.path.insert(0, str(PROJECT_ROOT))

from models.full_framework_densefpn import FasterRCNN_FullFramework_DenseFPN

# ── Dataset ────────────────────────────────────────────────────────────
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "visdrone_dataset", SCRIPT_DIR.parent / "4_visdrone_dataset.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
VisDroneDataset = _mod.VisDroneDataset

# ═══════════════════════ CONFIG ════════════════════════════════════════
CONFIG = {
    # Model
    'num_classes':               11,      # 10 VisDrone classes + background
    'fpn_channels':              256,
    'lambda_rec':                0.1,     # reconstruction loss weight
    'pretrained_backbone':       True,    # COCO-pretrained standard FPN
    'trainable_backbone_layers': 3,
    'use_checkpoint':            True,

    # Training
    'batch_size':                2,
    'accumulation_steps':        8,       # effective batch = 16
    'num_epochs':                50,
    'learning_rate':             1e-4,
    'backbone_lr':               1e-5,
    'weight_decay':              1e-4,
    'warmup_epochs':             3,
    'freeze_backbone_epochs':    5,
    'early_stopping_patience':   15,
    'max_grad_norm':             0.5,
    'augment_train':             True,

    # DataLoader
    'train_workers':             6,
    'val_workers':               4,

    # Paths
    'dataset_root':  str(PROJECT_ROOT.parent / 'dataset' / 'VisDrone-2018'),
    'output_dir':    str(PROJECT_ROOT / 'results' / 'outputs_full_framework_densefpn'),
    'resume_checkpoint': None,

    # Mixed precision
    'use_amp': True,
}

# ═══════════════════════ SEED ══════════════════════════════════════════
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ═══════════════════════ TRANSFORMS ════════════════════════════════════
class TrainAugmentation:
    """Box-aware hflip + PIL→tensor (avoids TF.to_tensor / numpy compat issue)."""
    def __call__(self, image, target):
        if random.random() > 0.5:
            w = image.width
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if len(target['boxes']) > 0:
                boxes = target['boxes'].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target['boxes'] = boxes
        # PIL → tensor [0, 1] via raw bytes — no numpy C-ext dependency
        w, h = image.size
        image = torch.ByteTensor(bytearray(image.tobytes())).reshape(h, w, 3).permute(2, 0, 1).float() / 255.0
        return image, target


def get_transforms(train: bool):
    if train and CONFIG['augment_train']:
        return TrainAugmentation()
    return None   # dataset handles PIL→tensor when transforms=None

# ═══════════════════════ COLLATE ═══════════════════════════════════════
def collate_fn(batch):
    return tuple(zip(*batch))

# ═══════════════════════ DATASET ═══════════════════════════════════════
def build_loaders():
    root = CONFIG['dataset_root']
    train_ds = VisDroneDataset(root, split='train', transforms=get_transforms(True))
    val_ds   = VisDroneDataset(root, split='val',   transforms=get_transforms(False))
    train_loader = DataLoader(
        train_ds, batch_size=CONFIG['batch_size'],
        shuffle=True, num_workers=CONFIG['train_workers'],
        collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG['batch_size'],
        shuffle=False, num_workers=CONFIG['val_workers'],
        collate_fn=collate_fn, pin_memory=True,
    )
    print(f"[Dataset] train={len(train_ds)}, val={len(val_ds)}")
    return train_loader, val_loader

# ═══════════════════════ OPTIMISER ═════════════════════════════════════
def build_optimiser(model: nn.Module):
    """Separate LR for backbone vs everything else."""
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Standard FPN backbone lives at base_model.backbone.*
        if 'base_model.backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    return torch.optim.AdamW(
        [
            {'params': head_params,     'lr': CONFIG['learning_rate']},
            {'params': backbone_params, 'lr': CONFIG['backbone_lr']},
        ],
        weight_decay=CONFIG['weight_decay'],
    )

# ═══════════════════════ SCHEDULER ═════════════════════════════════════
def build_scheduler(optimiser, steps_per_epoch: int):
    warmup_steps = CONFIG['warmup_epochs'] * steps_per_epoch
    return torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=[CONFIG['learning_rate'], CONFIG['backbone_lr']],
        epochs=CONFIG['num_epochs'],
        steps_per_epoch=steps_per_epoch,
        pct_start=warmup_steps / (CONFIG['num_epochs'] * steps_per_epoch),
        anneal_strategy='cos',
    )

# ═══════════════════════ TRAIN EPOCH ═══════════════════════════════════
def train_epoch(model, loader, optimiser, scheduler, scaler, epoch, writer):
    model.train()
    total_losses = {}
    optimiser.zero_grad()
    acc = CONFIG['accumulation_steps']

    max_steps = CONFIG.get('sanity_steps')  # None = unlimited

    for step, (images, targets) in enumerate(
        tqdm(loader, desc=f"Train E{epoch+1}", leave=False,
             total=max_steps if max_steps else len(loader))
    ):
        if max_steps and step >= max_steps:
            break

        images  = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        # Freeze backbone for early epochs
        freeze = epoch < CONFIG['freeze_backbone_epochs']
        for p in model.base_model.backbone.parameters():
            p.requires_grad_(not freeze)

        with torch.cuda.amp.autocast(enabled=CONFIG['use_amp']):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values()) / acc

        scaler.scale(loss).backward()

        if (step + 1) % acc == 0:
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), CONFIG['max_grad_norm']
            )
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad()
            scheduler.step()

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()

    n = min(step + 1, max_steps) if max_steps else len(loader)
    avg = {k: v / n for k, v in total_losses.items()}
    avg['total'] = sum(avg.values())

    gstep = epoch * len(loader)
    for k, v in avg.items():
        writer.add_scalar(f'train/{k}', v, gstep)

    return avg

# ═══════════════════════ VAL EPOCH ═════════════════════════════════════
def val_epoch(model, loader, epoch, writer):
    model.train()   # keep train mode to get losses
    total_losses = {}
    max_steps = CONFIG.get('sanity_steps')

    with torch.no_grad():
        for vstep, (images, targets) in enumerate(
            tqdm(loader, desc=f"Val   E{epoch+1}", leave=False,
                 total=max_steps if max_steps else len(loader))
        ):
            if max_steps and vstep >= max_steps:
                break
            images  = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            with torch.cuda.amp.autocast(enabled=CONFIG['use_amp']):
                loss_dict = model(images, targets)
            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.item()

    n = min(vstep + 1, max_steps) if max_steps else len(loader)
    avg = {k: v / n for k, v in total_losses.items()}
    avg['total'] = sum(avg.values())

    gstep = epoch * len(loader)
    for k, v in avg.items():
        writer.add_scalar(f'val/{k}', v, gstep)

    return avg

# ═══════════════════════ MAIN ══════════════════════════════════════════
def main():
    set_seed(42)
    out_dir = Path(CONFIG['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=CONFIG['num_epochs'])
    parser.add_argument('--sanity', action='store_true',
                        help='Quick sanity check: 3 epochs, 50 steps each')
    parser.add_argument('--sanity-steps', type=int, default=50,
                        help='Max steps per epoch in sanity mode')
    args = parser.parse_args()

    if args.sanity:
        CONFIG['num_epochs']      = 3
        CONFIG['sanity_steps']    = args.sanity_steps
        CONFIG['warmup_epochs']   = 0
        print(f"[Sanity] 3 epochs × {args.sanity_steps} steps each")
    else:
        CONFIG['num_epochs']   = args.epochs
        CONFIG['sanity_steps'] = None

    with open(out_dir / 'config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2)

    train_loader, val_loader = build_loaders()

    model = FasterRCNN_FullFramework_DenseFPN(
        num_classes=CONFIG['num_classes'],
        fpn_channels=CONFIG['fpn_channels'],
        lambda_rec=CONFIG['lambda_rec'],
        pretrained_backbone=CONFIG['pretrained_backbone'],
        trainable_backbone_layers=CONFIG['trainable_backbone_layers'],
        use_checkpoint=CONFIG['use_checkpoint'],
    ).to(DEVICE)

    optimiser = build_optimiser(model)
    steps_per_epoch = CONFIG.get('sanity_steps') or len(train_loader)
    scheduler = build_scheduler(optimiser, steps_per_epoch)
    scaler    = torch.cuda.amp.GradScaler(enabled=CONFIG['use_amp'])
    writer    = SummaryWriter(log_dir=str(out_dir / 'tensorboard'))

    # Resume
    start_epoch, best_val = 0, float('inf')
    ckpt_path = CONFIG.get('resume_checkpoint')
    if ckpt_path and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ck['model_state_dict'])
        optimiser.load_state_dict(ck['optimiser_state_dict'])
        start_epoch = ck['epoch'] + 1
        best_val    = ck.get('best_val_loss', float('inf'))
        print(f"[Resume] epoch {start_epoch}, best_val={best_val:.4f}")

    history  = []
    no_improve = 0

    print(f"\n[Train] {CONFIG['num_epochs']} epochs, "
          f"λ_rec={CONFIG['lambda_rec']}, "
          f"batch={CONFIG['batch_size']}×{CONFIG['accumulation_steps']} eff.")

    for epoch in range(start_epoch, CONFIG['num_epochs']):
        t0 = time.time()

        train_avg = train_epoch(
            model, train_loader, optimiser, scheduler, scaler, epoch, writer
        )
        val_avg = val_epoch(model, val_loader, epoch, writer)

        elapsed = time.time() - t0
        print(
            f"E{epoch+1:03d}/{CONFIG['num_epochs']} "
            f"| train_total={train_avg['total']:.4f} "
            f"| train_recon={train_avg.get('loss_reconstruction', 0):.4f} "
            f"| val_total={val_avg['total']:.4f} "
            f"| val_recon={val_avg.get('loss_reconstruction', 0):.4f} "
            f"| {elapsed/60:.1f}min"
        )

        row = {'epoch': epoch + 1, 'train': train_avg, 'val': val_avg,
               'elapsed_s': elapsed}
        history.append(row)
        with open(out_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

        # Checkpoint
        ck = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            'val_loss': val_avg['total'],
            'best_val_loss': best_val,
            'config': CONFIG,
        }
        torch.save(ck, out_dir / 'last_checkpoint.pth')

        if val_avg['total'] < best_val:
            best_val = val_avg['total']
            no_improve = 0
            torch.save(ck, out_dir / 'best_model.pth')
            print(f"  ↑ New best val={best_val:.4f}")
        else:
            no_improve += 1
            if no_improve >= CONFIG['early_stopping_patience']:
                print(f"[EarlyStop] {no_improve} epochs without improvement")
                break

        if args.sanity and (epoch + 1) >= 5:
            for k, v in {**train_avg, **val_avg}.items():
                if not isinstance(v, dict):
                    assert torch.isfinite(torch.tensor(float(v))), \
                        f"Non-finite in sanity check: {k}={v}"
            print("\n[Sanity] 5-epoch sanity check PASSED — all losses finite")
            break

    writer.close()
    print(f"\n[Done] Best val loss: {best_val:.4f}")
    print(f"       Checkpoints in: {out_dir}")


if __name__ == '__main__':
    main()
