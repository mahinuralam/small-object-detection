"""
Master Training Runner — Train All Pipeline Blocks
====================================================

Orchestrates training for the 2 trainable blocks of the 4-stage pipeline:

    Block 1 — CD-DPA  (Cascaded Deformable Dual-Path Attention detector)
    Block 2 — SR-TOD  (Self-Reconstructed Tiny Object Detector for weak tiles)

Each block can be trained independently or sequentially.
Produces training curves and saves checkpoints.

Usage:
    cd "simple implementation"

    # Train all blocks sequentially:
    python scripts/train/train_all_blocks.py --all

    # Train individual blocks:
    python scripts/train/train_all_blocks.py --block cddpa
    python scripts/train/train_all_blocks.py --block srtod

    # Quick sanity run (2 epochs each):
    python scripts/train/train_all_blocks.py --all --quick
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import json
import time
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

# ── Path setup ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models"))

# ── Dataset ────────────────────────────────────────────────────────
import importlib.util
_ds_path = PROJECT_ROOT / "data" / "visdrone_dataset.py"
if not _ds_path.exists():
    _ds_path = PROJECT_ROOT / "scripts" / "4_visdrone_dataset.py"
spec = importlib.util.spec_from_file_location("visdrone_dataset", _ds_path)
visdrone_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visdrone_module)
VisDroneDataset = visdrone_module.VisDroneDataset


def collate_fn(batch):
    return tuple(zip(*batch))


# ====================================================================
# Block 1 — CD-DPA Training
# ====================================================================

def train_cddpa(args):
    """Train CD-DPA detector (Stage 1 base detector)."""
    from models.cddpa_model import FasterRCNN_CDDPA

    device = torch.device(args.device)
    output_dir = Path(args.output_dir) / 'outputs_cddpa'
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = args.cddpa_epochs
    print("\n" + "=" * 70)
    print("BLOCK 1 — CD-DPA TRAINING")
    print("=" * 70)

    # Dataset
    train_ds = VisDroneDataset(root_dir=args.dataset_root, split='train', min_size=5)
    val_ds = VisDroneDataset(root_dir=args.dataset_root, split='val', min_size=5)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn,
                            pin_memory=True)
    print(f"  Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

    # Model
    model = FasterRCNN_CDDPA(
        num_classes=args.num_classes,
        enhance_levels=['0', '1', '2'],
        use_checkpoint=True,
        pretrained=True,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params)
    print(f"  Trainable params: {total_params:,}")

    optimizer = optim.AdamW(params, lr=args.cddpa_lr, weight_decay=1e-4)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,
                                                    total_iters=min(3, epochs))
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - 3), eta_min=1e-6
    )
    scaler = GradScaler()
    accumulation_steps = args.accumulation_steps

    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"CD-DPA Epoch {epoch}/{epochs}")
        for bi, (images, targets) in enumerate(pbar):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with autocast():
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values()) / accumulation_steps
            scaler.scale(loss).backward()
            if (bi + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(params, max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            train_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
        train_loss /= len(train_loader)

        # ── Validate ──
        model.train()  # Faster RCNN returns losses in train mode
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Val", leave=False):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                with autocast():
                    loss_dict = model(images, targets)
                val_loss += sum(l.item() for l in loss_dict.values())
        val_loss /= len(val_loader)

        if epoch <= 3:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        lr = optimizer.param_groups[0]['lr']

        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'val_loss': val_loss, 'lr': lr})
        print(f"  Epoch {epoch}: train={train_loss:.4f}  val={val_loss:.4f}  lr={lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_dir / 'best_model_cddpa.pth')
            print(f"  -> Saved best model (val={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    elapsed = (time.time() - t_start) / 60
    _save_history(history, output_dir / 'cddpa_training_log.json')
    _plot_curves(history, output_dir / 'cddpa_training_curves.png', 'CD-DPA Training')
    print(f"  CD-DPA done in {elapsed:.1f} min  |  best val={best_val_loss:.4f}")
    print(f"  Checkpoint: {output_dir / 'best_model_cddpa.pth'}")
    return output_dir / 'best_model_cddpa.pth'


# ====================================================================
# Block 2 — ReconstructionHead Training
# ====================================================================

class _SimpleImageDataset(torch.utils.data.Dataset):
    """Load images from directory for RH training (no labels needed)."""
    def __init__(self, image_dir, img_size=640):
        self.paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.paths.extend(list(Path(image_dir).glob(ext)))
            self.paths.extend(list(Path(image_dir).glob(ext.upper())))
        self.img_size = img_size
        print(f"  ImageDataset: {len(self.paths)} images from {image_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        img = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        if torch.rand(1) > 0.5:
            img = torch.flip(img, [2])
        return img


def _gaussian_kernel_1d(sigma):
    ks = int(6 * sigma + 1) | 1
    x = torch.arange(ks, dtype=torch.float32) - ks // 2
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _blur_batch(x, sigma=3.0):
    k = _gaussian_kernel_1d(sigma).to(x.device)
    pad = len(k) // 2
    C = x.shape[1]
    kh = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    kv = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    x = F.conv2d(x, kh, padding=(0, pad), groups=C)
    x = F.conv2d(x, kv, padding=(pad, 0), groups=C)
    return x


class _DeltaMapLoss(nn.Module):
    def __init__(self, sigma=3.0, global_weight=0.5):
        super().__init__()
        self.sigma = sigma
        self.gw = global_weight

    def forward(self, image, recon):
        delta = torch.abs(image - recon).mean(dim=1)
        blurred = _blur_batch(image, self.sigma)
        hf = torch.abs(image - blurred).mean(dim=1)
        B = hf.shape[0]
        hf_flat = hf.view(B, -1)
        hf_norm = (hf - hf_flat.min(1).values.view(B,1,1)) / \
                  (hf_flat.max(1).values.view(B,1,1) - hf_flat.min(1).values.view(B,1,1) + 1e-8)
        w_bg = 1.0 - hf_norm
        return (delta * w_bg).mean() + self.gw * delta.mean(), delta.detach()


class _P2Extractor(nn.Module):
    """Frozen backbone → P2 features."""
    def __init__(self, cddpa_ckpt=None, device='cuda'):
        super().__init__()
        self.device = device
        if cddpa_ckpt and Path(cddpa_ckpt).exists():
            from models.cddpa_model import FasterRCNN_CDDPA
            m = FasterRCNN_CDDPA(num_classes=11, pretrained=False, use_checkpoint=False)
            ckpt = torch.load(cddpa_ckpt, map_location=device)
            m.load_state_dict(ckpt.get('model_state_dict', ckpt))
            self.backbone = m.base_model.backbone
            self.transform = m.base_model.transform
            self.enhancers = m.enhancers
            self.enhance_levels = m.enhance_levels
            self._cddpa = True
            print(f"  P2Extractor: CD-DPA backbone from {cddpa_ckpt}")
        else:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            base = fasterrcnn_resnet50_fpn(weights='DEFAULT')
            self.backbone = base.backbone
            self.transform = base.transform
            self._cddpa = False
            print("  P2Extractor: pretrained ResNet50-FPN")
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.to(device)

    @torch.no_grad()
    def forward(self, images):
        img_list = [images[i] for i in range(images.shape[0])]
        image_list, _ = self.transform(img_list, None)
        features = self.backbone(image_list.tensors)
        if self._cddpa:
            for lvl in self.enhance_levels:
                if lvl in features:
                    features[lvl] = self.enhancers[lvl](features[lvl])
        return features['0']


def train_rh(args, cddpa_ckpt=None):
    """Train ReconstructionHead (Block 2)."""
    from models.enhancements.reconstruction_head import ReconstructionHead

    device = torch.device(args.device)
    output_dir = Path(args.output_dir) / 'outputs_rh'
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = args.rh_epochs
    print("\n" + "=" * 70)
    print("BLOCK 2 — RECONSTRUCTION HEAD TRAINING")
    print("=" * 70)

    # Dataset: images from VisDrone train split
    image_dir = Path(args.dataset_root) / 'VisDrone2019-DET-train' / 'images'
    if not image_dir.exists():
        image_dir = Path(args.dataset_root) / 'train' / 'images'
    ds = _SimpleImageDataset(image_dir, img_size=args.rh_img_size)
    t_size = int(0.9 * len(ds))
    v_size = len(ds) - t_size
    train_ds, val_ds = random_split(ds, [t_size, v_size])
    train_loader = DataLoader(train_ds, batch_size=args.rh_batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.rh_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Backbone (frozen)
    ckpt = cddpa_ckpt or args.cddpa_checkpoint
    p2_ext = _P2Extractor(cddpa_ckpt=ckpt, device=str(device))

    # RH model (trainable)
    rh = ReconstructionHead(in_channels=256, out_channels=3).to(device)
    n_params = sum(p.numel() for p in rh.parameters() if p.requires_grad)
    print(f"  RH params: {n_params:,}")

    criterion = _DeltaMapLoss(sigma=3.0, global_weight=0.5)
    optimizer = optim.Adam(rh.parameters(), lr=args.rh_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    best_val = float('inf')
    history = []

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        rh.train()
        t_loss, t_dvar = 0.0, 0.0
        for images in tqdm(train_loader, desc=f"RH Epoch {epoch}/{epochs}", leave=False):
            images = images.to(device)
            p2 = p2_ext(images)
            r_img = rh(p2)
            if r_img.shape[-2:] != images.shape[-2:]:
                r_img = F.interpolate(r_img, images.shape[-2:],
                                      mode='bilinear', align_corners=False)
            loss, delta = criterion(images, r_img)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(rh.parameters(), 5.0)
            optimizer.step()
            t_loss += loss.item()
            t_dvar += delta.var(dim=[1, 2]).mean().item()
        t_loss /= len(train_loader)
        t_dvar /= len(train_loader)

        rh.eval()
        v_loss, v_dvar = 0.0, 0.0
        with torch.no_grad():
            for images in tqdm(val_loader, desc="RH Val", leave=False):
                images = images.to(device)
                p2 = p2_ext(images)
                r_img = rh(p2)
                if r_img.shape[-2:] != images.shape[-2:]:
                    r_img = F.interpolate(r_img, images.shape[-2:],
                                          mode='bilinear', align_corners=False)
                loss, delta = criterion(images, r_img)
                v_loss += loss.item()
                v_dvar += delta.var(dim=[1, 2]).mean().item()
        v_loss /= len(val_loader)
        v_dvar /= len(val_loader)
        scheduler.step(v_loss)

        history.append({'epoch': epoch, 'train_loss': t_loss, 'val_loss': v_loss,
                        'train_dvar': t_dvar, 'val_dvar': v_dvar,
                        'lr': optimizer.param_groups[0]['lr']})
        print(f"  Epoch {epoch}: train={t_loss:.5f}  val={v_loss:.5f}  "
              f"Δvar_t={t_dvar:.6f}  Δvar_v={v_dvar:.6f}")

        if v_loss < best_val:
            best_val = v_loss
            torch.save({
                'epoch': epoch, 'model_state_dict': rh.state_dict(),
                'val_loss': v_loss, 'val_dvar': v_dvar,
            }, output_dir / 'best_reconstruction_head.pth')
            print(f"  -> Saved best RH (val={v_loss:.5f})")

    elapsed = (time.time() - t_start) / 60
    _save_history(history, output_dir / 'rh_training_log.json')
    _plot_curves(history, output_dir / 'rh_training_curves.png', 'RH Training')
    print(f"  RH done in {elapsed:.1f} min  |  best val={best_val:.5f}")
    return output_dir / 'best_reconstruction_head.pth'


# ====================================================================
# Block 3 — SR-TOD Training
# ====================================================================

def train_srtod(args):
    """Train SR-TOD detector (Stage 4 tile detector)."""
    from srtod_model import FasterRCNN_SRTOD

    device = torch.device(args.device)
    output_dir = Path(args.output_dir) / 'outputs_srtod'
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = args.srtod_epochs
    print("\n" + "=" * 70)
    print("BLOCK 3 — SR-TOD TRAINING")
    print("=" * 70)

    train_ds = VisDroneDataset(root_dir=args.dataset_root, split='train', min_size=5)
    val_ds = VisDroneDataset(root_dir=args.dataset_root, split='val', min_size=5)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn,
                            pin_memory=True)
    print(f"  Train: {len(train_ds)} images  |  Val: {len(val_ds)} images")

    model = FasterRCNN_SRTOD(
        num_classes=args.num_classes,
        learnable_thresh=0.0156862,  # 4/255
        pretrained_backbone=True,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params)
    print(f"  Trainable params: {total_params:,}")

    optimizer = optim.SGD(params, lr=args.srtod_lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val = float('inf')
    patience_counter = 0
    history = []

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_recon = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"SR-TOD Epoch {epoch}/{epochs}")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            t_recon += loss_dict.get('loss_reconstruction', torch.tensor(0.0)).item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{loss_dict.get("loss_reconstruction", torch.tensor(0.0)).item():.4f}'
            })
        t_loss /= len(train_loader)
        t_recon /= len(train_loader)

        # Validate
        model.train()
        v_loss, v_recon = 0.0, 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Val", leave=False):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                v_loss += sum(l.item() for l in loss_dict.values())
                v_recon += loss_dict.get('loss_reconstruction', torch.tensor(0.0)).item()
        v_loss /= len(val_loader)
        v_recon /= len(val_loader)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        thresh = model.learnable_thresh.item() if hasattr(model, 'learnable_thresh') else 0

        history.append({'epoch': epoch, 'train_loss': t_loss, 'val_loss': v_loss,
                        'train_recon': t_recon, 'val_recon': v_recon,
                        'lr': lr, 'thresh': thresh})
        print(f"  Epoch {epoch}: train={t_loss:.4f}  val={v_loss:.4f}  "
              f"recon_t={t_recon:.4f}  thresh={thresh:.6f}")

        if v_loss < best_val:
            best_val = v_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': v_loss, 'learnable_thresh': thresh,
            }, output_dir / 'best_model_srtod.pth')
            print(f"  -> Saved best SR-TOD (val={v_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    elapsed = (time.time() - t_start) / 60
    _save_history(history, output_dir / 'srtod_training_log.json')
    _plot_curves(history, output_dir / 'srtod_training_curves.png', 'SR-TOD Training')
    print(f"  SR-TOD done in {elapsed:.1f} min  |  best val={best_val:.4f}")
    return output_dir / 'best_model_srtod.pth'


# ====================================================================
# Utility helpers
# ====================================================================

def _save_history(history, path):
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  Log saved: {path}")


def _plot_curves(history, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = [h['epoch'] for h in history]
    t_loss = [h['train_loss'] for h in history]
    v_loss = [h['val_loss'] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, t_loss, 'b-', label='Train')
    ax1.plot(epochs, v_loss, 'r-', label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} — Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)

    lrs = [h['lr'] for h in history]
    ax2.plot(epochs, lrs, 'g-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title(f'{title} — LR Schedule')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  Curves saved: {path}")


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description='Train all pipeline blocks')

    # Which blocks to train
    parser.add_argument('--all', action='store_true', help='Train both blocks')
    parser.add_argument('--block', choices=['cddpa', 'srtod'],
                        help='Train a single block')
    parser.add_argument('--quick', action='store_true',
                        help='Quick sanity run (2 epochs each)')

    # Paths
    parser.add_argument('--dataset_root', type=str,
                        default=str(PROJECT_ROOT.parent / 'dataset' / 'VisDrone-2018'))
    parser.add_argument('--output_dir', type=str,
                        default=str(PROJECT_ROOT / 'results'))
    parser.add_argument('--cddpa_checkpoint', type=str, default=None,
                        help='Existing CD-DPA checkpoint (skip Block 1)')

    # Common
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=11)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--accumulation_steps', type=int, default=4)

    # Block-specific epochs/lr
    parser.add_argument('--cddpa_epochs', type=int, default=50)
    parser.add_argument('--cddpa_lr', type=float, default=1e-4)
    parser.add_argument('--srtod_epochs', type=int, default=50)
    parser.add_argument('--srtod_lr', type=float, default=0.005)

    args = parser.parse_args()

    if args.quick:
        args.cddpa_epochs = 2
        args.srtod_epochs = 2
        args.patience = 100  # no early stopping in quick mode

    if not args.all and args.block is None:
        parser.error("Specify --all or --block {cddpa,srtod}")

    print("=" * 70)
    print("MASTER TRAINING RUNNER — 4-Stage Pipeline Blocks")
    print("=" * 70)
    print(f"  Device     : {args.device}")
    print(f"  Dataset    : {args.dataset_root}")
    print(f"  Output     : {args.output_dir}")
    if args.quick:
        print(f"  Mode       : QUICK (2 epochs per block)")
    print("=" * 70)

    cddpa_ckpt = args.cddpa_checkpoint

    if args.all or args.block == 'cddpa':
        cddpa_ckpt = train_cddpa(args)

    if args.all or args.block == 'srtod':
        train_srtod(args)

    print("\n" + "=" * 70)
    print("ALL TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nCheckpoints saved in: {args.output_dir}/")
    print(f"  Block 1 (CD-DPA)  : outputs_cddpa/best_model_cddpa.pth")
    print(f"  Block 2 (SR-TOD)  : outputs_srtod/best_model_srtod.pth")
    print(f"\nNext steps:")
    print(f"  1. Evaluate:")
    print(f"     python scripts/eval/18_evaluate_cddpa.py")
    print(f"     python scripts/eval/12_evaluate_srtod.py")
    print(f"  2. Run confidence-guided SAHI inference:")
    print(f"     python scripts/inference/run_sahi_infer.py --image <path>")
    print("=" * 70)


if __name__ == '__main__':
    main()
