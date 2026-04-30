"""
V3 Full Framework Training.

Pipeline (single forward pass):
  Backbone + FPN
      → CD-DPA (P2, P3, P4)
      → SAHI weak-tile identification (no-grad, eval)
      → Upper stream: P2 weak tiles → ReconHead → Δ → DGFE → P2''
      → Lower stream: P3–P6 unchanged
      → DenseFPN fuses P2'' + P3–P6
      → RPN + RoI detection head

Initialised from Phase 2 checkpoint (47.71% mAP@0.50).
New modules (DenseFPN, DPA-P4) initialised from scratch.

Usage:
    torchrun --nproc_per_node=2 scripts/train/train_v3_full_framework.py
    torchrun --nproc_per_node=2 scripts/train/train_v3_full_framework.py --sanity
"""

import contextlib
import json
import os
import sys
import time
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from tqdm import tqdm

assert torch.cuda.is_available()

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
IS_DIST    = WORLD_SIZE > 1
IS_MAIN    = LOCAL_RANK == 0

if IS_DIST:
    dist.init_process_group("nccl")
    torch.cuda.set_device(LOCAL_RANK)

DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
torch.backends.cudnn.benchmark = True

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.enhancements.reconstruction_head import ReconstructionHead
from models.enhancements.dgfe_module import DGFEModule
from models.enhancements.dense_fpn import DenseFPN
from models.enhancements.cddpa_module import CDDPA

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "visdrone_dataset", SCRIPT_DIR.parent / "4_visdrone_dataset.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
VisDroneDataset = _mod.VisDroneDataset

# ═══════════════════════ CONFIG ════════════════════════════════════════
CONFIG = {
    'num_classes':   11,
    'lambda_rec':    0.1,

    'init_checkpoint': str(PROJECT_ROOT / 'results' / 'outputs_phase2_smart_recon' / 'best_model_phase2.pth'),
    'output_dir':      str(PROJECT_ROOT / 'results' / 'outputs_v3_full_framework'),
    'resume_checkpoint': None,

    # SAHI weak-tile identification (single scale for training speed)
    'tile_sizes':    (512,),
    'overlap':       0.35,
    'score_thresh':  0.008,
    'nms_iou':       0.35,
    'K':             5,
    'weak_tile_size': 512,

    # Training
    'num_epochs':         30,
    'batch_size':         1,
    'accumulation_steps': 8,        # eff batch = 1×2GPU×8 = 16
    'learning_rate':      3e-5,     # new modules: DenseFPN, DPA-P4, DGFE, ReconHead
    'backbone_lr':        1e-6,     # FPN + DPA-P2/P3 (fine-tune)
    'weight_decay':       1e-4,
    'max_grad_norm':      1.0,
    'warmup_epochs':      2,
    'use_amp':            True,

    'train_workers':  4,
    'val_workers':    2,
    'dataset_root':   str(PROJECT_ROOT.parent / 'dataset' / 'VisDrone-2018'),
}

# ═══════════════════════ SAHI HELPERS ══════════════════════════════════

def generate_tiles(img_w, img_h, tile_size=640, overlap=0.35):
    step = int(tile_size * (1 - overlap))
    tiles = []
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x2 = min(x + tile_size, img_w); y2 = min(y + tile_size, img_h)
            x1 = max(0, x2 - tile_size);   y1 = max(0, y2 - tile_size)
            tiles.append((x1, y1, x2, y2))
            if x2 == img_w: break
            x += step
        if y2 == img_h: break
        y += step
    return tiles


@torch.no_grad()
def sahi_batched(model, image, device, tile_size, overlap, score_thresh, tile_batch=8):
    """Batched SAHI — all tiles in a few forward calls."""
    _, img_h, img_w = image.shape
    tile_grid = generate_tiles(img_w, img_h, tile_size, overlap)
    padded = []
    for (x1, y1, x2, y2) in tile_grid:
        t = image[:, y1:y2, x1:x2]
        ph, pw = tile_size-(y2-y1), tile_size-(x2-x1)
        if ph > 0 or pw > 0:
            t = F.pad(t, (0, pw, 0, ph))
        padded.append(t)
    all_b, all_s, all_l = [], [], []
    for i in range(0, len(padded), tile_batch):
        batch = [p.to(device) for p in padded[i:i+tile_batch]]
        preds = model(batch)
        for j, pred in enumerate(preds):
            idx = i + j
            x1, y1, x2, y2 = tile_grid[idx]
            b, s, l = pred['boxes'].cpu(), pred['scores'].cpu(), pred['labels'].cpu()
            keep = s >= score_thresh
            b, s, l = b[keep], s[keep], l[keep]
            if b.shape[0] > 0:
                b[:, [0,2]] += x1; b[:, [1,3]] += y1
                b[:, [0,2]].clamp_(0, img_w); b[:, [1,3]].clamp_(0, img_h)
            all_b.append(b); all_s.append(s); all_l.append(l)
    if not all_b or not any(t.shape[0] > 0 for t in all_b):
        return torch.zeros(0,4), torch.zeros(0), torch.zeros(0, dtype=torch.long)
    return torch.cat(all_b), torch.cat(all_s), torch.cat(all_l)


@torch.no_grad()
def run_sahi(model, image, device, tile_sizes, overlap, score_thresh):
    """Full-image pass + batched SAHI tiles."""
    _, img_h, img_w = image.shape
    all_b, all_s, all_l = [], [], []
    p = model([image.to(device)])[0]
    bf, sf, lf = p['boxes'].cpu(), p['scores'].cpu(), p['labels'].cpu()
    keep = sf >= score_thresh
    if keep.any():
        all_b.append(bf[keep]); all_s.append(sf[keep]); all_l.append(lf[keep])
    for ts in tile_sizes:
        b, s, l = sahi_batched(model, image, device, ts, overlap, score_thresh)
        if b.shape[0] > 0:
            all_b.append(b); all_s.append(s); all_l.append(l)
    if not all_b:
        return torch.zeros(0,4), torch.zeros(0), torch.zeros(0, dtype=torch.long)
    return torch.cat(all_b), torch.cat(all_s), torch.cat(all_l)


def per_class_nms(all_b, all_s, all_l, nms_iou):
    if all_b.shape[0] == 0:
        return all_b, all_s, all_l
    fb, fs, fl = [], [], []
    for cls in all_l.unique():
        m = all_l == cls; b, s = all_b[m], all_s[m]
        keep = nms(b, s, nms_iou)
        fb.append(b[keep]); fs.append(s[keep]); fl.append(all_l[m][keep])
    return torch.cat(fb), torch.cat(fs), torch.cat(fl)


def select_weak_tiles(boxes, scores, img_h, img_w, tile_size, K):
    tiles = generate_tiles(img_w, img_h, tile_size, overlap=0.0)
    cell_scores = []
    for (x1, y1, x2, y2) in tiles:
        if boxes.shape[0] == 0:
            cell_scores.append((0.0, (x1,y1,x2,y2))); continue
        cx = (boxes[:,0]+boxes[:,2])/2; cy = (boxes[:,1]+boxes[:,3])/2
        inside = (cx>=x1)&(cx<x2)&(cy>=y1)&(cy<y2)
        s = scores[inside].sum().item() if inside.any() else 0.0
        cell_scores.append((s, (x1,y1,x2,y2)))
    cell_scores.sort(key=lambda x: x[0])
    return [coords for _, coords in cell_scores[:K]]


# ═══════════════════════ MODEL ═════════════════════════════════════════

class V3FullFramework(nn.Module):
    """
    V3 Full Framework:
      Backbone+FPN → CD-DPA(P2,P3,P4) → SAHI weak-tile ID
      Upper: P2 weak tiles → ReconHead → Δ → DGFE → P2''
      Lower: P3–P6 unchanged
      DenseFPN(P2'' + P3–P6) → RPN + RoI detection
    """
    def __init__(self, num_classes=11, lambda_rec=0.1):
        super().__init__()
        self.base_model = fasterrcnn_resnet50_fpn(weights=None)
        in_f = self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)

        # CD-DPA on P2, P3, P4 (full Cascaded Deformable DPA)
        self.cd_dpa = nn.ModuleDict({
            '0': CDDPA(channels=256, use_checkpoint=True),  # P2
            '1': CDDPA(channels=256, use_checkpoint=True),  # P3
            '2': CDDPA(channels=256, use_checkpoint=True),  # P4
        })

        # Upper stream: Recon + DGFE on P2 weak tiles
        self.recon = ReconstructionHead(in_channels=256)
        self.dgfe  = DGFEModule(error_channels=1, feat_channels=256)

        # DenseFPN neck — fuses P2'' (upper) + P3–P6 (lower)
        self.dense_fpn = DenseFPN(channels=256)

        self.lambda_rec = lambda_rec

    # ── Detection-only forward for SAHI inference ──────────────────────
    def detect_only(self, images):
        """Backbone + CD-DPA + standard detection. Used for SAHI weak-tile ID."""
        images_t, _ = self.base_model.transform(images, None)
        feats = self.base_model.backbone(images_t.tensors)
        for lvl in ('0', '1', '2'):
            feats[lvl] = self.cd_dpa[lvl](feats[lvl])
        proposals, _ = self.base_model.rpn(images_t, feats, None)
        dets, _      = self.base_model.roi_heads(feats, proposals, images_t.image_sizes, None)
        return self.base_model.transform.postprocess(
            dets, images_t.image_sizes,
            [(img.shape[1], img.shape[2]) for img in images],
        )

    @staticmethod
    def _build_img_inputs(original_images, images_t):
        bH, bW = images_t.tensors.shape[2:]
        batch = []
        for idx, img in enumerate(original_images):
            img = img.float()
            if img.max() > 1.0: img = img / 255.0
            if idx < len(images_t.image_sizes):
                th, tw = images_t.image_sizes[idx]
                if img.shape[1] != th or img.shape[2] != tw:
                    img = F.interpolate(img.unsqueeze(0), (th,tw),
                                        mode='bilinear', align_corners=False).squeeze(0)
            img = F.pad(img, (0, bW-img.shape[2], 0, bH-img.shape[1]))
            batch.append(img)
        return torch.stack(batch).to(images_t.tensors.device)

    def forward(self, images, targets, weak_tiles_per_img):
        original_images = [img.clone() for img in images]
        images_t, targets_p = self.base_model.transform(images, targets)
        img_inputs = self._build_img_inputs(original_images, images_t)

        # 1. Backbone + FPN
        features = self.base_model.backbone(images_t.tensors)

        # 2. CD-DPA on P2, P3, P4
        for lvl in ('0', '1', '2'):
            features[lvl] = self.cd_dpa[lvl](features[lvl])

        # 3. Upper stream: DGFE on weak P2 tiles
        B, C, fH, fW = features['0'].shape
        accumulated = torch.zeros_like(features['0'])
        rec_loss = features['0'].new_zeros(())
        n_tiles = 0

        for b in range(B):
            orig_h, orig_w = original_images[b].shape[1:]
            trans_h, trans_w = images_t.image_sizes[b]
            sy = trans_h / orig_h
            sx = trans_w / orig_w

            for (x1, y1, x2, y2) in weak_tiles_per_img[b]:
                tx1 = int(x1*sx); ty1 = int(y1*sy)
                tx2 = min(int(x2*sx), trans_w); ty2 = min(int(y2*sy), trans_h)
                fx1, fy1 = tx1//4, ty1//4
                fx2, fy2 = min(tx2//4, fW), min(ty2//4, fH)
                if fx2 <= fx1 or fy2 <= fy1:
                    continue

                p2_crop  = features['0'][b:b+1, :, fy1:fy2, fx1:fx2]
                img_crop = img_inputs[b:b+1, :, ty1:ty2, tx1:tx2]

                # Align img_crop to exact 4× P2 dims
                recon_h, recon_w = (fy2-fy1)*4, (fx2-fx1)*4
                if img_crop.shape[-2:] != (recon_h, recon_w):
                    img_crop = F.interpolate(img_crop, size=(recon_h, recon_w),
                                             mode='bilinear', align_corners=False)

                r_crop, delta = self.recon.forward_with_diff(p2_crop, img_crop)
                p2_enh = self.dgfe(delta, p2_crop)

                accumulated[b:b+1, :, fy1:fy2, fx1:fx2] += (p2_enh - p2_crop.detach())
                rec_loss = rec_loss + F.l1_loss(r_crop, img_crop)
                n_tiles += 1

        if n_tiles > 0:
            rec_loss = rec_loss / n_tiles

        # 4. P2'' = P2 + residuals from DGFE
        features['0'] = features['0'] + accumulated

        # 5. DenseFPN: fuse P2'' (upper) + P3–P6 (lower)
        features = self.dense_fpn(features)

        # 6. Detection
        if self.training:
            proposals, prop_losses = self.base_model.rpn(images_t, features, targets_p)
            _, det_losses = self.base_model.roi_heads(
                features, proposals, images_t.image_sizes, targets_p
            )
            losses = {**prop_losses, **det_losses}
            losses['loss_reconstruction'] = rec_loss * self.lambda_rec
            return losses
        else:
            proposals, _ = self.base_model.rpn(images_t, features, None)
            dets, _ = self.base_model.roi_heads(features, proposals, images_t.image_sizes, None)
            return self.base_model.transform.postprocess(
                dets, images_t.image_sizes,
                [(img.shape[1], img.shape[2]) for img in original_images],
            )


# ═══════════════════════ FREEZE ════════════════════════════════════════

def apply_freeze(model: nn.Module):
    """Freeze ResNet body layers 1-3. Everything else trains."""
    frozen = ['base_model.backbone.body.conv1', 'base_model.backbone.body.bn1',
              'base_model.backbone.body.layer1', 'base_model.backbone.body.layer2',
              'base_model.backbone.body.layer3']
    for name, p in model.named_parameters():
        if any(name.startswith(f) for f in frozen):
            p.requires_grad_(False)


# ═══════════════════════ OPTIMISER ═════════════════════════════════════

def build_optimiser(model):
    raw = model.module if IS_DIST else model
    new_modules, fine_tune, frozen_rest = [], [], []
    for name, p in raw.named_parameters():
        if not p.requires_grad: continue
        # New / heavily changed modules — full learning rate
        if any(name.startswith(x) for x in ('dense_fpn', 'recon', 'dgfe', 'cd_dpa')):
            new_modules.append(p)
        # Backbone FPN — fine-tune at lower lr
        else:
            fine_tune.append(p)
    return torch.optim.AdamW([
        {'params': new_modules, 'lr': CONFIG['learning_rate']},
        {'params': fine_tune,   'lr': CONFIG['backbone_lr']},
    ], weight_decay=CONFIG['weight_decay'])


def build_scheduler(opt, steps_per_epoch):
    total_steps = CONFIG['num_epochs'] * steps_per_epoch
    warmup_steps = min(CONFIG['warmup_epochs'] * steps_per_epoch, int(0.3*total_steps))
    pct_start = max(0.01, warmup_steps / total_steps)
    return torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=[CONFIG['learning_rate'], CONFIG['backbone_lr']],
        epochs=CONFIG['num_epochs'], steps_per_epoch=steps_per_epoch,
        pct_start=pct_start, anneal_strategy='cos',
    )


# ═══════════════════════ DATA ══════════════════════════════════════════

class TrainAug:
    def __call__(self, image, target):
        if random.random() > 0.5:
            w = image.width
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if len(target['boxes']) > 0:
                b = target['boxes'].clone(); b[:,[0,2]] = w-b[:,[2,0]]; target['boxes'] = b
        w, h = image.size
        image = torch.ByteTensor(bytearray(image.tobytes())).reshape(h,w,3).permute(2,0,1).float()/255.0
        return image, target

def collate_fn(batch): return tuple(zip(*batch))

def build_loaders():
    root = CONFIG['dataset_root']
    tr = VisDroneDataset(root, 'train', transforms=TrainAug())
    va = VisDroneDataset(root, 'val',   transforms=None)
    ts = DistributedSampler(tr, shuffle=True)  if IS_DIST else None
    vs = DistributedSampler(va, shuffle=False) if IS_DIST else None
    kw = dict(collate_fn=collate_fn, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    return (
        DataLoader(tr, CONFIG['batch_size'], sampler=ts, shuffle=(ts is None),
                   num_workers=CONFIG['train_workers'], **kw),
        DataLoader(va, CONFIG['batch_size'], sampler=vs, shuffle=False,
                   num_workers=CONFIG['val_workers'], **kw),
        ts,
    )


# ═══════════════════════ TRAIN / VAL ═══════════════════════════════════

def train_epoch(model, loader, opt, sched, scaler, epoch, writer):
    model.train()
    raw = model.module if IS_DIST else model
    total_losses = {}
    opt.zero_grad()
    acc = CONFIG['accumulation_steps']
    nan_skips = 0
    max_steps = CONFIG.get('sanity_steps', None)

    pbar = tqdm(loader, desc=f"Train E{epoch+1}", leave=False, disable=not IS_MAIN,
                total=max_steps or len(loader))

    for step, (images, targets) in enumerate(pbar):
        if max_steps and step >= max_steps:
            break

        images  = [img.to(DEVICE, non_blocking=True) for img in images]
        targets = [{k: v.to(DEVICE, non_blocking=True) for k,v in t.items()} for t in targets]

        # SAHI weak-tile selection (no-grad, eval mode, uses detect_only)
        weak_tiles_per_img = []
        raw.eval()
        for img in images:
            _, img_h, img_w = img.shape
            b, s, l = run_sahi(raw.detect_only, img, DEVICE,
                               CONFIG['tile_sizes'], CONFIG['overlap'], CONFIG['score_thresh'])
            b, s, l = per_class_nms(b, s, l, CONFIG['nms_iou'])
            weak = select_weak_tiles(b, s, img_h, img_w, CONFIG['weak_tile_size'], CONFIG['K'])
            weak_tiles_per_img.append(weak)
        raw.train()

        is_sync = (step+1) % acc == 0
        sync_ctx = contextlib.nullcontext() if (not IS_DIST or is_sync) else model.no_sync()

        with sync_ctx:
            with torch.amp.autocast('cuda', enabled=CONFIG['use_amp']):
                loss_dict = model(images, targets, weak_tiles_per_img)
                if not all(torch.isfinite(v) for v in loss_dict.values()):
                    nan_skips += 1; opt.zero_grad(); continue
                loss = sum(loss_dict.values()) / acc
            scaler.scale(loss).backward()

        if is_sync:
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
            if torch.isfinite(gn): scaler.step(opt)
            else: nan_skips += 1
            scaler.update(); opt.zero_grad(); sched.step()

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item()

    n_steps = min(len(loader), CONFIG.get('sanity_steps', len(loader)))
    avg = {k: v/max(n_steps,1) for k,v in total_losses.items()}
    avg['total'] = sum(avg.values())
    if IS_MAIN:
        for k, v in avg.items(): writer.add_scalar(f'train/{k}', v, epoch*n_steps)
    return avg


def val_epoch(model, loader, epoch, writer):
    model.train()
    total_losses = {}
    max_steps = CONFIG.get('sanity_steps', None)
    with torch.no_grad():
        for step, (images, targets) in enumerate(tqdm(loader, desc=f"Val E{epoch+1}",
                                                      leave=False, disable=not IS_MAIN,
                                                      total=max_steps or len(loader))):
            if max_steps and step >= max_steps:
                break
            images  = [img.to(DEVICE, non_blocking=True) for img in images]
            targets = [{k: v.to(DEVICE, non_blocking=True) for k,v in t.items()} for t in targets]
            with torch.amp.autocast('cuda', enabled=CONFIG['use_amp']):
                loss_dict = model(images, targets, [[] for _ in images])
            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0.0) + v.item()
    n_steps = min(len(loader), CONFIG.get('sanity_steps', len(loader)))
    avg = {k: v/max(n_steps,1) for k,v in total_losses.items()}
    avg['total'] = sum(avg.values())
    if IS_MAIN:
        for k, v in avg.items(): writer.add_scalar(f'val/{k}', v, epoch*n_steps)
    return avg


# ═══════════════════════ MAIN ══════════════════════════════════════════

def main():
    random.seed(42+LOCAL_RANK); torch.manual_seed(42+LOCAL_RANK)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',       type=int, default=CONFIG['num_epochs'])
    parser.add_argument('--resume',       type=str, default=None)
    parser.add_argument('--sanity',       action='store_true')
    parser.add_argument('--sanity-steps', type=int, default=4)
    args = parser.parse_args()
    if args.resume: CONFIG['resume_checkpoint'] = args.resume
    if args.sanity:
        CONFIG['num_epochs']   = 1
        CONFIG['sanity_steps'] = args.sanity_steps
    CONFIG['num_epochs'] = args.epochs if not args.sanity else CONFIG['num_epochs']

    out_dir = Path(CONFIG['output_dir'])
    if IS_MAIN: out_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    model = V3FullFramework(CONFIG['num_classes'], CONFIG['lambda_rec']).to(DEVICE)

    # Load Phase 2 checkpoint:
    #   - backbone, recon, dgfe weights transfer
    #   - cd_dpa: Phase2 had SimpleDPA (different arch) → skip, init CDDPA from scratch
    #   - dense_fpn: new module → init from scratch
    init_ckpt = CONFIG['init_checkpoint']
    if Path(init_ckpt).exists():
        ck = torch.load(init_ckpt, map_location='cpu', weights_only=False)
        state = {}
        for k, v in ck['model_state_dict'].items():
            nk = k.replace('rgr.', 'dgfe.').replace('enhancers.', 'cd_dpa.')
            # Skip SimpleDPA weights — CDDPA has different architecture
            if nk.startswith('cd_dpa.'):
                continue
            state[nk] = v
        missing, unexpected = model.load_state_dict(state, strict=False)
        if IS_MAIN:
            transferred = len(ck['model_state_dict']) - len([k for k in ck['model_state_dict']
                             if k.replace('rgr.','dgfe.').replace('enhancers.','cd_dpa.').startswith('cd_dpa.')])
            print(f"[Init] Phase2 weights loaded")
            print(f"  Transferred  : {transferred} keys  (backbone, recon, dgfe)")
            print(f"  Random init  : {len(missing)} keys (CD-DPA × 3 levels, DenseFPN)")

    apply_freeze(model)

    if IS_DIST:
        model = DDP(model, device_ids=[LOCAL_RANK], find_unused_parameters=False)

    train_loader, val_loader, train_sampler = build_loaders()
    steps_per_epoch = len(train_loader)
    opt    = build_optimiser(model)
    sched  = build_scheduler(opt, steps_per_epoch)
    scaler = torch.amp.GradScaler('cuda', enabled=CONFIG['use_amp'])
    writer = SummaryWriter(str(out_dir/'tensorboard')) if IS_MAIN else None

    start_epoch, best_val = 0, float('inf')
    if CONFIG.get('resume_checkpoint') and Path(CONFIG['resume_checkpoint']).exists():
        ck = torch.load(CONFIG['resume_checkpoint'], map_location=DEVICE)
        (model.module if IS_DIST else model).load_state_dict(ck['model_state_dict'])
        opt.load_state_dict(ck['optimiser_state_dict'])
        start_epoch = ck['epoch'] + 1
        best_val    = ck.get('best_val_loss', float('inf'))
        if IS_MAIN: print(f"[Resume] ep={start_epoch}  best_val={best_val:.4f}")

    if IS_MAIN:
        eff = CONFIG['batch_size'] * WORLD_SIZE * CONFIG['accumulation_steps']
        print(f"\n[V3 Full Framework]  {CONFIG['num_epochs']} epochs")
        print(f"  CD-DPA levels : P2, P3, P4")
        print(f"  SAHI scales   : {CONFIG['tile_sizes']}  K={CONFIG['K']}")
        print(f"  λ_rec={CONFIG['lambda_rec']}  eff_batch={eff}")
        total_params = sum(p.numel() for p in model.parameters()) / 1e6
        trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"  Params: {total_params:.2f}M total  {trainable:.2f}M trainable")

    history = []
    for epoch in range(start_epoch, CONFIG['num_epochs']):
        if train_sampler: train_sampler.set_epoch(epoch)
        t0 = time.time()
        tr  = train_epoch(model, train_loader, opt, sched, scaler, epoch, writer)
        val = val_epoch(model, val_loader, epoch, writer)
        elapsed = time.time() - t0

        if IS_MAIN:
            print(f"E{epoch+1:03d}/{CONFIG['num_epochs']} "
                  f"| train={tr['total']:.4f} recon={tr.get('loss_reconstruction',0):.4f} "
                  f"| val={val['total']:.4f} | {elapsed/60:.1f}min")

            raw = model.module if IS_DIST else model
            ck  = {'epoch': epoch, 'model_state_dict': raw.state_dict(),
                   'optimiser_state_dict': opt.state_dict(),
                   'val_loss': val['total'], 'config': CONFIG}
            torch.save(ck, out_dir/'last_checkpoint.pth')
            if val['total'] < best_val:
                best_val = val['total']
                torch.save(ck, out_dir/'best_model_v3.pth')
                print(f"  ↑ New best val={best_val:.4f}")

            history.append({'epoch': epoch+1, 'train': tr, 'val': val})
            with open(out_dir/'history.json', 'w') as f:
                json.dump(history, f, indent=2)

    if IS_DIST: dist.destroy_process_group()
    if IS_MAIN: print(f"\n[Done] best_val={best_val:.4f}  saved to {out_dir}")


if __name__ == '__main__':
    main()
