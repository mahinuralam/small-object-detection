"""
V3 Full Framework Evaluation.

Compares:
  1. V3 alone       (CD-DPA + SAHI + DGFE + DenseFPN)
  2. Baseline + V3 ensemble
  3. Baseline + Phase2 ensemble  (reference = 47.71%)
  4. Baseline + DPA ensemble     (reference = 47.38%)

Identical SAHI settings to scripts 24/25:
    tile_sizes=(512,640,768,960,1280), overlap=0.35,
    score_thresh=0.008, nms_iou=0.35, TTA=H-flip

Usage:
    python scripts/eval/26_v3_eval.py
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
from tqdm import tqdm
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.enhancements.reconstruction_head import ReconstructionHead
from models.enhancements.dgfe_module import DGFEModule
from models.enhancements.dense_fpn import DenseFPN
from models.enhancements.cddpa_module import CDDPA

spec = importlib.util.spec_from_file_location(
    "visdrone_dataset", Path(__file__).parent.parent / "4_visdrone_dataset.py"
)
visdrone_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visdrone_mod)
VisDroneDataset = visdrone_mod.VisDroneDataset

VISDRONE_CLASS_NAMES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor',
]


# ── Model definitions ──────────────────────────────────────────────────

class SimpleDPAModule(nn.Module):
    """SimpleDPA — used by Phase2 model."""
    def __init__(self, channels=256, reduction=16):
        super().__init__()
        self.edge_conv3 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.edge_conv5 = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels)
        self.spatial_att = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1), nn.ReLU(inplace=True),
            nn.Conv2d(channels//reduction, channels, 1), nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1), nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        e = self.edge_conv3(x) + self.edge_conv5(x)
        return self.fusion(torch.cat([e*self.spatial_att(e), x*self.channel_att(x)], 1)) + x


class V3Model(nn.Module):
    """V3 eval-mode model: CD-DPA(P2,P3,P4) + DGFE on P2 + DenseFPN."""
    def __init__(self, num_classes=11):
        super().__init__()
        self.base_model = fasterrcnn_resnet50_fpn(weights=None)
        in_f = self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
        self.cd_dpa = nn.ModuleDict({
            '0': CDDPA(channels=256, use_checkpoint=False),
            '1': CDDPA(channels=256, use_checkpoint=False),
            '2': CDDPA(channels=256, use_checkpoint=False),
        })
        self.recon    = ReconstructionHead(in_channels=256)
        self.dgfe     = DGFEModule(error_channels=1, feat_channels=256)
        self.dense_fpn = DenseFPN(channels=256)

    def forward(self, images, targets=None):
        original_images = images
        images_t, _ = self.base_model.transform(images, None)
        features = self.base_model.backbone(images_t.tensors)
        for lvl in ('0', '1', '2'):
            features[lvl] = self.cd_dpa[lvl](features[lvl])
        # No DGFE at eval time (no weak tiles) — P2 passes through unchanged
        features = self.dense_fpn(features)
        proposals, _ = self.base_model.rpn(images_t, features, None)
        dets, _      = self.base_model.roi_heads(features, proposals, images_t.image_sizes, None)
        return self.base_model.transform.postprocess(
            dets, images_t.image_sizes,
            [(img.shape[1], img.shape[2]) for img in original_images],
        )


class Phase2Model(nn.Module):
    """Phase 2 eval-mode model (SimpleDPA + DGFE)."""
    def __init__(self, num_classes=11):
        super().__init__()
        self.base_model = fasterrcnn_resnet50_fpn(weights=None)
        in_f = self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
        self.cd_dpa = nn.ModuleDict({
            '0': SimpleDPAModule(256),
            '1': SimpleDPAModule(256),
        })
        self.recon = ReconstructionHead(in_channels=256)
        self.dgfe  = DGFEModule(error_channels=1, feat_channels=256)

    def forward(self, images, targets=None):
        original_images = images
        images_t, _ = self.base_model.transform(images, None)
        features = self.base_model.backbone(images_t.tensors)
        for lvl in ('0', '1'):
            features[lvl] = self.cd_dpa[lvl](features[lvl])
        proposals, _ = self.base_model.rpn(images_t, features, None)
        dets, _      = self.base_model.roi_heads(features, proposals, images_t.image_sizes, None)
        return self.base_model.transform.postprocess(
            dets, images_t.image_sizes,
            [(img.shape[1], img.shape[2]) for img in original_images],
        )


def load_baseline(ckpt_path, device):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_f  = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 11)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    return model.to(device).eval()


def load_v3(ckpt_path, device):
    model = V3Model(11)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    return model.to(device).eval()


def load_phase2(ckpt_path, device):
    model = Phase2Model(11)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # Remap old key names
    state = {k.replace('rgr.', 'dgfe.').replace('enhancers.', 'cd_dpa.'): v
             for k, v in ck['model_state_dict'].items()}
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


# ── SAHI helpers ───────────────────────────────────────────────────────

def generate_tiles(img_w, img_h, tile_size=640, overlap=0.35):
    step = int(tile_size * (1 - overlap))
    tiles = []
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x2 = min(x+tile_size, img_w); y2 = min(y+tile_size, img_h)
            x1 = max(0, x2-tile_size);   y1 = max(0, y2-tile_size)
            tiles.append((x1, y1, x2, y2))
            if x2 == img_w: break
            x += step
        if y2 == img_h: break
        y += step
    return tiles


@torch.no_grad()
def sahi_single_scale(model, image, device, tile_size, overlap, score_thresh):
    _, img_h, img_w = image.shape
    tiles = generate_tiles(img_w, img_h, tile_size, overlap)
    all_b, all_s, all_l = [], [], []
    for (x1, y1, x2, y2) in tiles:
        tile = image[:, y1:y2, x1:x2]
        pw = tile_size-(x2-x1); ph = tile_size-(y2-y1)
        if pw > 0 or ph > 0:
            tile = F.pad(tile, (0, pw, 0, ph))
        p = model([tile.to(device)])[0]
        b, s, l = p['boxes'].cpu(), p['scores'].cpu(), p['labels'].cpu()
        keep = s >= score_thresh
        b, s, l = b[keep], s[keep], l[keep]
        b[:, 0] += x1; b[:, 2] += x1; b[:, 1] += y1; b[:, 3] += y1
        b[:, [0,2]].clamp_(0, img_w); b[:, [1,3]].clamp_(0, img_h)
        all_b.append(b); all_s.append(s); all_l.append(l)
    return torch.cat(all_b), torch.cat(all_s), torch.cat(all_l)


@torch.no_grad()
def collect_all_preds(model, image, device,
                      tile_sizes=(512,640,768,960,1280),
                      overlap=0.35, score_thresh=0.008, nms_iou=0.35):
    _, _, img_w = image.shape
    all_b, all_s, all_l = [], [], []
    p = model([image.to(device)])[0]
    bf, sf, lf = p['boxes'].cpu(), p['scores'].cpu(), p['labels'].cpu()
    keep = sf >= score_thresh
    if keep.any():
        all_b.append(bf[keep]); all_s.append(sf[keep]); all_l.append(lf[keep])
    for aug_idx, aug in enumerate([image, torch.flip(image, [2])]):
        for ts in tile_sizes:
            b, s, l = sahi_single_scale(model, aug, device, ts, overlap, score_thresh)
            if b.shape[0] > 0:
                if aug_idx == 1:
                    b2 = b.clone(); b2[:,0] = img_w-b[:,2]; b2[:,2] = img_w-b[:,0]; b = b2
                all_b.append(b); all_s.append(s); all_l.append(l)
    if not all_b:
        return torch.zeros(0,4), torch.zeros(0), torch.zeros(0, dtype=torch.long)
    return torch.cat(all_b), torch.cat(all_s), torch.cat(all_l)


def per_class_nms(all_b, all_s, all_l, nms_iou=0.35):
    if all_b.shape[0] == 0:
        return all_b, all_s, all_l
    fb, fs, fl = [], [], []
    for cls in all_l.unique():
        m = all_l == cls; b, s = all_b[m], all_s[m]
        keep = nms(b, s, nms_iou)
        fb.append(b[keep]); fs.append(s[keep]); fl.append(all_l[m][keep])
    return torch.cat(fb), torch.cat(fs), torch.cat(fl)


# ── Evaluation ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(models, dataset, device, label):
    metric = MeanAveragePrecision(
        iou_type='bbox', class_metrics=True,
        max_detection_thresholds=[1, 10, 500],
    )
    metric.warn_on_many_detections = False
    for idx in tqdm(range(len(dataset)), desc=label):
        image, target = dataset[idx]
        all_b, all_s, all_l = [], [], []
        for model in models:
            b, s, l = collect_all_preds(model, image, device)
            if b.shape[0] > 0:
                all_b.append(b); all_s.append(s); all_l.append(l)
        if not all_b:
            boxes = torch.zeros(0,4); scores = torch.zeros(0); labels = torch.zeros(0, dtype=torch.long)
        else:
            boxes, scores, labels = per_class_nms(torch.cat(all_b), torch.cat(all_s), torch.cat(all_l))
        metric.update(
            [{'boxes': boxes, 'scores': scores, 'labels': labels}],
            [{'boxes': target['boxes'].cpu(), 'labels': target['labels'].cpu()}],
        )
    return metric.compute()


def print_results(label, results):
    m50   = results['map_50'].item() * 100
    m75   = results['map_75'].item() * 100
    print(f'\n{"="*65}')
    print(f'RESULTS — {label}')
    print(f'{"="*65}')
    print(f'  mAP@0.50      : {m50:6.2f}%')
    print(f'  mAP@0.75      : {m75:6.2f}%')
    per = results.get('map_per_class')
    if isinstance(per, torch.Tensor) and per.numel() > 0:
        cls_ids = results.get('classes')
        print(f'\n  Per-class mAP@0.50:')
        for i, ap in enumerate(results.get('map_per_class_50', per)):
            cid = int(cls_ids[i].item()) if cls_ids is not None else i+1
            name = VISDRONE_CLASS_NAMES[cid-1] if 1 <= cid <= 10 else f'cls_{cid}'
            print(f'    {name:20s}: {ap.item()*100:6.2f}%')
    return {'label': label, 'mAP_50': round(m50/100, 6), 'mAP_75': round(m75/100, 6)}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU   : {torch.cuda.get_device_name(0)}')

    baseline_ckpt = PROJECT_ROOT / 'results' / 'outputs'     / 'best_model.pth'
    p2_ckpt       = PROJECT_ROOT / 'results' / 'outputs_phase2_smart_recon' / 'best_model_phase2.pth'
    v3_ckpt       = PROJECT_ROOT / 'results' / 'outputs_v3_full_framework'  / 'best_model_v3.pth'
    data_root     = str(PROJECT_ROOT.parent / 'dataset' / 'VisDrone-2018')

    print('\nLoading models...')
    m_base = load_baseline(baseline_ckpt, device)
    m_p2   = load_phase2(p2_ckpt, device)
    m_v3   = load_v3(v3_ckpt, device)
    print('  All models loaded')

    print('\nLoading dataset...')
    dataset = VisDroneDataset(root_dir=data_root, split='val', transforms=None)
    print(f'  {len(dataset)} images')

    all_results = []

    r1 = evaluate([m_v3],          dataset, device, 'V3 alone')
    all_results.append(print_results('V3 alone  (CD-DPA + DGFE + DenseFPN)', r1))

    r2 = evaluate([m_base, m_v3],  dataset, device, 'Baseline + V3')
    all_results.append(print_results('Ensemble: Baseline + V3', r2))

    r3 = evaluate([m_base, m_p2],  dataset, device, 'Baseline + Phase2 (ref)')
    all_results.append(print_results('Reference: Baseline + Phase2 (47.71%)', r3))

    out_path = PROJECT_ROOT / 'results' / 'eval_v3_val.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n  Saved: {out_path}')
    print('='*65)


if __name__ == '__main__':
    main()
