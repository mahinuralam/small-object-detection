"""
Ablation analysis: GFLOPs, Params, and mAP for each module stack.

Stacking order: Baseline → +CD-DPA → +SAHI → +Recon+RGR → +DenseFPN

Usage:
    cd "simple implementation"
    python scripts/eval/21_ablation_analysis.py
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from thop import profile as thop_profile, clever_format
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.enhancements.cddpa_module import CDDPA
from models.enhancements.dense_fpn import DenseFPN
from models.enhancements.reconstruction_module import ReconstructionModule
from models.enhancements.rgr_module import RGRModule

DEVICE   = torch.device('cpu')   # profiling on CPU for determinism
H, W     = 640, 640              # standard VisDrone crop / resize
CHANNELS = 256

# ── Feature map sizes at each FPN level for 640×640 input ─────────────
# P2: H/4,  P3: H/8,  P4: H/16, P5: H/32, pool: H/64
FEAT_SIZES = {
    '0':    (H // 4,  W // 4),
    '1':    (H // 8,  W // 8),
    '2':    (H // 16, W // 16),
    '3':    (H // 32, W // 32),
    'pool': (H // 64, W // 64),
}


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def dummy_feats(levels=None):
    levels = levels or list(FEAT_SIZES.keys())
    return {k: torch.zeros(1, CHANNELS, *FEAT_SIZES[k]) for k in levels}


def count_params(module: nn.Module) -> float:
    return sum(p.numel() for p in module.parameters()) / 1e6


def profile_module(module: nn.Module, *args) -> float:
    """Return GMACs (= GFLOPs/2 in multiply-add convention) × 2 → GFLOPs."""
    module.eval()
    with torch.no_grad():
        macs, _ = thop_profile(module, inputs=args, verbose=False)
    return macs * 2 / 1e9   # MACs → FLOPs → GFLOPs


# ── Wrappers for dict-input modules ────────────────────────────────────

class DenseFPNWrapper(nn.Module):
    def __init__(self, dfpn): super().__init__(); self.dfpn = dfpn
    def forward(self, p2, p3, p4, p5, pool):
        feats = {'0': p2, '1': p3, '2': p4, '3': p5, 'pool': pool}
        out = self.dfpn(feats)
        return out['0']   # thop needs a tensor return


class BodyWrapper(nn.Module):
    """Strip OrderedDict return so thop can trace it."""
    def __init__(self, body): super().__init__(); self.body = body
    def forward(self, x):
        out = self.body(x)
        return tuple(out.values())


class FPNWrapper(nn.Module):
    def __init__(self, fpn): super().__init__(); self.fpn = fpn
    def forward(self, p2, p3, p4, p5):
        feats = {'0': p2, '1': p3, '2': p4, '3': p5}
        out = self.fpn(feats)
        return out['0']


# ═══════════════════════════════════════════════════════════════════════
# GFLOPs per component
# ═══════════════════════════════════════════════════════════════════════

def get_backbone_fpn_gflops():
    base = fasterrcnn_resnet50_fpn(weights=None, trainable_backbone_layers=5)
    dummy = torch.zeros(1, 3, H, W)

    body    = BodyWrapper(base.backbone.body)
    gflops_body = profile_module(body, dummy)

    # Get actual feature maps for FPN input
    with torch.no_grad():
        feat_maps = base.backbone.body(dummy)
    fpn = FPNWrapper(base.backbone.fpn)
    p2 = feat_maps['0']; p3 = feat_maps['1']
    p4 = feat_maps['2']; p5 = feat_maps['3']
    gflops_fpn = profile_module(fpn, p2, p3, p4, p5)

    return gflops_body, gflops_fpn


def get_cddpa_gflops(levels=3):
    cddpa = CDDPA(channels=CHANNELS)
    # Use P2 (largest, dominant cost) as reference and scale by level count
    # P3 and P4 are 1/4 and 1/16 area of P2 respectively
    areas = [
        FEAT_SIZES['0'][0] * FEAT_SIZES['0'][1],
        FEAT_SIZES['1'][0] * FEAT_SIZES['1'][1],
        FEAT_SIZES['2'][0] * FEAT_SIZES['2'][1],
    ]
    p2_dummy = torch.zeros(1, CHANNELS, *FEAT_SIZES['0'])
    gflops_p2 = profile_module(cddpa, p2_dummy)
    # Scale per level by relative spatial area
    ref_area = areas[0]
    total = sum(gflops_p2 * (a / ref_area) for a in areas[:levels])
    return total


def get_densefpn_gflops():
    dfpn = DenseFPN(channels=CHANNELS)
    wrapper = DenseFPNWrapper(dfpn)
    f = dummy_feats()
    return profile_module(wrapper, f['0'], f['1'], f['2'], f['3'], f['pool'])


def get_recon_gflops():
    recon = ReconstructionModule(in_channels=CHANNELS)
    p2 = torch.zeros(1, CHANNELS, *FEAT_SIZES['0'])
    return profile_module(recon, p2)


def get_rgr_gflops():
    rgr = RGRModule(error_channels=3, feat_channels=CHANNELS)
    err  = torch.zeros(1, 3, *FEAT_SIZES['0'])
    feat = torch.zeros(1, CHANNELS, *FEAT_SIZES['0'])
    return profile_module(rgr, err, feat)


def get_rpn_roi_gflops():
    """
    RPN + ROI are roughly constant across all variants (~same proposals/boxes).
    Estimate empirically via a full model forward on a tiny dummy image.
    """
    model = fasterrcnn_resnet50_fpn(weights=None, trainable_backbone_layers=5)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 11)
    model.eval()

    dummy_img = [torch.zeros(3, H, W)]
    with torch.no_grad():
        # Profile full model, then subtract backbone
        class FullWrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x):
                imgs = [xi for xi in x]
                return self.m(imgs)
        full_g = profile_module(FullWrapper(model), torch.zeros(1, 3, H, W))

    body_g, fpn_g = get_backbone_fpn_gflops()
    return max(full_g - body_g - fpn_g, 0.0)


# ═══════════════════════════════════════════════════════════════════════
# Param counts
# ═══════════════════════════════════════════════════════════════════════

def param_counts():
    base = fasterrcnn_resnet50_fpn(weights=None, trainable_backbone_layers=5)
    in_f = base.roi_heads.box_predictor.cls_score.in_features
    base.roi_heads.box_predictor = FastRCNNPredictor(in_f, 11)

    p_base   = count_params(base)
    p_cddpa  = count_params(CDDPA(channels=CHANNELS)) * 3   # 3 levels
    p_dfpn   = count_params(DenseFPN(channels=CHANNELS))
    p_recon  = count_params(ReconstructionModule(in_channels=CHANNELS))
    p_rgr    = count_params(RGRModule(error_channels=3, feat_channels=CHANNELS))

    return p_base, p_cddpa, p_dfpn, p_recon, p_rgr


# ═══════════════════════════════════════════════════════════════════════
# mAP from saved eval JSONs
# ═══════════════════════════════════════════════════════════════════════

def load_map(json_path):
    p = Path(json_path)
    if not p.exists():
        return None, None
    with open(p) as f:
        d = json.load(f)
    return d.get('mAP_50'), d.get('mAP_50_95')


# ═══════════════════════════════════════════════════════════════════════
# SAHI overhead
# ═══════════════════════════════════════════════════════════════════════

VISDRONE_W, VISDRONE_H = 1920, 1080
SAHI_SLICE            = 640
SAHI_OVERLAP          = 0.2
SAHI_STRIDE           = int(SAHI_SLICE * (1 - SAHI_OVERLAP))
SAHI_SLICES           = (
    -(-VISDRONE_W // SAHI_STRIDE) *   # ceil div
    -(-VISDRONE_H // SAHI_STRIDE)
)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("\nProfiling components (CPU, 640×640 input) ...")

    # GFLOPs per component
    print("  backbone body ...", end=' ', flush=True)
    g_body, g_fpn = get_backbone_fpn_gflops()
    print(f"{g_body:.2f} + {g_fpn:.2f} GFLOPs")

    print("  CD-DPA (3 levels) ...", end=' ', flush=True)
    g_cddpa = get_cddpa_gflops(3)
    print(f"{g_cddpa:.2f} GFLOPs")

    print("  DenseFPN ...", end=' ', flush=True)
    g_dfpn = get_densefpn_gflops()
    print(f"{g_dfpn:.2f} GFLOPs")

    print("  ReconstructionModule ...", end=' ', flush=True)
    g_recon = get_recon_gflops()
    print(f"{g_recon:.2f} GFLOPs")

    print("  RGRModule ...", end=' ', flush=True)
    g_rgr = get_rgr_gflops()
    print(f"{g_rgr:.2f} GFLOPs")

    print("  RPN + ROI head ...", end=' ', flush=True)
    g_rpn_roi = get_rpn_roi_gflops()
    print(f"{g_rpn_roi:.2f} GFLOPs")

    # Param counts
    p_base, p_cddpa, p_dfpn, p_recon, p_rgr = param_counts()

    # mAP from saved results
    out_root = PROJECT_ROOT / 'results'
    map50_baseline,   map5095_baseline   = 0.3802, None   # from earlier eval
    map50_cddpa,      map5095_cddpa      = 0.4344, None   # from earlier eval
    map50_densefpn,   map5095_densefpn   = load_map(
        out_root / 'outputs_full_framework_densefpn' / 'eval_results_val.json'
    )

    # Build rows
    g_base = g_body + g_fpn + g_rpn_roi

    rows = [
        {
            'name':    'Baseline (Faster R-CNN R50-FPN)',
            'params':  p_base,
            'gflops':  g_base,
            'sahi_x':  1,
            'map50':   map50_baseline,
            'map5095': map5095_baseline,
        },
        {
            'name':    '+ CD-DPA (P2, P3, P4)',
            'params':  p_base + p_cddpa,
            'gflops':  g_base + g_cddpa,
            'sahi_x':  1,
            'map50':   map50_cddpa,
            'map5095': map5095_cddpa,
        },
        {
            'name':    '+ CD-DPA + SAHI inference',
            'params':  p_base + p_cddpa,
            'gflops':  g_base + g_cddpa,
            'sahi_x':  SAHI_SLICES,
            'map50':   None,
            'map5095': None,
        },
        {
            'name':    '+ CD-DPA + SAHI + Recon + RGR',
            'params':  p_base + p_cddpa + p_recon + p_rgr,
            'gflops':  g_base + g_cddpa + g_recon + g_rgr,
            'sahi_x':  SAHI_SLICES,
            'map50':   None,
            'map5095': None,
        },
        {
            'name':    '+ CD-DPA + SAHI + Recon + RGR + DenseFPN  (Full)',
            'params':  p_base + p_dfpn + p_cddpa + p_recon + p_rgr,
            'gflops':  g_base + g_dfpn + g_cddpa + g_recon + g_rgr,
            'sahi_x':  SAHI_SLICES,
            'map50':   map50_densefpn,
            'map5095': map5095_densefpn,
        },
    ]

    # ── Print table ──────────────────────────────────────────────────
    sep = '─' * 118
    hdr = (f"{'Model Variant':<50} {'Params':>9} {'GFLOPs†':>9} "
           f"{'SAHI×':>7} {'Total GFLOPs':>13} {'mAP@0.50':>10} {'mAP@0.50:0.95':>14}")

    print('\n' + '═' * 118)
    print('ABLATION TABLE  —  VisDrone-2018 Val Split  (input 640×640)')
    print('═' * 118)
    print(hdr)
    print(sep)
    for r in rows:
        map50_s   = f"{r['map50']*100:.2f}%"   if r['map50']   is not None else '—'
        map5095_s = f"{r['map5095']*100:.2f}%" if r['map5095'] is not None else '—'
        total_g   = r['gflops'] * r['sahi_x']
        sahi_s    = f"×{r['sahi_x']}" if r['sahi_x'] > 1 else '—'
        print(
            f"{r['name']:<50} "
            f"{r['params']:>7.2f}M "
            f"{r['gflops']:>9.1f} "
            f"{sahi_s:>7} "
            f"{total_g:>13.1f} "
            f"{map50_s:>10} "
            f"{map5095_s:>14}"
        )
    print(sep)
    print('  † GFLOPs per single forward pass at 640×640')
    print(f'  SAHI: {SAHI_SLICE}×{SAHI_SLICE}px slices, {int(SAHI_OVERLAP*100)}% overlap → {SAHI_SLICES} slices per 1920×1080 image')

    # ── Per-component delta table ────────────────────────────────────
    print('\nPer-component breakdown (640×640, single-pass):')
    print(f"  {'Component':<30} {'ΔParams':>9} {'ΔGFLOPs':>10}")
    print('  ' + '─' * 52)
    comps = [
        ('Backbone (ResNet50)',      0,      g_body),
        ('Standard FPN',            0,      g_fpn),
        ('RPN + ROI Head',          0,      g_rpn_roi),
        ('CD-DPA  (3 levels)',       p_cddpa, g_cddpa),
        ('DenseFPN neck',           p_dfpn,  g_dfpn),
        ('ReconstructionModule',    p_recon, g_recon),
        ('RGR Module',              p_rgr,   g_rgr),
    ]
    for name, dp, dg in comps:
        dp_s = f'+{dp:.2f}M' if dp > 0 else f'{p_base:.2f}M'
        print(f"  {name:<30} {dp_s:>9} {dg:>9.2f}G")

    # ── SAHI note ───────────────────────────────────────────────────
    print(f'\nSAHI settings (1920×1080 VisDrone images):')
    print(f'  Slice size : {SAHI_SLICE}×{SAHI_SLICE}px')
    print(f'  Overlap    : {int(SAHI_OVERLAP*100)}%  (stride {SAHI_STRIDE}px)')
    print(f'  Slices/img : {SAHI_SLICES}')
    print(f'  GFLOPs ×   : {SAHI_SLICES}  (per full-image inference pass)')

    print('\n' + '═' * 108 + '\n')


if __name__ == '__main__':
    main()
