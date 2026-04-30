"""
Full evaluation with COCO-style metrics using pycocotools.

Reports:
  AP@0.50:0.95  AP@0.50  AP@0.75
  APvt  (area <  100 px²  — very tiny, < 10×10)
  APt   (area  100–400    — tiny,      10×20px)
  APs   (area  400–1024   — small,     20×32px)
  Per-class AP@0.50:0.95 for all 10 VisDrone categories

Models evaluated:
  V3   — CD-DPA(P2,P3,P4) + DGFE + DenseFPN
  Baseline + V3 ensemble (best mAP configuration)

Usage:
    python scripts/eval/27_full_eval.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
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

# ── Area range thresholds (confirmed from VisDrone val distribution) ──
AREA_RANGES = {
    'all':  [0,      1e10],
    'vt':   [0,      100],      # very tiny  < 10×10 px
    't':    [100,    400],      # tiny        10–20 px
    's':    [400,    1024],     # small       20–32 px
}


# ── Model definitions ──────────────────────────────────────────────────

class SimpleDPAModule(nn.Module):
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
        self.recon     = ReconstructionHead(in_channels=256)
        self.dgfe      = DGFEModule(error_channels=1, feat_channels=256)
        self.dense_fpn = DenseFPN(channels=256)

    def forward(self, images, targets=None):
        original_images = images
        images_t, _ = self.base_model.transform(images, None)
        features = self.base_model.backbone(images_t.tensors)
        for lvl in ('0', '1', '2'):
            features[lvl] = self.cd_dpa[lvl](features[lvl])
        features = self.dense_fpn(features)
        proposals, _ = self.base_model.rpn(images_t, features, None)
        dets, _      = self.base_model.roi_heads(features, proposals, images_t.image_sizes, None)
        return self.base_model.transform.postprocess(
            dets, images_t.image_sizes,
            [(img.shape[1], img.shape[2]) for img in original_images],
        )


class Phase2Model(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.base_model = fasterrcnn_resnet50_fpn(weights=None)
        in_f = self.base_model.roi_heads.box_predictor.cls_score.in_features
        self.base_model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
        self.cd_dpa = nn.ModuleDict({'0': SimpleDPAModule(), '1': SimpleDPAModule()})
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
    state = {k.replace('rgr.','dgfe.').replace('enhancers.','cd_dpa.'): v
             for k, v in ck['model_state_dict'].items()}
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


# ── SAHI helpers ───────────────────────────────────────────────────────

def generate_tiles(img_w, img_h, tile_size=640, overlap=0.35):
    step = int(tile_size * (1 - overlap))
    tiles = []; y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x2 = min(x+tile_size, img_w); y2 = min(y+tile_size, img_h)
            x1 = max(0, x2-tile_size);   y1 = max(0, y2-tile_size)
            tiles.append((x1,y1,x2,y2))
            if x2 == img_w: break
            x += step
        if y2 == img_h: break
        y += step
    return tiles

@torch.no_grad()
def sahi_single_scale(model, image, device, tile_size, overlap, score_thresh):
    _, img_h, img_w = image.shape
    all_b, all_s, all_l = [], [], []
    for (x1,y1,x2,y2) in generate_tiles(img_w, img_h, tile_size, overlap):
        tile = image[:, y1:y2, x1:x2]
        pw = tile_size-(x2-x1); ph = tile_size-(y2-y1)
        if pw>0 or ph>0: tile = F.pad(tile,(0,pw,0,ph))
        p = model([tile.to(device)])[0]
        b,s,l = p['boxes'].cpu(), p['scores'].cpu(), p['labels'].cpu()
        keep = s>=score_thresh; b,s,l = b[keep],s[keep],l[keep]
        b[:,0]+=x1; b[:,2]+=x1; b[:,1]+=y1; b[:,3]+=y1
        b[:,[0,2]].clamp_(0,img_w); b[:,[1,3]].clamp_(0,img_h)
        all_b.append(b); all_s.append(s); all_l.append(l)
    return torch.cat(all_b), torch.cat(all_s), torch.cat(all_l)

@torch.no_grad()
def collect_all_preds(model, image, device,
                      tile_sizes=(512,640,768,960,1280),
                      overlap=0.35, score_thresh=0.008):
    _, _, img_w = image.shape
    all_b, all_s, all_l = [], [], []
    p = model([image.to(device)])[0]
    bf,sf,lf = p['boxes'].cpu(),p['scores'].cpu(),p['labels'].cpu()
    keep = sf>=score_thresh
    if keep.any():
        all_b.append(bf[keep]); all_s.append(sf[keep]); all_l.append(lf[keep])
    for aug_idx, aug in enumerate([image, torch.flip(image,[2])]):
        for ts in tile_sizes:
            b,s,l = sahi_single_scale(model, aug, device, ts, overlap, score_thresh)
            if b.shape[0]>0:
                if aug_idx==1:
                    b2=b.clone(); b2[:,0]=img_w-b[:,2]; b2[:,2]=img_w-b[:,0]; b=b2
                all_b.append(b); all_s.append(s); all_l.append(l)
    if not all_b:
        return torch.zeros(0,4), torch.zeros(0), torch.zeros(0,dtype=torch.long)
    return torch.cat(all_b), torch.cat(all_s), torch.cat(all_l)

def per_class_nms(all_b, all_s, all_l, nms_iou=0.35):
    if all_b.shape[0]==0: return all_b, all_s, all_l
    fb,fs,fl = [],[],[]
    for cls in all_l.unique():
        m=all_l==cls; b,s=all_b[m],all_s[m]
        keep=nms(b,s,nms_iou)
        fb.append(b[keep]); fs.append(s[keep]); fl.append(all_l[m][keep])
    return torch.cat(fb), torch.cat(fs), torch.cat(fl)


# ── Build COCO-format GT and predictions ──────────────────────────────

def build_coco_gt(dataset):
    """Convert VisDrone dataset to COCO-format ground truth dict."""
    coco_gt = {'images': [], 'annotations': [], 'categories': []}
    for cid, name in enumerate(VISDRONE_CLASS_NAMES, 1):
        coco_gt['categories'].append({'id': cid, 'name': name})

    ann_id = 1
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        img_id = int(target['image_id'].item())
        img_tensor, _ = dataset[idx]
        h, w = img_tensor.shape[1], img_tensor.shape[2]
        coco_gt['images'].append({'id': img_id, 'height': h, 'width': w})

        boxes  = target['boxes']
        labels = target['labels']
        for box, label in zip(boxes, labels):
            x1,y1,x2,y2 = box.tolist()
            bw, bh = x2-x1, y2-y1
            area = bw * bh
            coco_gt['annotations'].append({
                'id':          ann_id,
                'image_id':    img_id,
                'category_id': int(label.item()),
                'bbox':        [x1, y1, bw, bh],
                'area':        area,
                'iscrowd':     0,
            })
            ann_id += 1
    return coco_gt


@torch.no_grad()
def run_inference(models, dataset, device):
    """Run ensemble SAHI inference, return COCO-format predictions."""
    predictions = []
    for idx in tqdm(range(len(dataset)), desc='Inference'):
        image, target = dataset[idx]
        img_id = int(target['image_id'].item())

        all_b, all_s, all_l = [], [], []
        for model in models:
            b, s, l = collect_all_preds(model, image, device)
            if b.shape[0] > 0:
                all_b.append(b); all_s.append(s); all_l.append(l)

        if all_b:
            boxes, scores, labels = per_class_nms(
                torch.cat(all_b), torch.cat(all_s), torch.cat(all_l)
            )
            for box, score, label in zip(boxes, scores, labels):
                x1,y1,x2,y2 = box.tolist()
                predictions.append({
                    'image_id':    img_id,
                    'category_id': int(label.item()),
                    'bbox':        [x1, y1, x2-x1, y2-y1],
                    'score':       float(score.item()),
                })
    return predictions


# ── COCO evaluation with custom area ranges ────────────────────────────

def run_coco_eval(coco_gt_dict, predictions, label):
    """Run COCOeval with standard + custom (vt/t/s) area ranges."""
    coco_gt   = COCO()
    coco_gt.dataset = coco_gt_dict
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(predictions)
    ev = COCOeval(coco_gt, coco_dt, 'bbox')

    # Standard + custom area ranges
    ev.params.areaRng    = [
        [0,    1e10],   # all
        [0,    100],    # very tiny (vt)
        [100,  400],    # tiny (t)
        [400,  1024],   # small (s)
        [1024, 9216],   # medium (m)
    ]
    ev.params.areaRngLbl = ['all', 'vt', 't', 's', 'm']
    ev.params.maxDets    = [1, 10, 500]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    # Extract key metrics (iouThr=None → 0.50:0.95, iouThr=0.50, iouThr=0.75)
    def get_ap(iou_thr, area_lbl, max_det=500):
        p = ev.params
        iou_idx  = (0 if iou_thr is None
                    else [round(x,2) for x in p.iouThrs].index(round(iou_thr,2)))
        area_idx = p.areaRngLbl.index(area_lbl)
        det_idx  = p.maxDets.index(max_det)
        s = ev.eval['precision']
        if iou_thr is None:
            v = s[:, :, :, area_idx, det_idx]
        else:
            v = s[iou_idx, :, :, area_idx, det_idx]
        valid = v[v > -1]
        return float(valid.mean()) if valid.size else -1.0

    # Per-class AP@0.50:0.95
    def get_class_ap(cat_idx):
        s = ev.eval['precision'][:, :, cat_idx, 0, 2]  # all area, max_det=500
        valid = s[s > -1]
        return float(valid.mean()) if valid.size else -1.0

    results = {
        'label':    label,
        'AP':       get_ap(None,  'all') * 100,
        'AP50':     get_ap(0.50,  'all') * 100,
        'AP75':     get_ap(0.75,  'all') * 100,
        'APvt':     get_ap(None,  'vt')  * 100,
        'APt':      get_ap(None,  't')   * 100,
        'APs':      get_ap(None,  's')   * 100,
        'APm':      get_ap(None,  'm')   * 100,
    }

    # Per-class
    per_class = {}
    cat_ids = ev.params.catIds
    for i, cid in enumerate(cat_ids):
        name = VISDRONE_CLASS_NAMES[cid-1] if 1<=cid<=10 else f'cls_{cid}'
        per_class[name] = get_class_ap(i) * 100
    results['per_class_AP'] = per_class

    return results


def print_results(r):
    print(f'\n{"="*65}')
    print(f'RESULTS — {r["label"]}')
    print(f'{"="*65}')
    print(f'  AP@0.50:0.95  : {r["AP"]:6.2f}%')
    print(f'  AP@0.50       : {r["AP50"]:6.2f}%')
    print(f'  AP@0.75       : {r["AP75"]:6.2f}%')
    print(f'  APvt (<10×10) : {r["APvt"]:6.2f}%')
    print(f'  APt  (10-20px): {r["APt"]:6.2f}%')
    print(f'  APs  (20-32px): {r["APs"]:6.2f}%')
    print(f'  APm  (32-96px): {r["APm"]:6.2f}%')
    print(f'\n  Per-class AP@0.50:0.95:')
    for name, ap in r['per_class_AP'].items():
        print(f'    {name:20s}: {ap:6.2f}%')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    baseline_ckpt = PROJECT_ROOT / 'results' / 'outputs'     / 'best_model.pth'
    v3_ckpt       = PROJECT_ROOT / 'results' / 'outputs_v3_full_framework' / 'best_model_v3.pth'
    data_root     = str(PROJECT_ROOT.parent / 'dataset' / 'VisDrone-2018')

    print('\nLoading models...')
    m_base = load_baseline(baseline_ckpt, device)
    m_v3   = load_v3(v3_ckpt, device)
    print('  Loaded baseline + V3')

    print('\nLoading dataset...')
    dataset = VisDroneDataset(root_dir=data_root, split='val', transforms=None)
    print(f'  {len(dataset)} images')

    print('\nBuilding COCO ground truth...')
    coco_gt_dict = build_coco_gt(dataset)
    print(f'  {len(coco_gt_dict["annotations"])} GT boxes across {len(coco_gt_dict["images"])} images')

    all_results = []

    # V3 alone
    print('\n--- V3 alone ---')
    preds = run_inference([m_v3], dataset, device)
    r = run_coco_eval(coco_gt_dict, preds, 'V3 alone')
    print_results(r); all_results.append(r)

    # Baseline + V3 ensemble
    print('\n--- Baseline + V3 ensemble ---')
    preds = run_inference([m_base, m_v3], dataset, device)
    r = run_coco_eval(coco_gt_dict, preds, 'Baseline + V3 ensemble')
    print_results(r); all_results.append(r)

    out_path = PROJECT_ROOT / 'results' / 'eval_full_v3.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
