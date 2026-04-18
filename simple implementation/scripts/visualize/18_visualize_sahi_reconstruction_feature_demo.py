"""SAHI-style square-region reconstruction + feature highlight comparison.

This script creates paper-ready visuals that focus on a far tiny-object region:
1. Select tiny-object-heavy images.
2. Pick a tiny GT object farthest from image center, preferring cases baseline misses and CD-DPA hits.
3. Build a square SAHI-like focus tile around that object.
4. Degrade and reconstruct the tile.
5. Compute difference map + feature-enhancement map.
6. Compare baseline vs CD-DPA attention/boxes on full image and in the focus region.
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.cddpa_model import FasterRCNN_CDDPA


def load_visdrone_dataset_class():
    ds_path = PROJECT_ROOT / "scripts" / "4_visdrone_dataset.py"
    spec = importlib.util.spec_from_file_location("visdrone_dataset", ds_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VisDroneDataset


def parse_args():
    parser = argparse.ArgumentParser(description="SAHI square-region reconstruction feature demo")
    parser.add_argument("--data-root", type=str, default=str(PROJECT_ROOT.parent / "dataset" / "VisDrone-2018"))
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--baseline-checkpoint", type=str, default=str(PROJECT_ROOT / "results" / "outputs" / "best_model.pth"))
    parser.add_argument("--cddpa-checkpoint", type=str, default=str(PROJECT_ROOT / "results" / "outputs_cddpa" / "best_model_cddpa.pth"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--min-size", type=int, default=5)
    parser.add_argument("--tiny-max-side", type=int, default=24)
    parser.add_argument("--tiny-max-area", type=int, default=576)
    parser.add_argument("--tile-size", type=int, default=256, help="Square SAHI focus tile size")
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "visualizations_sahi_reconstruction_feature"),
    )
    return parser.parse_args()


def load_baseline_model(num_obj_classes: int, checkpoint_path: Path, device: torch.device):
    model = fasterrcnn_resnet50_fpn(weights=None, trainable_backbone_layers=5)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_obj_classes + 1)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_cddpa_model(num_classes_with_bg: int, checkpoint_path: Path, device: torch.device):
    model = FasterRCNN_CDDPA(
        num_classes=num_classes_with_bg,
        enhance_levels=["0", "1", "2"],
        use_checkpoint=False,
        pretrained=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def tensor_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    arr = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def normalize_map(attn: torch.Tensor) -> torch.Tensor:
    a_min = attn.amin(dim=(-2, -1), keepdim=True)
    a_max = attn.amax(dim=(-2, -1), keepdim=True)
    return (attn - a_min) / (a_max - a_min + 1e-8)


def compute_baseline_attention_map(model, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        img_list, _ = model.transform([image.to(device)], None)
        feats = model.backbone(img_list.tensors)

    level_weights = {"0": 0.5, "1": 0.3, "2": 0.2}
    maps = []
    for lvl, w in level_weights.items():
        if lvl in feats:
            m = normalize_map(feats[lvl].abs().mean(dim=1, keepdim=True))
            maps.append((w, m))

    h0, w0 = image.shape[-2:]
    if not maps:
        return torch.zeros((1, 1, h0, w0), device=device)

    acc = torch.zeros((1, 1, h0, w0), device=device)
    wsum = 0.0
    for w, m in maps:
        up = F.interpolate(m, size=(h0, w0), mode="bilinear", align_corners=False)
        acc += w * up
        wsum += w
    return normalize_map(acc / max(wsum, 1e-8))


def _dpa_stage_with_map(dpa_module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    identity = x

    offset = dpa_module.offset_conv(x)
    deform_feat = dpa_module.deform_conv(x, offset)

    edge3 = dpa_module.edge_conv3(deform_feat)
    edge5 = dpa_module.edge_conv5(deform_feat)
    edge_features = edge3 + edge5
    spatial_weight = dpa_module.spatial_att(edge_features)
    edge_enh = edge_features * spatial_weight

    channel_weight = dpa_module.channel_att(deform_feat)
    sem_enh = deform_feat * channel_weight

    combined = torch.cat([edge_enh, sem_enh], dim=1)
    out = dpa_module.fusion(combined) + identity

    c_scalar = channel_weight.mean(dim=1, keepdim=True)
    attn = normalize_map(0.7 * spatial_weight + 0.3 * c_scalar)
    return out, attn


def _cddpa_block_with_map(enhancer, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    f1, a1 = _dpa_stage_with_map(enhancer.stage1, x)
    f2, a2 = _dpa_stage_with_map(enhancer.stage2, f1)
    fused = torch.cat([f1, f2], dim=1)
    out = enhancer.relu(enhancer.fusion(fused) + x)
    feat_energy = normalize_map(out.abs().mean(dim=1, keepdim=True))
    attn = normalize_map(0.6 * ((a1 + a2) * 0.5) + 0.4 * feat_energy)
    return out, attn


def compute_cddpa_attention_map(model: FasterRCNN_CDDPA, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        img_list, _ = model.base_model.transform([image.to(device)], None)
        feats = model.base_model.backbone(img_list.tensors)

    level_weights = {"0": 0.5, "1": 0.3, "2": 0.2}
    maps = []

    with torch.no_grad():
        for lvl in model.enhance_levels:
            if lvl in feats:
                enhanced, attn = _cddpa_block_with_map(model.enhancers[lvl], feats[lvl])
                feats[lvl] = enhanced
                maps.append((level_weights.get(lvl, 0.2), attn))

    h0, w0 = image.shape[-2:]
    if not maps:
        return torch.zeros((1, 1, h0, w0), device=device)

    acc = torch.zeros((1, 1, h0, w0), device=device)
    wsum = 0.0
    for w, m in maps:
        up = F.interpolate(m, size=(h0, w0), mode="bilinear", align_corners=False)
        acc += w * up
        wsum += w
    return normalize_map(acc / max(wsum, 1e-8))


def filter_predictions(pred: Dict[str, torch.Tensor], score_thresh: float) -> Dict[str, np.ndarray]:
    scores = pred["scores"].detach().cpu().numpy()
    keep = scores >= score_thresh
    return {
        "boxes": pred["boxes"].detach().cpu().numpy()[keep],
        "labels": pred["labels"].detach().cpu().numpy()[keep],
        "scores": scores[keep],
    }


def draw_square(image_rgb: np.ndarray, square: Tuple[int, int, int, int], color=(255, 255, 255), width: int = 3) -> np.ndarray:
    x1, y1, x2, y2 = square
    img = Image.fromarray(image_rgb.copy())
    d = ImageDraw.Draw(img)
    d.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return np.array(img)


def _class_palette() -> List[Tuple[int, int, int]]:
    return [
        (255, 64, 64),
        (64, 255, 64),
        (64, 64, 255),
        (255, 255, 64),
        (255, 64, 255),
        (64, 255, 255),
        (255, 160, 64),
        (160, 64, 255),
        (64, 160, 255),
        (255, 180, 180),
    ]


def _color_for_label(label: int) -> Tuple[int, int, int]:
    palette = _class_palette()
    if label <= 0:
        return (220, 220, 220)
    return palette[(label - 1) % len(palette)]


def draw_gt_boxes(image_rgb: np.ndarray, boxes: np.ndarray, labels: np.ndarray, width: int = 2) -> np.ndarray:
    img = Image.fromarray(image_rgb.copy())
    d = ImageDraw.Draw(img)
    for b, lbl in zip(boxes, labels):
        x1, y1, x2, y2 = [int(round(v)) for v in b.tolist()]
        d.rectangle([x1, y1, x2, y2], outline=_color_for_label(int(lbl)), width=width)
    return np.array(img)


def crop_with_gt_boxes(
    image_rgb: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    square: Tuple[int, int, int, int],
    width: int = 2,
) -> np.ndarray:
    sx1, sy1, sx2, sy2 = square
    tile = image_rgb[sy1:sy2, sx1:sx2].copy()

    img = Image.fromarray(tile)
    d = ImageDraw.Draw(img)
    for b, lbl in zip(boxes, labels):
        x1, y1, x2, y2 = b.tolist()

        # Intersect GT box with SAHI square in global coordinates.
        ix1 = max(float(sx1), float(x1))
        iy1 = max(float(sy1), float(y1))
        ix2 = min(float(sx2), float(x2))
        iy2 = min(float(sy2), float(y2))
        if ix2 <= ix1 or iy2 <= iy1:
            continue

        # Convert to local tile coordinates.
        lx1 = int(round(ix1 - sx1))
        ly1 = int(round(iy1 - sy1))
        lx2 = int(round(ix2 - sx1))
        ly2 = int(round(iy2 - sy1))
        d.rectangle([lx1, ly1, lx2, ly2], outline=_color_for_label(int(lbl)), width=width)

    return np.array(img)


def apply_heat_overlay(image_rgb: np.ndarray, attn_01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = matplotlib.colormaps["jet"](np.clip(attn_01, 0.0, 1.0))[..., :3]
    heat = (heat * 255.0).astype(np.uint8)
    out = image_rgb.astype(np.float32) * (1.0 - alpha) + heat.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(0.0, (box_a[2] - box_a[0])) * max(0.0, (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0])) * max(0.0, (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def matches_any(gt_box: np.ndarray, pred_boxes: np.ndarray, thr: float = 0.2) -> bool:
    if pred_boxes.size == 0:
        return False
    return any(compute_iou(gt_box, pb) >= thr for pb in pred_boxes)


def tiny_gt_boxes(target: Dict[str, torch.Tensor], tiny_max_side: int, tiny_max_area: int) -> np.ndarray:
    boxes = target["boxes"].detach().cpu().numpy()
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    wh = boxes[:, 2:4] - boxes[:, 0:2]
    max_side = np.maximum(wh[:, 0], wh[:, 1])
    area = wh[:, 0] * wh[:, 1]
    keep = (max_side <= tiny_max_side) & (area <= tiny_max_area)
    return boxes[keep]


def pick_focus_tiny_box(
    image_hw: Tuple[int, int],
    tiny_boxes: np.ndarray,
    base_pred_boxes: np.ndarray,
    ours_pred_boxes: np.ndarray,
) -> Optional[np.ndarray]:
    if tiny_boxes.shape[0] == 0:
        return None

    h, w = image_hw
    cx_img = w * 0.5
    cy_img = h * 0.5

    scored = []
    for b in tiny_boxes:
        cx = 0.5 * (b[0] + b[2])
        cy = 0.5 * (b[1] + b[3])
        dist = math.sqrt((cx - cx_img) ** 2 + (cy - cy_img) ** 2)
        base_hit = matches_any(b, base_pred_boxes, thr=0.2)
        ours_hit = matches_any(b, ours_pred_boxes, thr=0.2)

        # Prefer baseline miss + ours hit, then farthest.
        priority = 2 if (not base_hit and ours_hit) else (1 if ours_hit else 0)
        scored.append((priority, dist, b))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


def build_square_around_box(box: np.ndarray, image_hw: Tuple[int, int], tile_size: int) -> Tuple[int, int, int, int]:
    h, w = image_hw
    cx = int(round((box[0] + box[2]) * 0.5))
    cy = int(round((box[1] + box[3]) * 0.5))

    half = tile_size // 2
    x1 = cx - half
    y1 = cy - half
    x2 = x1 + tile_size
    y2 = y1 + tile_size

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        shift = x2 - w
        x1 -= shift
        x2 = w
    if y2 > h:
        shift = y2 - h
        y1 -= shift
        y2 = h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # Enforce square final size when possible.
    side = min(x2 - x1, y2 - y1)
    x2 = x1 + side
    y2 = y1 + side
    return int(x1), int(y1), int(x2), int(y2)


def degrade_tile(tile_rgb: np.ndarray, downscale: int) -> np.ndarray:
    h, w = tile_rgb.shape[:2]
    sw = max(1, w // downscale)
    sh = max(1, h // downscale)
    low = cv2.resize(tile_rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    low = cv2.GaussianBlur(low, (3, 3), sigmaX=0.8)
    noise = np.random.normal(0.0, 4.0, size=low.shape).astype(np.float32)
    low = np.clip(low.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return cv2.resize(low, (w, h), interpolation=cv2.INTER_NEAREST)


def reconstruct_tile(tile_degraded: np.ndarray) -> np.ndarray:
    base = cv2.resize(tile_degraded, (tile_degraded.shape[1], tile_degraded.shape[0]), interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(base, (0, 0), sigmaX=1.0)
    recon = cv2.addWeighted(base, 1.5, blur, -0.5, 0)
    return np.clip(recon, 0, 255).astype(np.uint8)


def difference_map(original_tile: np.ndarray, recon_tile: np.ndarray) -> np.ndarray:
    diff = np.mean(np.abs(original_tile.astype(np.float32) - recon_tile.astype(np.float32)), axis=2)
    diff = diff - diff.min()
    diff = diff / (diff.max() + 1e-8)
    return diff


def feature_enhancement_map(recon_tile: np.ndarray, diff_map_01: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(recon_tile, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # Edge/detail response (high-frequency) from Sobel + Laplacian.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    detail = 0.7 * grad + 0.3 * lap

    # Local contrast term to produce broader coherent enhancement regions.
    local_mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    local_contrast = np.abs(gray - local_mean)

    # Keep it very close to difference-map style, with a slight structural boost.
    feat = 0.75 * diff_map_01 + 0.20 * detail + 0.05 * local_contrast

    # Mild smoothing and slightly deeper contrast than raw difference map.
    feat = cv2.GaussianBlur(feat, (0, 0), sigmaX=0.9)
    feat = np.clip(feat, 0.0, None)
    p05 = np.percentile(feat, 5)
    p95 = np.percentile(feat, 95)
    feat = (feat - p05) / (p95 - p05 + 1e-8)
    feat = np.clip(feat, 0.0, 1.0)
    feat = np.clip(np.power(feat, 0.88), 0.0, 1.0)
    return feat


def select_images(dataset, num_samples: int, tiny_max_side: int, tiny_max_area: int) -> List[int]:
    stats = []
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        tiny = tiny_gt_boxes(target, tiny_max_side, tiny_max_area)
        stats.append((idx, tiny.shape[0], target["boxes"].shape[0]))

    stats.sort(key=lambda x: (x[1], x[2]), reverse=True)
    selected = [s[0] for s in stats if s[1] > 0][:num_samples]
    return selected


def crop_local(arr: np.ndarray, square: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = square
    return arr[y1:y2, x1:x2]


def build_panel(
    image_rgb: np.ndarray,
    base_full: np.ndarray,
    ours_full: np.ndarray,
    base_attn_full: np.ndarray,
    ours_attn_full: np.ndarray,
    tile_original: np.ndarray,
    tile_degraded: np.ndarray,
    tile_recon: np.ndarray,
    diff_map_01: np.ndarray,
    feat_map_01: np.ndarray,
    out_path: Path,
):
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(base_full)
    axes[0, 1].set_title("Baseline view + SAHI square")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(ours_full)
    axes[0, 2].set_title("CD-DPA view + SAHI square")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(base_attn_full)
    axes[0, 3].set_title("Baseline attention")
    axes[0, 3].axis("off")

    axes[0, 4].imshow(ours_attn_full)
    axes[0, 4].set_title("CD-DPA attention")
    axes[0, 4].axis("off")

    axes[1, 0].imshow(tile_original)
    axes[1, 0].set_title("SAHI square (original)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(tile_degraded)
    axes[1, 1].set_title("Degraded")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(tile_recon)
    axes[1, 2].set_title("Reconstructed")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(diff_map_01, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1, 3].set_title("Difference map")
    axes[1, 3].axis("off")

    axes[1, 4].imshow(feat_map_01, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1, 4].set_title("Feature enhancement")
    axes[1, 4].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_final_detection_demo(
    full_with_gt: np.ndarray,
    sahi_tile: np.ndarray,
    out_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(full_with_gt)
    axes[0].set_title("Final demo: GT boxes")
    axes[0].axis("off")

    axes[1].imshow(sahi_tile)
    axes[1].set_title("SAHI square crop + GT boxes")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    VisDroneDataset = load_visdrone_dataset_class()
    dataset = VisDroneDataset(root_dir=args.data_root, split=args.split, min_size=args.min_size)

    baseline_ckpt = Path(args.baseline_checkpoint)
    cddpa_ckpt = Path(args.cddpa_checkpoint)
    if not baseline_ckpt.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_ckpt}")
    if not cddpa_ckpt.exists():
        raise FileNotFoundError(f"CD-DPA checkpoint not found: {cddpa_ckpt}")

    print("=" * 90)
    print("SAHI SQUARE RECONSTRUCTION + FEATURE HIGHLIGHT DEMO")
    print("=" * 90)
    print(f"Data root: {args.data_root}")
    print(f"Split: {args.split}")
    print(f"Output dir: {out_dir}")
    print(f"Device: {device}")

    baseline_model = load_baseline_model(num_obj_classes=10, checkpoint_path=baseline_ckpt, device=device)
    cddpa_model = load_cddpa_model(num_classes_with_bg=11, checkpoint_path=cddpa_ckpt, device=device)

    selected = select_images(dataset, args.num_samples, args.tiny_max_side, args.tiny_max_area)
    print(f"Selected {len(selected)} tiny-object-heavy images")

    summary = []

    for rank, ds_idx in enumerate(selected, start=1):
        image_t, target = dataset[ds_idx]
        image_t = image_t.to(device)
        h, w = image_t.shape[-2:]

        with torch.no_grad():
            pred_base = baseline_model([image_t])[0]
            pred_ours = cddpa_model([image_t])[0]

        base_det = filter_predictions(pred_base, args.score_thresh)
        ours_det = filter_predictions(pred_ours, args.score_thresh)

        tiny_boxes = tiny_gt_boxes(target, args.tiny_max_side, args.tiny_max_area)
        focus_box = pick_focus_tiny_box((h, w), tiny_boxes, base_det["boxes"], ours_det["boxes"])
        if focus_box is None:
            print(f"[{rank}/{len(selected)}] skip idx={ds_idx}: no tiny GT candidate")
            continue

        square = build_square_around_box(focus_box, (h, w), args.tile_size)

        attn_base = compute_baseline_attention_map(baseline_model, image_t, device).squeeze().detach().cpu().numpy()
        attn_ours = compute_cddpa_attention_map(cddpa_model, image_t, device).squeeze().detach().cpu().numpy()

        image_rgb = tensor_to_uint8(image_t.detach().cpu())
        gt_boxes_all = target["boxes"].detach().cpu().numpy()
        gt_labels_all = target["labels"].detach().cpu().numpy()

        base_full = draw_square(image_rgb, square)
        ours_full = draw_square(image_rgb, square)

        base_attn_full = apply_heat_overlay(draw_square(image_rgb, square), attn_base, alpha=0.45)
        ours_attn_full = apply_heat_overlay(draw_square(image_rgb, square), attn_ours, alpha=0.45)

        tile_original = crop_local(image_rgb, square)

        tile_degraded = degrade_tile(tile_original, args.downscale)
        tile_recon = reconstruct_tile(tile_degraded)

        diff_01 = difference_map(tile_original, tile_recon)
        feat_01 = feature_enhancement_map(tile_recon, diff_01)

        stem = f"sample_{rank:03d}_idx_{ds_idx:05d}"
        panel_path = out_dir / f"{stem}_sahi_recon_feature_panel.png"
        build_panel(
            image_rgb=image_rgb,
            base_full=base_full,
            ours_full=ours_full,
            base_attn_full=base_attn_full,
            ours_attn_full=ours_attn_full,
            tile_original=tile_original,
            tile_degraded=tile_degraded,
            tile_recon=tile_recon,
            diff_map_01=diff_01,
            feat_map_01=feat_01,
            out_path=panel_path,
        )

        final_demo_path = out_dir / f"{stem}_final_detection_demo.png"
        full_with_gt = draw_gt_boxes(image_rgb, gt_boxes_all, gt_labels_all)
        sahi_tile = crop_with_gt_boxes(image_rgb, gt_boxes_all, gt_labels_all, square)
        build_final_detection_demo(
            full_with_gt=full_with_gt,
            sahi_tile=sahi_tile,
            out_path=final_demo_path,
        )

        base_hit = matches_any(focus_box, base_det["boxes"], thr=0.2)
        ours_hit = matches_any(focus_box, ours_det["boxes"], thr=0.2)

        print(
            f"[{rank}/{len(selected)}] {panel_path.name} | "
            f"focus_box={focus_box.round(1).tolist()} base_hit={base_hit} ours_hit={ours_hit}"
        )

        summary.append(
            {
                "sample_rank": rank,
                "dataset_index": int(ds_idx),
                "panel": panel_path.name,
                "final_demo": final_demo_path.name,
                "focus_box": [float(x) for x in focus_box.tolist()],
                "baseline_hit": bool(base_hit),
                "ours_hit": bool(ours_hit),
                "num_baseline_det": int(len(base_det["boxes"])),
                "num_ours_det": int(len(ours_det["boxes"])),
            }
        )

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("-" * 90)
    print(f"Saved panels and summary to: {out_dir}")


if __name__ == "__main__":
    main()
