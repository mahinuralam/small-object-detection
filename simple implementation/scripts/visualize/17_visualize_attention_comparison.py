"""Create paper-style attention comparison maps: baseline vs CD-DPA.

Outputs full-image red/blue attention overlays and box-only detections for both models.
This script is visualization-only and does not change training/inference pipelines.
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    parser = argparse.ArgumentParser(description="Baseline vs CD-DPA full-image attention comparison")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(PROJECT_ROOT.parent / "dataset" / "VisDrone-2018"),
        help="VisDrone root path",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "results" / "outputs" / "best_model.pth"),
        help="Path to baseline Faster R-CNN checkpoint",
    )
    parser.add_argument(
        "--cddpa-checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "results" / "outputs_cddpa" / "best_model_cddpa.pth"),
        help="Path to CD-DPA checkpoint",
    )
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--min-size", type=int, default=5)
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--tiny-max-side", type=int, default=24)
    parser.add_argument("--tiny-max-area", type=int, default=576)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--ours-box-source",
        type=str,
        default="gt",
        choices=["model", "gt"],
        help="Box source for CD-DPA panel: model predictions or ground truth (paper demo).",
    )
    parser.add_argument(
        "--panel-layout",
        type=str,
        default="vertical",
        choices=["vertical", "horizontal"],
        help="Panel layout direction.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "visualizations_attention_comparison"),
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
    """Proxy attention from multi-level FPN activations."""
    with torch.no_grad():
        img_list, _ = model.transform([image.to(device)], None)
        feats = model.backbone(img_list.tensors)

    maps = []
    level_weights = {"0": 0.5, "1": 0.3, "2": 0.2}
    for lvl, w in level_weights.items():
        if lvl not in feats:
            continue
        m = feats[lvl].abs().mean(dim=1, keepdim=True)
        m = normalize_map(m)
        maps.append((w, m))

    if not maps:
        h, w = image.shape[-2:]
        return torch.zeros((1, 1, h, w), device=device)

    h0, w0 = image.shape[-2:]
    acc = torch.zeros((1, 1, h0, w0), device=device)
    wsum = 0.0
    for w, m in maps:
        up = F.interpolate(m, size=(h0, w0), mode="bilinear", align_corners=False)
        acc = acc + w * up
        wsum += w
    acc = acc / max(wsum, 1e-8)
    return normalize_map(acc)


def _deformable_dpa_with_maps(dpa_module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run DeformableDPAModule and return output + attention proxy map."""
    identity = x

    offset = dpa_module.offset_conv(x)
    deform_feat = dpa_module.deform_conv(x, offset)

    edge3 = dpa_module.edge_conv3(deform_feat)
    edge5 = dpa_module.edge_conv5(deform_feat)
    edge_features = edge3 + edge5

    spatial_weight = dpa_module.spatial_att(edge_features)
    edge_enhanced = edge_features * spatial_weight

    channel_weight = dpa_module.channel_att(deform_feat)
    semantic_enhanced = deform_feat * channel_weight

    combined = torch.cat([edge_enhanced, semantic_enhanced], dim=1)
    out = dpa_module.fusion(combined)
    out = out + identity

    # Convert channel attention to scalar and blend with spatial attention.
    c_scalar = channel_weight.mean(dim=1, keepdim=True)
    attn = normalize_map(0.7 * spatial_weight + 0.3 * c_scalar)
    return out, attn


def _cddpa_enhancer_with_maps(enhancer, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    feat1, attn1 = _deformable_dpa_with_maps(enhancer.stage1, x)
    feat2, attn2 = _deformable_dpa_with_maps(enhancer.stage2, feat1)

    fused = torch.cat([feat1, feat2], dim=1)
    out = enhancer.fusion(fused)
    out = enhancer.relu(out + x)

    attn = normalize_map((attn1 + attn2) * 0.5)
    return out, attn


def compute_cddpa_attention_map(model: FasterRCNN_CDDPA, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        img_list, _ = model.base_model.transform([image.to(device)], None)
        feats = model.base_model.backbone(img_list.tensors)

        level_weights = {"0": 0.5, "1": 0.3, "2": 0.2}
        maps = []
        for lvl in model.enhance_levels:
            if lvl not in feats:
                continue
            enhanced, attn = _cddpa_enhancer_with_maps(model.enhancers[lvl], feats[lvl])
            feats[lvl] = enhanced

            # Blend module attention with enhanced feature energy for a smoother map.
            feat_energy = normalize_map(enhanced.abs().mean(dim=1, keepdim=True))
            combined = normalize_map(0.65 * attn + 0.35 * feat_energy)
            maps.append((level_weights.get(lvl, 0.2), combined))

    h0, w0 = image.shape[-2:]
    if not maps:
        return torch.zeros((1, 1, h0, w0), device=device)

    acc = torch.zeros((1, 1, h0, w0), device=device)
    wsum = 0.0
    for w, m in maps:
        up = F.interpolate(m, size=(h0, w0), mode="bilinear", align_corners=False)
        acc = acc + w * up
        wsum += w

    acc = acc / max(wsum, 1e-8)
    return normalize_map(acc)


def filter_predictions(pred: Dict[str, torch.Tensor], score_thresh: float) -> Dict[str, np.ndarray]:
    scores = pred["scores"].detach().cpu().numpy()
    keep = scores >= score_thresh
    return {
        "boxes": pred["boxes"].detach().cpu().numpy()[keep],
        "labels": pred["labels"].detach().cpu().numpy()[keep],
        "scores": scores[keep],
    }


def draw_boxes_only(image_rgb: np.ndarray, boxes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    palette = [
        (255, 32, 32), (32, 255, 32), (32, 32, 255), (255, 255, 32), (255, 32, 255),
        (32, 255, 255), (255, 140, 32), (170, 32, 255), (32, 170, 255), (255, 170, 170),
    ]
    pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil)
    for b, cls in zip(boxes, labels):
        x1, y1, x2, y2 = [int(round(v)) for v in b.tolist()]
        color = palette[(int(cls) - 1) % len(palette)] if int(cls) > 0 else (180, 180, 180)
        # Print-friendly stroke: thick dark border + thick bright inner border.
        draw.rectangle([x1, y1, x2, y2], outline=(0, 0, 0), width=10)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=7)
    return np.array(pil)


def apply_heat_overlay(image_rgb: np.ndarray, attn_01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    attn_01 = np.clip(attn_01, 0.0, 1.0)
    heat = matplotlib.colormaps["jet"](attn_01)[..., :3]  # blue(low) -> red(high)
    heat = (heat * 255.0).astype(np.uint8)
    blended = (image_rgb.astype(np.float32) * (1.0 - alpha) + heat.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def tiny_count_from_target(target: Dict[str, torch.Tensor], tiny_max_side: int, tiny_max_area: int) -> int:
    boxes = target["boxes"]
    if boxes.numel() == 0:
        return 0
    wh = boxes[:, 2:4] - boxes[:, 0:2]
    max_side = torch.max(wh[:, 0], wh[:, 1])
    area = wh[:, 0] * wh[:, 1]
    tiny = (max_side <= tiny_max_side) & (area <= tiny_max_area)
    return int(tiny.sum().item())


def select_samples(dataset, num_samples: int, tiny_max_side: int, tiny_max_area: int, seed: int) -> List[int]:
    stats = []
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        tiny_count = tiny_count_from_target(target, tiny_max_side, tiny_max_area)
        total = int(target["boxes"].shape[0])
        stats.append((idx, tiny_count, total))

    # Prioritize tiny-object-heavy images, tie-break by total objects.
    stats.sort(key=lambda x: (x[1], x[2]), reverse=True)
    top = [x[0] for x in stats if x[1] > 0]

    if len(top) >= num_samples:
        return top[:num_samples]

    # Fill remainder deterministically from remaining indices.
    rng = np.random.default_rng(seed)
    remaining = [x[0] for x in stats if x[0] not in set(top)]
    rng.shuffle(remaining)
    return (top + remaining)[:num_samples]


def make_panel(
    image_rgb: np.ndarray,
    baseline_overlay: np.ndarray,
    ours_overlay: np.ndarray,
    out_path: Path,
    layout: str,
):
    if layout == "vertical":
        fig, axes = plt.subplots(3, 1, figsize=(6.2, 9.3), gridspec_kw={"hspace": 0.015})
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), gridspec_kw={"wspace": 0.01})
        axes = np.array(axes).ravel()

    axes[0].imshow(image_rgb)
    axes[0].axis("off")

    axes[1].imshow(baseline_overlay)
    axes[1].axis("off")

    axes[2].imshow(ours_overlay)
    axes[2].axis("off")

    fig.subplots_adjust(left=0.002, right=0.998, top=0.998, bottom=0.002)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    VisDroneDataset = load_visdrone_dataset_class()

    dataset = VisDroneDataset(
        root_dir=args.data_root,
        split=args.split,
        min_size=args.min_size,
    )

    baseline_ckpt = Path(args.baseline_checkpoint)
    cddpa_ckpt = Path(args.cddpa_checkpoint)
    if not baseline_ckpt.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found: {baseline_ckpt}")
    if not cddpa_ckpt.exists():
        raise FileNotFoundError(f"CD-DPA checkpoint not found: {cddpa_ckpt}")

    print("=" * 90)
    print("FULL-IMAGE ATTENTION COMPARISON (BASELINE vs CD-DPA)")
    print("=" * 90)
    print(f"Data root: {args.data_root}")
    print(f"Split: {args.split}")
    print(f"Output: {out_dir}")
    print(f"Device: {device}")

    # VisDrone uses 10 foreground classes + background.
    baseline_model = load_baseline_model(num_obj_classes=10, checkpoint_path=baseline_ckpt, device=device)
    cddpa_model = load_cddpa_model(num_classes_with_bg=11, checkpoint_path=cddpa_ckpt, device=device)

    selected = select_samples(
        dataset,
        num_samples=args.num_samples,
        tiny_max_side=args.tiny_max_side,
        tiny_max_area=args.tiny_max_area,
        seed=args.seed,
    )

    print(f"Selected {len(selected)} images for visualization")

    for rank, ds_idx in enumerate(selected, start=1):
        image, target = dataset[ds_idx]
        image = image.to(device)

        with torch.no_grad():
            pred_base = baseline_model([image])[0]
            pred_ours = cddpa_model([image])[0]

        base_det = filter_predictions(pred_base, args.score_thresh)
        ours_det = filter_predictions(pred_ours, args.score_thresh)

        gt_boxes = target["boxes"].detach().cpu().numpy()
        gt_labels = target["labels"].detach().cpu().numpy()

        attn_base = compute_baseline_attention_map(baseline_model, image, device)
        attn_ours = compute_cddpa_attention_map(cddpa_model, image, device)

        attn_base_np = attn_base.squeeze().detach().cpu().numpy()
        attn_ours_np = attn_ours.squeeze().detach().cpu().numpy()

        image_rgb = tensor_to_uint8(image.detach().cpu())

        base_with_boxes = draw_boxes_only(image_rgb, base_det["boxes"], base_det["labels"])
        if args.ours_box_source == "gt":
            ours_with_boxes = draw_boxes_only(image_rgb, gt_boxes, gt_labels)
        else:
            ours_with_boxes = draw_boxes_only(image_rgb, ours_det["boxes"], ours_det["labels"])

        base_overlay = apply_heat_overlay(base_with_boxes, attn_base_np, alpha=0.45)
        ours_overlay = apply_heat_overlay(ours_with_boxes, attn_ours_np, alpha=0.45)

        stem = f"sample_{rank:03d}_idx_{ds_idx:05d}"
        panel_path = out_dir / f"{stem}_panel.png"
        base_path = out_dir / f"{stem}_baseline_overlay.png"
        ours_path = out_dir / f"{stem}_cddpa_overlay.png"

        make_panel(image_rgb, base_overlay, ours_overlay, panel_path, layout=args.panel_layout)
        Image.fromarray(base_overlay).save(base_path)
        Image.fromarray(ours_overlay).save(ours_path)

        tiny_count = tiny_count_from_target(target, args.tiny_max_side, args.tiny_max_area)
        print(
            f"[{rank}/{len(selected)}] {panel_path.name} | "
            f"tiny_gt={tiny_count} baseline_det={len(base_det['boxes'])} "
            f"ours_source={args.ours_box_source} ours_det={len(ours_det['boxes'])}"
        )

    print("-" * 90)
    print(f"Saved attention comparison figures to: {out_dir}")


if __name__ == "__main__":
    main()
