"""Tiny object reconstruction demo for paper visualization.

This script is intentionally separate from the actual SR-TOD implementation.
It demonstrates a simple tiny-object reconstruction pipeline on samples from
TRAIN split images that were used to train the reconstructor.

Pipeline (demo only):
1. Select tiny-object boxes from VisDrone train split
2. Crop object patch with context
3. Degrade patch (downsample + blur + noise)
4. Reconstruct patch (bicubic upsample + unsharp enhancement)
5. Visualize full image, crop, degraded, reconstructed, and diff heatmap
"""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_visdrone_dataset_class():
    dataset_path = PROJECT_ROOT / "scripts" / "4_visdrone_dataset.py"
    spec = importlib.util.spec_from_file_location("visdrone_dataset", dataset_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VisDroneDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny object reconstruction demo visualizer")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(PROJECT_ROOT.parent / "dataset" / "VisDrone-2018"),
        help="Path to VisDrone-2018 root",
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-size", type=int, default=3, help="Keep boxes >= this size in dataset loader")
    parser.add_argument("--tiny-max-side", type=int, default=24, help="Max tiny-object side length")
    parser.add_argument("--tiny-max-area", type=int, default=576, help="Max tiny-object area (24x24 default)")
    parser.add_argument("--context-pad", type=int, default=16, help="Context padding around tiny object")
    parser.add_argument("--downscale", type=int, default=4, help="Degradation downscale factor")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results" / "visualizations_reconstruction_demo"),
    )
    return parser.parse_args()


def tensor_to_uint8_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def clamp_box(box: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return x1, y1, x2, y2


def make_context_box(x1: int, y1: int, x2: int, y2: int, pad: int, w: int, h: int) -> Tuple[int, int, int, int]:
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(w, x2 + pad)
    cy2 = min(h, y2 + pad)
    return cx1, cy1, cx2, cy2


def degrade_patch(patch: np.ndarray, downscale: int) -> np.ndarray:
    ph, pw = patch.shape[:2]
    small_w = max(1, pw // downscale)
    small_h = max(1, ph // downscale)

    low = cv2.resize(patch, (small_w, small_h), interpolation=cv2.INTER_AREA)
    low = cv2.GaussianBlur(low, (3, 3), sigmaX=0.8)

    noise = np.random.normal(0, 4.0, size=low.shape).astype(np.float32)
    low_noisy = np.clip(low.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    degraded = cv2.resize(low_noisy, (pw, ph), interpolation=cv2.INTER_NEAREST)
    return degraded


def reconstruct_patch(degraded: np.ndarray, downscale: int) -> np.ndarray:
    ph, pw = degraded.shape[:2]
    # Simulate a lightweight reconstruction path: smooth upsample + sharpening.
    up = cv2.resize(
        degraded,
        (pw, ph),
        interpolation=cv2.INTER_CUBIC if downscale > 1 else cv2.INTER_LINEAR,
    )
    blur = cv2.GaussianBlur(up, (0, 0), sigmaX=1.0)
    recon = cv2.addWeighted(up, 1.5, blur, -0.5, 0)
    recon = np.clip(recon, 0, 255).astype(np.uint8)
    return recon


def abs_diff_heatmap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)), axis=2)
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    return diff


def collect_tiny_candidates(dataset, tiny_max_side: int, tiny_max_area: int) -> List[Dict]:
    candidates: List[Dict] = []
    for idx in range(len(dataset)):
        image, target = dataset[idx]
        boxes = target["boxes"].cpu().numpy()
        labels = target["labels"].cpu().numpy()

        if len(boxes) == 0:
            continue

        for bi, box in enumerate(boxes):
            w = float(box[2] - box[0])
            h = float(box[3] - box[1])
            area = w * h
            if max(w, h) <= tiny_max_side and area <= tiny_max_area:
                candidates.append(
                    {
                        "dataset_index": idx,
                        "box_index": bi,
                        "box": box.tolist(),
                        "label": int(labels[bi]),
                        "width": w,
                        "height": h,
                        "area": area,
                    }
                )
    return candidates


def visualize_sample(
    image: np.ndarray,
    box: np.ndarray,
    label_name: str,
    out_path: Path,
    context_pad: int,
    downscale: int,
):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clamp_box(np.array(box), w, h)
    cx1, cy1, cx2, cy2 = make_context_box(x1, y1, x2, y2, context_pad, w, h)

    full_with_box = image.copy()
    cv2.rectangle(full_with_box, (x1, y1), (x2, y2), (0, 255, 255), 2)

    context_crop = image[cy1:cy2, cx1:cx2].copy()
    local_x1, local_y1 = x1 - cx1, y1 - cy1
    local_x2, local_y2 = x2 - cx1, y2 - cy1

    crop_with_box = context_crop.copy()
    cv2.rectangle(crop_with_box, (local_x1, local_y1), (local_x2, local_y2), (0, 255, 255), 1)

    object_patch = context_crop[local_y1:local_y2, local_x1:local_x2].copy()
    if object_patch.size == 0:
        return

    degraded = degrade_patch(object_patch, downscale=downscale)
    recon = reconstruct_patch(degraded, downscale=downscale)
    heat = abs_diff_heatmap(object_patch, recon)

    paste_demo = full_with_box.copy()
    blend_patch = cv2.addWeighted(object_patch, 0.35, recon, 0.65, 0)
    paste_demo[y1:y2, x1:x2] = blend_patch

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Tiny Object Reconstruction Demo | class={label_name}", fontsize=14)

    axes[0, 0].imshow(cv2.cvtColor(full_with_box, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Full image with tiny-object box")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(crop_with_box, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Context crop")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(cv2.cvtColor(object_patch, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title("Original tiny object")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Degraded tiny object")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(cv2.cvtColor(recon, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Demo reconstruction")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(heat, cmap="hot")
    axes[1, 2].set_title("|Original - Reconstruction| heatmap")
    axes[1, 2].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    paste_out = out_path.with_name(out_path.stem + "_paste.jpg")
    cv2.imwrite(str(paste_out), cv2.cvtColor(paste_demo, cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    np.random.seed(args.seed)

    VisDroneDataset = load_visdrone_dataset_class()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TINY OBJECT RECONSTRUCTION DEMO (PAPER VISUALIZATION)")
    print("=" * 80)
    print(f"Data root: {args.data_root}")
    print(f"Split: {args.split}")
    print(f"Output dir: {output_dir}")

    dataset = VisDroneDataset(
        root_dir=args.data_root,
        split=args.split,
        min_size=args.min_size,
    )
    print(f"Loaded {len(dataset)} images from {args.split} split")

    print("Collecting tiny-object candidates...")
    candidates = collect_tiny_candidates(
        dataset,
        tiny_max_side=args.tiny_max_side,
        tiny_max_area=args.tiny_max_area,
    )
    print(f"Found {len(candidates)} tiny-object candidates")

    if len(candidates) == 0:
        print("No tiny candidates found with current thresholds.")
        return

    n = min(args.num_samples, len(candidates))
    selected_idx = np.random.choice(len(candidates), size=n, replace=False)
    selected = [candidates[i] for i in selected_idx]

    metadata = {
        "config": vars(args),
        "num_candidates": len(candidates),
        "num_selected": n,
        "samples": [],
    }

    class_names = dataset.CLASS_NAMES

    for i, item in enumerate(selected, start=1):
        image_tensor, _ = dataset[item["dataset_index"]]
        image = tensor_to_uint8_image(image_tensor)

        label = item["label"]
        label_name = class_names[label] if 0 <= label < len(class_names) else f"class_{label}"

        out_path = output_dir / f"tiny_recon_{i:03d}.png"
        visualize_sample(
            image=image,
            box=np.array(item["box"], dtype=np.float32),
            label_name=label_name,
            out_path=out_path,
            context_pad=args.context_pad,
            downscale=args.downscale,
        )

        item_copy = dict(item)
        item_copy["label_name"] = label_name
        item_copy["figure"] = out_path.name
        item_copy["paste_figure"] = out_path.with_name(out_path.stem + "_paste.jpg").name
        metadata["samples"].append(item_copy)

        print(
            f"[{i}/{n}] saved {out_path.name} | img_idx={item['dataset_index']} "
            f"label={label_name} box={item['box']}"
        )

    meta_path = output_dir / "demo_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("-" * 80)
    print(f"Saved {n} demo figures to: {output_dir}")
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
