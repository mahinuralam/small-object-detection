# Complete Framework Guide: CD-DPA + SAHI + DGFE + DenseFPN
**Last Updated**: April 30, 2026
**Project**: Small Object Detection for UAV Imagery (VisDrone Dataset)

---

## Framework Overview

The base detection pipeline (CD-DPA + SAHI) detects most objects. Reconstruction
and DGFE are applied **only on weak tiles where SAHI fails** — as little as possible.
DenseFPN fuses the DGFE-enhanced P2'' with P3–P6 for final detection.

### Full Pipeline

```
Input Image
     │
ResNet50 Backbone + Standard FPN
     │
CD-DPA on P2, P3, P4   (Cascaded Deformable Dual-Path Attention)
     │
SAHI  →  base detections D_base
      →  K=5 weak tiles  (bottom-K by detection confidence score)
     │
     ├── P3–P6  ──────────────────────────────────────────→ DenseFPN
     │                                                          │
     └── P2 weak tiles:                                        │
             P2_crop → ReconHead → r_crop                      │
                                 → Δ = |r_crop − img| / 3      │
                                 → DGFE(Δ, P2_crop) → P2''    │
             residual paste: P2_final = P2 + accumulated ──────┘
                                                          │
                                                    DenseFPN
                                                          │
                                                   RPN + RoI Detection
```

### Key Principles

| Principle | Implementation |
|-----------|---------------|
| Minimal reconstruction | K=5 tiles only (42% of 12 canonical cells) — 58% compute saved vs exhaustive |
| P2 upper stream only | ReconHead/DGFE at highest resolution; P3–P6 unmodified |
| DenseFPN fuses streams | P2'' + P3–P6 enriched via dense cross-stage connections |
| CD-DPA on P2, P3, P4 | Deformable attention on fine-to-medium feature levels |
| SAHI dual purpose | Detection oracle AND weak-region identifier in one pass |

---

## Module Descriptions

### CD-DPA — Cascaded Deformable Dual-Path Attention
**File:** `models/enhancements/cddpa_module.py`

Applied to P2, P3, P4 after FPN. Two cascaded stages per level:
```
Stage 1: offset_conv(256→18) → DeformConv(256→256)
         Edge path:     DW-Conv(3×3) + DW-Conv(5×5) → spatial attention (B,1,H,W)
         Semantic path: GAP → FC(256→16→256) → channel attention (B,256,1,1)
         Fusion: Cat([edge×spatial, feat×channel]) → Conv1×1 + BN + ReLU + residual
Stage 2: same as Stage 1
Final:   Cat(stage1, stage2) → FusionConv(512→256) + residual
```
Params: 2.285M per level × 3 = **6.855M total**

---

### ReconstructionHead
**File:** `models/enhancements/reconstruction_head.py`

Takes P2 crop → reconstructs RGB tile. Used only during training on weak tiles.
```
P2_crop (256, H/4, W/4)
  → Up(256→128) → DoubleConv  → (128, H/2, W/2)
  → Up(128→64)  → DoubleConv  → (64,  H,   W)
  → Conv(64→3) + Sigmoid       → r_crop (3, H, W)

Δ = |r_crop − img_crop| / 3    → (1, H, W),  high where reconstruction failed
```
Params: **1.026M** | Applied at inference: **No** (training only)

---

### DGFE — Difference map Guided Feature Enhancement
**File:** `models/enhancements/dgfe_module.py`

Dual-gate attention driven by Δ. Amplifies P2 features both **globally** (channel)
and **spatially** (where Δ is high = where tiny objects are).

```
Δ (B,1,H,W) → resize to P2 dims (B,1,fH,fW)
    │
    ├── Channel gate (what channels matter — global):
    │     path3: DW-Conv(3×3) → GAP → FC(1→256) → ReLU → FC(256→256)
    │     path5: DW-Conv(5×5) → GAP → FC(1→256) → ReLU → FC(256→256)
    │     ch_attn = sigmoid(path3 + path5)  →  (B, 256, 1, 1)
    │
    └── Spatial gate (where to amplify — preserves H×W structure of Δ):
          Conv(1→16, 3×3) → ReLU → Conv(16→8, 3×3) → ReLU → Conv(8→1, 1×1) → Sigmoid
          sp_attn  →  (B, 1, fH, fW)
          Init: bias=-6 → sp_attn≈0 at start → no disruption to existing weights

P2_enhanced = P2_crop × ch_attn × (1 + sp_attn)
```

**Residual form `(1 + sp_attn)` rationale:**
- At init: `sp_attn ≈ 0` → `(1+0) = 1` → identical to old behaviour, no disruption
- During training: `sp_attn` grows at high-Δ locations → additive spatial boost
- Guarantee: P2 never suppressed below original value (minimum multiplier = 1.0)

Params: **0.134M** | Applied at inference: **No** (training only)

---

### DenseFPN
**File:** `models/enhancements/dense_fpn.py`

Dense cross-stage fusion neck. Receives P2'' (upper stream) + P3–P6 (lower stream).
```
Phase 1 (top-down dense):
  M5 = L5
  M4 = L4 + Up(M5)
  M3 = L3 + Up(M4) + Up(M5)
  M2 = L2 + Up(M3) + Up(M4) + Up(M5)   ← P2'' injected as L2

Phase 2 (bottom-up):
  P2_out = M2
  P3_out = M3 + Down(P2_out)
  P4_out = M4 + Down(P3_out)
  P5_out = M5 + Down(P4_out)
```
Params: **4.396M**

---

## Mathematical Formulation

### Weak Tile Score
```
score(cell_i) = Σ s_j   for all detections j whose box-centre falls in cell_i
weak_tiles = BottomK({ score(cell_i) }, K=5)
```

### Difference Map
```
Δ(x,y) = (1/3) · Σ_{c∈{R,G,B}} |r_crop(x,y,c) − img_crop(x,y,c)|
Δ ∈ ℝ^{1×h×w},  values ∈ [0,1]
```

### DGFE Enhancement (new)
```
ch_attn = σ(path3(Δ_resized) + path5(Δ_resized))       ∈ ℝ^{256×1×1}
sp_attn = σ(SpatialConvStack(Δ_resized))                ∈ ℝ^{1×H×W}
P2'' = P2_crop × ch_attn × (1 + sp_attn)
```

### Differentiable Residual Paste
```
accumulated[:, :, fy1:fy2, fx1:fx2] += (P2'' − P2_crop.detach())
P2_final = P2 + accumulated
```

### Training Loss
```
L = L_det + λ_rec · (1/K) · Σ_{k=1}^{K} L1(r_crop_k, img_crop_k)
λ_rec = 0.1
```

---

## Training Strategy

### Checkpoint Hierarchy

| Checkpoint | mAP@0.50 | Used for |
|------------|----------|----------|
| `outputs/best_model.pth` | 38.02% | Baseline reference |
| `outputs_dpa/best_model_dpa.pth` | 38.79% | DPA baseline |
| `outputs_phase2_smart_recon/best_model_phase2.pth` | 47.71%* | Phase 2 init |
| `outputs_v3_full_framework/best_model_v3.pth` | 48.16%* | V3 init (current) |

*ensemble with baseline, 5-scale SAHI eval

### Freeze Policy

| Layers | Status |
|--------|--------|
| ResNet body (conv1, layer1–3) | Frozen (~14.9M params) |
| ResNet layer4 + FPN | Trainable |
| CD-DPA (P2, P3, P4) | Trainable |
| ReconHead | Trainable (training only) |
| DGFE (channel + spatial) | Trainable (training only) |
| DenseFPN | Trainable |
| RPN + RoI heads | Trainable |

### Config (V3 + Spatial DGFE)

```python
num_epochs        = 15        # fine-tune from V3 checkpoint
batch_size        = 1         # per GPU
accumulation_steps = 8        # eff batch = 16 (2 GPU)
learning_rate     = 3e-5      # new/changed modules
backbone_lr       = 1e-6      # FPN fine-tune
lambda_rec        = 0.1
tile_sizes        = (512,)    # single-scale SAHI during training
K                 = 5         # weak tiles per image
init_checkpoint   = 'outputs_v3_full_framework/best_model_v3.pth'
output_dir        = 'outputs_v3_full_framework/'
```

### Launch
```bash
torchrun --nproc_per_node=2 scripts/train/train_v3_full_framework.py --epochs 15
```

---

## Evaluation Results

### Metrics (VisDrone-2018 val, 548 images)

| Model | AP@0.50:0.95 | AP@0.50 | AP@0.75 | APvt | APt | APs |
|-------|-------------|---------|---------|------|-----|-----|
| Baseline FasterRCNN | — | 38.02% | 19.25% | — | — | — |
| Baseline + DPA + SAHI (ref) | — | 47.38% | 24.22% | — | — | — |
| Phase 2 ensemble | — | 47.71% | 24.79% | — | — | — |
| **V3 alone** | **26.30%** | **47.67%** | **25.65%** | **10.86%** | **17.99%** | **24.74%** |
| **V3 ensemble** | **26.20%** | **48.16%** | **25.01%** | **9.78%** | **17.58%** | **24.59%** |

Area thresholds (VisDrone val distribution):
- APvt: area < 100 px² (< 10×10) — 9.4% of GT boxes
- APt:  100–400 px² (10–20px)    — 31.8% of GT boxes
- APs:  400–1024 px² (20–32px)   — 26.6% of GT boxes

### GFLOPs (single forward pass, 540×960 input)

| Component | GFLOPs |
|-----------|--------|
| FasterRCNN base | 93 |
| CD-DPA (P2+P3+P4) | 90 |
| DenseFPN | 67 |
| ReconHead + DGFE (K=5, training only) | 282 |
| **Inference total (no recon)** | **~250** |
| **5-scale SAHI eval (25 passes)** | **~2,870** |

### Parameters

| Module | Params |
|--------|--------|
| FasterRCNN base | 41.35M |
| CD-DPA × 3 | 6.86M |
| DenseFPN | 4.40M |
| ReconHead | 1.03M |
| DGFE (ch + spatial) | 0.13M |
| **Total** | **53.76M** |

---

## Project Structure

```
models/enhancements/
  cddpa_module.py          # CD-DPA (deformable dual-path attention)
  reconstruction_head.py   # ReconHead: P2_crop → r_crop + Δ
  dgfe_module.py           # DGFE: Δ → channel gate × (1 + spatial gate)
  dense_fpn.py             # DenseFPN neck

scripts/train/
  train_v3_full_framework.py    # Main training script (V3 + spatial DGFE)

scripts/eval/
  27_full_eval.py               # Full eval: AP, APvt, APt, APs, per-class
  26_v3_eval.py                 # Quick mAP@0.50/0.75 eval

scripts/visualize/
  19_visualize_v3_pipeline.py   # SAHI weak tiles + Recon + DGFE heatmaps

results/
  outputs_v3_full_framework/best_model_v3.pth    # V3 (48.16%)
  outputs_phase2_smart_recon/best_model_phase2.pth
  eval_full_v3.json                              # Full metrics JSON
```

---

## Implementation Notes

### DGFE Spatial Gate — Why Residual Form

Old: `P2 × ch_attn`  — channel-only, spatially blind  
New: `P2 × ch_attn × (1 + sp_attn)`  — spatially aware

The `(1 + sp_attn)` residual is critical:
- `sp_attn` init bias = −6 → sigmoid(−6) ≈ 0.002 at start
- Existing V3 weights trained without spatial gate are preserved
- Spatial gate grows gradually as it learns to activate at high-Δ locations
- Never suppresses features (minimum multiplier = 1.0)

### Weak Tile Coverage (1920×1080)
- 12 canonical 512px cells
- K=5 selected → 42% of cells → 58% compute saved vs exhaustive reconstruction

### SAHI Training vs Eval
- Training: 1-scale 512px (speed) → ~0.15s/image after warmup
- Eval: 5-scale (512–1280px) + H-flip TTA → 25 passes per image
