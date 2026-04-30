# Small Object Detection — VisDrone Benchmark Results

**Dataset:** VisDrone-2018 val split (548 images, 10 object classes)  
**Hardware:** 2× NVIDIA GeForce RTX 3090 (24 GB each)  
**Eval:** 5-scale SAHI (512–1280px) + H-flip TTA, score_thresh=0.008, nms_iou=0.35

---

## Current Best Results

### V3 Full Framework (pycocotools, script 27)

| Model | AP@0.50:0.95 | AP@0.50 | AP@0.75 | APvt | APt | APs | APm |
|-------|-------------|---------|---------|------|-----|-----|-----|
| **V3 alone** | **26.30%** | **47.67%** | **25.65%** | **10.86%** | **17.99%** | **24.74%** | **34.66%** |
| **Baseline + V3 ensemble** | **26.20%** | **48.16%** | **25.01%** | **9.78%** | **17.58%** | **24.59%** | **34.80%** |

Area thresholds: APvt < 100px² | APt 100–400px² | APs 400–1024px² | APm 1024–9216px²

### V3 Per-class AP@0.50:0.95

| Class | V3 alone | Ensemble |
|-------|----------|----------|
| car | 56.67% | 55.62% |
| bus | 39.67% | 39.64% |
| van | 31.08% | 31.01% |
| pedestrian | 30.60% | 30.32% |
| motor | 25.16% | 25.00% |
| truck | 23.39% | 23.92% |
| people | 18.37% | 18.19% |
| tricycle | 17.82% | 17.99% |
| bicycle | 10.40% | 10.46% |
| awning-tricycle | 9.82% | 9.81% |

---

## Model Progression

| Model | Params | mAP@0.50 | mAP@0.75 | Notes |
|-------|--------|----------|---------|-------|
| Baseline FasterRCNN | 41.4M | 38.02% | 19.25% | ResNet50+FPN, reference |
| + SimpleDPA (P2,P3) | 42.2M | 38.79% | — | Dual-path attention |
| Baseline + DPA + 5-scale SAHI | — | 47.38% | 24.22% | Ensemble, script 24 reference |
| Phase 2 (DGFE weak tiles) | 42.8M | 47.71%* | 24.79%* | Ensemble with baseline |
| **V3 (CD-DPA + DGFE + DenseFPN)** | **53.8M** | **48.16%*** | **25.01%*** | **Ensemble, current best** |
| V3 + Spatial DGFE | 53.8M | TBD | TBD | Training in progress (15 ep) |

*ensemble with baseline, 5-scale SAHI + TTA

---

## SAHI Ablation (Baseline Model)

| Method | Tile Size(s) | mAP@0.50 | Δ |
|--------|-------------|----------|---|
| Full image only | — | 38.13% | — |
| 1-scale SAHI | 512 | 44.01% | +5.88% |
| 3-scale SAHI | 512+640+768 | 46.20% | +8.07% |
| 5-scale + full-image | 512–1280 | 46.96% | +8.83% |
| **5-scale + TTA + ensemble** | 512–1280 | **47.38%** | **+9.25%** |

---

## GFLOPs Profile (540×960 input, measured with thop)

| Component | GFLOPs | Params |
|-----------|--------|--------|
| FasterRCNN base | 93 | 41.35M |
| CD-DPA × 3 (P2,P3,P4) | 90 | 6.86M |
| DenseFPN | 67 | 4.40M |
| ReconHead + DGFE (K=5 tiles, training only) | 282 | 1.16M |
| **Inference total (no recon)** | **~250** | **53.76M** |
| **5-scale SAHI eval (25 passes)** | **~2,870** | — |

---

## DGFE Architecture Change (April 30, 2026)

**Problem:** Original DGFE used channel-only attention — GAP destroyed all spatial structure of Δ, making before/after P2 features look identical spatially.

**Fix:** Added spatial attention branch:
```
P2'' = P2 × ch_attn(B,256,1,1) × (1 + sp_attn(B,1,H,W))
```
- `ch_attn`: existing channel gate (what channels to amplify)
- `sp_attn`: new spatial gate (where to amplify — high at high-Δ locations)
- Residual form `(1 + sp_attn)`: starts at ~0, never suppresses features
- New params: +1,304 (negligible)

Retraining from V3 checkpoint: 15 epochs, ~12 hours total.

---

## Training Notes

### DDP + Gradient Accumulation
```python
is_sync = (step + 1) % accumulation_steps == 0
sync_ctx = contextlib.nullcontext() if (not IS_DIST or is_sync) else model.no_sync()
with sync_ctx:
    loss = model(images, targets) / accumulation_steps
    scaler.scale(loss).backward()
```

### Key Hyperparameters (V3)
- λ_rec = 0.1, K = 5 weak tiles, tile_size = 512px (training SAHI)
- Frozen: ResNet layer1–3; Trainable: layer4, FPN, CD-DPA, DenseFPN, ReconHead, DGFE
- Batch = 1/GPU, accumulation = 8 → eff batch = 16

*Last updated: 2026-04-30*
