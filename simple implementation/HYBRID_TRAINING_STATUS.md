# Hybrid Detector Training Status
**Started**: February 3, 2026
**Model**: MSFE (P3, P4) + RGD (P2)

## Configuration
- **Dataset**: VisDrone-2018 (6,471 train / 548 val)
- **Batch Size**: 2 (reduced for memory)
- **Max Epochs**: 50
- **Early Stopping**: 10 epochs patience
- **Reconstruction Weight**: 0.2 (vs 1.0 in RGD-only)
- **LR Schedule**: Cosine Annealing with Warm Restarts
  - Initial LR: 0.005
  - T_0: 10 epochs
  - T_mult: 2
  - eta_min: 1e-6

## Architecture Details

### P2 Enhancement (RGD)
- Feature Reconstructor: P2 → RGB reconstruction
- DGFF: Difference-guided feature fusion
- Learnable threshold: 0.0156862 (initial)
- Purpose: Very tiny objects (<16px)

### P3 Enhancement (MSFE)
- Multi-scale convolutions (3×3, 5×5)
- Spatial attention
- Channel attention (SE-style)
- Purpose: Tiny objects (16-32px, 52% of dataset)

### P4 Enhancement (MSFE)
- Multi-scale convolutions (3×3, 5×5)
- Spatial attention
- Channel attention (SE-style)
- Purpose: Small objects (32-64px, 33% of dataset)

## Training Progress

### Initial Observations (First 7 Epochs)

| Epoch | Train Total | Train Det | Val Total | Val Det | Thresh | Status |
|-------|-------------|-----------|-----------|---------|---------|--------|
| 1 | 1.7924 | 1.5612 | 1.5446 | 1.3767 | 0.014470 | ✓ Best |
| 2 | 1.5451 | 1.3228 | 1.4480 | 1.2801 | 0.013372 | ✓ Best |
| 3 | 1.4430 | 1.2207 | 1.3828 | 1.2149 | 0.012428 | ✓ Best |
| 4 | 1.3660 | 1.1436 | 1.3453 | 1.1774 | 0.011655 | ✓ Best |
| 5 | 1.3026 | 1.0799 | 1.3180 | 1.1501 | 0.011054 | ✓ Best |
| 6 | 1.2472 | 1.0249 | 1.2865 | 1.1186 | 0.010615 | ✓ Best |
| 7 | (running...) | | | | | |

### Key Metrics

**Detection Loss Progress**:
- Epoch 1 → 6: 1.3767 → 1.1186 (18.7% improvement)
- Consistent improvement every epoch
- No plateaus yet

**Reconstruction Loss**:
- Train: ~0.222 (stable)
- Val: ~0.168 (stable)
- Much lower impact due to weight=0.2

**Learnable Threshold**:
- Epoch 1: 0.014470
- Epoch 6: 0.010615
- Trend: Decreasing (more selective enhancement)

**Training Speed**:
- ~7.6 min/epoch
- ~274s for validation
- Total: ~8 min/epoch

## Comparison with Previous Models

| Model | Best Val Loss | mAP@0.5 | Training Time | Memory |
|-------|---------------|---------|---------------|--------|
| Baseline | - | 38.02% | 3.2h (14 epochs) | ~18GB |
| MSFE | 0.9242 | 43.44% | 3.5h (21 epochs) | ~23GB |
| RGD | 1.4344* | 38.83% | 7.0h (24 epochs) | ~25GB |
| **Hybrid** | **1.1186** (det) | TBD | ~6.7h (est. 50 epochs) | ~31GB |

*Note: RGD validation loss includes reconstruction, not directly comparable

## Expected Performance

Based on analysis:
- **Predicted mAP@0.5**: 44-46%
- **Improvement over baseline**: +6-8 percentage points
- **Rationale**: 
  - P2 (RGD): Handles very tiny objects (21%)
  - P3, P4 (MSFE): Handles tiny/small objects (85%)
  - Complementary mechanisms at optimal scales
  - Reduced reconstruction weight allows better detection focus

## Monitor Commands

```bash
# Check training progress
tail -f logs/train_hybrid.log

# Check if process is running
ps aux | grep "12_train_hybrid.py" | grep -v grep

# View current epoch progress
tail -50 logs/train_hybrid.log

# Check GPU memory
nvidia-smi

# View training output (errors)
tail -50 logs/train_hybrid_output.log
```

## Files

- **Model**: `models/hybrid_detector.py`
- **Training Script**: `scripts/train/12_train_hybrid.py`
- **Logs**: `logs/train_hybrid.log`, `logs/train_hybrid_output.log`
- **Checkpoints**: `results/outputs_hybrid/`
- **Best Model**: `results/outputs_hybrid/best_model_hybrid.pth`

## Next Steps

1. ✅ Training started successfully
2. ⏳ Monitor for ~6-7 hours (50 epochs max, early stop at 10)
3. ⏳ Evaluate on validation set
4. ⏳ Compare with MSFE and RGD standalone
5. ⏳ Generate visualizations
6. ⏳ Analyze per-class performance
7. ⏳ Document final results

---

**Status**: 🟢 Training in progress  
**Last Updated**: February 3, 2026  
**Expected Completion**: ~6-7 hours from start
