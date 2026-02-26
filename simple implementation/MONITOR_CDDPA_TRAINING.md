# CD-DPA Training Monitor

**Status**: ✅ Training Started  
**Start Time**: February 5, 2026  
**Expected Duration**: 8-10 hours  
**Target**: 48-50% mAP@0.5

---

## Quick Status Check

```bash
# View live training progress
tail -f results/cddpa_training.log

# Check last 50 lines
tail -n 50 results/cddpa_training.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check if still running
ps aux | grep 14_train_cddpa.py
```

---

## Training Configuration

**Model**: FasterRCNN with CD-DPA  
**Enhanced Levels**: P2, P3, P4  
**Total Parameters**: 48.20M (6.86M CD-DPA)

**Memory Optimizations**:
- ✅ Mixed Precision (FP16)
- ✅ Gradient Checkpointing  
- ✅ Gradient Accumulation (4 steps)
- ✅ Effective Batch Size: 16

**Dataset**:
- Train: 6,471 images (1,618 batches)
- Val: 548 images (137 batches)

**Training**:
- Epochs: 50
- Learning Rate: 1e-4
- Warmup: 3 epochs
- Early Stopping: 15 epochs patience

---

## Expected Timeline

| Time | Milestone |
|------|-----------|
| 0h | ✅ Training started |
| ~0.5h | Epoch 1 complete |
| ~1.5h | Epoch 3 complete (warmup done) |
| ~4h | Epoch 20 (halfway) |
| ~8-10h | Training complete (or early stop) |

---

## Performance Targets

| Model | mAP@0.5 | Status |
|-------|---------|--------|
| Baseline | 38.02% | Reference |
| SimplifiedDPA | 43.44% | Current Best |
| **CD-DPA** | **48-50%** | **Training...** |

**Expected Improvement**: +4-6% over SimplifiedDPA

---

## What to Watch For

### ✅ Good Signs
- Loss decreasing steadily
- GPU memory ~20-22GB (safe margin)
- Speed ~1.0-1.5 it/s
- Validation loss improving

### ⚠️ Warning Signs
- Loss not decreasing after 5 epochs
- GPU memory >23GB (might OOM)
- Speed <0.5 it/s (too slow)
- Validation loss increasing (overfitting)

### 🛑 Stop If
- OOM error (kill and reduce batch_size)
- Loss becomes NaN (restart with lower lr)
- No improvement after 15 epochs (early stop will handle this)

---

## After Training Completes

### 1. Check Results
```bash
cd scripts/eval
python 18_evaluate_cddpa.py
```

### 2. Compare Performance
```bash
cat results/outputs_cddpa/final_metrics.json
```

### 3. Check Best Model
```bash
ls -lh results/outputs_cddpa/best_model.pth
```

Expected output: Model achieves 48-50% mAP@0.5

---

## Emergency Controls

### Pause Training
```bash
# Send SIGTERM (graceful stop)
pkill -SIGTERM -f 14_train_cddpa.py

# Best model checkpoint should be saved
```

### Resume Training
The script saves checkpoints every epoch. To resume:
1. Modify script to load from checkpoint
2. Or use the best model so far

### Kill Training
```bash
# Force kill if needed
pkill -9 -f 14_train_cddpa.py
```

---

## Files Being Generated

```
results/
├── outputs_cddpa/
│   ├── best_model.pth           # Best validation loss
│   ├── checkpoint_epoch_XX.pth  # Periodic checkpoints
│   ├── training_log.json        # Epoch-by-epoch metrics
│   └── config.json              # Training configuration
├── cddpa_training.log           # Full training output
└── ...

logs/
└── train_cddpa.log              # Structured log file
```

---

## Next Steps After Training

1. **Evaluate**: Run evaluation script to get mAP@0.5
2. **Compare**: Check against baseline (38.02%) and SimplifiedDPA (43.44%)
3. **Analyze**: If <48%, review training curves and consider tuning
4. **Document**: If ≥48%, prepare paper contributions

---

**Training Terminal ID**: `9828f8f8-61d5-4809-91bb-776b870b4ac6`

**Last Check**: Training started successfully at Epoch 1, loss ~3.8
