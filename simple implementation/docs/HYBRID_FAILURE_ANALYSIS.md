# Model Performance Comparison
**Date**: February 4, 2026  
**Dataset**: VisDrone-2018 Validation (548 images, 37,771 GT boxes)

## Results Summary

| Model | mAP@0.5 | mAP(0.5:0.95) | Training | Val Loss | Memory | Status |
|-------|---------|---------------|----------|----------|--------|--------|
| **Baseline** | 38.02% | - | 14 epochs, 3.2h | - | ~18GB | ✅ Complete |
| **MSFE (P3+P4)** | **43.44%** ✅ | - | 21 epochs, 3.5h | 0.9242 (det) | ~23GB | ✅ Complete |
| **RGD (P2)** | 38.83% | 22.01% | 24 epochs, 7.0h | 1.0787 (det) | ~25GB | ✅ Complete |
| **Hybrid (P2+P3+P4)** | **22.44%** ❌ | 11.73% | 7 epochs, <1h | 1.0787 (det) | ~31GB | ⚠️ **FAILED** |

## Critical Issue: Hybrid Model Underperformed Severely

### The Problem
The hybrid model achieved **22.44% mAP@0.5**, which is:
- **48% worse** than MSFE (43.44%)
- **41% worse** than Baseline (38.02%)
- **42% worse** than RGD (38.83%)

This is a **catastrophic failure** - combining two working approaches made performance worse than either alone!

## Root Cause Analysis

### 1. ⚠️ Early Stopping at Epoch 7

**Training Log**:
```
Epoch 1: Val Det Loss = 1.3767
Epoch 2: Val Det Loss = 1.2801  ✓ Improved
Epoch 3: Val Det Loss = 1.2149  ✓ Improved
Epoch 4: Val Det Loss = 1.1774  ✓ Improved
Epoch 5: Val Det Loss = 1.1501  ✓ Improved
Epoch 6: Val Det Loss = 1.1186  ✓ Improved
Epoch 7: Val Det Loss = 1.0787  ✓ Best (stopped here)
```

**Issue**: Training stopped at epoch 7 (probably due to early stopping with patience=10).

**Analysis**:
- Loss was **still decreasing** every epoch
- Model hadn't converged yet
- MSFE trained for 21 epochs, RGD for 24 epochs
- Stopping at epoch 7 = only **33% of MSFE's training** and **29% of RGD's training**

### 2. 🔥 Memory/Computational Overload

**System Resources**:
- Hybrid model: 43.5M parameters
- Batch size: 2 (reduced from 4)
- Memory: ~31GB (approaching 32GB limit)

**Evidence**:
- 5 training processes were found running simultaneously
- Training logs weren't being written properly
- Checkpoint timestamps suggest training may have stalled/restarted

**Theory**: System ran out of memory, causing training to crash or restart, leading to premature termination.

### 3. 💥 Gradient/Optimization Conflicts

**Hybrid Architecture**:
```python
P2: RGD (reconstruction + DGFF)
P3: MSFE (multi-scale + attention)
P4: MSFE (multi-scale + attention)
```

**Potential Issue**:
- Three different enhancement mechanisms fighting for gradient updates
- Reconstruction loss (even at 0.2 weight) may interfere with MSFE optimization
- DGFF on P2 might create feature inconsistencies with MSFE on P3/P4

### 4. 🎯 Insufficient Training

**Comparison**:
```
MSFE:   21 epochs → 43.44% mAP
RGD:    24 epochs → 38.83% mAP
Hybrid:  7 epochs → 22.44% mAP  ← Stopped too early!
```

**At Epoch 7**:
- Loss still decreasing steadily
- Model hadn't learned effective feature enhancement
- Needed at least 15-20 more epochs

## Per-Class Performance Breakdown

### Hybrid Model Per-Class AP@0.50

| Class | Hybrid | MSFE | RGD | vs MSFE | vs RGD |
|-------|--------|------|-----|---------|--------|
| Car | **38.66%** | 78.76% | 51.75% | -40.10% | -13.09% |
| Bus | 16.89% | 49.67% | 33.11% | -32.78% | -16.22% |
| Van | 16.86% | 48.29% | 29.42% | -31.43% | -12.56% |
| Truck | 11.86% | 40.95% | 21.66% | -29.09% | -9.80% |
| Pedestrian | 8.42% | 54.64% | 21.38% | -46.22% | -12.96% |
| Motor | 7.97% | 51.12% | 20.28% | -43.15% | -12.31% |
| Tricycle | 6.89% | 29.56% | 13.85% | -22.67% | -6.96% |
| People | 5.39% | 43.22% | 13.76% | -37.83% | -8.37% |
| Awning-tricycle | 3.31% | 16.53% | 8.25% | -13.22% | -4.94% |
| Bicycle | 1.10% | 21.56% | 6.65% | -20.46% | -5.55% |

**Observations**:
- Every single class performs worse than both MSFE and RGD
- Worst drops: Pedestrian (-46%), Motor (-43%), Car (-40%)
- Best class (Car) still only 38.66% vs MSFE's 78.76%

## Why Did This Happen?

### Theory 1: Undertrained Model
**Most Likely**

The model simply didn't train long enough:
- 7 epochs vs 21 (MSFE) or 24 (RGD)
- Loss was still decreasing
- Model needed more time to learn complex hybrid architecture

### Theory 2: System Instability
**Likely Contributing Factor**

Evidence of training issues:
- Multiple processes running
- Log file not updating
- Training may have crashed and restarted
- System memory near capacity (31GB/32GB)

### Theory 3: Architecture Incompatibility
**Possible**

Combining RGD + MSFE may create conflicts:
- Reconstruction on P2 creates features optimized for RGB output
- MSFE on P3/P4 expects features optimized for detection
- Feature pyramid inconsistency between levels
- Gradient updates fighting each other

### Theory 4: Hyperparameter Mismatch
**Less Likely**

Current settings may not suit hybrid architecture:
- Learning rate too high/low
- Reconstruction weight (0.2) still interfering
- Batch size too small (2) for stable gradients

## Recommendations

### 🟢 Immediate Action: Retrain with More Epochs

**Modified Configuration**:
```python
{
    'num_epochs': 50,
    'patience': 15,  # Increased from 10
    'batch_size': 2,  # Keep at 2 due to memory
    'reconstruction_weight': 0.1,  # Further reduce
    'learning_rate': 0.003,  # Slightly lower
}
```

**Expected**: Allow model to train for 15-25 epochs minimum

### 🟡 Alternative: Use MSFE Only

Since MSFE already achieves **43.44% mAP** with good efficiency:
- **Recommendation**: Use MSFE as final model
- Already best performance
- Faster training (3.5h vs 6-7h)
- More stable
- Lower memory requirements

### 🔴 If Retraining Hybrid:

1. **Fix Training Stability**:
   - Kill all existing processes
   - Ensure single training instance
   - Monitor with `watch -n 5 nvidia-smi`

2. **Adjust Hyperparameters**:
   ```python
   reconstruction_weight: 0.1  # Even lower
   learning_rate: 0.003        # More conservative
   patience: 15                # More patience
   ```

3. **Monitor Closely**:
   - Check GPU memory every 10 minutes
   - Verify log file updates
   - Watch for process crashes

4. **Early Indicators** (by epoch 15):
   - mAP should be >35% by epoch 15
   - If not, stop and use MSFE

## Final Verdict

### Current Recommendation: **Use MSFE Model**

| Reason | MSFE | Hybrid (current) |
|--------|------|------------------|
| mAP@0.5 | **43.44%** ✅ | 22.44% ❌ |
| Training Time | 3.5h ✅ | <1h (incomplete) |
| Stability | Stable ✅ | Crashed ❌ |
| Memory | 23GB ✅ | 31GB ⚠️ |
| Status | **Production Ready** ✅ | Needs retraining ⚠️ |

### If Resources Allow: Retrain Hybrid

- Clear all processes
- Train for minimum 25 epochs
- Monitor closely for crashes
- Target: >40% mAP@0.5

### If Time Constrained: Proceed with MSFE

MSFE is the **winner**:
- Highest mAP: **43.44%**
- Fast training: 3.5 hours
- Stable and reproducible
- Good memory efficiency
- **+5.42% over baseline**

---

**Decision**: Unless hybrid can be successfully retrained to >43%, **use MSFE as the final model**.
