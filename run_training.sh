#!/bin/bash
# CD-DPA Training Launcher
set -euo pipefail

# Use env passed as first arg, default to base (the CUDA-capable env on this machine).
ENV_NAME="${1:-base}"

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
cd "/home/mahin/Documents/notebook/small-object-detection/simple implementation/scripts/train"
echo "[launcher] env=$(conda info --envs | grep '*' | awk '{print $1}')"
echo "[launcher] python=$(which python)"
python -c "import torch; assert torch.cuda.is_available(), 'NO CUDA in selected env'; print(f'[launcher] CUDA OK: {torch.cuda.get_device_name(0)}')"
exec env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u 14_train_cddpa.py 2>&1 | tee ../../results/outputs_cddpa/train.log
