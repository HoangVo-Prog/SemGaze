#!/usr/bin/env bash
set -euo pipefail

export JAVA_HOME=/data/shared/cvpr/hoang/tools/java17
export PATH="$JAVA_HOME/bin:$PATH"

python -u evaluate_flat.py \
  --checkpoint runs/YOUR_RUN/checkpoints/YOUR_CHECKPOINT \
  --config configs/flat_single.yaml \
  --output-dir runs/eval_metrics_only \
  --skip-loss