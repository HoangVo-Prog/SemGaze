#!/usr/bin/env bash
set -euo pipefail

export JAVA_HOME=/data/shared/cvpr/hoang/tools/java17
export PATH="$JAVA_HOME/bin:$PATH"

python -u train_flat.py \
  --config configs/flat_single.yaml \
  --output-dir runs/flat