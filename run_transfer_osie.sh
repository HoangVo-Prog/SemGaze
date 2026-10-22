#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Checkpoint & Dataset
export CHECKPOINT="${CHECKPOINT:-/home/cvpr2027/hoang/main/checkpoint-6050}"
export SCANDIFF_DATA="${SCANDIFF_DATA:-/data/shared/cvpr/nghiant/dataset}"

# Evaluation
export DATASET=osie
export SCOPE="${SCOPE:-unseen}"
export K_VALUES="${K_VALUES:-1 5 10}"
export DRAWS="${DRAWS:-1}"

# Batch size per K
export BATCH_SIZE_BY_K="${BATCH_SIZE_BY_K:-1:64,5:16,10:16}"

exec bash "$HERE/run_transfer_dataset.sh" "$@"
