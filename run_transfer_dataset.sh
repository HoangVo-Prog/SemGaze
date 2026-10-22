#!/usr/bin/env bash
# Standalone FS-PSP-split transfer evaluation for ONE target dataset.
# Invoked by run_transfer_osie.sh or run_transfer_coco_freeview.sh.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${DATASET:?Use run_transfer_osie.sh or run_transfer_coco_freeview.sh}"
if [[ "$DATASET" != 'osie' && "$DATASET" != 'coco_freeview' ]]; then
  echo "Unsupported DATASET=$DATASET; must be osie or coco_freeview" >&2
  exit 2
fi
: "${CHECKPOINT:?Set CHECKPOINT=/absolute/path/to/semgaze/checkpoint}"
: "${SCANDIFF_DATA:?Set SCANDIFF_DATA=/absolute/path/to/ScanDiff/data}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SCOPE="${SCOPE:-unseen}"
K_VALUES="${K_VALUES:-1 5 10}"
DRAWS="${DRAWS:-1}"
SEED="${SEED:-42}"
# Conservative-ish starting point; tune downward if your GPU runs out of memory.
BATCH_SIZE_BY_K="${BATCH_SIZE_BY_K:-1:64,5:16,10:16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_FIXATIONS="${MAX_FIXATIONS:-10}"
MODEL_DEVICE="${MODEL_DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/transfer_${DATASET}_${SCOPE}}"
DRY_RUN="${DRY_RUN:-0}"
BUCKET_BY_FIXATION_COUNT="${BUCKET_BY_FIXATION_COUNT:-1}"

read -r -a K_ARRAY <<< "$K_VALUES"
if (( ${#K_ARRAY[@]} == 0 )); then
  echo 'K_VALUES must be nonempty, e.g. K_VALUES="1 5 10"' >&2
  exit 2
fi

cmd=("$PYTHON_BIN" -u "$SCRIPT_DIR/evaluate_transfer_freeview.py"
  --checkpoint "$CHECKPOINT"
  --scandiff-data "$SCANDIFF_DATA"
  --dataset "$DATASET"
  --scope "$SCOPE"
  --split-protocol fspsp-main
  --k-values "${K_ARRAY[@]}"
  --draws "$DRAWS"
  --seed "$SEED"
  --batch-size "$BATCH_SIZE"
  --batch-size-by-k "$BATCH_SIZE_BY_K"
  --max-fixations "$MAX_FIXATIONS"
  --device "$MODEL_DEVICE"
  --output-dir "$OUTPUT_DIR"
)
if [[ "$DRY_RUN" == 1 ]]; then cmd+=(--dry-run); fi
if [[ "$BUCKET_BY_FIXATION_COUNT" == 0 ]]; then cmd+=(--no-bucket-by-fixation-count); fi
# Forward optional evaluator flags such as --fspsp-subject-id-base 0.
cmd+=("$@")
printf '[TRANSFER] dataset=%s scope=%s K=(%s) BS_BY_K=%s draws=%s\n' \
  "$DATASET" "$SCOPE" "$K_VALUES" "$BATCH_SIZE_BY_K" "$DRAWS"
exec "${cmd[@]}"
