#!/usr/bin/env bash
# One-shot COCO-FreeView only; no OSIE data loaded/evaluated.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DATASET=coco_freeview
exec bash "$HERE/run_transfer_dataset.sh" "$@"
