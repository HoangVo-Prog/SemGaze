#!/usr/bin/env bash
# One-shot OSIE only; no COCO-FreeView data loaded/evaluated.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DATASET=osie
exec bash "$HERE/run_transfer_dataset.sh" "$@"
