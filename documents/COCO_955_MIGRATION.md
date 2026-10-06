# COCO-Search18 95/5 query-coverage migration

Protocol: `cocosearch18_semgaze_master_955_v1`.

## Persisted inputs

- `data/COCO_Search18/split_95_5/all/train.json`: 26,125 records, 4,048 images.
- `data/COCO_Search18/split_95_5/all/test.json`: 1,375 records, 213 images.
- `data/COCO_Search18/split_95_5/master_split_manifest.json` and `split_index.json`:
  shared master partition and variant integrity metadata.
- `split_95_5/{all,tp_only,ta_only}/split_manifest.json`: frozen support draws,
  record checksums, and variant membership. All three variants share the master.

The existing persisted split and `prepare_cocosearch18_splits_95_5.py` already
implement the new split generation. Runtime verifies their checksums and partition;
it never resplits data. Old split roots and validation configuration fail clearly.

## Runtime changes

`semgaze/data/cocosearch18.py` verifies source, variant and shared metadata and
loads train/test only. `semgaze/data/fewshot.py` builds a K-independent query
universe from seen-subject all/train records. It contains 18,508 queries across
subjects 1/2/3/4/5/6/10. Each epoch is one shuffled permutation. K is an independent
uniform draw from 1..10 per query; supports use distinct same-subject train images,
exclude the query image, resolve one record per image, and randomize record order.

`semgaze/training/batching.py` consumes up to B*G consecutive queries per optimizer
window, resamples only supports on overflow, and groups accepted episodes by K.
It retains every partial batch and weights physical forwards by actual episode
count. Mixed K may need more than G physical forwards. A collation failure rolls
back the sampler window and aborts. Retry exhaustion does not claim infeasibility
was exhaustively proved; it reports the query/K and fails without substituting it.

`train_flat.py` and `semgaze/training/loop.py` derive the scheduler/warmup horizon
from `ceil(18508/(B*G))*num_train_epochs`. `max_steps` is only a debugging cap.
Test evaluation runs after a completed traversal, before its checkpoint.
`semgaze/evaluation/test.py`, `predictions.py`, `evaluate_flat.py`, and
`scripts/benchmark_test_a100.py` use test queries and all ten frozen support draws
for each K=1/5/10. There are 407 eligible unseen test queries and 12,210 evaluation
episodes per complete evaluation. Supports are never resampled or reordered.
Test metrics and configuration use `test` names; generic annotation/configuration
validation functions keep their names.

Checkpoints retain the exact permutation, cursor, epoch, query identity list and
sampler RNG alongside existing optimizer, scheduler, Python/Torch/CUDA RNG and
model state. Resume restores the scheduler and checks optimizer-window geometry;
it rejects legacy or corrupt sampler states instead of restarting an epoch.

## Usage and compatibility

```powershell
.venv/Scripts/python train_flat.py --config configs/flat_single.yaml --num-train-epochs 10 --output-dir runs/coco955
.venv/Scripts/python evaluate_flat.py --checkpoint runs/coco955/checkpoint-18508 --output-dir runs/test955 --split test --path both --semantic-max-new-tokens 32
```

`--epochs` remains a CLI spelling of `--num-train-epochs`; it has coverage semantics.
There are no validation-dataset aliases. Old YAML `epochs`, `steps_per_epoch`,
`validation`, and `validation_scope` fields are rejected. The throughput harness
retains a `sampling_group_size` argument for workload replay; it never forces K.
Historical performance reports and plans are labelled as predating the migration.

The trainer currently supports one process. Distributed launches fail before
training because uneven-rank execution is not implemented; there is no padded or
dropped distributed query traversal. Old checkpoints without query-coverage state
cannot resume. Schedule and optimizer-window geometry must match on resume.

## Verification and limits

Regression tests cover actual persisted counts and master partitions, source/file
checksums, the full real query universe, deterministic continuation, K distribution,
same-subject distinct-image supports, frozen evaluation draw order, overflow
query/K preservation, partial batches, epoch boundaries, checkpoint restoration,
and model/loss/gradient/generation parity on a small real Hugging Face model.

The local COCO image directories contain no JPEGs. Full InternVL 8B training and
GPU throughput cannot be verified here. No architecture, LoRA, loss, tokenization,
generation policy, or AiR-D/reference implementation was changed for this migration.

Verification commands (PowerShell):

```powershell
$env:TMP = (Resolve-Path .cache/protocol-tmp).Path
$env:TEMP = $env:TMP
.venv/Scripts/python -m pytest tests -q --basetemp=.cache/protocol-pytest-N
.venv/Scripts/python scripts/smoke_flat.py --contract-only
.venv/Scripts/python -m compileall -q semgaze scripts train_flat.py evaluate_flat.py
```

Use a fresh test output directory; local temporary-file capture is routed to the
workspace to avoid the system temporary-directory permission and capacity limits.
