# SemGaze flat single-output implementation

The package implements one connected episode: ordered same-subject demonstrations,
teacher-forced XYD WHERE, query `<END_FIX>` states, shared Linear + LayerNorm `P_E`,
one flat WHAT/WHY/HOW response, and one joint backward. Only `flat_single_output`
is accepted. Scientific contracts remain in `documents/`.

## Setup and verification

Use Python 3.11+ and an environment with the appropriate PyTorch 2.11 / torchvision
0.26 build for your device. The tested HF/PEFT versions are pinned in `pyproject.toml`.
From the repository root:

```powershell
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install -e '.[test]'
.venv/Scripts/python -m pytest tests -q
.venv/Scripts/python scripts/smoke_flat.py --contract-only
.venv/Scripts/python scripts/smoke_flat.py --with-model
```

`--with-model` initializes the configured fresh or continued LoRA trainably and performs no optimizer step
unless `--step` is supplied. It saves resolved configuration and preflight provenance
under `runs/`. It fails on missing dependencies, LFS pointer files, incompatible
tokenizers/templates, incompatible adapter tensors, unsupported devices, or unintended trainables.

`tests/test_model_path.py` uses a small, randomly initialized **real HF InternVL +
PEFT** solely to test implementation mechanics on CPU. This is not the baseline
initialization and does not satisfy the released 8B milestone. It covers semantic-only
gradients through WHERE, direct state/image conditioning, tied/untied token rows,
unchanged old vocabulary weights after AdamW, checkpoint/resume, and generation.
`tests/test_base_processor.py` additionally checks the real base processor when it
is cached at `.cache/semgaze-base-processor`; it otherwise skips without downloading.

## Current runtime gates

The checkout inspected on 2026-10-04 has Git LFS pointer files in place of the
released adapter weights and tokenizer JSON, CPU-only PyTorch, no cached 8B model
weights, and empty `data/COCO_Search18/images/{tp,ta}/` directories. The reference
checkout was not changed. Full released-model smoke and training remain unverified.

Persisted annotations contain no coordinate-frame dimensions. Before real-data
runs, set `data.annotation_frame` to a verified `{width: ..., height: ...}` or to
`original_image` only if installed images are the original annotation images.
The normalized fixture already declares its dimensions and needs no such inference.

The adapter reads `data.split_root`, verifies source/split checksums and
frozen manifest membership, maps `prediction.fixations/regions/how`, and preserves
`data.duration.source_field` as milliseconds (default `T`). It never recreates splits or trial filtering. Empty images
or unresolved frame metadata fail explicitly.

## Training and resume

```powershell
.venv/Scripts/python train_flat.py --config configs/flat_single.yaml --output-dir runs/experiment --num-train-epochs 10 --semantic-max-new-tokens 512
.venv/Scripts/python train_flat.py --config configs/flat_single.yaml --output-dir runs/resumed --num-train-epochs 10 --semantic-max-new-tokens 512 --resume runs/experiment/checkpoint-100
```

These are example horizons, not scientific defaults. Every training run executes
the fixture gradient gate before optimization. `per_device_train_batch_size` is a
true physical GPU episode batch: one WHERE backbone forward and one semantic
backbone forward per batch. `gradient_accumulation_steps` counts these physical
batches per optimizer update. Each branch averages shifted supervised-token NLL
within each episode, then averages episode losses; long targets get no extra
weight. Effective batch size is physical B * accumulation (world size is currently 1).
WHERE overflow resamples only the rejected complete episode;
semantic overflow fails. By default, checkpoints include processor/tokenizer, PEFT, separate
END_FIX rows, P_E, optimizer, scheduler, RNG/sampler state, resolved config, and
split identity. Checkpoint component flags control saved artifacts; incomplete exports cannot reconstruct the full model. Resume restores optimizer moments, RNG and history, while applying the resolved optimizer hyperparameters and schedule. Model/token structure and optimizer implementation must remain compatible.

### Training throughput and profiling

For the real server-side A100 B=1/B=2 benchmark, use
[`scripts/benchmark_a100.py`](scripts/benchmark_a100.py). The
[server guide](documents/A100_BENCHMARK.md) gives exact commands and describes
real training episodes, isolated variants, OOM handling, and JSON/CSV results.
It has no fixture or CPU fallback. The older profiler below remains available
for the original-path and tiny-model engineering comparisons.

Training computes full-vocabulary logits only at supervised predictor positions
and reads final LM hidden states directly. The deprecated
`where.supervision.output_hidden_states` field remains readable in old checkpoints;
training always disables all-layer output collection. Trainable parameter names
and checkpoint tensors are unchanged. Keep non-reentrant checkpointing enabled
until the target GPU is profiled.

`length_aware_batching: true` stably groups already-sampled episodes by K and coarse
length only **within one optimizer minibatch**. It preserves every accepted episode,
support order and sampler RNG state. Floating-point summation order and dropout
RNG assignment can differ with physical batching; deterministic parity tests disable
dropout while production retains its configured probabilities. Prediction snapshots
retain their original sampling-order membership.

`reuse_query_vision: true` reuses this batch's frozen projected query image features
when preprocessing is identical and vision has no active stochastic dropout. It
automatically disables reuse if either vision producer becomes trainable. Images
are decoded once per optimizer minibatch, and no image/language cache survives an
optimizer step. Native preprocessing is still run per prompt. Gradient diagnostics
run at smoke/preflight, the first optimizer step, and optionally every
`gradient_diagnostics_every` steps (0 disables periodic scans). Aggregate non-finite
gradient checking runs at every optimizer step, including when clipping is disabled.

Use `configs/flat_throughput.yaml` for the required released DeepGaze adapter and
bf16 A100 run. The fresh-initialization example remains a separate user choice.
The profiling command synchronizes only at explicit profiling boundaries, records
live/peak allocated and reserved memory separately, component timings, episode
lengths, images, K, fixation counts, supervised counts, attention backends and
optimizer-state bytes. It never empties the allocator cache.

```powershell
python scripts/profile_training.py --config configs/flat_throughput.yaml --variant baseline --batch-size 1 --output runs/profile-old-b1.json
python scripts/profile_training.py --config configs/flat_throughput.yaml --variant optimized --batch-size 1 --output runs/profile-new-b1.json
python scripts/profile_training.py --config configs/flat_throughput.yaml --variant optimized --batch-size 2 --output runs/profile-new-b2.json
```

Repeat with `--checkpointing off` for the requested checkpointing matrix, and with
larger B only if memory headroom permits. `--workload episodes.json` accepts an
array of normalized episodes (exactly `--batch-size` entries); retain the same
episode pool and seed for comparisons. The default is the K=1 fixture. Use
`--tiny` for CPU engineering measurements; these cannot establish 8B/A100 capacity
or speed. No attention backend, allocator, optimizer precision, or checkpointing
retuning is selected without target measurements. See
`documents/TRAINING_THROUGHPUT_AUDIT.md` and `documents/TRAINING_THROUGHPUT_RESULTS.md`.

### Query-coverage epochs, test evaluation and resume

COCO-Search18 uses `data/COCO_Search18/split_95_5/all/train.json` and
`test.json`, with a shared image partition in `master_split_manifest.json` and
`split_index.json` in the parent directory. There is no validation split.
The persisted master contains 4,048 train images / 213 test images and
26,125 train records / 1,375 test records.

`Q_train` consists of every eligible train record whose subject is outside
`{7,8,9}`. Each epoch shuffles this fixed universe and consumes every query once.
Each query independently draws K uniformly from 1..10, then distinct same-subject
train support images excluding the query image, one record per image, and a
random support order. Overflow retries retain the query and K and resample only
supports. Exhausting the retry budget fails with the query ID and K.

Set `training.num_train_epochs` or `--num-train-epochs`. `--epochs` is an equivalent
CLI spelling. `--max-steps` is a debugging cap; a partial epoch never triggers
epoch-end evaluation. The removed `steps_per_epoch` configuration is rejected.

An optimizer window takes at most B*G consecutive queries and groups their
already constructed episodes by K into physical batches of at most B. This can
require more than G physical forwards when K differs. Each batch is weighted by
its actual episode count. The tail is retained; no query is padded or dropped.
Scheduler length and warmup derive from `ceil(|Q_train|/(B*G))*num_train_epochs`.

After each completed epoch, `evaluation.strategy: epoch` evaluates unseen-subject
test queries at K=1/5/10, with all ten persisted exclusive draws per K. Frozen
support order is preserved. `test_where`, `test_flat`, `test_total`, and
`test_what/why/how` describe the existing teacher-forced losses; section metrics
are diagnostic slices of one flat response. They do not change the objective or
add a checkpoint selection rule. `test.loss`, `test.prediction`, and `test.cache`
configure execution. `evaluation.strategy: 'no'` is available for smoke tests
and throughput benchmarks.

Optional generation uses the same test draws and writes
`predictions/epoch-0001/{train,test}.jsonl`. Test rows include `draw_id` and ordered
support IDs. Training predictions reuse the most recent accepted episodes, up
to `train_batches*B`; they never consume the sampler. An interrupted run may
have fewer recent examples after resume. `train_batches: 0` and
`test_scope: none` disable generation while retaining test loss evaluation.
`semantic_max_new_tokens` remains an explicit run choice. WHERE generation and
GT-WHERE-conditioned semantic generation retain their existing definitions.

Checkpoints save epoch index, cursor, exact query permutation, query IDs, K/support
RNG, optimizer, scheduler and model RNG state. Resume continues the next query.
Old samplers/checkpoints without coverage state fail clearly. Training schedule
and optimizer-window geometry must match on resume. The current trainer supports
one process; distributed launches fail before training because uneven-rank
execution is not implemented. No distributed sampler pads or drops queries.

See [the protocol migration notes](documents/COCO_955_MIGRATION.md) for validation
and remaining runtime limits.

## Evaluation

`evaluate_flat.py` loads a saved checkpoint, consumes every eligible query and
the configured K support sets in the saved manifest, and exports predictions and draw-level validity
diagnostics. It runs free-running WHERE separately from GT-trajectory-conditioned
flat semantic generation. It does not tune on evaluation subjects.

```powershell
.venv/Scripts/python evaluate_flat.py --checkpoint runs/experiment/checkpoint-1000 --output-dir runs/evaluation --split test --path both --semantic-max-new-tokens 512
```

The semantic budget above is an explicit example setting, not a canonical budget.
SM/MM/SED implementations, invalid-trajectory treatment, semantic text metrics,
and the checkpoint-selection scalar are not yet frozen. No canonical metric table
or automatic best-checkpoint selection is claimed by this implementation.

See `IMPLEMENTATION_PLAN_FLAT.md` for phase status, concrete verification results,
and the remaining released-model gates.

Evaluation inherits the checkpoint configuration. `--config` may supply `runtime`
and `evaluation` overrides, including a new output directory, device, K subset and
semantic generation budget. Explicit CLI flags override these values. Model/data
identity stays attached to the checkpoint; missing frozen support blocks and
manifest/hash mismatches fail instead of inventing replacement supports.
