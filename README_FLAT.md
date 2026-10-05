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
.venv/Scripts/python train_flat.py --config configs/flat_single.yaml --output-dir runs/experiment --max-steps 1000 --steps-per-epoch 100 --semantic-max-new-tokens 512
.venv/Scripts/python train_flat.py --config configs/flat_single.yaml --output-dir runs/resumed --max-steps 1000 --steps-per-epoch 100 --semantic-max-new-tokens 512 --resume runs/experiment/checkpoint-100
```

These are example horizons, not scientific defaults. Every training run executes
the fixture gradient gate before optimization. Episodes are accumulated sequentially
with an explicit episode mean. WHERE overflow resamples the complete episode;
semantic overflow fails. By default, checkpoints include processor/tokenizer, PEFT, separate
END_FIX rows, P_E, optimizer, scheduler, RNG/sampler state, resolved config, and
split identity. Checkpoint component flags control saved artifacts; incomplete exports cannot reconstruct the full model. Resume restores optimizer moments, RNG and history, while applying the resolved optimizer hyperparameters and schedule. Model/token structure and optimizer implementation must remain compatible.

### Epoch validation and logging

The sampler draws episodes with replacement, so epochs are explicitly bounded by
optimizer steps. Set `training.steps_per_epoch` or `--steps-per-epoch`; the example
above defines ten epochs of 100 optimizer steps each. Set `training.max_steps`, or `training.epochs` together with `steps_per_epoch`. An explicit max-steps horizon takes precedence and may end in a partial epoch. This changes neither subject/K/query sampling nor accumulation.
`evaluation.strategy: epoch` validates at the end of **every** epoch, before the
epoch checkpoint is saved when `checkpoint.save_at_epoch_end` is enabled. `evaluation.strategy` also supports `steps` with `evaluation.every_steps`, or the quoted string `"no"` to disable evaluation.

Validation uses the full seen-subject validation query set at `evaluation.k_values` (default 1/5/10), with frozen same-subject train supports from the configured manifest. It never resamples/truncates frozen
episodes or uses unseen subjects for validation. No gradients or updates occur;
model/projector train/eval modes are restored even if validation fails.

The human-readable epoch summary and `trainer_log.jsonl` expose:

- `eval_where`: existing query-response WHERE mean NLL;
- `eval_what`, `eval_why`, `eval_how`: diagnostic token-NLL means for the respective
  sections of the **same single teacher-forced flat response**;
- `eval_flat`: existing full flat response-token mean NLL;
- `eval_total`: `lambda_where * eval_where + lambda_sem * eval_flat`.

Section diagnostics include ordinary line prefixes and their trailing newlines;
native EOS belongs to HOW. A tokenizer token spanning a section boundary is
assigned by its starting character. All supervised flat tokens are counted exactly
once. These diagnostics do not add semantic forwards, rebalance the flat loss,
or define generation-quality metrics. All reported epoch losses are means of
per-episode losses, with additional per-K summaries in `eval_by_k`. No automatic
best-checkpoint selection is introduced.

Each checkpoint includes `trainer_state.json` with the complete structured
`log_history`. Resume carries that history into the new run's JSONL log and resumes
the absolute epoch schedule without repeating completed epoch evaluations.

### Epoch autoregressive predictions

With the reference prediction settings, every evaluation generates:

- Train: exactly one batch of `per_device_train_batch_size` episodes, reusing the
  first batch of accepted episodes from that epoch's final optimizer step. Later
  gradient-accumulation batches are excluded. There are no extra sampler draws.
- Validation: every seen query at K=1/5/10 with the same frozen supports as loss
  validation. No sampling, truncation, or cap is applied to validation coverage.

Both responses use native HF autoregressive generation separately: WHERE receives
no query GT coordinates/durations; semantic output uses states from GT WHERE and
gold WHY groups, as required by the primary inference contract. Predicted WHERE
is never passed to semantic inference. Generation preserves parameters, gradients,
module modes and training RNG. Context overflow fails without skipping a query.

Declare `evaluation.predictions.semantic_max_new_tokens` or pass
`--semantic-max-new-tokens`; the example 512 above is an explicit runtime choice,
not a canonical default. The resolved choice is saved and may change on resume.
The WHERE budget retains its existing tokenizer-measured policy.

Each epoch writes `predictions/epoch-0001/{train,validation}.jsonl`. Each line has
query/subject/K, ordered support IDs, split identity, and `WHERE: {GT, PRED}` and
`SEMANTIC: {GT, PRED}`, plus existing parser diagnostics and conditioning provenance.
Raw malformed predictions are preserved. Files are streamed through `.partial`
paths and renamed on successful completion. A failed generation emits no success
event or completed epoch checkpoint. Text is stored in full in the files; the
console prints counts and paths:

```text
Epoch 1 | step 100 | autoregressive predictions:
  train: 1 batch (1 queries) -> .../predictions/epoch-0001/train.jsonl
  validation: 2 seen queries x K=1,5,10 (6 episodes) -> .../predictions/epoch-0001/validation.jsonl
  Each record: WHERE GT/PRED; SEMANTIC GT/PRED. WHERE is free-running; semantic uses GT WHERE states + gold WHY groups.
```

The counts above illustrate a two-query fixture, not the benchmark size.
The `epoch_predictions` event in `trainer_log.jsonl` and checkpoint
`trainer_state.json` records counts, absolute file paths, budget and conditioning.
Prediction files remain in the run directory; retain that directory alongside
checkpoints when moving runs. Resume retains earlier history and generates only
new epochs. Training losses and the existing `epoch_validation` event are unchanged.

Configuration follows `YAML -> missing-field defaults -> CLI overrides -> validation -> runtime`.
`configs/defaults.yaml` supplies only absent fields; explicit `null` values are never
silently replaced. The reference experiment now declares its optimizer engineering
values explicitly. `max_grad_norm: null` disables clipping; unresolved required
values fail with a field-specific error. Each run saves `resolved_config.json`.
`runtime.output_dir`, `logging.every_steps`, `logging.filename`, checkpoint save
cadence/components, and prediction batch count/scope can all be set in YAML.
Prediction `train_batches: 0` and `validation_scope: none` disable generation without
disabling loss validation. Multiple train batches reuse accepted training episodes,
without extra sampling. Step-based predictions use distinct `predictions/step-*`
directories. Batch size and accumulation contribute their product to the optimizer
step's episode mean; episodes run sequentially to retain their multimodal contexts.

See [the configuration audit](documents/CONFIGURATION_AUDIT.md) for the complete
field-to-consumer inventory, removed enforcement, and remaining implementation limits.

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
