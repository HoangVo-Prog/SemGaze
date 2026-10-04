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

`--with-model` loads the released adapter trainably and performs no optimizer step
unless `--step` is supplied. It saves resolved configuration and preflight provenance
under `runs/`. It fails on missing dependencies, LFS pointer files, incompatible
tokenizers/templates, wrong adapters, unsupported devices, or unintended trainables.

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

The adapter reads the existing `all` splits, verifies source/split checksums and
frozen manifest membership, maps `prediction.fixations/regions/how`, and preserves
`T` as milliseconds. It never recreates splits or trial filtering. Empty images
or unresolved frame metadata fail explicitly.

## Training and resume

```powershell
.venv/Scripts/python train_flat.py --config configs/flat_single.yaml --output-dir runs/experiment --max-steps 1000
.venv/Scripts/python train_flat.py --config configs/flat_single.yaml --output-dir runs/resumed --max-steps 1000 --resume runs/experiment/checkpoint-100
```

These are example horizons, not scientific defaults. Every training run executes
the fixture gradient gate before optimization. Episodes are accumulated sequentially
with an explicit episode mean. WHERE overflow resamples the complete episode;
semantic overflow fails. Checkpoints include processor/tokenizer, PEFT, separate
END_FIX rows, P_E, optimizer, scheduler, RNG/sampler state, resolved config, and
split identity. Resume preserves the scheduler horizon.

The unresolved optimizer engineering knobs resolve explicitly to Adam betas
`(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`, max gradient norm `1.0`, and
gradient checkpointing `false`. Resolutions are recorded under
`engineering_resolutions` in each run's `resolved_config.json`. Non-reentrant
gradient checkpointing is supported when explicitly enabled. Dataset coordinate
metadata and evaluation metric choices are not filled in by defaults.

## Evaluation

`evaluate_flat.py` loads a saved checkpoint, consumes every eligible query and
the frozen K=1/5/10 support sets, and exports predictions and draw-level validity
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
