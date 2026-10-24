# WHERE-only checkpoint -> Joint training (weight-only staged initialization)

The `--resume` and `--init-from-checkpoint` workflows have different contracts:

- **`--resume`** restores the complete optimizer, scheduler, sampler and RNG state
  and requires the original architecture, objectives and training schedule.
- **`--init-from-checkpoint`** restores checkpoint model weights in the SOURCE
  architecture first, adapts trainable rows/projector for the TARGET config,
  then starts a NEW training run at step 0 with fresh optimizer, scheduler and
  sampler. The newly configured number of epochs is additional training.

## Examples

### WHERE-only + END_FIX -> Joint + END_FIX (explicit semantic)

```bash
python -u train_flat.py \
  --config configs/flat_single.yaml \
  --init-from-checkpoint runs/intergaze_v3_where_only/checkpoint-STEP \
  --num-train-epochs 1 \
  --output-dir runs/joint_from_where_end_fix
```

### WHERE-only without END_FIX -> Joint without END_FIX

```bash
python -u train_flat.py \
  --config configs/ablation_architecture/no_end_fix_token.yaml \
  --init-from-checkpoint runs/intergaze_v3_where_only_no_end_fix/checkpoint-STEP \
  --num-train-epochs 1 \
  --output-dir runs/joint_from_where_no_end_fix
```

**Evaluation note:** the repository's `no_end_fix_token.yaml` currently uses
`evaluation.strategy: steps`; set it to `last` in a stage-two copy if you
want evaluation automatically at the end of one new epoch.

### WHERE-only (either END_FIX mode) -> Joint with shared-state semantic

Choose an appropriate validated shared-state YAML configuration, and ensure
`checkpoint.save_projector: true`. If shared states were **not** present in
stage one, the semantic projector is initialized from the new stage's seed.
Its weights do not come from WHERE-only. Even if the source used a
shared-state architecture with `lambda_sem: 0`, its untrained projector is
reset rather than misrepresented as learned.

## Compatibility contract

The source checkpoint is loaded with its own saved processor, vocabulary,
LoRA and trainable END_FIX rows. The split-manifest identity must match.
The following are allowed in the target:

- `lambda_where`, `lambda_sem`, learning rate, optimizer, batch/K distribution,
  epochs, scheduler, evaluation and logging options.
- `where.use_end_fix_token: true/false`, provided the corresponding
  `where.supervision.supervise_query_end_fix` and `state.readout_position`
  reflect the chosen mode.
- `semantic.conditioning` switches between `explicit_scanpath_text` and
  `shared_states`. A newly required projector starts from a fresh initialization.

Backbone ID, language LoRA geometry, frozen-module policy, END_FIX token string,
WHERE coordinate/duration codec, and experiment protocol must remain compatible.
The checkpoint's original `model` provenance is retained in the target run.

The previous training optimizer, scheduler, sampler, RNG and history are not
carried forward. The stage-two LR schedule starts from step 0, so **one new
stage-two epoch** is distinct from continuing the previous cosine scheduler.
No source checkpoint files are modified.

Every stage-two checkpoint stores `diagnostics.staged_initialization` in
`metadata.json` with source checkpoint, source step, both architectures,
original/new loss weights, and whether the projector was initialized anew or removed.
The same audit metadata is emitted to the new run's `preflight.json`.

## Sanity checks

1. For an explicit-semantic target, the projector is `Identity` and is not
   part of optimization; WHERE-only and Joint differ by semantic supervision.
2. For a shared-state target, the projector is `Linear + LayerNorm`; it is
   included in the optimizer whenever `lambda_sem > 0`.
3. For no-END_FIX targets, special-token rows remain frozen. For END_FIX
   targets, the restored rows are trainable (regardless of stage-one mode).
4. After the transition, inspect `preflight.json`, `resolved_config.json`,
   and the new `trainer_log.jsonl`; its first `train_step` is step 1.
5. Compare staged training to fresh-Joint with matched evaluation protocol,
   noting that stage-two learns from an additional earlier WHERE-only epoch.

CPU tests: `python -m pytest -q tests/test_staged_init.py`.
A full 8B checkpoint/GPU integration test is still necessary before relying
on large multi-stage experiments.
