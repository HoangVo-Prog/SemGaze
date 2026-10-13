# SemGaze no-attn — Fixation SS + Late-Fixation CE patch (v2, installer fix)

Source baseline: `HoangVo-Prog/SemGaze`, branch `no-attn`, commit `b2215a65aa41a1a9f2555dd41296789f235f31a9`.

## v2 hotfix for `where forward return: expected exactly one anchor, found 0`

- Corrected upstream's exact 23-space indentation in `semgaze/where/forward.py` return line (the first installer mistakenly required 24 spaces).
- All original source anchors in the installer were rechecked against the upstream `no-attn` source.
- Included missing `semgaze/semantic/flat/diagnostics.py` for WHAT/WHY/HOW logging.
- Repaired SS + LWCE coupling: response character offsets now remain aligned after the sampled fixation has been masked from CE.
- Six CPU weighted-loss/span-mask tests passed, and WHAT/WHY/HOW diagnostic partition smoke passed. GPU InternVL execution is still unverified.

**If you only ran `--dry-run`, no source files were modified.** Replace the extracted `semgaze_ss_lwce_patch` folder with this v2 folder, then rerun the exact same dry-run and apply commands. Do not apply the installer twice.


This package is a **source patch installer**, not a live modification of the user's GitHub repository. It rewrites seven existing files using exact anchored source replacements, and creates three helper modules, two test modules, and four full YAML configs. The installer is atomic in preparation: it validates all anchors and new-file collisions before writing anything. Make a Git branch before applying.

## Apply

```bash
git switch no-attn
git switch -c feat/fixation-ss-lwce
python /path/to/semgaze_ss_lwce_patch/apply_semgaze_patch.py --repo . --dry-run
python /path/to/semgaze_ss_lwce_patch/apply_semgaze_patch.py --repo .
python -m compileall -q semgaze/where semgaze/training semgaze/model train_flat.py
pytest -q tests/test_ss_lwce_patch.py tests/test_ss_policy.py tests/test_config_plumbing.py tests/test_model_path.py

git add -N semgaze/where/ss.py semgaze/where/weighting.py \
  semgaze/semantic/flat/diagnostics.py \
  tests/test_ss_lwce_patch.py tests/test_ss_policy.py \
  configs/flat_continue_sft.yaml configs/flat_ss_only.yaml \
  configs/flat_lwce_only.yaml configs/flat_ss_lwce.yaml
git diff --check
git diff > semgaze-ss-lwce.diff
```

## Run three matched adaptation experiments

The new `checkpoint_loading.init_from` config key is the preferred way to adapt from an SFT checkpoint. It initializes **LoRA + END_FIX rows + semantic projector weights**, but creates a fresh optimizer, scheduler and sampler at step 0. Use the same source checkpoint for all three.

1. Set `checkpoint_loading.init_from` in each YAML file to the actual path, e.g. `runs/2e/checkpoint-6050`.
2. Use identical `data.fewshot`, `training.num_train_epochs`, `training.learning_rate`, batch/accumulation and evaluation protocol across all experiment YAMLs.
3. Run `python train_flat.py --config configs/flat_continue_sft.yaml --output-dir runs/continued_sft_control`.
4. Run `python train_flat.py --config configs/flat_ss_only.yaml --output-dir runs/ss_only`.
5. Run `python train_flat.py --config configs/flat_lwce_only.yaml --output-dir runs/lwce_only`.
6. Optional combined: `python train_flat.py --config configs/flat_ss_lwce.yaml --output-dir runs/ss_lwce`.

`checkpoint_loading.resume_from` or `--resume` is **strict recovery of the same run**, restoring optimizer/scheduler/sampler/RNG; it rejects intervention flag changes. Use `--init-from-checkpoint` or YAML `init_from` to start an intervention from baseline weights instead. Both cannot be set simultaneously.

## Semantics and guardrails

- LWCE is per-episode normalized and affects WHERE only; punctuation/structural tokens outside a canonical complete fixation keep unit weight.
- SS chooses at most one fixation per selected episode, generates it causally with a GT prefix and greedy autoregressive decoding, and masks its predicted tokens from CE. Remaining GT labels are trained on the mixed history.
- Invalid generation / missing END_FIX / overflow falls back to normal teacher forcing for that episode, with failure reasons logged.
- Original GT WHERE states are used for semantic WHAT/WHY/HOW generation, preserving state-projector supervision.
- Logging: `loss_where`, `loss_where_gt`, `loss_where_gt_unweighted`, `loss_where_mixed`, `loss_what`, `loss_why`, `loss_how`, `loss_flat`, `loss_total`, `ss_eligible`, `ss_attempted`, `ss_accepted`, `ss_changed`, `ss_positions`, `ss_probability`, `ss_failure_reasons`, `lwce_enabled`, `lwce_alpha`, `ss_enabled`.
- `--resume` and `--init-from-checkpoint` can also override or supply the checkpoint selection at the command line.

## Validation caveats

Six CPU unit tests were executed successfully in a standalone module smoke harness; full InternVL forward/generation, preflight, checkpoint loading, and CUDA memory tests **have not been executed**. Run fixture smoke and a short GPU training smoke with `--max-steps` on your actual system before training full epochs. The sparse SS implementation is exact for its *single sampled replacement*, not a full multi-fixation stochastic rollout. It can increase step time and memory.

## CLI checkpoint examples

```bash
# New adapter/projector/END_FIX-weight adaptation from a common SFT checkpoint:
python train_flat.py --config configs/flat_ss_only.yaml \
  --init-from-checkpoint runs/2e/checkpoint-6050 --output-dir runs/adapt_ss

# Strict same-run resume after an interrupted SS adaptation:
python train_flat.py --config configs/flat_ss_only.yaml \
  --resume runs/adapt_ss/checkpoint-100 --output-dir runs/adapt_ss_resumed
```

Note: `--resume` does not add training steps beyond the original schedule.
The optimizer horizon (`num_train_epochs`, `max_steps`) must match the saved
run exactly. To start additional training from the saved weights, use
`--init-from-checkpoint` instead.
