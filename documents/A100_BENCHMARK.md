> Historical implementation/performance report. Split, sampling, schedule and evaluation examples below predate the 95/5 query-coverage migration. Use [COCO_955_MIGRATION.md](COCO_955_MIGRATION.md) and the current YAML/CLI for new runs. Historical measurements are not evidence for current sampling throughput.

# Server-side SemGaze A100 same-K benchmark

`scripts/benchmark_a100.py` runs the production SemGaze pipeline on real training
episodes. It has not been run or performance-validated on an A100 locally.

## Run on the training server

From the repository root, activate the prepared server Python environment with
the pinned project dependencies and CUDA-enabled PyTorch. Install the configured
model/adapter/tokenizer payloads, persisted COCO-Search18 splits, curated source
referenced by the split manifest, and real images.

Compare B=1 baseline, B=2 same-K, and B=4 same-K if it fits, with checkpointing ON:

```bash
python scripts/benchmark_a100.py \
  --config configs/flat_throughput.yaml \
  --output-dir runs/a100-same-k \
  --effective-batch-size 4 \
  --include-b4 \
  --warmup-steps 3 \
  --steps 30
```

Use a new or empty output directory. Append `--include-checkpointing-off` to also
attempt all three sizes with checkpointing OFF. Omit `--include-b4` to test only
B=1/B=2. `--effective-batch-size` must be divisible by the largest requested B;
if omitted, it defaults to configured B times accumulation. B=4 OOM is recorded
without losing B=1/B=2 results. Any attempted failure causes exit code 1 after
results are saved; it does not prevent subsequent variants from running.

Use `--device cuda:1` for another visible GPU. Run directly, not with `torchrun`.
For data stored elsewhere append, for example:

```bash
  --images-root /data/COCO_Search18/images --split-root /data/COCO_Search18/split/all
```

The config must select `OpenGVLab/InternVL3_5-8B-HF` and bf16. **Source discrepancy:**
`configs/flat_throughput.yaml` currently selects `initialization_adapter: null` /
`adapter_load_mode: fresh`, despite its old header describing a released adapter.
The benchmark preserves the selected initialization. To measure an existing
released DeepGaze production run, pass that production configuration with
`--config`; no model initialization or optimizer setting is changed here.

## Focused batch-construction audit and refinement

The previous `sample_optimizer_batches` independently sampled every episode,
sorted the whole optimizer minibatch by K and coarse length, then cut it into
physical batches. A K group whose size was not divisible by B could leave a
mixed-K chunk. Sorting did not guarantee homogeneous K.

Training now draws its first episode normally, using the existing subject-then-K
sampling order and configured K probabilities. It holds that K for the remaining
episodes in the physical batch, retaining conditional subject/query/support
sampling and support order. Context-overflow retries keep the selected K,
including when the first episode overflows. The existing retry bound still
fails clearly if that K cannot fit. Each new physical batch draws K again.
Zero-probability K values remain excluded. Length sorting is confined to a same-K
batch, and no episodes move across independently drawn K groups or updates.

The requested within-batch K correlation changes B>1 seeded trajectories; it
preserves the configured K marginal distribution and conditional episode rules.
B=1 retains its original RNG order when no context overflow occurs. Holding K on
retries prevents context rejection from preferentially resampling smaller K.
The sampler checkpoint remains its original Python RNG state, with no new queues.
Model, precision, losses, gradient paths and optimizer logic are unchanged. Generic
collation/forward helpers still support mixed K for mathematical parity tests.

## Comparable real workloads

The benchmark loads the same persisted training split and normalization as
`train_flat.py`. K is random with the configured probabilities; there are no
forced quotas, reduced resolutions, or OOM-driven changes to episode content.
Actual query/support IDs, rejection counts and K histograms are saved.

To avoid confounding throughput with different episode draws, all variants use
**the same same-K groups of the largest requested B**. For the command above,
real episodes are sampled in groups of four, sorted within those groups, then
split into physical batches of 1, 2 or 4. The production batch builder exposes
this benchmark-only grouping override. Default production training draws K once
per physical batch; in the comparison, adjacent smaller batches share a group's
K. This is a controlled workload replay, not a claim that production B=1 draws
four correlated K values. B=1 is the current source's singleton baseline, not a
historical pre-Section-A implementation.

| Physical B | Accumulation | Episodes per optimizer update |
|---|---:|---:|
| 1 | 4 | 4 |
| 2 | 2 | 4 |
| 4 | 1 | 4 |

All variants start from the same configured seed/initial model in fresh worker
processes. The script checks accepted query/support IDs and order at common
optimizer steps, dataset identity, and model provenance. Mismatches cause a
nonzero exit. Dropout RNG assignment can differ by batch size, so training
trajectories are not claimed bitwise identical. No dropout, attention backend,
TF32, optimizer, loss weight, cache, or allocator setting is tuned.

Thirty measured updates process 120 real episodes after 12 warmup episodes.
Increase `--steps` for more K coverage; absent configured K values are reported.
The configured training scheduler horizon is preserved; when no horizon is
configured, warmup plus measured steps supplies it. A benchmark longer than an
explicitly configured horizon is rejected.

## Timing, VRAM and padding definitions

- `episodes_per_second` is completed measured episodes divided by total measured
  optimizer-step wall time. `seconds_per_episode` is its reciprocal. Step times
  include sampling, CPU preprocessing, transfers, all accumulated physical
  WHERE/semantic forwards and backward passes, clipping, optimizer and scheduler.
  Model/data loading, report writing and console output are excluded.
- Stage timings include WHERE forward, semantic preparation/forward, backward,
  clipping, optimizer and scheduler, summed per optimizer update. Mean, median
  and p95 total step times are reported. Nested stage times are inclusive.
- CUDA synchronization occurs only at explicit profiling boundaries. This
  instrumentation can reduce natural overlap. Each variant has a fresh process
  and allocator; memory peaks reset after loading and after warmup. There are no
  per-step `empty_cache()` calls.
- Peak allocated/reserved bytes cover the measured window, including model,
  optimizer and activations. Loading/warmup snapshots are separate. Per-step
  phase peaks are cumulative within that phase. Reserved bytes include cached
  allocator blocks from warmup.
- For each WHERE/semantic physical forward, token slots are B times that batch's
  maximum sequence length. Padding tokens are slots minus valid sequence lengths;
  padding fraction is total padding divided by total slots, weighted by tokens.
  B=1 has zero padding. These describe sequence padding, not image feature memory.
- WHERE and semantic statistics include token totals, padded slots, padding count
  and fraction, unpadded length min/mean/median/p95/max, and mean/max padded length.
  Semantic lengths include inserted fixation states. Statistics appear per step
  and across completed measured steps, alongside raw per-episode lengths and K.
- Warmup really trains using the existing semantic-to-WHERE gradient path.
  Diagnostics follow the production first-step/periodic policy. Gradient clipping
  and aggregate non-finite checks stay enabled at each optimizer update.

## Results to send back

Send **`runs/a100-same-k/results.json` and `runs/a100-same-k/results.csv`**.
JSON contains resolved configs, machine/GPU/driver/library information, source
hash and revision, attention backends, model/adapter provenance, optimizer state
size, dataset identity, times, VRAM, padding statistics, query/support identities,
K, image/fixation counts and supervised token counts. CSV contains one summary
row per variant plus raw length/count arrays. Times are seconds, VRAM is bytes,
and padding fractions range from 0 to 1.

Each variant also has its request, result, console log and model preflight files.
Results persist after every completed update and variant. Status is `ok`, `oom`,
`failed` or `not_run`. A killed process is not automatically labeled CUDA OOM.
Partial rates are marked; measured peaks may include the failed step. Warmup
failure leaves measured metrics null and reports failure memory separately.
Send the variant log too if it failed. GPU identity is recorded rather than
assuming the server actually provides an A100 40GB.

For an unmeasured output plan without loading data, model or CUDA:

```bash
python scripts/benchmark_a100.py --plan-only \
  --include-b4 --effective-batch-size 4 \
  --output-dir runs/a100-same-k-plan
```

Example JSON/CSV under `documents/a100_benchmark_example/` are explicitly
`plan_only`, `not_run` templates with null measurements. Local tests cover
sampling, workload identity, padding arithmetic, OOM isolation and CPU small-HF
loss/gradient/batching regressions. They do not validate A100 performance or
capacity, or the real 8B model's target-hardware behavior.

Local verification for this refinement: **52 passed** (CPU), including existing
fp32/bf16 loss/state/gradient parity, episode-mean batching equivalence,
semantic-to-WHERE gradients, sampling and RNG resume, B=1/2/4 orchestration,
padding calculations and failure persistence. The first attempt hit Windows
temporary-directory permissions; the run below used a workspace-local directory:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_same_k_batching.py tests/test_benchmark_a100.py tests/test_training_throughput.py tests/test_epoch_validation.py tests/test_epoch_predictions.py -q --basetemp=D:/Programming/Python/SemGaze/runs/pytest-same-k-20261005 --tb=short
```

The B=1/2/4 `--plan-only` command also passed. No A100 measurements are available
locally; throughput and VRAM remain unmeasured until the server run.
