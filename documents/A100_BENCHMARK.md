# Server-side SemGaze A100 benchmark

`scripts/benchmark_a100.py` is a standalone command in this repository. It imports
the production SemGaze pipeline, not the old fixture profiler or test utilities.
The benchmark has not been run or performance-validated on an A100 locally.

## Run on the training server

Run these commands from the repository root, using the server's prepared Python
environment with the pinned project dependencies and its CUDA-enabled PyTorch
build. The released adapter/tokenizer payloads, base model or HF cache, persisted
COCO-Search18 splits, curated source used by the split manifest, and real images
must be installed. The ordinary production loader checks the released adapter
payload, tokenizer compatibility, loaded LoRA tensors, bf16 and frozen modules.
The benchmark fails explicitly if those prerequisites are missing.

Required B=1/B=2, gradient checkpointing ON:

```bash
python scripts/benchmark_a100.py \
  --config configs/flat_throughput.yaml \
  --output-dir runs/a100-benchmark-on \
  --effective-batch-size 2 \
  --warmup-steps 3 \
  --steps 30
```

Required variants plus optional checkpointing OFF attempts:

```bash
python scripts/benchmark_a100.py \
  --config configs/flat_throughput.yaml \
  --output-dir runs/a100-benchmark-all \
  --effective-batch-size 2 \
  --warmup-steps 3 \
  --steps 30 \
  --include-checkpointing-off
```

Choose one command; the second includes the first command's variants. Use a new
output directory for each run. The script refuses to overwrite earlier results.
Use `--device cuda:1` to select another visible GPU. Do not launch with `torchrun`.
If the server stores data elsewhere, append explicit path overrides, for example:

```bash
  --images-root /data/COCO_Search18/images --split-root /data/COCO_Search18/split/all
```

Otherwise these paths come from your config. A custom `--config` must still select
`OpenGVLab/InternVL3_5-8B-HF`, bf16, and the released DeepGaze adapter with
`adapter_load_mode: continue_trainable`. Keep the verified annotation coordinate
frame in that configuration. No dataset files or model/checkpoint weights are
written by the benchmark.

## Comparable real workloads

The script reads and validates the same persisted training split as `train_flat.py`,
normalizes real records with `CocoSearch18Adapter`, and uses the unchanged
`TrainingEpisodeSampler` and `sample_optimizer_batches`. It does not use fixture
episodes, force K quotas, reduce image resolution, or remove long episodes to
avoid OOM. The production WHERE context-overflow rejection/resampling rule remains
active and its rejection count is recorded. Other errors, including semantic
overflow, fail the variant instead of silently changing the workload.

The explicit effective optimizer batch of 2 means:

| Physical episode batch | Accumulation | Episodes per optimizer update |
|---|---:|---:|
| B=1 | 2 | 2 |
| B=2 | 1 | 2 |

Both variants sample the same ordered episode stream with the configured seed.
Their optimizer minibatch membership, schedule horizon, warmup updates and
measured updates are identical. Existing within-minibatch K/length grouping is
preserved. Batching still changes the assignment of dropout RNG draws; the
benchmark does not disable dropout or claim bitwise-identical training trajectories.
An explicitly configured training horizon is used for the scheduler unchanged;
otherwise the benchmark's warmup + measured step count supplies the required
horizon. The selected horizon is saved. The script rejects a benchmark longer
than an explicitly configured training horizon.

The default run measures 60 real episodes per variant, after six warmup episodes.
Use more `--steps` for broader workload coverage. Actual K histograms and any
configured K values absent from the measured sample are saved; the script never
resamples to manufacture coverage. Omit `--effective-batch-size` to use the
configured B*accumulation, which must be divisible by 2. Any explicit override is
recorded as a benchmark setting.

Each variant is a **fresh subprocess** that loads the same configured initial
model, released LoRA, trainable END_FIX rows, and P_E with the same initialization
seed. Optimizer state and the CUDA allocator are isolated between variants. The
script compares accepted query/support IDs and their order at every common
optimizer step, along with dataset and model provenance. A mismatch is reported
and the command exits nonzero. No attention kernel, TF32 setting, allocator
configuration, optimizer, dropout probability, loss weight, or cache setting is
tuned by this script.

## Timing and memory definitions

- Every measured step runs production collation, physical WHERE and semantic
  forwards, the joint backward, aggregate finite-gradient checking/clipping,
  `optimizer.step()`, and `scheduler.step()`.
- Throughput is **completed measured episodes / sum of measured optimizer-step
  wall times**. Seconds/episode is its reciprocal. Step time includes sampling,
  CPU preprocessing, transfers and all accumulated physical batches. Model/data
  loading, report serialization and console logging are outside this window.
- `where_forward_mean_sec` and `semantic_forward_mean_sec` time their existing
  backbone-and-loss stage boundaries, summed over the physical microbatches in
  an optimizer step, then averaged over completed measured steps. Backward and
  optimizer-step metrics use the corresponding explicit boundaries. Sampling,
  semantic preparation, clipping and scheduler timings are saved separately.
- CUDA synchronization occurs only at profiling boundaries. This is an
  instrumented benchmark: boundary synchronization can reduce natural overlap.
  Nested `where`/`semantic` parent timings are inclusive; do not add them to
  their child preparation/forward timings.
- Peaks reset after model loading and again after warmup, in each new process.
  `peak_memory_allocated_bytes` and `peak_memory_reserved_bytes` cover the measured
  window, including the live model, optimizer state and activations. Warmup and
  loading snapshots are separate. Per-step `phase_memory` peaks are cumulative
  within that phase, not independent per-step peaks. Reserved memory includes
  allocator blocks retained from warmup. There are no `empty_cache()` calls.
- Gradient diagnostics use the production policy: first warmup update and any
  configured periodic interval. Finite checking/clipping remains enabled each
  optimizer step. Warmup really trains; no synthetic input or frozen objective
  replaces the semantic-to-WHERE gradient path.

## Files to send back

Send **`results.json` and `results.csv`** from your output directory.

`results.json` contains the complete configuration, machine/GPU/driver/library
metadata, source hash and Git revision/status, actual attention backends,
checkpointed modules, model commit, adapter file hashes, optimizer state size,
dataset manifest identity, summary metrics, per-step times, query/support IDs,
K, both sequence lengths, image counts/order, fixation counts and supervised
token counts. It also reports whether the detected GPU is an A100 with roughly
40 GiB; it does not mislabel another GPU's results.

`results.csv` has one row per variant, with summary measurements, actual length
and count arrays, and failure status. Times are seconds and memory is bytes.

Each variant also has `request.json`, `result.json`, `console.log`, and the normal
model preflight/resolved-config files. Results are persisted after each completed
optimizer step and after each variant. Send the variant log too if it failed.

Status is `ok`, `oom`, `failed`, or `not_run`. An OOM in an optional OFF variant
does not prevent subsequent variants from running. A crashed/killed worker is
reported as failed with its exit code; GPU OOM is not inferred from a process
kill alone. Rates after a partial measured run are explicitly marked partial;
peaks may include the failed measured step. If warmup fails, measured metrics
stay null and the failure snapshot is reported separately. Any attempted failure
causes the overall command to exit nonzero **after saving the results**.

## Local verification and example format

Local verification covers CLI parsing, unchanged configuration fields, result
math/serialization, process failure handling, mocked CUDA OOM/peak-reset logic,
and B=1/B=2 orchestration on a small CPU HF/PEFT graph. These checks do not validate
A100 execution, capacity, speed, or released 8B behavior.

To inspect the output structure without using CUDA, data or model weights:

```bash
python scripts/benchmark_a100.py --plan-only --include-checkpointing-off \
  --output-dir runs/a100-benchmark-plan
```

The example files under `documents/a100_benchmark_example/` have `plan_only: true`,
`status: not_run`, and null performance values. They are templates, not measurements.
