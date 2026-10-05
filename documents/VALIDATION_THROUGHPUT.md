# Validation throughput implementation and server acceptance

Status: validation batching implemented and locally verified; pending A100 throughput validation.
The audit was saved in `runs/validation_throughput_audit.md` before any validation behavior changes.
The current persisted split has **1751** eligible seen-subject queries and **5253** episodes for K=1,5,10.
Counts are checked from the loaded checkpoint's split and manifest, not hard-coded.

## Implementation

Validation schedules the existing `WhereBatch` representation and calls `forward_where_batch`,
the shared projector, and `forward_flat_batch`. It never calls the training backward/optimizer wrapper.
Semantic component diagnostics partition the existing selected-position NLL using the unchanged
character-offset definitions. Native full semantic logits/duplicate CE remain only in the explicit
serial reference used for server acceptance. Every episode has a K/query identifier and contributes
equally, including the last short batch. K and support selection are never randomized.

Optional stable length windows use actual native WHERE token lengths. Both semantic and WHERE
padding, canonical-order padding, and actual physical sizes are recorded by the benchmark. Subject
boundaries do not restrict a physical batch. Configured windows are bounded; prediction file order
remains canonical even when internal execution is reordered.

The invocation-local bounded cache reuses native image preprocessing and frozen native projected
visual features. Text/image expansion still runs through the original processor. It stores independent
feature rows, not views retaining whole old batch allocations. It never stores LM states or prefix KV.
WHERE query features feed the existing semantic reuse path. Caches are closed before training resumes.
Support text serialization/tokenization still occurs per episode; this is CPU preprocessing, not serial
full-model execution. Profile `validation_collation` before considering additional text-prefix reuse.

Prediction now batches same-K WHERE and semantic generation in both epoch output and `evaluate_flat.py`.
WHERE subgroups retain exactly the original count-dependent max-new-token budget. This can limit
physical WHERE generation batch size when oracle lengths differ. Generation uses left padding,
greedy decoding, the same EOS, native KV cache, and unchanged parsers. Semantic generation still uses
GT teacher-forced WHERE states and gold WHY groups. No stochastic decoding mode was added.

## Configuration

`configs/defaults.yaml` contains independent `validation.loss` and `validation.prediction` controls:
`execution`, `batch_size_by_k`, `bucket_by_length`, and `bucket_window`.
Defaults are B=1 for every K and bucketing off; these are conservative defaults, not A100 selections.
Cache controls are `frozen_visual_features`, `support_preprocessing`, `max_entries`, and `prefix_kv`.
Prefix KV must remain false until separate server evidence and parity justify implementing it.
`execution: serial` is the singleton optimized runtime path; benchmark `--mode serial` is the native
semantic reference. Old checkpoints receive missing configuration defaults on load.

## File-by-file changes

All rows preserve scientific behavior (no change to workload, targets, loss weights, component spans,
state semantics, support order, or decoding settings). Floating-point kernel parity still needs A100 verification.

| File | Why / change | Ownership |
|---|---|---|
| `.gitignore` | Keep the mandatory audit available in Git while other run artifacts remain ignored | Artifact plumbing |
| `runs/validation_throughput_audit.md` | Pre-implementation call graph, reusable contracts and bottlenecks | Audit |
| `configs/defaults.yaml` | Conservative independent loss/prediction K sizes and bounded cache settings | Validation configuration |
| `semgaze/model/config.py` | Validate settings and normalize YAML integer/JSON string K keys | Shared configuration |
| `semgaze/evaluation/batching.py` | Deterministic complete same-K scheduling, optional native-length windows | Validation scheduling over shared packing |
| `semgaze/evaluation/validation.py` | Shared physical forwards, one semantic NLL decomposition, exact accumulator, mode restoration, native serial reference | Validation execution/metrics |
| `semgaze/model/visual_cache.py` | Bounded native preprocessing and frozen feature reuse | Optional inference cache |
| `semgaze/where/collator.py` | Optional native processor cache facade; original batch format unchanged | Shared helper |
| `semgaze/where/forward.py` | Optional visual reuse and explicit inference use_cache override | Shared helper; defaults retain training graph |
| `semgaze/semantic/flat/forward.py` | Expose diagnostic offsets only when requested; generalize preparation to generation; explicit inference reuse control | Shared helper |
| `semgaze/where/generation.py` | Same-K generation, budget subgroups, left padding, shared image packing | Prediction |
| `semgaze/evaluation/flat.py` | Batched GT-state extraction and semantic generation using shared insertion | Prediction |
| `semgaze/evaluation/predictions.py` | Batch execution with canonical file order, RNG/mode restoration | Prediction scheduling |
| `evaluate_flat.py` | Use batch generation for standalone validation/test export; allow validation overrides | Evaluation CLI |
| `scripts/benchmark_validation_a100.py` | CUDA-only serial/batched quick/full runs, parity, fresh-process sweeps, OOM reporting, optional generation parity | Server tooling; reuses training report/profiler utilities |
| `tests/test_validation_throughput.py` | Real small-HF same-K parity, variable lengths/states, cache/generation checks, exact short-batch metrics | CPU correctness |
| `tests/test_benchmark_validation.py` | Identity/parity rejection, padding, OOM candidate selection, CPU rejection and mocked report wiring | Tool correctness |
| `tests/test_epoch_validation.py` | Point execution mocks at the shared batch path; retain explicit serial prediction regression | Existing regressions |
| `tests/test_epoch_predictions.py` | Explicitly select retained serial reference for singleton API tests | Existing regressions |
| `tests/test_config_plumbing.py` | Batch entrypoint mocks and standalone config propagation | Existing regressions |

## Server commands (Bash, from repository root)

Use the same **complete SemGaze checkpoint** for every run (processor, adapter, rows/projector, metadata,
resolved config). Set the path once. The optional `--config` supplies validation controls only; the
checkpoint owns all scientific settings, model, tokenizer, processor, data, loss weights and precision.
Keep the source checkout and environment identical across comparisons. Output directories must be empty.

```bash
export CHECKPOINT=/absolute/path/to/semgaze/checkpoint
```

Quick serial reference (deterministic length-spread queries from every seen subject):

```bash
python scripts/benchmark_validation_a100.py --checkpoint "$CHECKPOINT" --config configs/flat_throughput.yaml --mode serial --scope quick --cache off --output-dir runs/validation-serial-quick
```

Quick physical batching. These sizes are test candidates, not final recommendations:

```bash
python scripts/benchmark_validation_a100.py --checkpoint "$CHECKPOINT" --config configs/flat_throughput.yaml --mode same-k-batched --scope quick --batch-k1 4 --batch-k5 2 --batch-k10 2 --cache on --bucket --output-dir runs/validation-batched-quick
python scripts/benchmark_validation_a100.py --compare runs/validation-serial-quick/serial_quick.json runs/validation-batched-quick/batched_quick.json --tolerance 0.002 --output-dir runs/validation-parity-quick
```

The tolerance is an explicit acceptance threshold for per-episode **absolute** metric differences.
All six metrics and state/support identity must pass. Do not proceed with sweeps on a parity failure;
inspect its episode records rather than merely increasing tolerance. If a candidate OOMs, the report
identifies its K/B; explicitly choose smaller candidates and new output paths, without reducing query coverage.

Run each sweep in fresh processes. Full-scope sweeps cover all 1751 queries at the selected K:

```bash
python scripts/benchmark_validation_a100.py --checkpoint "$CHECKPOINT" --config configs/flat_throughput.yaml --mode same-k-batched --scope full --k 1 --sweep 1,2,4,8 --cache on --bucket --output-dir runs/validation-sweep-k1
python scripts/benchmark_validation_a100.py --checkpoint "$CHECKPOINT" --config configs/flat_throughput.yaml --mode same-k-batched --scope full --k 5 --sweep 1,2,4 --cache on --bucket --output-dir runs/validation-sweep-k5
python scripts/benchmark_validation_a100.py --checkpoint "$CHECKPOINT" --config configs/flat_throughput.yaml --mode same-k-batched --scope full --k 10 --sweep 1,2 --cache on --bucket --output-dir runs/validation-sweep-k10
```

Each sweep picks the highest measured throughput among completed candidates with peak reserved memory
at most 90% of device capacity (`--memory-fraction` controls this explicit policy). OOMs remain in the
report and never silently trigger smaller batches. Selection fails if no useful candidate remains.
Read the measured choices and require all three before full validation:

```bash
export B1=$(python -c 'import json; x=json.load(open("runs/validation-sweep-k1/sweep.json"))["chosen_batch_size"]; assert x is not None; print(x)')
export B5=$(python -c 'import json; x=json.load(open("runs/validation-sweep-k5/sweep.json"))["chosen_batch_size"]; assert x is not None; print(x)')
export B10=$(python -c 'import json; x=json.load(open("runs/validation-sweep-k10/sweep.json"))["chosen_batch_size"]; assert x is not None; print(x)')
test -n "$B1" && test -n "$B5" && test -n "$B10"
```

Complete native reference and optimized validation, with full episode-by-episode comparison:

```bash
python scripts/benchmark_validation_a100.py --checkpoint "$CHECKPOINT" --config configs/flat_throughput.yaml --mode serial --scope full --cache off --output-dir runs/validation-serial-full
python scripts/benchmark_validation_a100.py --checkpoint "$CHECKPOINT" --config configs/flat_throughput.yaml --mode same-k-batched --scope full --batch-k1 "$B1" --batch-k5 "$B5" --batch-k10 "$B10" --cache on --bucket --reference runs/validation-serial-full/serial_full.json --tolerance 0.002 --output-dir runs/validation-batched-full
```

Cache and padding controls can be compared without changing the workload:

```bash
python scripts/benchmark_validation_a100.py --checkpoint "$CHECKPOINT" --scope quick --batch-k1 4 --batch-k5 2 --batch-k10 2 --cache off --bucket --reference runs/validation-serial-quick/serial_quick.json --output-dir runs/validation-cache-off-quick
python scripts/benchmark_validation_a100.py --compare runs/validation-cache-off-quick/report.json runs/validation-batched-quick/report.json --output-dir runs/validation-cache-comparison
python scripts/benchmark_validation_a100.py --checkpoint "$CHECKPOINT" --scope quick --batch-k1 4 --batch-k5 2 --batch-k10 2 --cache on --reference runs/validation-serial-quick/serial_quick.json --output-dir runs/validation-no-bucket-quick
python scripts/benchmark_validation_a100.py --checkpoint "$CHECKPOINT" --scope quick --batch-k1 4 --batch-k5 2 --batch-k10 2 --cache on --bucket --profile --output-dir runs/validation-profile-quick
```

The profiler synchronizes at explicit stage boundaries and can perturb throughput. Use unprofiled runs
for batch selection; use `stage_times` for identifying remaining costs. Serial semantic reference is
timed end-to-end and intentionally retains the original native CE/diagnostic overhead.

Generation parity uses a separately specified semantic budget (the project has no default). Set it
to the exact budget used by your run; this command times loss validation first, then compares serial
and batched greedy prediction outside the measured interval:

```bash
export SEMANTIC_BUDGET=$(python -c 'import json,os; x=json.load(open(os.path.join(os.environ["CHECKPOINT"],"resolved_config.json")))["evaluation"]["predictions"]["semantic_max_new_tokens"]; assert isinstance(x,int) and x>0, "Set your explicit semantic decoding budget"; print(x)')
python scripts/benchmark_validation_a100.py --checkpoint "$CHECKPOINT" --scope quick --batch-k1 "$B1" --batch-k5 "$B5" --batch-k10 "$B10" --cache on --bucket --prediction-parity-budget "$SEMANTIC_BUDGET" --output-dir runs/validation-generation-parity
```

## Expected artifacts

- Every run: `<output-dir>/report.json`, `<output-dir>/report.md`.
- Successful quick runs: `runs/validation-serial-quick/serial_quick.json`, `runs/validation-batched-quick/batched_quick.json`.
- Quick comparison: `runs/validation-parity-quick/parity_quick.json`, `runs/validation-parity-quick/parity_quick.md`.
- Sweeps: `runs/validation-sweep-k{1,5,10}/sweep.json`, `sweep.md`, and per-candidate `k<K>-b<B>/report.json`/`report.md`.
- Full runs: `runs/validation-serial-full/serial_full.json`, `runs/validation-batched-full/batched_full.json`, plus each directory's reports.
- Full comparison: `runs/validation-batched-full/parity_quick.json`/`parity_quick.md` (the stable comparison filename; its episode count records full scope).
- Generation comparison: `runs/validation-generation-parity/prediction_parity.json`, `prediction_parity.md`.
- Failed runs: `report.json`/`report.md` with status/error and K/requested B; successful aliases are not emitted.

Reports include checkpoint content hash, manifest and scientific-config identity, source/environment,
GPU/dtype/backend, complete workload, physical sizes, real/padded tokens, canonical-vs-bucketed padding,
time/rates by K, allocated/reserved CUDA peaks, cache entries/hits/misses/rates and per-episode metrics.
Comparison reports include measured overall/per-K timing ratios. Sweep reports contain measured choices.

## Local verification and pending server work

CPU tests use a real small HF InternVL/PEFT model plus synthetic isolation fixtures. They verify source,
tensor, mask, selected-NLL, component, state/image isolation, generation, mode/RNG restoration, training
gradient and checkpoint contracts. CUDA telemetry in the report-wiring test is explicitly mocked; it is
not a performance measurement. Local dependencies were installed only under `runs/validation-test-deps`.

Final local full-suite result: **134 passed, 0 skipped, 0 failed**. The suite emitted 301 NumPy
`__array_wrap__` deprecation warnings from the installed dependency stack. The complete output is
`runs/validation-local-tests-final.txt`. Python compilation and Git whitespace checks also pass.

Run the local suite with:

```powershell
$env:PYTHONPATH='runs/validation-test-deps'
python -m pytest -q --basetemp runs/pytest-validation-final
```

The local environment has CPU-only Torch 2.11.0, so production CUDA kernels, A100 throughput/peak
memory, useful B1/B5/B10, production greedy parity, cache speedup, full wall-time reduction and final
full-split numerical parity remain pending server execution. No production batch sizes are selected
locally. Prefix KV caching remains unimplemented as required until profiling justifies it.
