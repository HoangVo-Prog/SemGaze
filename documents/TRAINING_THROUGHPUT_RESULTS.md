# Section A implementation and verification

Status: physical batching and exact-objective patches implemented. **A100 40GB
acceptance remains blocked**, not passed. This host has PyTorch 2.11.0+cpu, no
CUDA, and Git LFS pointers in place of the released adapter and tokenizer payloads.
The real-model preflight failure is saved in `profile_a100_blocked.json`.

## A. Audit findings

The mandatory pre-refactor audit is in `TRAINING_THROUGHPUT_AUDIT.md`, including
the call graph, shapes, batch restrictions, original allocation estimates and
source discrepancies. Confirmed problems were:

| Problem and source | Evidence / impact |
|---|---|
| `training/loop.py:run_training_loop` | B * accumulation calls to the B=1 step; no physical GPU batching |
| `where/forward.py:forward_where` and native HF conditional-generation forward | full [B,L,V] logits and all LM layer outputs despite query-only loss/readout |
| `model/trainable_tokens.py:RowHead.forward` | `cat` allocated another full logits tensor; HF CE then upcast to FP32 |
| `state/insertion.py:insert_states`, semantic preparation | explicit single-sample insertion and scalar prompt metadata |
| `training/flat_step.py:run_flat_training_step` | per-episode all-parameter finite scan plus repeated diagnostic reductions forced CUDA synchronization |
| WHERE/semantic native collation and forwards | repeated query decode, preprocessing and frozen visual encoding |

The extractor already supported variable N; its caller was the restriction.
Default YAML and the user-edited example selected fresh adapters; the required
continued-adapter configuration is now separately provided. No user adapter
selection was silently overwritten. Active tiny-model text and vision backends
are both SDPA; the real 8B backends could not be inspected at runtime.

## B. Changes implemented

All changes below preserve the scientific objective and trainable parameter names.
No model, precision, adapter tensor, LoRA architecture, supervision, loss weight,
sampling distribution, semantic architecture, or WHERE readout definition changed.

| Files | Implementation | Scientific behavior changed? |
|---|---|---|
| `model/selected_loss.py`, `model/trainable_tokens.py` | full-V logits on supervised shifted predictors only; token NLL -> episode mean -> batch mean; in-place replacement of END_FIX output column | No |
| `where/forward.py` | direct final native backbone tensor, all-layer LM outputs disabled; one WHERE forward per physical batch | No |
| `where/collator.py`, `state/extractor.py` | right-padded text, flattened image order with per-sample assertions, query-only variable-N states, host metadata/index discovery | No |
| `state/insertion.py`, `semantic/flat/forward.py` | differentiable insertion and right-padding, one semantic backbone forward, same selected loss helper | No |
| `training/flat_step.py`, `training/loop.py` | one backward per physical batch; accumulation scales by physical-batch count; first-step/debug gradient diagnostics; aggregate non-finite checking at every optimizer step | No |
| `training/batching.py` | stable K/coarse-length grouping of accepted samples inside the current optimizer minibatch only; sampler and support order unchanged | No; dropout RNG assignment and floating-point summation order may differ |
| WHERE collation / semantic preparation | decoded-image reuse within optimizer minibatch; query projected visual features reused only for frozen deterministic producers with identical preprocessed pixels | No |
| `model/build.py`, `model/checkpoint.py`, `training/profiling.py`, `scripts/profile_training.py` | actual backend/checkpoint/optimizer metadata; explicit stage timers and allocated/reserved memory profiling; original-path comparison reference | No |
| configs, README, smoke and tests | true physical-B semantics, conventional accumulation, checkpointing retained, separate released-adapter acceptance config | No |

`configs/flat_throughput.yaml` selects InternVL3_5-8B-HF, released DeepGaze LoRA,
bf16, B=2 and checkpointing. The old `output_hidden_states: true` checkpoint field
is accepted but no longer requests all LM outputs. No validation fan-out or
validation-throughput redesign was introduced; shared WHERE helpers also serve
existing validation callers. Generation APIs remain supported.

## C. Exact-semantics verification

Tests use real HF InternVL/Qwen3/vision layers, PEFT LoRA, RowEmbedding/RowHead,
and the native processor on a small random architecture. This is engineering
verification, not validation of released 8B weights.

- B=1 original-vs-selected WHERE, semantic and total loss parity: FP32 and bf16,
  checkpointing both on/off.
- Final selected fixation hidden-state parity: same counts, positions and order.
- Every trainable gradient compared, including early/middle/late LoRA, input and
  output END_FIX rows and P_E; representative gradients explicitly nonzero.
- FP32 tolerance: rtol=2e-4, atol=2e-6. Bf16: rtol=0.04, atol=0.002.
- Mixed K=1/2, N=4/3, differing WHERE/semantic lengths: batched branch loss equals
  the mean of independent episode losses. A separate unequal-token-count test
  explicitly rejects global-token-mean equivalence.
- Mixed B=2 gradient accumulation agrees with the original sequential episode
  mean; semantic-only losses backpropagate into both samples' WHERE states.
- Sample-specific insertion/padding checked; changing only B's image pixels
  leaves A's final fixation states bitwise unchanged.
- Hooks verify exactly two language forwards and one frozen vision forward per
  physical B=2 step; head inputs are [M,D], not [B,L,D].
- Sampling membership, support order and final sampler RNG state match exactly;
  sorting never crosses optimizer boundaries.
- B=2 optimizer/checkpoint round trip, existing tied/untied row tests, unchanged
  old vocabulary weights, validation/generation and native processor contracts.
- Contract smoke and Python compilation pass. Full released-model smoke is blocked
  by the same CUDA/LFS prerequisites as the A100 profile.

Full-suite command: `.venv/Scripts/python.exe -m pytest -q --basetemp=.cache/pytest-acceptance`.
**Final result: 88 passed, 177 NumPy deprecation warnings, 82.01 seconds.**
No tests were skipped in this prepared virtual environment. Contract smoke,
`compileall`, and `git diff --check` also passed. The original pre-patch suite was
74 passed / 2 failed; both failures depended on mutable example-config values and
were fixed by making those tests declare their intended cache/budget settings.

## D. Performance results

The pre-patch production-entrypoint CPU baseline is retained in
`profile_baseline_cpu.json` (1.859 episodes/s). Subsequent original-path reference
and optimized measurements use the same seeded tiny model, fixture, FP32, two CPU
threads, and non-reentrant checkpointing. Per episode: K=1, WHERE length 948,
semantic length 751, two WHERE images, four fixations, 59 supervised WHERE tokens,
161 supervised semantic tokens. V=400 in this engineering model.

Initial fixed-learning-rate measurements (2 warmup steps, 10 measured steps):

| Variant | Physical B | K | Peak allocated VRAM | Peak reserved VRAM | Episodes/s | Seconds/episode |
|---|---:|---:|---|---|---:|---:|
| Original reference | 1 | 1 | N/A: CPU | N/A: CPU | 2.008 | 0.498 |
| Optimized | 1 | 1 | N/A: CPU | N/A: CPU | 0.914 | 1.094 |
| Optimized | 2 | 1 | N/A: CPU | N/A: CPU | 3.153 | 0.317 |

The slower B=1 result was repeated with 30 measured steps: **1.826 episodes/s,
0.548 seconds/episode** (`profile_optimized_b1_repeat_cpu.json`). Both results are
retained; these measurements do not establish a B=1 speedup. Final profiles add
scheduler stepping and finer preparation/forward timing; see the final table below.

Final scheduler-inclusive profiles (same 2 warmup + 10 measured steps per variant):

| Variant | Physical B | K | Peak allocated VRAM | Peak reserved VRAM | Episodes/s | Seconds/episode |
|---|---:|---:|---|---|---:|---:|
| Original reference | 1 | 1 | N/A: CPU | N/A: CPU | 2.448 | 0.408 |
| Optimized | 1 | 1 | N/A: CPU | N/A: CPU | 2.183 | 0.458 |
| Optimized | 2 | 1 | N/A: CPU | N/A: CPU | 0.864 | 1.157 |

Raw reports are `profile_final_baseline_b1_cpu.json`,
`profile_final_optimized_b1_cpu.json`, and `profile_final_optimized_b2_cpu.json`.
The large variation between trials prevents a reliable CPU throughput improvement
claim. In particular the final B=2 run regressed; the earlier 3.153 result must not
be treated as established speedup. No additional tuning is justified by these
unstable tiny-model CPU timings.

These timings do not estimate real 8B/A100 speed. VRAM is unavailable, not zero.
Real-model B=1 old, B=1 optimized and B=2 optimized, checkpoint on/off and higher-B
capacity remain unmeasured. No claim that B=2 fits 40GB is made.

## E. Remaining bottlenecks and acceptance work

In the final measured CPU B=2 workload, backward accounted for 11.907/23.144
seconds (51.4%). WHERE collation was 2.872 seconds, WHERE forward+loss 3.732,
semantic preparation 1.550, semantic forward+loss 2.451, and optimizer/scheduler
0.497 seconds. These nested stages must not be summed with their parent timers.
The earlier B=2 trial also spent most time in backward (46.5%). The profiler
does not support ranking GPU bottlenecks on this host. Tiny optimizer state was
75,144 bytes; no optimizer-format change is justified by these data.

Required acceptance still needs the actual released adapter/tokenizer payloads,
base model, and a CUDA A100 40GB runtime. The provided profiler records real live
and reserved peaks, exact lengths/counts, backend and optimizer metadata. Run
the B=1/B=2 checkpoint matrix on the same saved representative workload before
choosing batch size or checkpoint settings. Broader feature caching, attention
kernel changes, preprocessing workers/prefetch, and allocator tuning were not
enabled without supporting target-hardware measurements. No scientific trade-off
was introduced to work around the missing hardware.
