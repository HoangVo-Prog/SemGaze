# AGENTS.md — SemGaze Validation Parity Root-Cause Audit

## Objective

Audit the current SemGaze validation pipeline and identify the **first concrete cause** of the parity failure between:

- serial validation, and
- same-K physically batched validation.

The current observed failure already includes a non-trivial difference in:

```text
eval_where max diff ~= 0.0241
```

Therefore, the investigation must start from the **WHERE path**.

Do **not** begin by auditing WHAT/WHY/HOW metric slicing, final aggregation, or tolerance settings unless the WHERE path has first been proven equivalent.

The main deliverable is a **server-runnable benchmark/report** that isolates the root cause with the smallest possible number of experiments.

Do not modify the training or validation scientific behavior yet.

---

# 1. Scope and Constraints

## Primary scope

Audit:

```text
serial validation
vs
same-K batched validation
```

with special attention to:

```text
WHERE forward
visual preprocessing/cache
visual feature cache
physical batching
padding / masks
per-episode WHERE loss
semantic path only after WHERE parity is established
```

## Files available for benchmark/instrumentation

Prefer modifying:

```text
scripts/benchmark_validation_a100.py
```

Only modify another script if absolutely necessary:

```text
scripts/benchmark_a100.py
scripts/profile_training.py
scripts/smoke_flat.py
```

Do not change the actual SemGaze training/evaluation implementation merely to make parity pass.

Instrumentation changes inside library code are allowed only if they are:

1. audit-only,
2. gated behind an explicit debug/audit flag,
3. behavior-preserving when disabled.

Prefer keeping all instrumentation in `scripts/benchmark_validation_a100.py`.

---

# 2. Existing Evidence That Must Guide the Audit

Do not restart the investigation from every possible hypothesis.

The current evidence already narrows the problem substantially.

## 2.1 WHERE is the first known divergence

Serial and batched validation already disagree on:

```text
eval_where
```

Therefore the initial discrepancy occurs before:

```text
WHAT slicing
WHY slicing
HOW slicing
semantic metric aggregation
final averaging
```

Do not spend time on those downstream components until WHERE parity is understood.

## 2.2 Low-priority suspects

The following should not be treated as primary hypotheses unless new evidence directly points back to them:

```text
episode reorder from bucketing
dropout / train-vs-eval mode mismatch
final episode averaging
WHAT/WHY/HOW offset slicing
generic Python overhead
```

The current code already contains substantial checks around these areas.

## 2.3 Highest-priority validation suspects

Investigate in this order:

### H1 — Different validation implementation even at physical B=1

The serial reference and batched evaluator do not necessarily execute identical code paths.

A same-K batched run with:

```text
physical batch = 1
cache = off
bucket = off
```

must therefore be tested before attributing the mismatch to actual multi-example batching.

### H2 — Visual feature cache changes the model execution path

The current validation cache can call:

```text
get_image_features(...)
```

separately and later fuse the cached visual features into the language-model input.

That is not equivalent to merely caching decoded/preprocessed images.

The current CLI also appears to couple:

```text
preprocessing cache
visual feature cache
```

under one cache flag.

These two caches must be isolatable.

### H3 — Physical B>1 changes WHERE numerics

If:

```text
serial cache-off
vs
batched-B1 cache-off
```

passes, but:

```text
batched-B2/B4 cache-off
```

fails, investigate:

```text
padding
attention masks
position-related inputs
image ordering / image offsets
pixel slices
batch-shape-dependent BF16 kernels
per-episode hidden states / token NLL
```

Do not immediately call the difference "expected BF16 noise".

A max WHERE loss difference around `0.024` is too large to waive without locating the first divergent tensor.

---

# 3. Required Audit Strategy

The audit must be **decision-tree based**, not an open-ended profiler exercise.

The goal is to stop as soon as one branch clearly identifies the source.

Implement the following benchmark modes in:

```text
scripts/benchmark_validation_a100.py
```

if they are not already directly supported.

---

# 4. Required Experiment Matrix

Use the exact same checkpoint, dataset subset, support sets, episode IDs, and deterministic ordering for every comparison.

## A — Serial baseline

```text
mode: serial
cache: off
bucket: off
```

Output directory example:

```text
runs/parity-a-serial
```

This is the reference.

---

## B0 — Batched implementation, physical B=1

```text
mode: same-k-batched
physical batch: 1
cache: off
bucket: off
```

Compare episode-by-episode against A.

This is the highest-value discriminator.

### Interpretation

If B0 fails WHERE parity:

```text
The problem is NOT physical batching.
There is a code-path difference between serial validation and the batched evaluator.
```

Stop and trace that difference before testing B2/B4.

If WHERE passes but semantic losses fail:

```text
The WHERE path is equivalent.
The discrepancy is downstream in the semantic serial-reference vs selected-loss/batched semantic path.
```

Only then audit semantic parity.

If B0 fully passes:

```text
Proceed to physical batching.
```

---

## B1 — Batched physical B>1, cache OFF

Use the production candidate batch sizes:

```text
K=1  -> B=4
K=5  -> B=2
K=10 -> B=2
```

or allow explicit CLI override.

Keep:

```text
cache: off
bucket: off
```

Compare episode-by-episode against A.

### Interpretation

If B0 passes but B1 fails:

```text
The root cause is triggered by physical batching itself:
padding / masks / image packing / batch-dependent kernels / numerical behavior.
```

If B1 passes:

```text
Physical batching itself is parity-safe.
Proceed to cache isolation.
```

---

## C — Bucketing only

Only run if B1 passes.

Use:

```text
same physical batch sizes
cache: off
bucket: on
```

If C fails while B1 passes:

```text
The problem is scheduling / packing / bucket-order related.
```

Do not investigate cache until this is resolved.

---

## D — Preprocessing cache only

Add independent cache controls if they do not already exist:

```text
--preprocess-cache on/off
--visual-cache on/off
```

Run:

```text
preprocess cache: on
visual feature cache: off
```

All other settings should match the passing B1/C configuration.

If D passes:

```text
Image decoding / preprocessing caching is parity-safe.
```

---

## E — Visual feature cache enabled

Run:

```text
preprocess cache: on
visual feature cache: on
```

If D passes but E fails:

```text
The root cause is localized to the visual-feature cache path,
most likely get_image_features()/feature reuse/fusion or batch-shape-dependent vision features.
```

This should be considered a strong localization result.

---

# 5. Required First-Divergence Trace

Do not compare only final scalar losses.

For a tiny deterministic fixture of 2–4 same-K episodes, implement an optional audit mode such as:

```bash
--trace-parity
```

The trace should compare one target episode run:

```text
alone
```

versus the exact same episode embedded inside a physical batch.

Use a short and a long episode in the same batch if possible.

Before the model forward, verify exact equality for the target episode.

Required checks:

```text
episode_id
K
support_ids
input_ids over the real sequence
attention_mask over the real sequence
labels over the real sequence
padding tail
pixel_values for the episode
image paths
image ordering
image offsets / image boundaries
supervised WHERE token positions
state count
```

Padding invariants must be explicitly checked:

```text
attention_mask == 0 on padded tokens
labels == -100 on padded tokens
```

Report PASS/FAIL for each invariant.

---

# 6. Tensor-Level Trace Order

If inputs are identical but WHERE output differs, compare tensors in this order and stop at the first meaningful divergence:

```text
1. visual features per image
2. fused multimodal input / image-inserted embeddings if accessible
3. final WHERE hidden states at supervised token positions
4. WHERE supervised-token logits or selected logits if available
5. WHERE per-token NLL
6. WHERE per-episode loss
7. query <END_FIX> hidden/state representation
8. projected R states
9. semantic fused inputs
10. semantic supervised logits
11. semantic per-token NLL
12. WHAT / WHY / HOW losses
```

The report must explicitly say:

```text
FIRST_DIVERGENCE = <stage>
```

Do not dump huge tensors.

For each compared tensor report:

```text
shape
dtype
max_abs_diff
mean_abs_diff
allclose result
first differing index if practical
```

If exact tensor access requires touching library code, prefer lightweight forward hooks or audit-only helper functions rather than changing model behavior.

---

# 7. BF16 Numerical-Difference Test

Only consider "batch-shape-dependent BF16 numerics" after proving all of the following for the target episode:

```text
input_ids identical
attention_mask identical
labels identical
pixel_values identical
image order identical
image offsets identical
supervised token positions identical
```

Then locate the first model tensor that diverges.

If the first divergence occurs inside a standard model forward while all inputs are identical, record:

```text
candidate cause:
batch-shape-dependent BF16 / CUDA kernel numerical behavior
```

But do not automatically declare it acceptable.

Report both:

```text
absolute error
relative error
effect on final episode loss
```

The report should distinguish:

```text
bitwise mismatch
small numerical mismatch
scientifically/materially significant mismatch
```

Do not change the parity tolerance as part of this audit.

---

# 8. Visual Cache Audit Requirements

If visual feature cache is implicated, inspect and report:

```text
whether serial and cached paths invoke the same vision model operations
whether vision features are computed at B=1 vs B>1
feature tensor shape per image
feature dtype
feature device
feature ordering
feature reuse key
fusion/insertion offsets
whether cached features are detached
whether cached feature values differ from direct forward features
```

For the same image, compare:

```text
direct feature computation
vs
cached feature computation
```

under the same checkpoint and preprocessing.

Also compare the same image when feature extraction is performed:

```text
alone
vs
inside a larger image batch
```

This determines whether the vision tower itself is batch-shape dependent.

Do not redesign the cache yet.

---

# 9. Semantic Audit — Only If WHERE Passes

Do not start this section unless WHERE parity passes.

If:

```text
WHERE parity = PASS
semantic parity = FAIL
```

then audit the two semantic implementations:

```text
serial reference semantic path
vs
forward_flat_batch / selected supervised-position loss path
```

For the same episode compare:

```text
semantic input ids
semantic labels
supervised token indices
selected logits
per-token NLL
per-component token ranges
WHAT loss
WHY loss
HOW loss
flat loss
```

Verify mathematically that the reduction is:

```text
mean token loss per episode
then mean over episodes
```

and not:

```text
global token average across the physical batch
```

Do not change the loss implementation during the audit.

---

# 10. Server Benchmark Interface

The benchmark script must support a simple server workflow.

Prefer one command that can run the complete validation isolation suite, for example:

```bash
python scripts/benchmark_validation_a100.py \
  --checkpoint "$CHECKPOINT" \
  --config "$CFG" \
  --audit-parity-suite \
  --scope quick \
  --output-dir runs/validation-parity-audit
```

If implementing one orchestration flag would make the script messy, a small sequence of explicit commands is acceptable.

The important requirement is that the script itself writes machine-readable and human-readable reports.

---

# 11. Required Report Files

The server run must produce at least:

```text
report.json
report.md
```

Optionally:

```text
episodes.jsonl
tensor_trace.json
```

## report.json

Must include:

```text
git commit
dirty-tree status if available
checkpoint path
config path
GPU model
torch version
CUDA version
transformers version
dtype
attention backend if detectable
scope
number of episodes
episode IDs
K distribution
batch sizes
cache settings
bucketing setting
```

For every experiment include:

```text
status
episodes_per_second
wall time
peak allocated GiB
peak reserved GiB
parity pass/fail
max diff per metric
mean diff per metric
episode with maximum diff
```

For parity, report at least:

```text
eval_where
eval_flat
eval_what
eval_why
eval_how
```

If other validation losses exist, include them too.

---

# 12. Human-Readable Final Diagnosis

`report.md` must finish with exactly one compact diagnosis section containing:

```text
## Root-Cause Verdict
```

The verdict must assign the result to the narrowest supported category.

Use categories such as:

```text
A. serial-vs-batched implementation mismatch at B=1
B. physical batching / padding / packing mismatch
C. bucketing / scheduling mismatch
D. preprocessing-cache mismatch
E. visual-feature-cache mismatch
F. batch-shape-dependent model numerics
G. semantic-path mismatch after WHERE parity
H. unresolved
```

For the selected category include:

```text
confidence: high / medium / low
first divergent stage
minimum reproducer
evidence
what has been ruled out
next implementation target
```

Do not conclude multiple vague causes if one branch has already isolated the failure.

---

# 13. Stop Conditions

The audit should stop early when possible.

Examples:

### Stop case 1

```text
B0 WHERE fails
```

Then do not benchmark B2/B4/cache combinations yet.

Trace serial vs batched-B1 and identify the first code/tensor divergence.

### Stop case 2

```text
B0 passes
B1 fails
```

Then do not spend time on visual cache yet.

Trace physical-batching inputs and WHERE tensors.

### Stop case 3

```text
B1 passes
D passes
E fails
```

Root cause is sufficiently localized to the visual-feature-cache path.

Do not profile unrelated validation components.

---

# 14. Training Audit Policy

Training is **not the primary task in this AGENTS.md**.

Do not automatically rerun broad training profiling.

Only audit training if the validation investigation reveals a cause that plausibly affects the shared training path, or if validation parity is resolved and there is still a specific unresolved training question that requires code inspection.

Examples that justify a training follow-up:

```text
shared InternVL batch-shape-dependent behavior
shared vision-tower batch scaling
shared multimodal fusion bug
shared padding/mask construction bug
shared selected-loss bug
```

Examples that do NOT justify expanding into training:

```text
validation-only visual cache bug
validation bucketing bug
validation report aggregation bug
validation-only semantic reference path mismatch
```

If a training follow-up becomes necessary, first inspect whether the existing:

```text
scripts/benchmark_a100.py
```

already records enough evidence.

Only modify `scripts/benchmark_a100.py` if a specific missing discriminator is required.

Do not add broad profiling "just in case".

---

# 15. If Training Does Need One Additional Audit

Only if justified by validation findings, the preferred training discriminator is:

```text
gradient checkpointing ON vs OFF
B1 vs B2
same effective optimizer batch
same sampled workload
```

The key question is:

```text
Does the B>1 backward regression disappear when gradient checkpointing is disabled?
```

Do not expand to Nsight or large profiler work unless this simple factorial is inconclusive.

---

# 16. Non-Goals

Do not:

```text
change model architecture
change the WHERE formulation
change semantic supervision
change K sampling
change support-set policy
change optimizer batch semantics
raise parity tolerance to make tests pass
silently switch precision
disable model components
change dataset samples between compared runs
optimize throughput before parity is understood
```

Do not "fix" the code during the initial audit phase.

First produce evidence.

---

# 17. Implementation Discipline

Before editing:

1. Read the entire current:
   ```text
   scripts/benchmark_validation_a100.py
   ```
2. Trace the called validation functions into the SemGaze source.
3. Identify which required audit capabilities already exist.
4. Reuse existing report/parity infrastructure where possible.
5. Make the smallest benchmark-only patch necessary.

Do not duplicate functionality that already exists.

Keep normal benchmark behavior backward-compatible.

New audit flags must default to OFF.

---

# 18. Required Deliverables from Codex

Return:

1. A short static audit summary of the current validation call graph.
2. The exact benchmark/instrumentation changes made.
3. The changed files.
4. A concise explanation of why each change is needed.
5. Exact server commands to run.
6. Expected output files.
7. A decision table explaining how to interpret each possible result.

Do not claim the root cause before the server report is available.

The local machine may not have a usable GPU, so any CUDA-dependent conclusion must come from the generated server report.

---

# 19. Preferred Server Commands

The final implementation should make the following workflow possible.

Set:

```bash
export CHECKPOINT=/data/shared/cvpr/hoang/semgaze/runs/flat-smoke-gc-2/checkpoint-1
export CFG=configs/flat_throughput.yaml
```

Then ideally:

```bash
python scripts/benchmark_validation_a100.py \
  --checkpoint "$CHECKPOINT" \
  --config "$CFG" \
  --audit-parity-suite \
  --scope quick \
  --output-dir runs/validation-parity-audit
```

If an orchestration flag is not implemented, provide the minimum explicit command sequence for:

```text
A  serial cache-off
B0 batched B=1 cache-off
B1 batched production B cache-off
C  bucket-only, only if B1 passes
D  preprocess-cache only, only if previous stages pass
E  visual-feature-cache on, only if D passes
```

The commands must be copy-paste runnable on the server.

---

# 20. Success Criterion

The audit is successful when the report can answer:

```text
What is the earliest operation where serial and optimized validation cease to be equivalent?
```

and reduce the cause to one narrow component or execution difference.

A successful result is not:

```text
"There are numerical differences somewhere."
```

A successful result is something like:

```text
"Serial and batched-B1 are identical. Batched-B2 first diverges in WHERE hidden states despite identical logical inputs; the first mismatch occurs inside the InternVL forward, before semantic projection."
```

or:

```text
"Physical batching is parity-safe. Preprocessing cache is parity-safe. Enabling visual-feature cache changes per-image vision features; the first divergence occurs in get_image_features(), which then propagates to eval_where."
```

or:

```text
"Batched-B1 already differs from serial before physical batching is introduced; the first mismatch is the serial-reference WHERE construction versus forward_where_batch()."
```

That level of localization is sufficient to begin a separate implementation/fix phase.
