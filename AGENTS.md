# AGENTS.md — SemGaze Validation Throughput
## Reuse the Implemented Same-K Physical-Batch Training Infrastructure

## 0. Mission

Resolve the remaining **validation throughput** bottleneck in the current SemGaze flat pipeline.

Training throughput has already been refactored to support:

```text
true physical episode batching
same-K batches
K re-sampled between training microbatches
batched WHERE
batched <END_FIX> state extraction
batched projector
batched semantic state insertion
batched semantic forward
episode-mean loss reduction
differentiable semantic -> WHERE gradient flow
```

Therefore, **do not design another generic batching system for validation**.

The validation implementation should reuse the already-tested same-K physical-batch path wherever possible.

The required workflow is:

```text
PHASE 1 — AUDIT / VALIDATE CURRENT CODE
    inspect the post-training-throughput repository
    verify exactly which training batching helpers can be reused
    identify validation-only serialization / recomputation
    write an audit report
    do not patch validation behavior before the audit is complete

PHASE 2 — IMPLEMENT
    reuse same-K physical-batch infrastructure
    implement batched teacher-forced validation
    preserve exact per-episode metrics
    implement validation-specific scheduling / caching
    add server benchmark tooling

PHASE 3 — SERVER VALIDATION
    the local Codex environment is not authoritative for A100 performance
    run generated commands on the A100 server
    produce JSON + Markdown reports
    choose final K-specific batch sizes from measured server results
```

Do not claim throughput or A100 memory improvements from local static analysis alone.

---

# 1. What changed because same-K physical batching already exists

The original validation-throughput plan assumed batching infrastructure still had to be designed.

That is no longer true.

The validation plan must now follow these rules.

## 1.1 Reuse, do not duplicate

Before adding any validation batching code, locate the implemented training equivalents of:

```text
same-K batch construction
batched WHERE collation
batched multimodal image packing
batched WHERE forward
selective supervised-position logits / NLL
batched <END_FIX> extraction
batched projector
batched semantic insertion
batched semantic forward
per-episode loss reconstruction
```

Validation should call or lightly generalize these helpers.

Do not create:

```text
a second WHERE batch format
a second multimodal packing implementation
a second state insertion implementation
a second selected-logit implementation
a second per-episode CE implementation
```

unless the audit proves that the training helper is intrinsically training-only.

## 1.2 Validation does NOT use random K per batch

Training behavior:

```text
sample one K for the physical microbatch
construct a same-K physical batch
next microbatch may sample a different K
```

Validation behavior is deterministic:

```text
evaluate all K=1 episodes
evaluate all K=5 episodes
evaluate all K=10 episodes
```

or another deterministic same-K schedule with identical coverage.

Validation must never randomly choose which K to evaluate.

All configured K values and all validation queries must still be evaluated.

## 1.3 Same-K is now the canonical validation batching constraint

Every physical validation batch should contain one K only.

Example:

```text
batch A: K=1 only
batch B: K=1 only
...
batch M: K=5 only
...
batch N: K=10 only
```

Do not mix K values inside one physical batch unless a future benchmark proves a compelling reason and the change is separately justified.

---

# 2. Scientific invariants — MUST NOT CHANGE

## 2.1 Evaluation workload

Preserve:

```text
same validation split
same subjects
same queries
same K values
same frozen support sets
same support ordering
same checkpoint
same model
same tokenizer
same processor
same WHERE targets
same flat semantic targets
```

Representative workload:

```text
queries = 1751
K = [1, 5, 10]
episodes = 5253
```

Use the actual canonical count from the repository/config if it differs.

Do not reduce workload to gain throughput.

## 2.2 Episode definition

One validation episode remains:

```text
frozen same-subject K-shot supports
+
one validation query
+
teacher-forced WHERE
+
query <END_FIX> state extraction
+
shared projector
+
flat semantic forward
```

Do not modify the few-shot protocol.

## 2.3 Exact metric definitions

Preserve:

```text
eval_where
eval_flat
eval_what
eval_why
eval_how
eval_total
```

and any currently reported validation metrics.

Do not redefine component spans or loss weights.

## 2.4 Episode-mean aggregation

This is mandatory.

For metric `m`:

```text
eval_m = mean over episodes(m_episode)
```

Do not convert this into:

```text
global token mean
batch mean weighted by token count
mean of batch means when batch sizes differ
mean weighted by fixation count
```

The accumulator must track:

```text
sum of per-episode metric values
number of episodes
```

or an equivalent exact formulation.

The last partial batch must not receive the same weight as a full batch unless it contains the same number of episodes.

## 2.5 WHERE state semantics

Teacher-forced validation must continue to extract:

```text
final LM-layer hidden state
at each query-assistant <END_FIX>
in chronological fixation order
```

Do not include support states, padding, or another sample's states.

## 2.6 Validation must not build training graphs

Use:

```python
model.eval()
torch.inference_mode()
```

where compatible.

There is no backward pass.

Validation throughput work must not modify training gradient semantics.

---

# 3. Local environment constraint

Codex is auditing / implementing on a local machine that is not the production A100 environment.

The local environment may not have CUDA, A100, 40GB VRAM, production InternVL weights, representative server I/O, or the same fused kernels.

Local work may establish:

```text
source correctness
shape correctness
mask correctness
metric correctness
batch completeness
serial-vs-batched numerical parity using available fixtures
```

Local work may NOT establish:

```text
production throughput
A100 peak memory
maximum safe validation batch size
optimal K-specific batch size
full validation wall time
CUDA allocator behavior
```

All production-performance conclusions require server reports.

---

# 4. PHASE 1 — mandatory audit before implementation

Do not patch first.

Inspect the current repository **after the training-throughput implementation**.

Write the audit result before modifying the validation execution path.

Recommended artifact:

```text
runs/validation_throughput_audit.md
```

---

# 5. Audit the implemented training same-K batch path first

Locate the real current symbols.

Do not rely only on old filenames or old AGENTS.md assumptions.

Document:

```text
training batch object / dataclass
same-K batch builder
WHERE batch collator
WHERE forward
state extractor
projector
semantic batch builder
semantic insertion
semantic forward
selected-logit / NLL helper
per-episode loss helper
```

For every reusable component state:

```text
source location
input contract
output contract
training-only dependency?
safe for inference?
change required for validation?
```

Preferred architecture:

```text
validation episode scheduler
        ↓
existing same-K batch collator
        ↓
existing batched WHERE path
        ↓
existing batched state path
        ↓
existing batched semantic path
        ↓
validation-only metric accumulator
```

---

# 6. Audit the current validation call graph

Trace the actual current path:

```text
validation trigger
-> K loop
-> query loop
-> frozen support lookup
-> episode construction
-> WHERE preparation
-> WHERE forward
-> WHERE loss
-> state extraction
-> projector
-> semantic preparation
-> semantic forward
-> flat / WHAT / WHY / HOW decomposition
-> metric aggregation
-> progress output
```

Record every point where validation is still serialized.

Specifically search for:

```text
for query in ...
for episode in ...
shape[0] == 1
batch_size == 1
[0]
unsqueeze(0)
single episode
one episode
```

Classify each finding as:

```text
already fixed by shared training batching
validation-specific serial loop
metric-only Python loop
harmless metadata loop
full-model serial execution
```

Only full-model serial execution is a major throughput blocker.

---

# 7. Audit actual post-training-optimization loss computation

Training throughput work may already have removed:

```text
full-sequence full-vocabulary logits
duplicate CE
all-layer hidden-state materialization
```

Do not implement these again blindly.

Verify whether validation uses the optimized path or falls back to an older model call.

Check:

```text
WHERE selected supervised logits?
semantic selected supervised logits?
per-token NLL available?
per-episode NLL reconstruction available?
```

Validation should use the optimized exact-semantics path.

If validation still calls native `outputs.loss` in a way that recreates full `[B,L,V]` logits, migrate it to the shared optimized loss helper.

---

# 8. Audit WHAT / WHY / HOW decomposition

Determine exactly how current validation computes:

```text
eval_flat
eval_what
eval_why
eval_how
```

Preferred target:

```text
semantic selected supervised logits
        ↓
one unreduced token-NLL tensor
        ↓
existing component masks/spans
        ↓
per-episode flat/what/why/how metrics
```

Do not run a second full CE solely for diagnostic decomposition.

Do not alter component boundaries.

---

# 9. Audit validation episode ordering

Because training uses random same-K batches while validation is deterministic, validation needs its own scheduler.

Audit the current canonical order of:

```text
K
subject
query
```

Determine whether execution order can be changed internally without changing scientific results.

Every episode must carry a stable identifier.

Final metrics are order-independent, but progress logs, prediction-file order, and parity reports should remain reproducible.

---

# 10. Audit fixed-support reuse

For validation, support sets are frozen.

For each `(subject, K)` determine:

```text
number of validation queries
support record IDs
support images
support prompt serialization
support assistant text
processed support pixel tensors
projected support visual features
```

Identify what is recomputed for every query.

This is validation-specific and may not have been solved by training throughput work.

---

# 11. Audit query-image reuse

Determine whether the optimized training path already reuses query visual features between WHERE and semantic.

If yes, reuse the same mechanism in validation.

If no, determine whether:

```text
vision tower
native multimodal projector
```

are frozen and therefore eligible for exact feature reuse.

Do not create a second caching mechanism if training already has one.

---

# 12. Audit sequence-length waste inside same-K batches

Same-K batching eliminates the largest K-driven variation, but episodes within one K can still vary due to support fixation lengths, query fixation count, WHERE text length, and semantic target length.

Measure / expose:

```text
unpadded WHERE length
padded WHERE length
unpadded semantic length
padded semantic length
fixation count
```

Compute padding efficiency:

```text
sum(real tokens) / sum(padded tokens)
```

Do not assume length bucketing is necessary until the distribution is inspected.

---

# 13. Audit prediction separately

Teacher-forced validation and autoregressive prediction are different workloads.

Audit prediction independently:

```text
WHERE generation
parsing
semantic generation
prediction serialization
```

Training physical batching helpers may help with prompt construction and multimodal packing, but teacher-forced loss helpers are not automatically appropriate for generation.

Do not block teacher-forced validation optimization on generation refactoring.

---

# 14. Audit report gate

Before substantive validation refactoring, write:

```text
runs/validation_throughput_audit.md
```

with:

```text
1. current validation call graph
2. reused training same-K batching symbols
3. remaining serial full-model calls
4. current per-episode metric semantics
5. current WHAT/WHY/HOW decomposition path
6. repeated support computation
7. repeated query computation
8. same-K padding statistics available locally
9. proposed validation patch order
10. items requiring A100 validation
```

Then continue to implementation.

Do not wait for human approval unless source is missing or internally inconsistent.

---

# 15. PHASE 2 — implementation priorities

Use this order:

```text
P0. reuse same-K training batch execution in teacher-forced validation
P0. exact per-episode validation accumulator
P0. reuse optimized selected-logit / NLL path
P0. configurable validation batch size by K
P1. deterministic same-K length bucketing
P1. fixed-support preprocessing / visual-feature reuse
P1. query visual-feature reuse
P1. efficient progress reporting
P1. server profiling/report tooling
P1. batched autoregressive prediction
P2. support-prefix LM KV caching only if profiling still justifies it
```

---

# 16. P0 — reuse the training same-K physical-batch path

Do not implement a second physical batch executor.

Validation should construct multiple validation episodes with the same K and pass them through the same batch infrastructure already used by training.

Conceptually:

```text
validation episodes with K = k
        ↓
same-K physical batch builder
        ↓
batched WHERE forward
        ↓
batched query <END_FIX> extraction
        ↓
shared projector
        ↓
batched semantic state insertion
        ↓
batched semantic forward
        ↓
per-episode validation metrics
```

Differences from training:

```text
model.eval()
inference_mode
no backward
no optimizer
no scheduler
no gradient clipping
no random K selection
```

The full InternVL model must not be invoked once per episode inside the batch loop.

---

# 17. P0 — deterministic validation scheduler

Recommended execution structure:

```text
for K in configured_K_values:
    collect every canonical validation episode for this K
    deterministically schedule same-K batches
    execute all batches
```

Within each K, preserve stable episode IDs.

Recommended first implementation:

```text
K
    -> subject
        -> stable query list
        -> optional stable length bucketing
        -> physical batches
```

Grouping by `(subject, K)` is useful because support sets are fixed and repeated.

However, do not unnecessarily force tiny subject-only batches if the shared batching implementation can combine same-K episodes across subjects efficiently.

Preferred policy:

```text
same-K is mandatory
same-subject grouping is a cache optimization, not a scientific requirement
```

Use profiling to decide whether subject grouping should constrain the physical batch.

---

# 18. P0 — K-specific physical batch sizes

Validation must support separate batch sizes:

```yaml
validation:
  loss_batch_size_by_k:
    1: <B1>
    5: <B5>
    10: <B10>
```

or the repository's equivalent config format.

Do not guess final values.

Local defaults should be conservative.

Final values must be chosen using A100 reports.

Likely:

```text
B1 >= B5 >= B10
```

but this is a hypothesis, not a hard-coded rule.

---

# 19. P0 — exact per-episode accumulator

Do not average batch means.

For every physical batch obtain:

```text
where_loss_per_episode     [B]
flat_loss_per_episode      [B]
what_loss_per_episode      [B]
why_loss_per_episode       [B]
how_loss_per_episode       [B]
total_loss_per_episode     [B]
```

Accumulator:

```python
sum_where += where_loss_per_episode.sum()
sum_flat += flat_loss_per_episode.sum()
...
count += B
```

Final:

```python
eval_where = sum_where / count
...
```

This correctly handles the last short batch.

Store per-episode metrics in debug/parity mode.

---

# 20. P0 — reuse the exact optimized NLL path

If the training throughput implementation already computes vocabulary logits only at supervised predictor positions, validation must reuse it.

Do not regress validation to:

```text
full [B,L,V] logits
native global CE
duplicate component CE
```

Required flow:

```text
final hidden states
-> gather supervised predictor states
-> existing full-vocabulary LM / RowHead
-> unreduced CE
-> restore token NLL by episode
```

Teacher-forced validation does not need logits for ignored prompt/support/padding positions.

---

# 21. P0 — component metrics from one semantic NLL

Use one semantic per-token NLL tensor to derive:

```text
flat
WHAT
WHY
HOW
```

for every episode.

Do not recompute semantic forward.

Do not recompute full CE independently for each component.

Keep exact existing component masks.

---

# 22. P1 — length bucketing inside same K

Because K is already homogeneous, length bucketing is now a secondary optimization.

Do not over-engineer it before measuring padding waste.

Simple deterministic strategy:

```text
within one K:
    compute estimated WHERE length
    stable-sort within a bounded window
    batch nearby lengths together
```

Optional tie-breakers:

```text
query fixation count
semantic length
```

Requirements:

```text
no episode dropped
no duplicate episode
no K redistribution
same support assignment
stable episode IDs
```

Final metrics remain independent of execution order.

---

# 23. P1 — fixed-support cache

Validation has a major reuse pattern that training does not:

```text
(subject, K) support set is fixed across many queries
```

Implement cache only for deterministic/frozen work.

Potential reusable artifacts:

```text
support record resolution
support prompt serialization
support tokenization
decoded/preprocessed support image tensors
frozen projected support image features
```

Prefer reusing existing training visual-cache infrastructure if present.

Do not cache:

```text
trainable LM support hidden states
query-dependent states
anything whose producer changes during validation
```

For a fixed validation checkpoint, frozen visual features are safe if the producer is frozen.

---

# 24. P1 — query visual reuse

If the query image is consumed by both WHERE and semantic forward:

```text
do not re-run a frozen vision tower twice
```

Reuse the same projected visual feature when mathematically identical.

Do not change the numerical representation consumed by the LM.

---

# 25. P1 — inference mode and model mode

Validation entrypoint must preserve the caller's prior model mode when validation is invoked during training.

Conceptually:

```python
was_training = model.training
model.eval()

with torch.inference_mode():
    run_validation()

if was_training:
    model.train()
```

Do not leave the model in eval mode when training resumes.

---

# 26. P1 — progress reporting

Keep progress by K.

Example:

```text
[EVAL][LOSS] K=1 | 176/1751 | 10.1% | batch=8 | elapsed=... | ETA=...
```

Update at batch or coarse-interval granularity.

Do not force CUDA synchronization for every episode solely for logging.

---

# 27. P1 — teacher-forced validation and prediction stay separate

## Teacher-forced validation

Use:

```text
same-K physical batching
known targets
selected supervised logits
large batches
inference_mode
normally use_cache=False
```

## Autoregressive prediction

Use:

```text
same-K prompt batches
KV cache
generation loop
smaller batch sizes if required
length-aware scheduling
existing decoding/stopping rules
```

Teacher-forced optimization is the first priority.

---

# 28. P1 — same-K batched autoregressive prediction

After teacher-forced validation is correct, adapt prediction.

Batch only same-K prompts.

Preserve exactly:

```text
generation mode
temperature
top-p / top-k if present
max_new_tokens
EOS / stopping criteria
WHERE parser
semantic parser
```

For greedy decoding, outputs should match serial decoding within the same model/kernel environment.

For stochastic decoding, preserve per-sample RNG semantics where practical and document any unavoidable RNG-ordering difference.

---

# 29. P2 — support-prefix LM KV caching

This is no longer an initial recommendation.

Same-K physical batching + fixed visual reuse should be implemented and benchmarked first.

Only implement prefix KV reuse if server profiling shows support-prefix LM computation remains dominant.

Potential reuse unit:

```text
(subject, K)
```

Before implementation validate:

```text
same support input IDs
same support image features
same position IDs
same multimodal placeholder layout
compatible cache_position
compatible query suffix attention mask
safe batch expansion
```

Do not ship prefix caching without numerical parity tests.

---

# 30. Server benchmark tooling is mandatory

Codex cannot conclude validation throughput from local execution.

Inspect existing training benchmark tooling first.

There may already be infrastructure such as:

```text
scripts/benchmark_a100.py
configs/flat_throughput.yaml
```

Reuse common profiler/report code where practical.

Either extend the existing benchmark script with validation mode or create a dedicated:

```text
scripts/benchmark_validation_a100.py
```

Choose whichever produces cleaner code.

Do not duplicate generic GPU/report utilities unnecessarily.

---

# 31. Required benchmark modes

The server tool must support:

```text
serial reference
same-K batched validation
quick parity subset
full validation workload
K-specific batch sizes
optional cache on/off comparison
```

Recommended CLI concept:

```bash
python scripts/benchmark_validation_a100.py \
  --config configs/flat_throughput.yaml \
  --mode serial \
  --scope quick \
  --output-dir runs/validation-serial-quick
```

```bash
python scripts/benchmark_validation_a100.py \
  --config configs/flat_throughput.yaml \
  --mode same-k-batched \
  --scope quick \
  --batch-k1 4 \
  --batch-k5 2 \
  --batch-k10 1 \
  --output-dir runs/validation-batched-quick
```

Adapt argument names to actual repository conventions.

---

# 32. Quick server parity test

Before full benchmarking, run a deterministic subset.

Include:

```text
multiple K=1 episodes
multiple K=5 episodes
multiple K=10 episodes
different fixation counts
different WHERE lengths
different semantic lengths
```

Compare serial vs same-K batched execution episode-by-episode.

Required fields:

```text
episode_id
K
where_loss
flat_loss
what_loss
why_loss
how_loss
total_loss
number of extracted query states
```

Output:

```text
serial_quick.json
batched_quick.json
parity_quick.md
```

Report:

```text
max absolute difference
mean absolute difference
tolerance
pass/fail
```

Do not use aggregate-only parity as the first correctness check.

---

# 33. K-specific batch sweep on server

Find the largest **useful** physical batch for each K, not merely the largest batch that fits.

Example candidate grids:

```text
K=1:  B = 1, 2, 4, 8
K=5:  B = 1, 2, 4
K=10: B = 1, 2
```

Expand only if measured headroom suggests it.

Record:

```text
episodes/sec
seconds/episode
peak allocated VRAM
peak reserved VRAM
padding efficiency
OOM
```

Choose the batch that maximizes throughput with safe memory headroom.

A larger batch that is slower is not preferred.

---

# 34. Same-K benchmark comparison

Because same-K batching was introduced to avoid cross-K padding waste, validation reports should expose this directly.

For each K report:

```text
mean unpadded WHERE tokens / episode
mean padded WHERE tokens / episode
padding efficiency
mean semantic unpadded tokens
mean semantic padded tokens
physical batch size
```

If length bucketing is enabled, report before/after padding efficiency.

---

# 35. Full validation benchmark

After quick parity passes and K batch sizes are chosen, run the complete workload.

Expected representative workload:

```text
K=1: 1751 episodes
K=5: 1751 episodes
K=10: 1751 episodes
TOTAL: 5253 episodes
```

Use actual canonical counts if different.

Do not stop after a small benchmark.

---

# 36. Required JSON report fields

Each server run must record at least:

## Environment

```text
timestamp
git commit
dirty working tree
config path
checkpoint / adapter
base model
dtype
GPU name
GPU total memory
PyTorch version
Transformers version
CUDA version
attention backend
```

## Execution

```text
mode
inference_mode
same-K batching enabled
length bucketing enabled
visual cache enabled
support cache enabled
prefix KV cache enabled
```

## Workload

```text
query count
K values
episode count total
episode count by K
```

## Batching

```text
configured physical batch size by K
actual batch count by K
mean actual batch size by K
minimum / maximum actual batch size
```

## Padding

```text
WHERE real tokens
WHERE padded tokens
WHERE padding efficiency
semantic real tokens
semantic padded tokens
semantic padding efficiency
```

## Performance

```text
total wall time
wall time by K
episodes/sec overall
episodes/sec by K
seconds/episode overall
```

## Memory

```text
peak torch.cuda.memory_allocated
peak torch.cuda.memory_reserved
```

## Cache

where enabled:

```text
entries
hits
misses
hit rate
```

Do not infer live tensor memory from `nvidia-smi` alone.

---

# 37. Markdown report

Each benchmark should also emit a human-readable report.

Recommended table:

```text
| Mode | K | Physical B | Episodes | Pad eff. | Peak alloc | Peak reserved | Episodes/s |
|---|---:|---:|---:|---:|---:|---:|---:|
```

Full report should include:

```text
serial baseline
same-K batched results
speedup by K
overall speedup
chosen B1/B5/B10
parity result
cache result
remaining bottleneck
```

---

# 38. Required local tests

Local tests must not require production GPU.

## 38.1 Reuse-contract test

Ensure validation calls the shared same-K batch path rather than a separate batch implementation.

## 38.2 Same-K scheduler test

Input episodes containing K=1, K=5, and K=10.

Verify every emitted physical batch contains exactly one K.

## 38.3 Completeness test

Verify scheduler:

```text
drops zero episodes
duplicates zero episodes
returns every episode exactly once
```

## 38.4 Unequal final-batch metric test

Example:

```text
B=4
5 episodes total
```

Verify final validation result is:

```text
mean of 5 episode metrics
```

not mean of two batch means.

## 38.5 Variable-length test within same K

Use same-K episodes with different WHERE lengths, semantic lengths, and fixation counts.

Verify padding, masks, and state extraction.

## 38.6 State isolation test

Synthetic states must prove sample A receives A states only and sample B receives B states only.

## 38.7 Multimodal isolation test

Use distinguishable image features and verify no cross-sample assignment.

## 38.8 Component NLL parity

Verify batched flat/WHAT/WHY/HOW per-episode values match serial reference definitions.

## 38.9 Training regression test

Because validation reuses training batch infrastructure, run existing training correctness tests after any shared-helper change.

Validation optimization must not regress training same-K batches, training loss, gradient flow, or checkpoint compatibility.

---

# 39. Validation configuration

Prefer validation-specific controls without redefining training behavior.

Example:

```yaml
validation:
  loss:
    execution: same_k_batched
    batch_size_by_k:
      1: 4
      5: 2
      10: 1
    bucket_by_length: true

  prediction:
    execution: same_k_batched
    batch_size_by_k:
      1: 2
      5: 1
      10: 1

  cache:
    frozen_visual_features: true
    support_preprocessing: true
    prefix_kv: false

  profiling:
    enabled: false
```

Use actual config conventions.

Do not hard-code production batch sizes in Python.

---

# 40. Failure conditions

Fail clearly on:

```text
mixed K inside a supposedly same-K batch
wrong validation episode count
missing K
duplicate episode
missing episode
support mismatch
query state-count mismatch
cross-sample image mapping mismatch
non-finite metric
metric accumulator count mismatch
report write failure
```

On OOM:

```text
report K
report requested physical batch
write failed candidate to report
exit cleanly
```

Do not silently reduce batch size unless explicit auto-backoff mode is enabled.

---

# 41. Recommended implementation order

Use this order unless the source audit proves a dependency differs.

```text
0. Inspect the implemented training same-K batch path
1. Audit current validation path
2. Write runs/validation_throughput_audit.md
3. Route teacher-forced validation through shared same-K batch helpers
4. Add exact per-episode validation accumulator
5. Reuse shared selected-logit / NLL helpers
6. Compute WHAT/WHY/HOW from one semantic NLL
7. Add validation batch_size_by_k config
8. Add local same-K scheduler / parity tests
9. Add deterministic within-K length bucketing
10. Add support/query frozen-feature reuse where not already shared
11. Add server quick serial-vs-batched parity tooling
12. Add server K-specific batch sweep
13. Add full validation benchmark/report
14. Optimize autoregressive prediction with same-K generation batches
15. Only after profiling, consider prefix KV caching
```

---

# 42. What should be removed from the OLD validation AGENTS.md

Do not carry forward old assumptions that are now obsolete.

Remove or rewrite recommendations that imply:

```text
validation must invent its own generic physical batch representation
validation must first make state extraction batch-aware from scratch
validation must implement semantic insertion batching independently
validation should support mixed-K batches
```

Those belong to the already-completed training-throughput refactor.

Validation should now be an **execution/scheduling/reuse layer over the shared same-K batch core**.

---

# 43. What remains valid from the OLD validation plan

Keep:

```text
K-specific validation batch sizes
teacher-forced validation separated from autoregressive prediction
exact episode-mean aggregation
single-pass NLL decomposition
length bucketing
frozen visual-feature reuse
fixed-support reuse
coarse progress logging
serial-reference parity
server A100 batch sweep
JSON + Markdown reports
no reduced evaluation coverage
```

These remain valid and are now easier to implement because the shared physical-batch core already exists.

---

# 44. Final Codex deliverable

After implementation, return:

## A. Audit

```text
shared training helpers reused
validation-specific serial bottlenecks found
loss path
support/query reuse opportunities
```

## B. Code changes

For each file:

```text
why changed
what changed
shared helper vs validation-specific code
scientific behavior changed? yes/no
```

## C. Local verification

```text
tests passed
tests skipped
why GPU tests could not run locally
```

Never fabricate A100 results.

## D. Exact server commands

Provide copy-paste commands for:

```text
quick serial reference
quick same-K batched
quick parity comparison
K=1 sweep
K=5 sweep
K=10 sweep
full optimized validation
```

Use the actual implemented CLI.

## E. Expected outputs

List exact report paths.

## F. Pending A100 verification

Explicitly state which conclusions cannot be made until server reports exist.

---

# 45. Definition of locally complete

```text
[ ] post-training-throughput source was audited first
[ ] validation audit report exists
[ ] validation reuses shared same-K batch infrastructure
[ ] no duplicate generic batch implementation was introduced
[ ] teacher-forced validation executes true B>1 same-K model batches
[ ] no full InternVL call remains inside a per-episode validation loop
[ ] validation batch size is configurable independently for K=1/5/10
[ ] exact episode-mean aggregation is preserved
[ ] WHAT/WHY/HOW semantics are preserved
[ ] mixed sequence lengths within same K work
[ ] mixed fixation counts within same K work
[ ] all validation episodes are evaluated exactly once
[ ] local correctness tests pass
[ ] training regression tests pass
[ ] server serial-reference mode exists
[ ] server same-K batched mode exists
[ ] server parity report exists
[ ] server K-specific sweep tooling exists
[ ] full benchmark report tooling exists
[ ] exact A100 commands are documented
```

---

# 46. Definition of experimentally resolved

Validation throughput is not experimentally resolved until the A100 server confirms:

```text
[ ] quick serial vs same-K batched parity passes
[ ] K=1 batch sweep completed
[ ] K=5 batch sweep completed
[ ] K=10 batch sweep completed
[ ] final B1/B5/B10 selected from measured throughput + memory
[ ] full canonical validation workload completes
[ ] all K values are present
[ ] all expected episodes are present
[ ] final eval metrics match reference within tolerance
[ ] total wall time is materially lower
[ ] peak allocated/reserved VRAM are recorded
[ ] JSON and Markdown reports are saved
```

Until then describe the state as:

```text
validation batching implemented and locally verified,
pending A100 throughput validation
```
