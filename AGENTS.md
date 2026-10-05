# AGENTS.md — SemGaze Flat Training Throughput Audit + Patch

## 0. Mission

Your task is to **fully resolve Section A — Training Throughput** for the current SemGaze flat training pipeline.

You must work in **two mandatory phases**:

1. **AUDIT FIRST**
   - Inspect the actual current source.
   - Identify the real training hot path, memory-retention points, batch-size-1 assumptions, redundant work, and loss-reduction semantics.
   - Produce an evidence-based audit before changing behavior.

2. **PATCH SECOND**
   - Implement the highest-priority exact-semantics optimizations from the audit.
   - Add true physical episode batching.
   - Preserve all current scientific behavior unless a change is explicitly marked as a scientific trade-off.
   - Do not stop after proposing changes. Patch the code, add tests, and provide verification evidence.

Do **not** implement validation-throughput changes unless they are required as shared infrastructure for training batching. The primary scope of this task is **training**.

---

# 1. Scientific invariants — MUST NOT CHANGE

The following are method-defining invariants.

## 1.1 Model / precision / training method

- Base model: `OpenGVLab/InternVL3_5-8B-HF`
- Precision: bf16
- Training: LoRA
- Initialization: released DeepGaze visual-search adapter
- GPU target: NVIDIA A100 40GB
- Gradient checkpointing currently enabled
- WHERE and semantic branches use the **same InternVL model**

Do not silently replace the model, quantize it, change precision, or switch to QLoRA.

---

## 1.2 Gradient-flow invariant

The semantic loss must continue to backpropagate through the WHERE hidden states.

The required path is:

```text
L_semantic
    -> semantic InternVL computation
    -> inserted projected fixation states R
    -> projector P_E
    -> WHERE <END_FIX> hidden states F
    -> WHERE InternVL computation
```

Therefore:

```text
DO NOT detach F
DO NOT detach R
DO NOT run WHERE under no_grad
DO NOT cache trainable WHERE states across optimizer steps
DO NOT turn semantic supervision into a separate frozen-target objective
```

Caching or reusing features is allowed only where the producing module is frozen and doing so preserves the same mathematical input to the trainable path.

---

## 1.3 Episode semantics

One training episode remains:

```text
same-subject support set
+
one query
+
teacher-forced WHERE
+
query <END_FIX> state extraction
+
projector
+
flat semantic forward
```

Do not change:

- few-shot support construction;
- same-subject constraint;
- support order;
- K sampling distribution;
- query sampling;
- requested fixation count;
- WHERE serialization;
- semantic target;
- loss weights;
- supervision masks;
- semantic architecture.

---

## 1.4 Loss-reduction invariant

This is critical.

The existing implementation effectively computes a loss per episode and then averages episode losses across the optimizer minibatch.

True vectorized batching must preserve:

```text
L_batch = mean_b(L_episode_b)
```

Do **not** silently replace this with a global token mean across all samples.

For variable-length samples:

```text
1. compute token NLL;
2. aggregate supervised tokens within each episode exactly as the old code does;
3. obtain one scalar loss per episode;
4. average episode losses across the physical batch.
```

This applies to WHERE and flat semantic loss.

Add explicit tests for this.

---

## 1.5 WHERE state invariant

The state for fixation `t` remains the **final LM-layer hidden state** at the corresponding query-assistant `<END_FIX>` position.

Do not:

- use an earlier Transformer layer;
- use support `<END_FIX>` states;
- change token positions;
- pool multiple tokens;
- replace the state with logits or embeddings.

The implementation may change **how the final hidden tensor is obtained**, but not what state is selected.

---

# 2. Current known concerns to verify against source

Treat these as hypotheses to confirm or reject from the actual repository.

Do not assume they are correct until you inspect the code.

Expected areas:

```text
semgaze/training/
semgaze/where/
semgaze/state/
semgaze/semantic/flat/
semgaze/model/
configs/
scripts/
tests/
```

Likely relevant files include:

```text
semgaze/training/loop.py
semgaze/training/flat_step.py

semgaze/where/collator.py
semgaze/where/forward.py

semgaze/state/extractor.py
semgaze/state/insertion.py
semgaze/state/projector.py

semgaze/semantic/flat/forward.py

semgaze/model/build.py
semgaze/model/trainable_tokens.py

scripts/smoke_flat.py
configs/flat_single.yaml
```

Search the repository rather than trusting this list.

---

# 3. Phase I — mandatory source audit

Do not patch first.

Before editing source, inspect and document the current training call graph.

## 3.1 Reconstruct the exact training hot path

Trace from the public training entrypoint to:

```text
episode sampling
-> episode collation
-> WHERE processor/tokenization
-> WHERE model forward
-> WHERE loss
-> <END_FIX> state extraction
-> P_E
-> semantic preparation
-> semantic model forward
-> semantic loss
-> total loss
-> backward
-> gradient checks
-> gradient clipping
-> optimizer.step()
-> scheduler.step()
```

For every stage, identify:

- input shapes;
- batch dimension assumptions;
- sequence lengths;
- image count;
- whether tensors require grad;
- whether tensors are retained for backward;
- whether a full vocabulary projection is performed;
- whether CUDA synchronization occurs;
- whether CPU preprocessing blocks GPU execution.

---

## 3.2 Audit the meaning of `per_device_train_batch_size`

Confirm exactly whether it currently means:

```text
true vectorized physical batch
```

or:

```text
number of sequential episode forwards/backwards accumulated before optimizer.step()
```

Identify all source locations responsible for the current behavior.

---

## 3.3 Audit batch-size-1 assumptions

Search for all of the following patterns:

```text
shape[0] == 1
batch_size == 1
assert ... == 1
tensor[0]
processor_output[0]
unsqueeze(0)
single-episode
single episode
len(...) == 1
scalar-only metadata
one query
one episode
```

Audit at least:

- WHERE collator;
- WHERE forward;
- state extraction;
- state insertion;
- semantic preparation;
- training step;
- loss functions;
- gradient diagnostics;
- image handling.

Create a concrete list:

```text
file
symbol
assumption
why it prevents physical batching
required replacement
```

---

## 3.4 Audit full-vocabulary logits

Determine exactly how logits are currently produced for WHERE and semantic forwards.

Verify:

- vocabulary size;
- logits shape;
- dtype before loss;
- whether the native HF loss upcasts to FP32;
- whether `RowHead` creates a second `[B,L,V]` tensor;
- which token positions actually have `labels != -100`;
- whether full logits are needed after loss computation.

Measure or estimate memory for representative sequence lengths.

At minimum report:

```text
B
L
V
dtype
size of one logits tensor
additional RowHead allocation
loss upcast allocation
```

Do not implement vocabulary truncation or sampled softmax.

The only acceptable exact optimization is to avoid computing/storing vocabulary logits for positions that cannot contribute to the current loss.

---

## 3.5 Audit hidden-state retention

Find all uses of:

```python
output_hidden_states=True
outputs.hidden_states
hidden_states[-1]
```

Determine whether requesting hidden states materializes outputs from every LM layer.

Verify whether the underlying InternVL/Qwen model can expose the final hidden state without collecting the full hidden-state tuple.

Required target behavior:

```text
same final LM hidden tensor
same <END_FIX> extraction
no all-layer hidden-state output collection
```

---

## 3.6 Audit gradient checkpointing

Determine:

- exactly which model blocks are checkpointed;
- whether non-reentrant checkpointing is used;
- whether requesting hidden states interferes with expected checkpointing memory savings;
- whether semantic backprop through WHERE triggers the expected recomputation.

Do not disable checkpointing as the first optimization.

Checkpointing should remain enabled until the new memory profile is known.

---

## 3.7 Audit vision computation

Determine whether:

- the vision tower is frozen;
- the native multimodal projector is frozen;
- WHERE encodes all support + query images;
- semantic re-encodes the query image;
- the query image is redundantly decoded/preprocessed;
- frozen support images are recomputed every episode;
- any frozen image features can be safely reused without changing gradients.

Important distinction:

```text
Caching frozen visual features is allowed.
Detaching WHERE fixation states is not allowed.
```

---

## 3.8 Audit attention backend

Log / inspect the actual text and vision attention implementations.

Check:

```text
eager
sdpa
flash_attention_2
```

Do not assume Flash Attention is active.

Do not install or enable a different attention implementation until after confirming the current backend and compatibility.

---

## 3.9 Audit CPU / synchronization overhead

Search for training-hot-path operations such as:

```python
bool(cuda_tensor)
tensor.item()
torch.isfinite(...).all()
tensor.ne(0).any()
print(...)
decode(...)
PIL.Image.open(...)
processor(...)
```

Identify operations that force GPU synchronization after every episode/backward.

Specifically audit repeated trainable-parameter gradient scans.

---

## 3.10 Establish baseline measurements

Before patches, add or run lightweight instrumentation sufficient to capture:

```text
peak memory allocated
peak memory reserved
time / episode
episodes / second
WHERE forward time
semantic forward time
backward time
optimizer-step time
sequence length
number of images
K
number of fixations
number of supervised WHERE tokens
number of supervised semantic tokens
```

Use CUDA synchronization only around explicit profiling boundaries.

Do not leave expensive fine-grained synchronization enabled in production training.

---

# 4. Audit report gate

Before substantive refactoring, produce a concise audit report containing:

```text
1. actual call graph;
2. confirmed batch-size-1 assumptions;
3. confirmed largest VRAM allocations;
4. confirmed redundant computation;
5. confirmed synchronization points;
6. exact loss-reduction semantics;
7. proposed patch order;
8. expected risks.
```

Then continue directly to patching.

Do not wait for human confirmation unless blocked by missing source or an impossible ambiguity.

---

# 5. Phase II — required patches

The implementation target is the following priority order.

---

# 5.1 P0 — selective supervised-position vocabulary logits

## Problem

The model must not materialize full `[B,L,V]` logits for positions that do not contribute to the loss.

This is especially wasteful because support turns, prompts, image/context tokens, and padding are ignored by labels.

---

## Required behavior

Preserve the exact causal-LM objective.

For standard shifted causal LM labels:

```python
predictor_hidden = final_hidden[:, :-1, :]
targets = labels[:, 1:]
valid = targets != -100
```

For each episode, compute vocabulary logits only for:

```python
predictor_hidden[b][valid[b]]
```

The classifier must remain the same trainable LM head / `RowHead`.

The softmax vocabulary remains **full vocabulary**.

Forbidden:

```text
target-only logits
restricted vocabulary
sampled softmax
approximate CE
top-k vocabulary CE
detached hidden states
```

---

## Required loss reduction

Return per-token NLL first.

Then reconstruct per-episode mean loss.

Example conceptual structure:

```python
token_nll = cross_entropy(
    selected_logits,
    selected_targets,
    reduction="none",
)

episode_loss[b] = token_nll_for_b.mean()
batch_loss = torch.stack(episode_loss).mean()
```

If the previous single-episode implementation has different boundary handling, match it exactly.

---

## RowHead requirement

Audit the current trainable `<END_FIX>` row mechanism.

Do not replace its scientific behavior.

However, avoid constructing an additional full `[B,L,V]` tensor merely to substitute one vocabulary row.

The head should support selected hidden states:

```text
[M, d] -> [M, V]
```

where `M` is only the number of supervised predictor positions.

Preserve gradients to:

- all current trainable LoRA parameters;
- trainable `<END_FIX>` row;
- projector;
- any other currently trainable parameters.

---

## Verification

For fixed seeds and one episode:

```text
old WHERE loss ≈ new WHERE loss
old semantic loss ≈ new semantic loss
old total loss ≈ new total loss
```

Compare gradients for representative trainable parameters:

```text
LoRA parameter
END_FIX trainable row
P_E parameter
```

Use reasonable bf16 tolerances.

---

# 5.2 P0 — remove all-layer hidden-state materialization

## Problem

WHERE currently needs only the final LM hidden tensor for `<END_FIX>` extraction.

Do not request/store every intermediate Transformer hidden state.

---

## Required implementation

Refactor WHERE to obtain the same final language hidden state directly from the backbone.

Target behavior:

```text
final_hidden: [B, L, d]
```

Then use the existing query-only `<END_FIX>` selection logic.

Set:

```python
output_hidden_states=False
```

unless a specific compatibility reason requires otherwise.

---

## Verification

For the same episode and same model state:

```text
old selected F
new selected F
```

must match numerically within bf16 tolerance.

The number/order of extracted fixation states must remain identical.

---

# 5.3 P0 — remove hot-path gradient synchronization scans

## Problem

Do not scan every trainable parameter with CUDA reductions after every episode if those checks exist only as diagnostics.

---

## Required implementation

Keep comprehensive gradient diagnostics in:

```text
smoke tests
preflight
first real training step
explicit debug mode
optional sparse periodic diagnostics
```

Do not perform repeated per-parameter:

```python
torch.isfinite(...).all()
grad.ne(0).any()
bool(cuda_tensor)
.item()
```

in the normal per-episode hot path.

Preserve existing gradient clipping and non-finite failure behavior at an appropriate aggregate level.

---

## Scientific behavior

No optimization tensor may be modified differently.

This is execution-only cleanup.

---

# 5.4 P1 — true physical WHERE batching

Implement a new batch-capable collation/forward path.

Do not fake batching by looping over samples internally.

---

## Batched episode representation

The batch object must retain per-sample metadata, including at least:

```text
episode ID / record ID
K
query fixation count N_b
WHERE sequence length
query assistant span / supervision mask
image count
image ordering
query image index
support image indices
semantic target metadata
```

---

## WHERE text tensors

Produce standard padded tensors:

```text
input_ids      [B, Lmax]
attention_mask [B, Lmax]
labels         [B, Lmax]
```

Requirements:

```text
padding attention_mask = 0
padding labels = -100
support/query masking unchanged
```

Use right padding unless the underlying current causal behavior requires another convention.

---

## Multimodal image batching

Preserve exact image order within each episode.

The implementation must know which image features belong to which sample.

Do not allow image placeholders from one sample to consume image features from another sample.

Add explicit assertions covering:

```text
number of image placeholders
number of image feature groups
per-sample image count
support/query image ordering
```

---

# 5.5 P1 — batch-aware WHERE state extraction

The extractor may output either:

```text
List[Tensor[N_b, d]]
```

or:

```text
padded_states [B, Nmax, d]
state_mask    [B, Nmax]
```

but it must preserve per-episode ordering and exact counts.

Required assertion:

```text
selected query END_FIX count for sample b == N_b
```

Never include:

```text
support END_FIX
padding END_FIX
masked history END_FIX
```

---

# 5.6 P1 — batched projector

Apply the same shared projector to every valid fixation state.

Padding must not become semantic state.

Either:

```text
project list elements independently
```

or:

```text
project padded tensor and mask padding correctly
```

Do not change projector architecture or parameter sharing.

---

# 5.7 P1 — batch-aware semantic state insertion

Remove any `batch_size == 1` restriction.

For each sample:

1. construct the exact same semantic prompt/target as before;
2. locate that sample's semantic insertion positions;
3. insert only that sample's projected state sequence;
4. preserve differentiability;
5. right-pad the final fused semantic embedding sequence across batch.

Expected semantic tensors may look like:

```text
inputs_embeds   [B, Smax, d]
attention_mask  [B, Smax]
labels          [B, Smax]
```

Do not copy or detach states in a way that breaks:

```text
semantic loss -> inserted states -> P_E -> WHERE
```

Add an explicit gradient-flow regression test.

---

# 5.8 P1 — true physical semantic batching

Run one semantic InternVL forward for the physical batch rather than one forward per episode.

Use the same selective supervised-position vocabulary-logit strategy as WHERE.

Compute:

```text
per-episode semantic loss
then batch mean
```

Do not use a global supervised-token mean across all episodes.

---

# 5.9 P1 — end-to-end physical training batch

The final training step should conceptually become:

```text
episodes = sample B episodes

where_batch = collate_where_batch(episodes)

where_result = forward_where_batch(where_batch)
    -> per-episode WHERE losses
    -> differentiable F_b

R_b = P_E(F_b)

semantic_batch = prepare_semantic_batch(
    episodes,
    R_b,
)

semantic_result = forward_semantic_batch(semantic_batch)
    -> per-episode semantic losses

episode_total_b =
    lambda_where * where_loss_b
    + lambda_semantic * semantic_loss_b

loss = mean(episode_total_b)

loss.backward()

gradient clipping

optimizer.step()
scheduler.step()
optimizer.zero_grad()
```

There must be **one vectorized WHERE model forward** and **one vectorized semantic model forward** for a physical batch, except where model internals naturally execute subcomponents.

A Python loop used only for metadata assembly or per-sample indexing is acceptable.

A Python loop invoking full InternVL separately for each episode is not true physical batching.

---

# 5.10 P1 — separate physical batch size from gradient accumulation

The configuration must have unambiguous meanings.

Recommended semantics:

```yaml
training:
  per_device_train_batch_size: <true physical GPU episode batch>
  gradient_accumulation_steps: <number of physical batches accumulated>
```

Then:

```text
effective optimizer batch =
    per_device_train_batch_size
    * gradient_accumulation_steps
    * world_size
```

Do not preserve the current misleading interpretation if the source currently uses `per_device_train_batch_size` as a sequential episode count.

Update README/config comments accordingly.

---

# 5.11 P1 — length-aware / K-aware training batches

Once true batching works, reduce padding waste.

Do not alter the K distribution.

Allowed strategy:

1. sample episodes according to the current frozen sampling logic;
2. place already-sampled episodes into a short-lived batching buffer;
3. group episodes with similar estimated cost;
4. preserve every sampled episode;
5. do not bias/drop/resample K values.

Cost signals may include:

```text
K
number of images
estimated WHERE token length
query fixation count
semantic target length
```

Do not reorder across optimizer steps in a way that changes which episodes belong to a configured minibatch unless the resulting training semantics are intentionally equivalent and documented.

Start simple:

```text
K-aware queues
+
coarse length buckets
```

before implementing a complex token-budget scheduler.

---

# 5.12 P1/P2 — frozen visual feature reuse

First confirm from source that the relevant components are frozen.

If both are frozen:

```text
vision tower
native multimodal projector
```

then frozen projected visual features may be reused.

---

## Safe first optimization

Within the same episode / physical batch:

```text
compute query visual features once
reuse for WHERE and semantic
```

if the exact same preprocessing and feature representation are consumed.

This avoids the semantic branch re-encoding the query image.

---

## Optional later cache

A dataset-level feature cache may be added for frozen image features.

Cache key must include everything needed to guarantee feature identity, e.g.:

```text
image identity/path
image preprocessing configuration
processor/model revision
image size / dynamic tiling configuration
```

Do not cache trainable language states.

Do not silently cache stale features if the vision path becomes trainable in another configuration.

Fail or disable the cache automatically when its producer has trainable parameters.

---

# 5.13 P2 — attention backend

After the exact batching changes work:

1. record the active text attention backend;
2. record the active vision attention backend;
3. benchmark current behavior;
4. if text attention is `eager`, test SDPA;
5. optionally test Flash Attention 2 on the A100.

Do not make Flash Attention a hard dependency unless there is a demonstrated benefit and compatibility is verified.

Keep a safe fallback.

Numerical kernels may not be bitwise identical, but this is not a scientific-method change if model inputs/objective remain unchanged.

Document the backend used in run metadata.

---

# 5.14 P2 — CPU preprocessing / transfer overlap

Only after GPU-side issues are resolved:

- avoid duplicate query image loading;
- use DataLoader workers or equivalent preprocessing workers;
- use pinned host memory where useful;
- use `non_blocking=True` H2D transfers where safe;
- prefetch upcoming physical batches;
- move invariant serialization/tokenization assertions out of the normal hot path when covered by tests.

Do not let asynchronous preprocessing make episode sampling nondeterministic.

---

# 5.15 P2 — allocator behavior

Add optional profiling for:

```python
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

Distinguish:

```text
live tensor memory
vs
allocator-reserved memory
```

Do not call:

```python
torch.cuda.empty_cache()
```

after every episode or every step.

Allocator configuration changes must be optional and documented.

---

# 6. Gradient checkpointing policy

Keep gradient checkpointing enabled during the main refactor.

After the P0/P1 memory reductions are implemented, benchmark:

```text
B=1, checkpointing on
B=1, checkpointing off

B=2, checkpointing on
B=2, checkpointing off

larger B if safe
```

Choose throughput based on:

```text
episodes/sec
not merely step/sec
```

Do not disable checkpointing globally unless the resulting configuration fits the target A100 40GB with meaningful safety headroom.

---

# 7. Optimizer-memory policy

Do not introduce optimizer quantization as a primary fix.

First inspect current optimizer state memory.

Because LoRA is used, optimizer-state memory may be much smaller than activation/logit memory.

Only consider alternative optimizer-state formats if profiling proves optimizer memory is material after P0/P1 fixes.

Any optimizer change must be labeled separately because it can affect numerical behavior.

---

# 8. Required tests

Do not consider the patch complete without tests.

## 8.1 B=1 forward parity

Same model state, same episode:

```text
old WHERE loss vs new WHERE loss
old semantic loss vs new semantic loss
old total loss vs new total loss
old extracted F vs new F
```

---

## 8.2 B=1 gradient parity

Compare representative gradients:

```text
one early LoRA parameter
one middle LoRA parameter
one late LoRA parameter
END_FIX trainable row
P_E parameter
```

All required gradient paths must remain nonzero where expected.

---

## 8.3 Batched loss equivalence

For episodes A and B:

```text
batched_loss(A,B)
```

must match:

```text
(loss(A) + loss(B)) / 2
```

within numerical tolerance.

Test episodes with different:

```text
K
WHERE lengths
fixation counts
semantic lengths
```

---

## 8.4 State extraction correctness

For a mixed batch:

```text
N = [N1, N2, ...]
```

verify that every sample returns exactly its own query `<END_FIX>` states in chronological order.

---

## 8.5 Semantic insertion correctness

Verify:

- state sequence is inserted into the correct sample;
- padding does not receive states;
- state order is preserved;
- gradients flow from semantic loss into WHERE.

---

## 8.6 Multimodal isolation test

Batch two deliberately different episodes/images.

Ensure sample A never receives image features from sample B.

---

## 8.7 Sampling invariance

With the same seed/config, batching infrastructure must not change:

```text
subject sampling
K sampling
query sampling
support sampling
support order
```

unless a batching buffer intentionally reorders already-sampled episodes.

If order changes only for throughput but episode membership is identical, document it.

---

## 8.8 Smoke tests

Update:

```text
scripts/smoke_flat.py
```

or the repository's actual smoke suite to cover:

```text
B=1
B=2
mixed K if supported
gradient path
checkpoint save/load if batching metadata changes
```

---

# 9. Profiling / acceptance criteria

The final report must include before/after measurements on the same representative workload.

At minimum:

```text
physical batch size
K
sequence length
number of images
fixation count
peak allocated VRAM
peak reserved VRAM
episodes / second
seconds / episode
```

Measure at least:

```text
B=1 old
B=1 optimized
B=2 optimized
```

If B>2 fits, include it.

---

## Target outcomes

Primary target:

```text
true B >= 2 physical training on A100 40GB
```

without changing scientific behavior.

Secondary target:

```text
higher episodes/sec than the original sequential implementation
```

The patch is not successful if it merely changes the meaning of batch size while invoking full model forwards sequentially.

---

# 10. Implementation order

Use this order unless source inspection shows a concrete reason to change it.

```text
0. Instrument baseline
1. Selective supervised-position logits + exact per-episode CE
2. Remove all-layer hidden-state capture
3. Remove hot-path gradient sync diagnostics
4. Generalize batch data structures while still testing B=1
5. Batched WHERE collation + forward
6. Batched <END_FIX> extraction
7. Batched semantic state insertion
8. Batched semantic forward
9. End-to-end true physical B=2 training
10. K/length-aware batching
11. Reuse frozen query visual features
12. Optional broader frozen visual cache
13. Attention backend tuning
14. CPU prefetch / pinned transfer
15. Gradient-checkpointing retuning
16. Allocator tuning if still needed
```

Do not begin with:

```text
detach
lower K
reduce validation/training data
change supervision
change semantic architecture
lower image resolution
QLoRA
8-bit optimizer
loss approximation
```

unless later profiling proves one is necessary and it is explicitly presented as a scientific/numerical trade-off.

---

# 11. Code-quality requirements

## 11.1 Prefer narrow reusable helpers

Examples:

```text
compute_selected_causal_nll(...)
collate_where_batch(...)
extract_query_states_batch(...)
prepare_semantic_batch(...)
insert_states_batch(...)
```

Avoid duplicating old and new loss logic in multiple places.

---

## 11.2 Explicit shapes

Use comments/type hints/assertions for important tensors:

```text
input_ids:      [B, L]
labels:         [B, L]
final_hidden:   [B, L, D]
selected_state: [N_b, D]
inputs_embeds:  [B, S, D]
```

---

## 11.3 No silent fallback to serial execution

If physical batching cannot handle a case, either:

```text
raise a clear error
```

or explicitly use a debug compatibility mode.

Do not advertise `per_device_train_batch_size > 1` while internally running one model forward per episode.

---

## 11.4 Preserve checkpoint compatibility

Do not rename existing trainable parameters unnecessarily.

Existing DeepGaze LoRA loading and SemGaze checkpoint resume should continue to work.

If checkpoint metadata must be extended, maintain backward-compatible defaults.

---

# 12. Final deliverable

After patching, return a report with these sections.

## A. Audit findings

For each confirmed issue:

```text
Problem
Source location
Evidence
VRAM impact
Speed impact
```

---

## B. Changes implemented

For each patch:

```text
Files changed
Implementation summary
Scientific behavior changed? yes/no
```

---

## C. Exact-semantics verification

Report:

```text
B=1 loss parity
B=1 hidden-state parity
gradient parity
batched loss equivalence
sampling invariance
multimodal isolation
```

---

## D. Performance results

Use a compact table:

```text
Variant | Physical B | K | Peak allocated | Peak reserved | episodes/s
```

---

## E. Remaining bottlenecks

List only bottlenecks supported by the new profiler.

Do not speculate from `nvidia-smi` alone.

---

# 13. Definition of done

This task is complete only when all of the following are true:

```text
[ ] actual source was audited before refactor
[ ] full-sequence vocabulary logits are removed from the training hot path where unnecessary
[ ] all-layer hidden-state output capture is removed from WHERE
[ ] hot-path gradient diagnostic synchronizations are removed/reduced
[ ] WHERE supports true B>1 physical batching
[ ] state extraction supports variable N_b
[ ] semantic state insertion supports B>1
[ ] semantic forward supports true B>1
[ ] loss reduction remains episode-mean
[ ] semantic gradients still reach WHERE
[ ] per_device_train_batch_size means true physical batch size
[ ] gradient_accumulation_steps has conventional meaning
[ ] B=1 parity tests pass
[ ] B=2 equivalence tests pass
[ ] smoke tests pass
[ ] A100 40GB memory/throughput profile is reported
```

Do not stop at a partial refactor that leaves the model execution serial.
