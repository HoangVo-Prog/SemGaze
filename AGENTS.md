# Objective

Audit first, then implement only the three evaluation-system optimizations specified here for the current `flat` branch of SemGaze.

This file is an implementation specification, not permission to change the method. The intended integrated evaluation cycle remains:

```text
training
→ evaluation loss pass
→ prediction/generation pass
```

The implementation target is to remove three exact-semantics redundancies:

```text
1. recomputing GT teacher-forced WHERE + SemGaze projector during prediction
   when the identical projected GT representation R was already produced by
   the immediately preceding evaluation-loss pass;

2. moving `pixel_values` CPU→GPU for images whose final frozen visual features
   are already present in `InferenceVisualCache`;

3. discarding the frozen visual cache after loss and warming a second visual
   cache again during prediction in the same integrated evaluation cycle.
```

Do not implement any other optimization in this task.

## Audited current source

The current source architecture was audited before writing this specification. The relevant current files/functions are:

```text
train_flat.py
    main()
    -> run_training_loop(...)

semgaze/training/loop.py
    run_training_loop(...)
        -> evaluate_test_epoch(...)
        -> predict_epoch(...)

semgaze/evaluation/test.py
    test_mode(...)
    batch_losses(...)
    evaluate_test_episode(...)
    evaluate_test_epoch(...)

semgaze/evaluation/predictions.py
    _prediction_record(...)
    prediction_batches(...)
    predict_epoch(...)

semgaze/evaluation/flat.py
    evaluate_flat_batch(...)
    evaluate_flat_episode(...)

semgaze/evaluation/where.py
    evaluate_where_episode(...)

semgaze/where/forward.py
    forward_where(...)
    forward_where_batch(...)

semgaze/where/generation.py
    generate_where_batch(...)
    generate_where(...)

semgaze/state/extractor.py
    extract_query_states(...)

semgaze/state/projector.py
    build_projector(...)

semgaze/semantic/flat/forward.py
    prepare_flat_inputs(...)
    prepare_semantic_batch(...)
    forward_flat_batch(...)
    frozen_vision_is_reusable(...)

semgaze/model/visual_cache.py
    InferenceVisualCache
        processor_for(...)
        fuse(...)
        statistics(...)
        close(...)

semgaze/where/collator.py
    load_images(...)
    collate_native(...)
    collate_where(...)
    to_model_device(...)
    pack_where_batch(...)
    collate_where_batch(...)

semgaze/evaluation/batching.py
    episode_id(...)
    schedule_batches(...)

evaluate_flat.py
    standalone prediction entrypoint
```

Important source facts that the implementation must preserve:

1. `batch_losses()` currently performs:

```text
forward_where_batch(...)
→ extract query END_FIX states
→ bundle.projector(...)
→ per-episode projected states R
→ forward_flat_batch(...)
```

The current projected representation is created at:

```python
states = bundle.projector(torch.cat(where.states)).split(counts)
```

and each element has shape:

```text
[N_fixations, d_model]
```

2. `evaluate_flat_batch()` currently repeats:

```text
collate GT WHERE
→ forward_where_batch(...)
→ bundle.projector(...)
→ prepare_semantic_batch(...)
→ semantic generate(...)
```

Therefore the GT WHERE forward and SemGaze projector are currently recomputed during prediction.

3. Canonical semantic prediction intentionally uses GT XYD/END_FIX states, not the free-running predicted scanpath. This invariant must not change.

4. `prepare_semantic_batch()` currently obtains query-image reuse from a `WhereOutput`:

```text
where.query_image_features
where.batch.image_cache
```

and verifies that WHERE query preprocessing equals semantic query preprocessing before inserting the query visual feature.

This dependency must be handled explicitly when cached `R` allows prediction to skip the GT WHERE forward.

5. `forward_where_batch()` currently calls `to_model_device(...)` on the complete non-label WHERE input before `visual_cache.fuse(...)`.

6. `generate_where_batch()` similarly calls `to_model_device(...)` before `cache.fuse(...)`.

Therefore the current final-feature cache can avoid the frozen vision forward on a hit but cannot avoid the already-performed pixel H2D transfer.

7. `InferenceVisualCache.fuse()` currently keys final visual features using image path plus the already-moved pixel tensor's dtype/device. It selects visual misses only after `pixel_values` are on the model device.

8. `evaluate_test_epoch()` currently creates and closes its own `InferenceVisualCache`.

9. `predict_epoch()` currently creates and closes a second `InferenceVisualCache`.

10. `evaluate_flat.py` is a standalone prediction entrypoint. It creates one prediction-local `InferenceVisualCache` and closes it after standalone prediction. This behavior must remain valid.

11. `serial_reference_losses()` in `semgaze/evaluation/test.py` is a useful reference path. Do not optimize away its independent recomputation; keep it suitable for parity/reference checks unless a separate test-only reason requires otherwise.

Before implementing, run:

```bash
git rev-parse HEAD
git status --short
```

Record the inspected commit in the implementation report. If the functions above have materially changed since this specification was produced, re-audit those call sites before editing.

# Non-negotiable scientific invariants

The implementation must preserve all scientific and model-facing behavior.

Do not change:

```text
COCO-Search18 split membership
seen/unseen subject definitions
test query set
support-set membership
support-set order
support draws
draw IDs
same-subject support rule
query/support exclusion rules
K values or K meaning
number of support draws
WHERE prompt construction
WHERE tokenization
WHERE XYD serialization
END_FIX semantics
GT END_FIX state readout position
SemGaze projector weights or computation
semantic prompt construction
semantic state insertion positions
semantic use of canonical GT WHERE states
semantic gold WHY-group behavior
WHERE generation algorithm
semantic generation algorithm
generation budgets
generation EOS behavior
prediction text
prediction serialization schema
loss definitions
loss aggregation
metrics
metric aggregation
model architecture
model weights
input resolution
image preprocessing semantics
number/order of multimodal image tokens
physical batching policy
training behavior
```

Specific invariants for these fixes:

```text
R must remain the exact projected GT WHERE representation.

R must not be:
- averaged;
- truncated;
- quantized;
- cast to a lower precision;
- recomputed from predicted fixations;
- replaced by raw WHERE hidden states;
- modified by a new normalization;
- detached as a new semantic operation.

Evaluation already runs under inference/no-grad semantics. Do not introduce
a `.detach()` transformation as part of the cache design.

The semantic branch must still see the same chronological R rows at the same
state insertion positions.

A visual-cache hit must return the same native frozen visual feature that the
current vision tower + native multimodal projector would produce from the
canonical preprocessing.

Image feature order must remain exactly:
support 1, support 2, ..., support K, query
within each episode, and episode order must remain the current physical-batch
flattening order.

No cache may silently survive into an incompatible model/checkpoint,
processor/preprocessing configuration, dtype, device, or evaluation cycle.
```

The implementation must be fail-closed. If compatibility cannot be proven, treat the entry as a miss or raise a clear error. Never silently reuse a questionable cache entry.

# Current evaluation architecture

## Integrated training-time evaluation

`run_training_loop()` currently does, on an evaluation boundary:

```text
evaluate_test_epoch(...)
→ append/log loss summary
→ predict_epoch(...)
→ append/log prediction summary
```

The same model/projector parameters are used for both calls. No optimizer step occurs between them.

This is the correct place to own resources whose lifetime is exactly one integrated evaluation cycle.

## Evaluation-loss path

Current loss path:

```text
evaluate_test_epoch
→ frozen_episode(...)
→ schedule_batches(...)
→ batch_losses(...)
→ forward_where_batch(...)
→ extract_query_states(...)
→ bundle.projector(...)
→ forward_flat_batch(...)
→ selected WHERE/semantic NLL
```

`batch_losses()` already has the exact per-episode projected GT representation needed later by semantic prediction.

## Prediction path

Current same-K batched prediction path:

```text
predict_epoch
→ frozen_episode(...)
→ prediction_batches(...)

prediction_batches:
    generate_where_batch(...)          # free-running WHERE
    evaluate_flat_batch(...)           # semantic path

evaluate_flat_batch:
    collate GT WHERE again
    → forward_where_batch(...)
    → bundle.projector(...)
    → prepare_semantic_batch(...)
    → bundle.model.generate(...)
```

The free-running WHERE output remains required. Only the second GT teacher-forced WHERE + projector computation is redundant when compatible `R` is available from the loss pass.

## Current visual-feature path

Current loss WHERE path is effectively:

```text
CPU collation/preprocessing
→ batch.inputs contains CPU pixel_values
→ to_model_device(all non-label inputs)
→ pixel_values now on GPU
→ InferenceVisualCache.fuse(...)
→ final-feature hit/miss lookup
→ frozen vision only for misses
```

Current WHERE generation follows the same problematic ordering:

```text
CPU collation
→ left-pad
→ to_model_device(...)
→ InferenceVisualCache.fuse(...)
```

The required refactor is to perform final-feature hit/miss resolution before moving pixel tensors to the GPU.

## Current cache lifecycle

Current integrated lifecycle:

```text
evaluate_test_epoch:
    create visual cache A
    use A
    close A

predict_epoch:
    create visual cache B
    use B
    close B
```

Target integrated lifecycle:

```text
run_training_loop:
    create visual cache V
    create projected-R cache R_cache

    evaluate_test_epoch(..., visual_cache=V, projected_r_cache=R_cache)
    predict_epoch(..., visual_cache=V, projected_r_cache=R_cache)

    close R_cache
    close V
```

Standalone paths must still own their own cache when no external cache is supplied.

# Required Change 1 — Cross-pass GT R reuse

## Current behavior

CURRENT:

```text
Loss:
GT WHERE teacher-forced forward
→ final-layer query END_FIX hidden states
→ SemGaze projector
→ R
→ semantic loss

Prediction:
free-running WHERE generation
→ another GT WHERE teacher-forced forward
→ another SemGaze projector call
→ R again
→ semantic generation
```

The second GT WHERE + projector computation is redundant in the integrated loss→prediction cycle because:

```text
- the model/projector weights do not change between the two passes;
- the frozen evaluation episode is the same;
- semantic prediction intentionally consumes canonical GT WHERE states;
- the loss pass has already computed the exact projected R required.
```

## Target behavior

TARGET:

```text
Loss:
GT WHERE
→ END_FIX hidden states
→ projector
→ R
→ store exact per-episode R in an evaluation-cycle cache
→ semantic loss uses the same R object/value as before

Prediction:
free-running WHERE generation remains unchanged
→ look up exact frozen episode in projected-R cache
→ if hit:
       do NOT run GT WHERE
       do NOT run SemGaze projector
       use cached R for semantic generation
  else:
       run the current GT WHERE + projector fallback
→ semantic generation
```

The cache is an optimization only. A miss must reproduce current behavior exactly.

### Required new cache abstraction

Add a dedicated evaluation-only cache, preferably in:

```text
semgaze/evaluation/cache.py
```

Recommended public shape:

```python
class ProjectedWhereCache:
    def __init__(self, *, bundle, split_manifest_identity, cycle_id):
        ...

    def key_for(self, episode):
        ...

    def put(self, episode, projected_r):
        ...

    def get(self, episode):
        ...

    def contains(self, episode):
        ...

    def close(self):
        ...
```

Equivalent naming is acceptable, but the semantics below are mandatory.

Do not put projected R into `InferenceVisualCache`; the lifetimes and compatibility rules are different.

### What is cached

Cache the projected representation produced by the existing projector:

```python
R = bundle.projector(where_state)
```

Cache one tensor per evaluation episode:

```text
shape = [N_fixations, d_model]
dtype = exact projector output dtype
values = exact projector output values
```

Do not cache only raw END_FIX hidden states, because the target optimization explicitly removes both the redundant WHERE forward and the redundant projector call.

### Storage

The projected-R cache must survive the complete loss pass until prediction consumes it.

Do not retain an unbounded evaluation's `R` tensors on GPU by default.

Use exact-value CPU storage for the cross-pass cache unless the existing implementation can prove a bounded GPU lifetime:

```text
- preserve dtype exactly;
- do not quantize;
- do not convert bf16→fp16/fp32 or vice versa;
- do not average/compress;
- do not serialize through text/numpy;
- do not call detach as a semantic operation;
- copy/move under inference mode only.
```

A CPU device copy preserves tensor values and is acceptable. On retrieval, the existing state insertion path may move the exact tensor to the embedding device/dtype as it already does.

If implementation chooses another storage location, parity requirements below still apply and the cache must not cause evaluation OOM.

## Cache key / ownership

### Episode identity

The cache key must uniquely encode the exact WHERE episode, not just query ID.

At minimum include:

```text
query:
    record_id
    stimulus_id
    subject
    image_path
    task
    condition
    image_width
    image_height
    x_px
    y_px
    duration_ms

supports, IN ORDER:
    for every support:
        record_id
        stimulus_id
        subject
        image_path
        task
        condition
        image_width
        image_height
        x_px
        y_px
        duration_ms

episode:
    K = len(supports)
    draw_id
```

Support order is part of the key.

The exact query GT XYD trajectory is part of the key because the hidden states at supervised query END_FIX positions depend on the complete teacher-forced query response.

The exact support XYD trajectories are part of the key because support assistant responses are in the WHERE conversation.

Do not use Python's randomized `hash()` as the sole persistent identity. A tuple/dataclass of immutable fields is sufficient for an in-process dictionary; if a digest is used, use a deterministic collision-resistant digest over a canonical serialization.

### Cache provenance

Episode identity alone is not sufficient.

Bind each `ProjectedWhereCache` instance to exactly one evaluation cycle and one live model state.

The cache must retain/check provenance including:

```text
live model object identity
live SemGaze projector object identity
processor/tokenizer object identity or equivalent strict identity
END_FIX ID
END_FIX token
protocol version
split_manifest_identity
evaluation cycle identity
```

For integrated training evaluation, `cycle_id` should include the current evaluation boundary, e.g.:

```text
(epoch, global_step)
```

The primary safety mechanism is lifecycle:

```text
create only after the optimizer step that precedes evaluation
do not perform optimizer updates while the cache is live
close it before training resumes
never store it globally
```

Do not rely on the cache remaining valid after any optimizer step.

### Why include both K/draw and support identities

`draw_id` is not enough by itself.

Two episodes must not collide when:

```text
same query, different K
same query, same K, different draw
same query, same K/draw metadata but different ordered supports
same support set in a different order
same record ID with mutated WHERE-relevant content in a unit test
```

The key must make all of these distinct.

## Fallback behavior

BACKWARD-COMPATIBILITY BEHAVIOR:

If any of the following is true:

```text
no ProjectedWhereCache supplied
cache provenance incompatible
episode key absent
episode key invalid
standalone prediction executed without preceding loss
training prediction episode has no matching test-loss entry
```

then semantic prediction must execute the current baseline path:

```text
forward_where / forward_where_batch
→ extract GT END_FIX states
→ bundle.projector
→ semantic generation
```

Do not fail standalone prediction merely because `R` is absent.

A provenance mismatch must never fall through to a wrong cache hit. It may be treated as a miss or as a clear compatibility error depending on API location.

### Mixed hit/miss physical batches

`evaluate_flat_batch()` must support a physical group in which some episodes hit the R cache and others miss.

Required behavior:

```text
lookup all episode keys first

hits:
    take exact cached R

misses:
    construct GT WHERE work only for misses
    run GT WHERE only for misses
    project only misses
    optionally populate the cache with newly produced R

restore R list to the original episode order

semantic generation:
    consume R in original episode order
```

Do not recompute GT WHERE for the full physical group merely because one episode misses.

## Necessary semantic query-vision decoupling

WHY SAFE / REQUIRED ARCHITECTURAL DETAIL:

Current `prepare_semantic_batch()` uses a full `WhereOutput` to obtain:

```text
where.query_image_features
where.batch.image_cache
```

If all R entries hit, there is no GT `WhereOutput`.

Do not solve this by running a new GT WHERE forward only to obtain query vision; that would defeat Change 1.

Refactor semantic preparation so query visual reuse can come from either:

```text
A. existing WhereOutput, for training/loss/current fallback paths; or
B. compatible InferenceVisualCache, for optimized prediction with cached R.
```

Recommended API direction:

```python
prepare_semantic_batch(
    bundle,
    queries,
    states,
    *,
    where=None,
    visual_cache=None,
    generation_budget=None,
    reuse_query_vision=None,
)
```

Equivalent design is acceptable.

For the `visual_cache` path:

```text
- collate the canonical semantic native prompt exactly as today;
- obtain the query's final frozen visual feature from InferenceVisualCache;
- if it is absent, compute/cache that query visual feature using the exact
  current preprocessing;
- preserve the current host-side preprocessing identity/equality check;
- masked-scatter the same feature into the same semantic image-token slots;
- then insert R at the same state boundaries.
```

The semantic language-model input must be equivalent to the baseline input.

Do not synthesize image tokens or bypass the native processor in a way that could change tokenization.

## Files/functions affected

FILES/FUNCTIONS AFFECTED:

Required audit/edit targets:

```text
semgaze/evaluation/cache.py
    new ProjectedWhereCache / episode-key helper

semgaze/evaluation/test.py
    batch_losses(...)
    evaluate_test_epoch(...)
    possibly evaluate_test_episode(...) only for cache ownership consistency
    serial_reference_losses(...) should remain an independent reference

semgaze/evaluation/flat.py
    evaluate_flat_batch(...)
    evaluate_flat_episode(...)

semgaze/evaluation/predictions.py
    prediction_batches(...)
    predict_epoch(...)

semgaze/semantic/flat/forward.py
    prepare_semantic_batch(...)
    possibly a small helper for query-feature injection independent of WhereOutput

semgaze/training/loop.py
    run_training_loop(...)
    create/pass/close the cycle-scoped projected-R cache

semgaze/model/visual_cache.py
    expose safe query-feature retrieval needed when no WhereOutput exists
```

Avoid changing `_prediction_record()` output fields.

## Required parity checks

PARITY TEST:

For deterministic frozen test episodes covering K=1, K=5, and K=10:

1. Compute baseline prediction using forced R-cache miss/current recomputation.
2. Compute optimized prediction using R captured from the preceding loss pass.
3. Assert:

```text
cached R == recomputed R
```

Use `torch.equal` when both paths use the same kernel/shape execution. If bitwise equality is not guaranteed by the same hardware/backend, use the narrowest justified numerical tolerance and document why. Do not use a loose tolerance.

Also assert:

```text
state count identical
state order identical
semantic insertion positions identical
semantic attention_mask identical
semantic input embeddings equivalent
semantic generated token IDs exactly identical
decoded semantic text exactly identical
serialized semantic prediction exactly identical
loss values unchanged
existing metrics unchanged
```

Generated token IDs must be exactly equal. Do not accept "close" token IDs.

Capture generated token tensors in test instrumentation; do not add token IDs to the production JSON schema solely for testing.

Required key/fallback tests:

```text
integrated loss→prediction uses cached R for matching test episodes

standalone prediction with no R cache recomputes GT WHERE and succeeds

training predictions without a matching loss-cache key recompute and succeed

different K cannot collide

different draw_id cannot collide

different ordered support IDs cannot collide

same support membership in a different order cannot collide

different query GT trajectory cannot collide

different support GT trajectory cannot collide

different split_manifest_identity cannot reuse the cache

different live model/projector/cycle cannot reuse the cache
```

Use call counters/mocks around `forward_where_batch()` and `bundle.projector` to prove that an all-hit semantic prediction group invokes neither redundant operation.

# Required Change 2 — Cache lookup before pixel H2D

## Current behavior

CURRENT:

`forward_where_batch()` currently moves all non-label inputs to the model device before calling `visual_cache.fuse()`.

Conceptually:

```text
batch.inputs['pixel_values'] on CPU
→ to_model_device(...)
→ all pixel_values moved to GPU
→ visual_cache.fuse(...)
→ feature hit/miss lookup
→ frozen vision executes only for misses
```

`generate_where_batch()` has the same ordering.

`InferenceVisualCache.fuse()` currently receives already-device-resident pixels and forms final-feature keys from:

```text
(path, pixel dtype, pixel device)
```

Therefore even a final visual-feature hit already paid pixel H2D.

## Target behavior

TARGET:

```text
image/preprocessing identities
→ validate cache provenance
→ final visual-feature lookup on host metadata first
→ identify hit indices and miss indices

hits:
    no pixel H2D
    no frozen vision execution

misses:
    use canonical preprocessed CPU pixel rows
    collate only the miss rows for GPU vision work
    move only miss pixels CPU→GPU
    run frozen vision only for misses
    cache exact output feature

merge:
    reconstruct final features in the original flattened image order
    inject them into the exact same image-token positions
```

The text/token inputs may be moved independently as they are today. The prohibited operation is moving hit-image `pixel_values` to GPU.

### Minimum required call-order change

In both loss WHERE and WHERE generation, do not call the generic `to_model_device()` on `pixel_values` before visual cache lookup.

Separate:

```text
text/model inputs
host pixel inputs
```

before device transfer.

Recommended structure:

```python
host_inputs = ...
host_pixels = ...

if visual_cache is active and final-feature caching is enabled:
    model_inputs = to_model_device(inputs_without_pixel_values, bundle.model)
    fused_inputs, features = visual_cache.fuse(
        bundle,
        model_inputs,
        image_paths,
        host_pixel_values=host_pixels,
    )
else:
    model_inputs = to_model_device(full_inputs, bundle.model)
```

Equivalent APIs are acceptable, but `InferenceVisualCache` must receive host pixels or per-image host pixel rows and must select misses before H2D.

## Hit/miss batching behavior

### Required visual-cache primitive

Refactor `InferenceVisualCache` so the feature lookup/computation is reusable independently of LM input fusion.

Recommended internal/public helper:

```python
get_or_compute_visual_features(
    bundle,
    *,
    paths,
    host_pixel_values_or_rows,
)
```

Required behavior:

```text
validate cache/bundle/preprocessing compatibility

derive the target vision device/dtype from the bound model,
not from an already-moved pixel tensor

build ordered keys for every image

resolve hits without touching GPU pixels

deduplicate misses when the same compatible image identity repeats

gather/concatenate only miss pixel rows

move only the miss batch to target device/dtype

native.get_image_features(...) only on misses

clone/store each newly produced feature so an entry does not retain an
unrelated full-batch allocation

assemble output features in exact original key order
```

`fuse()` may become a thin wrapper:

```text
get/compute ordered features
→ create text embeddings
→ masked_scatter image features
→ return inputs_embeds + ordered features
```

### Preprocessing behavior

A final-feature cache hit must not invoke the underlying native image processor again.

The current preprocessing cache may be retained, but the implementation must ensure the following invariant:

```text
visual-feature hit
⇒ canonical preprocessing row is already available/compatible
⇒ original image preprocessing is not re-executed
```

If the current independent LRU stores can violate this implication, couple the lifetime/identity of a visual entry with its canonical preprocessing entry, or treat the visual entry as unusable when the matching preprocessing identity is absent.

Do not replace the native processor with synthetic image-token expansion.

The native processor must still define the canonical text/image expansion. Preserve:

```text
crop_to_patches=False
448×448 one-tile behavior
native processor image-token expansion
tokenizer/chat-template behavior
```

### Host collation

Do not concatenate all hit pixels into a GPU-bound visual tensor.

It is acceptable for the canonical `NativeBatch` to retain CPU preprocessing tensors for correctness checks, but the visual compute batch passed to `.to(cuda)` / `get_image_features()` must contain misses only.

Prefer changing evaluation-aware packing so it can retain per-image CPU rows in flattened order and concatenate only miss rows when the visual cache asks for them.

Do not change training collation behavior.

### Ordering

For a physical batch, define the flattened image order exactly as current `pack_where_batch()`:

```text
episode 0 support 0
episode 0 support 1
...
episode 0 query
episode 1 support 0
...
episode B-1 query
```

If hit/miss resolution produces:

```text
keys = [k0, k1, k2, k3]
hits = {k0, k2}
misses = [k1, k3]
```

the returned feature tensor must still be:

```text
[f(k0), f(k1), f(k2), f(k3)]
```

not hit-first or miss-first order.

The following existing validations must continue to hold:

```text
feature group count == flattened image count
feature sequence length == processor.image_seq_length
image placeholder count == feature_count * image_seq_length
query_image_index still points to the query's feature
```

## Cache compatibility / identity

WHY SAFE:

Current cache lifetime is invocation-local, but Change 3 will make it span loss→prediction. Strengthen cache binding rather than introducing a persistent/global store.

Each `InferenceVisualCache` instance must be bound to one compatible live visual context.

At minimum validate:

```text
same live model object
same native vision_tower object
same native multi_modal_projector object
same processor/image_processor object
same native preprocessing signature
same target visual device
same target visual dtype
same processor.image_seq_length
same processor.image_token_id
frozen_vision_is_reusable(model) is still true
model is in inference/eval-compatible state
autograd is disabled for feature reuse
```

Because the cache must never be reused across a different live model object, this also prevents silent cross-checkpoint reuse.

Do not create a global on-disk or process-global feature cache in this task.

### Preprocessing signature

Bind cache entries to the exact native preprocessing contract used by `collate_native()`.

At minimum include or validate the equivalent of:

```text
processor/image_processor identity
crop_to_patches=False
size.height=448
size.width=448
one tile / num_patches == 1
```

Do not assume path alone is sufficient if preprocessing settings can differ.

### Target dtype/device

Do not derive visual cache compatibility from already-transferred `pixel_values`.

Derive target device/dtype from the bound live model, then transfer miss pixels to that exact target.

If the bound model/device/dtype changes while the cache is live, fail closed; do not reuse old visual entries.

## Files/functions affected

FILES/FUNCTIONS AFFECTED:

```text
semgaze/model/visual_cache.py
    InferenceVisualCache.__init__ / binding logic
    processor_for(...)
    new/get-or-compute feature helper
    fuse(...)
    statistics(...)
    close(...)

semgaze/where/forward.py
    forward_where_batch(...)
    move text inputs separately from host pixels
    call visual cache before pixel H2D

semgaze/where/generation.py
    generate_where_batch(...)
    same early lookup/miss-only H2D behavior

    generate_where(...)
    if serial prediction is wired to use a supplied visual cache, preserve
    baseline behavior when cache is None

semgaze/where/collator.py
    collate_native(...)
    pack_where_batch(...)
    only as needed to expose canonical per-image host pixel rows and strict
    preprocessing identity without changing native tokenization

semgaze/semantic/flat/forward.py
    prepare_semantic_batch(...)
    query-feature retrieval from shared visual cache when cached R removes
    the GT WhereOutput

semgaze/evaluation/where.py
    evaluate_where_episode(...)
    only if optional cache propagation is added for serial execution

semgaze/evaluation/flat.py
    evaluate_flat_batch(...)
    evaluate_flat_episode(...)
    propagate shared visual cache to semantic query-feature reuse
```

Do not change training forward paths when no `InferenceVisualCache` is supplied.

## Backward-compatibility behavior

BACKWARD-COMPATIBILITY BEHAVIOR:

```text
visual cache None:
    current full pixel transfer + native vision path remains valid

visual cache supplied but feature caching disabled:
    current full native visual computation remains valid

visual cache miss:
    exact current preprocessing
    exact target device/dtype
    exact native.get_image_features result
    exact feature placement

visual cache hit:
    same final feature value
    no corresponding pixel H2D
    no corresponding get_image_features call
```

Do not require cache presence for correctness.

## Required parity checks

PARITY TEST:

### Single miss

Start with an empty compatible cache.

Assert:

```text
canonical preprocessing executes
pixel H2D executes for the image
native.get_image_features executes
returned visual feature equals baseline no-cache visual feature
feature is stored
```

### Single hit

Run the same compatible image again.

Assert:

```text
final visual cache hit occurs
underlying native image preprocessing does not execute again
no corresponding pixel H2D occurs
native.get_image_features does not execute
returned feature is identical to the previously stored feature
LM-facing inputs_embeds are equivalent to baseline
```

Use test instrumentation/call counters around the miss-only pixel-transfer helper and `native.get_image_features()`.

### Mixed-hit physical batch

Warm only a strict subset of images, then evaluate a batch containing hits and misses.

Assert:

```text
only miss images are included in the GPU pixel transfer batch
only miss images are passed to native.get_image_features
each miss is computed once
hits are not recomputed
final ordered feature tensor matches the original flattened image order
query_image_index still selects the correct query feature
image placeholder mapping is unchanged
WHERE outputs/loss are unchanged
```

Test repeated compatible paths in the same physical batch and ensure deduplicated miss computation does not change final repeated positions.

### Compatibility rejection

Assert no reuse across:

```text
different live model object/checkpoint bundle
different native vision producer object
different processor object
different preprocessing signature
different target dtype
different target device
```

A mismatch must be a miss or clear error, never a hit.

# Required Change 3 — Persistent visual cache across loss and prediction

## Current lifecycle

CURRENT:

```text
evaluate_test_epoch(...)
    cache = InferenceVisualCache(...)
    ...
    cache.close()

predict_epoch(...)
    cache = InferenceVisualCache(...)
    ...
    cache.close()
```

This means prediction cannot reuse final frozen visual features retained at the end of the immediately preceding loss phase.

`evaluate_flat.py` separately owns a standalone prediction cache, which is correct for standalone operation.

## Target lifecycle

TARGET for the integrated training evaluation cycle:

```text
evaluation boundary begins

create one compatible InferenceVisualCache

evaluate_test_epoch(..., visual_cache=shared_cache)
    does not close externally owned cache

predict_epoch(..., visual_cache=shared_cache)
    uses entries warmed by loss
    does not close externally owned cache

finally:
    shared_cache.close()

evaluation boundary ends
training may resume
```

The shared cache must not survive into another evaluation cycle.

### Combined lifecycle with R cache

Recommended integrated orchestration:

```python
visual_cache = make_visual_cache(bundle)
projected_r_cache = ProjectedWhereCache(
    bundle=bundle,
    split_manifest_identity=split_manifest_identity,
    cycle_id=(sampler.epoch, global_step),
)
try:
    summary = evaluate_test_epoch(
        ...,
        visual_cache=visual_cache,
        projected_r_cache=projected_r_cache,
    )

    predictions = predict_epoch(
        ...,
        visual_cache=visual_cache,
        projected_r_cache=projected_r_cache,
    )
finally:
    projected_r_cache.close()
    visual_cache.close()
```

Use a context manager if it makes exception-safe ownership clearer.

If prediction is disabled, do not create a projected-R cache solely for unused data. `evaluate_test_epoch()` may own its normal standalone visual cache.

## Ownership API

Add explicit optional ownership to the high-level evaluation functions.

Recommended signatures:

```python
def evaluate_test_epoch(
    bundle,
    train_by_id,
    test_records,
    manifest,
    *,
    episode_callback=None,
    profiler=None,
    visual_cache=None,
    projected_r_cache=None,
):
    ...
```

```python
def predict_epoch(
    bundle,
    train_batch,
    train_by_id,
    test_records,
    manifest,
    *,
    epoch,
    step,
    split_manifest_identity,
    visual_cache=None,
    projected_r_cache=None,
):
    ...
```

Equivalent naming is acceptable.

Ownership rule:

```text
if visual_cache is None:
    function creates a cache
    function owns it
    function closes it in finally

if visual_cache is supplied:
    function validates compatibility
    function uses it
    function MUST NOT close it
```

Apply the same principle to any helper that gains optional cache ownership.

Do not use a module-global singleton.

### Cache statistics

`evaluate_test_epoch()` may continue taking a snapshot from `cache.statistics()` for its existing diagnostics even when the cache is externally owned.

Do not reset shared-cache statistics between loss and prediction solely to preserve old counters.

If logging behavior is adjusted, keep scientific metrics unchanged and keep cache diagnostics clearly separate from scientific metrics.

## Standalone compatibility

BACKWARD-COMPATIBILITY BEHAVIOR:

### Standalone loss

Calling:

```python
evaluate_test_epoch(..., visual_cache=None)
```

must:

```text
create its own compatible visual cache
use it
close it in finally
return the same loss/metric result schema
```

No projected-R cache is required when no later prediction consumer exists.

### Standalone prediction through `predict_epoch`

Calling:

```python
predict_epoch(..., visual_cache=None, projected_r_cache=None)
```

must:

```text
create/own/close its own visual cache
use current GT WHERE recomputation whenever R is absent
produce identical predictions
```

### Standalone CLI `evaluate_flat.py`

Preserve the current standalone CLI behavior.

It may continue to explicitly construct one prediction-local `InferenceVisualCache` and pass it to `prediction_batches()`, or it may use a new context/factory helper.

It must not assume an earlier loss pass or an R cache exists.

### Low-level/reference functions

Direct no-cache calls must remain correct.

Do not make correctness depend on integrated orchestration.

## Files/functions affected

FILES/FUNCTIONS AFFECTED:

```text
semgaze/training/loop.py
    run_training_loop(...)
    integrated owner of the shared visual cache and projected-R cache

semgaze/evaluation/test.py
    evaluate_test_epoch(...)
    optional externally owned visual cache
    optional projected-R sink

semgaze/evaluation/predictions.py
    predict_epoch(...)
    optional externally owned visual cache
    optional projected-R source
    prediction_batches(...) propagates both

evaluate_flat.py
    preserve standalone prediction ownership/cleanup

semgaze/model/visual_cache.py
    explicit compatibility validation
    close() remains idempotent/safe

semgaze/evaluation/cache.py
    projected-R cache lifecycle
```

## Required parity checks

PARITY TEST:

Instrument cache object identity and call counts.

For integrated evaluation:

```text
loss receives visual_cache object V
prediction receives the exact same object V
id(loss_cache) == id(prediction_cache)
prediction sees visual entries already created during loss
V is not closed between loss and prediction
V is closed after the complete integrated cycle
```

On an injected exception during loss:

```text
shared cache is closed
training does not continue with a leaked cache
```

On an injected exception during prediction:

```text
shared cache is closed
projected-R cache is cleared
```

Standalone:

```text
standalone evaluate_test_epoch owns/closes its cache

standalone predict_epoch owns/closes its cache

standalone evaluate_flat.py closes its cache

an externally supplied cache is never closed by the callee
```

Also assert that reuse does not cross two consecutive evaluation cycles:

```text
cycle N cache is closed
cycle N+1 creates a distinct cache
no entry from N is addressable in N+1
```

# Implementation order

Implement in this order. Do not combine all edits into one unverified patch.

## Step 1 — Add cache provenance/ownership primitives

1. Add `ProjectedWhereCache` and exact episode-key construction.
2. Strengthen `InferenceVisualCache` binding/compatibility rules.
3. Add optional external-cache ownership semantics to high-level evaluation APIs without changing the current compute path.
4. Add unit tests for ownership and key collision rules.

At the end of Step 1, predictions/losses must still be baseline-equivalent even if no optimization has fired yet.

## Step 2 — Refactor final visual-feature lookup before pixel H2D

1. Refactor `InferenceVisualCache` to resolve ordered hits/misses from host metadata.
2. Separate text device transfer from pixel device transfer in `forward_where_batch()`.
3. Do the same in `generate_where_batch()`.
4. Transfer/collate only visual misses to GPU.
5. Preserve exact feature ordering and image-token placement.
6. Add mixed-hit and compatibility tests.

Do not touch batching sizes or scheduling.

## Step 3 — Share the visual cache across integrated loss→prediction

1. Change `evaluate_test_epoch()` and `predict_epoch()` to accept externally owned caches.
2. Move integrated cache creation to `run_training_loop()` around both phases.
3. Keep standalone ownership fallback.
4. Verify warmed loss entries are visible to prediction.
5. Verify exception-safe cleanup.

At this point Change 2 and Change 3 should work independently of R reuse.

## Step 4 — Capture projected R during evaluation loss

In `batch_losses()`:

```text
run current WHERE
compute current projector output
split current R per episode
store exact R for each exact frozen episode in ProjectedWhereCache
continue semantic loss with the same in-memory R as baseline
```

Do not read back from the cache during the same loss forward. Caching must be a side effect only; the loss path should consume the freshly computed tensor exactly as before.

## Step 5 — Reuse R during semantic prediction

1. Modify `evaluate_flat_batch()` to look up R before constructing GT WHERE work.
2. Compute GT WHERE/projector only for R-cache misses.
3. Merge hit/miss R values back to original episode order.
4. Refactor semantic query-vision reuse so it can source frozen query visual features from the shared `InferenceVisualCache` when no `WhereOutput` exists.
5. Preserve fallback behavior.
6. Apply equivalent optional behavior to `evaluate_flat_episode()` / serial execution if needed so execution mode does not silently disable correctness or cache semantics.

## Step 6 — Run full parity suite

Do not consider implementation complete until all required tests and acceptance criteria below pass.

# Required tests

Add focused tests rather than relying only on an end-to-end long A100 run.

Suggested files:

```text
tests/test_projected_where_cache.py
tests/test_visual_cache_h2d.py
tests/test_evaluation_cache_lifecycle.py
tests/test_evaluation_reuse_parity.py
```

Names may differ.

## 1. Projected R key tests

Construct deterministic synthetic/frozen episodes and assert:

```text
identical exact episode → identical key

different query record → different key
different query trajectory → different key
different K → different key
different draw_id → different key
different support record → different key
different support trajectory → different key
different support order → different key
```

Assert cache provenance rejects a different:

```text
model object
projector object
processor identity
split_manifest_identity
cycle_id
```

## 2. Loss capture test

Instrument `batch_losses()` with a supplied projected-R cache.

Assert:

```text
one entry stored per evaluated episode
stored R shape == [N_fixations, d_model]
stored dtype == original R dtype
stored values == original R values
loss path still consumes freshly computed R
loss outputs unchanged
```

## 3. R-hit prediction test

Warm the R cache through loss.

Then run semantic prediction for the identical episode/group.

Count calls to:

```text
forward_where_batch
bundle.projector
```

For an all-hit semantic group, assert the GT semantic path adds:

```text
0 GT WHERE forwards
0 projector calls
```

Free-running WHERE generation remains present when `path='both'`.

## 4. R-miss fallback test

Run standalone prediction with no R cache and with an empty R cache.

Assert current GT WHERE/projector computation occurs and outputs match baseline.

## 5. Mixed R hit/miss batch test

Warm only some episodes.

Assert:

```text
GT WHERE runs only for misses
projector runs only for misses
R is restored to original episode order
semantic outputs equal baseline for every row
```

## 6. Semantic-input parity test

For the same episodes, compare baseline recomputed-R semantic preparation against optimized cached-R + shared-visual-cache preparation.

Assert:

```text
input embedding shape identical
attention_mask identical
state insertion positions identical
image-token feature positions identical
inputs_embeds torch.equal where possible
otherwise strict documented allclose tolerance
```

Do not accept a difference caused by changed preprocessing/tokenization.

## 7. Generated-token parity test

Run baseline and optimized generation with identical checkpoint/hardware/config.

Capture raw generated semantic token IDs before decode.

Assert:

```text
token IDs exactly equal
decoded text exactly equal
parse result exactly equal
serialized SEMANTIC.PRED exactly equal
```

## 8. Integrated serialized-record parity

Compare baseline and optimized `_prediction_record()` outputs for deterministic episodes.

Ignore only intentionally non-record timing diagnostics if any are outside the record.

Assert the JSON-serializable prediction record is otherwise exactly equal.

## 9. Loss/metric parity

Compare baseline vs optimized integrated evaluation summaries.

Assert all scientific values are unchanged:

```text
test_where
test_what
test_why
test_how
test_flat
test_total
per-K aggregation
existing metrics when present
```

Cache diagnostics may be additive but must not be mixed into scientific metrics.

## 10. Visual miss test

Empty visual cache:

```text
preprocessing executes
miss pixel is transferred to GPU
vision executes
feature equals baseline
```

## 11. Visual hit test

Second access to the exact compatible image:

```text
feature hit
underlying preprocessing not re-executed
no pixel H2D for the hit
no vision call for the hit
returned feature identical
```

## 12. Mixed visual hit/miss test

Warm a subset of images in a same-K physical batch.

Assert:

```text
GPU pixel batch contains only misses
vision call receives only misses
final feature count equals original flattened image count
final feature order equals original flattened image order
query_image_index remains correct
WHERE LM image-token positions unchanged
```

Cover K=1, K=5, and K=10.

## 13. Repeated-image/dedup test

If the same compatible path appears more than once in a physical visual request:

```text
compute an uncached feature once
reuse it in every corresponding ordered output position
do not reorder image positions
```

## 14. Visual compatibility tests

A cache created for one context must not produce a hit for an incompatible:

```text
model/checkpoint object
vision_tower object
native multimodal projector object
processor/image_processor
preprocessing signature
device
dtype
image_seq_length
image_token_id
```

## 15. Shared lifecycle test

Integrated cycle:

```text
create V
loss receives V
prediction receives V
prediction observes warmed entries
close V after prediction
```

Use the same test for projected-R cache lifecycle.

## 16. Standalone ownership tests

Assert:

```text
evaluate_test_epoch(cache=None)
    creates and closes own visual cache

predict_epoch(cache=None)
    creates and closes own visual cache

evaluate_flat.py-style standalone prediction
    does not require projected-R cache
    recomputes GT WHERE for semantic prediction
    closes its local visual cache

callee receiving an external cache
    never closes the external cache
```

## 17. Exception cleanup tests

Inject an exception:

```text
during loss
during prediction
```

Assert integrated externally owned caches are closed by their owner in `run_training_loop()`.

## 18. End-to-end deterministic smoke

On a small deterministic subset containing at least:

```text
one K=1 episode
one K=5 episode
one K=10 episode
at least two draws for the same query where available
```

run:

```text
baseline loss→prediction
optimized loss→prediction
```

Assert all parity requirements above.

# Acceptance criteria

The task is complete only when all of the following are true.

## Cross-pass R reuse

```text
[ ] evaluation loss stores the exact projected GT R for each exact frozen
    evaluation episode when a projected-R cache is supplied

[ ] integrated prediction reuses that R

[ ] an R hit skips the redundant GT WHERE teacher-forced forward

[ ] an R hit skips the redundant SemGaze projector call

[ ] free-running WHERE prediction remains unchanged

[ ] semantic prediction still uses canonical GT WHERE-derived R

[ ] mixed cache hit/miss groups work correctly

[ ] standalone prediction falls back to baseline recomputation

[ ] keying includes query identity, ordered supports, K, draw, and exact
    WHERE-relevant trajectories

[ ] cache provenance prevents cross-cycle/model/projector/split reuse

[ ] R is not quantized, averaged, truncated, or otherwise altered
```

## Pixel H2D avoidance

```text
[ ] final visual-feature lookup occurs before pixel H2D

[ ] visual-feature hits transfer zero corresponding image pixels to GPU

[ ] visual-feature hits execute zero corresponding frozen vision work

[ ] only visual misses are assembled into the GPU visual-compute batch

[ ] final visual features are restored to the original flattened image order

[ ] query feature indexing remains correct

[ ] multimodal token count/order is unchanged

[ ] native preprocessing semantics are unchanged

[ ] visual cache compatibility is strict for model/processor/preprocessing/
    device/dtype

[ ] no global or stale cross-checkpoint cache is introduced
```

## Shared visual-cache lifecycle

```text
[ ] integrated loss and prediction receive the same InferenceVisualCache instance

[ ] prediction can hit entries warmed during loss

[ ] evaluate_test_epoch does not close an externally owned cache

[ ] predict_epoch does not close an externally owned cache

[ ] standalone loss still creates/owns/closes its own cache

[ ] standalone prediction still creates/owns/closes its own cache

[ ] shared cache closes after the complete integrated evaluation cycle

[ ] exception paths close the shared cache

[ ] a later evaluation cycle receives a new cache
```

## Scientific parity

```text
[ ] evaluation query/support episodes unchanged
[ ] K/draw semantics unchanged
[ ] WHERE GT serialization unchanged
[ ] WHERE generated token IDs/text unchanged
[ ] semantic insertion positions unchanged
[ ] semantic generated token IDs/text unchanged
[ ] serialized prediction records unchanged
[ ] loss values unchanged
[ ] metrics unchanged
[ ] no training behavior changed
```

## Final implementation report

After implementation, report:

```text
audited git commit
files changed
exact ownership API introduced
exact projected-R key definition
where R is stored and why it preserves exact values
how semantic query vision is obtained when GT WhereOutput is skipped
how visual hits are resolved before H2D
how mixed hit/miss image order is restored
which standalone paths were tested
parity test results
any remaining limitations
```

Do not report speculative speedup numbers as acceptance evidence. The acceptance condition is removal of the three redundant operations with exact scientific parity.

# Explicitly out of scope

Do not implement or redesign any of the following in this task:

```text
FlashAttention
FlashAttention2
varlen attention
torch.compile
StaticCache
DynamicCache tuning
custom greedy decoding
prefix-KV reuse
generation API replacement
WHERE generation algorithm changes
semantic generation algorithm changes
WHERE generation-length changes
semantic generation-length changes
EOS/stopping changes
K changes
support-draw changes
support sampling changes
evaluation-sample reduction
metric changes
approximate metrics
metric batching redesign
physical batch-size tuning
batching-policy tuning
length-bucketing tuning
training batching changes
training sampling changes
training loss changes
training backward changes
model architecture changes
LoRA changes
projector architecture changes
input-resolution changes
persistent/global/disk-backed visual caches
cross-process caches
cross-checkpoint cache reuse
```

Do not opportunistically clean up unrelated code.

## Implemented production architecture notes

The implemented CUDA training path keeps dense multimodal SemGaze inputs and
image-feature insertion, then uses the Qwen3 language model's native
Transformers FlashAttention-2 route: dense Q/K/V projections and RoPE, native
variable-length unpadding with cumulative sequence lengths, grouped-query
attention, and native repadding before `o_proj`. External sequence indexing,
labels, `<END_FIX>` extraction, episode boundaries, and the semantic branch
remain dense. Gradient checkpointing remains enabled and semantic gradients
continue through WHERE states.

CPU preparation runs as one authoritative producer for optimizer window N+1
feeding a bounded depth-2 queue while the GPU consumer executes window N.
Sampling, retries, support order, and physical grouping are unchanged. Checkpoint
progress belongs to the consumed consumer sampler state; queued windows are
discarded and regenerated after resume. Transferable CPU tensors are pinned
selectively and moved with existing non-blocking transfers; Python/PIL metadata
is not pinned.

The implementation should be narrowly scoped to:

```text
1. exact cross-pass GT projected-R reuse;
2. final frozen visual-cache lookup before pixel H2D;
3. one explicit shared frozen visual cache across integrated loss→prediction.
```
