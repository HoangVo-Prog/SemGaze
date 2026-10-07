# AGENTS.md — SemGaze Evaluation Metrics Implementation Contract

## 0. Authority, repository baseline, and task boundary

This file is the implementation contract for adding evaluation metrics to the SemGaze `flat` evaluation pipeline.

Implementation baseline is **pinned** to:

```text
b84e9c752124457244ee34b93e70f72be5690914
```

Before editing code, the implementing agent MUST run:

```bash
git rev-parse HEAD
git status --short
git show -s --format='%H %D %s' HEAD
```

The starting `HEAD` MUST equal:

```text
b84e9c752124457244ee34b93e70f72be5690914
```

If it does not, **STOP** and report the mismatch. Do not silently implement against a different revision.

The task is **evaluation-only**.

Do not change scientific behavior, training behavior, support sampling, K semantics, model architecture, optimization, generation semantics, or split semantics.

Primary runtime configuration:

```text
configs/flat_single.yaml
```

Target metrics:

```text
WHERE / scanpath
  SM   ↑
  MM   ↑
  SED  ↓
  LL   ↑
  IG   ↑

SEMANTIC / WHAT
  BERTScore-F1 ↑
  CIDEr-R      ↑

SEMANTIC / WHY
  BERTScore-F1 ↑
  CIDEr-R      ↑

SEMANTIC / HOW
  BERTScore-F1 ↑
  CIDEr-R      ↑
```

Canonical evaluation ordering remains:

```text
training
  -> evaluation loss
  -> prediction generation
  -> metric computation
```

Redundant computation may be reused only when the reused quantity is mathematically identical.

Central invariant:

```text
One logical prediction is generated once.
Metrics consume the already-generated prediction.
```

`SM`, `MM`, `SED`, BERTScore-F1, and CIDEr-R MUST NOT trigger another WHERE or semantic generation pass.

`LL` and `IG` are probability metrics and MUST NOT be computed from generated coordinates.

---

# 1. Non-negotiable scientific invariants

The implementation MUST NOT change:

```text
model architecture
LoRA configuration
trainable parameters
optimizer
training loss
existing evaluation loss
support sampling
persisted support draws
K semantics
query membership
seen/unseen subject protocol
WHERE generation semantics
oracle-length conditioning
semantic conditioning
semantic generation format
existing prediction text
existing prediction diagnostics
train/test split semantics
gold WHY-group semantics
```

Current semantic prediction conditioning MUST remain:

```text
teacher-forced GT XYD states + gold WHY groups
```

Do not condition semantic generation on the free-running predicted scanpath.

The current evaluation cycle conceptually executes:

```text
LOSS
  1. teacher-forced WHERE forward
  2. teacher-forced semantic forward

PREDICTION
  3. free-running WHERE generation
  4. teacher-forced GT-WHERE forward
  5. semantic generation
```

Metric integration MUST add:

```text
0 new autoregressive VLM generation passes
```

---

# 2. Mandatory preflight gates

No agent may skip these gates by assumption.

## Gate A — ISP-SENet provenance

Before implementing `SM`, `MM`, or `SED`, locate the exact local ISP-SENet / Few-shot Personalized Scanpath Prediction reference source.

Run from the pinned checkout:

```bash
git ls-files | grep -Ei \
  'few-shot|ISP|GazeformerISP|scanmatch.py|visual_attention_metrics.py|multimatch'
```

Record:

```text
actual local path
SemGaze HEAD
upstream repository
upstream commit if recoverable
SHA256 for relevant local source files
whether local source is byte-identical or behaviorally identical to upstream
```

Expected scientific reference is the official CVPR 2025 repository:

```text
cvlab-stonybrook/few-shot-scanpath
```

with COCO-Search18 reference paths equivalent to:

```text
ISP/COCO_Search18/GazeformerISP/src/test.py
ISP/COCO_Search18/GazeformerISP/src/utils/evaluation.py
ISP/COCO_Search18/GazeformerISP/src/utils/evaltools/scanmatch.py
ISP/COCO_Search18/GazeformerISP/src/utils/evaltools/visual_attention_metrics.py
```

Do not install or substitute an unrelated ScanMatch/VAME implementation.

If no verified local copy exists, vendor the exact official files into a clearly marked `third_party` directory with provenance and hashes.

`SM/MM/SED` implementation MUST remain disabled until Gate A passes.

## Gate B — coordinate parity and prediction-bin inverse

ISP COCO-Search18 evaluation uses:

```text
width  = 512
height = 320
```

SemGaze emits prediction coordinates in integer bins:

```text
0..99
```

SemGaze forward coordinate encoding is lossy and many-to-one.

Therefore there are two separate contracts:

```text
A. raw GT coordinate -> ISP 512x320 frame
B. SemGaze predicted bin -> ISP 512x320 frame
```

### Gate B1 — GT transform

Match real records existing in both SemGaze and ISP data using at least:

```text
subject
task
image/name
condition if available
```

Compare full fixation sequences and derive the exact raw-GT -> ISP transform from the actual preprocessing source.

The parity fixture MUST cover:

```text
interior coordinates
x=0 / y=0
max image edges
portrait and landscape cases if present
```

GT MUST be converted from raw coordinates directly whenever raw coordinates are available.

Do not unnecessarily round-trip GT through SemGaze `0..99` bins.

### Gate B2 — prediction-bin inverse — HARD OWNER GATE

GT parity does **not** mathematically identify a unique inverse from a quantized SemGaze bin back to a pixel coordinate.

The agent MUST search for a historical/canonical inverse or representative mapping in:

```text
SemGaze source
DeepGaze source
COCO-Search18 preprocessing
ISP preprocessing
prior project fixtures/configuration
```

If an exact historical/canonical prediction-bin inverse is found, record its provenance and prove it with fixtures.

If no such inverse exists, **STOP SM/MM/SED production implementation** and report:

```text
BLOCKED: predicted-bin -> ISP-frame representative is scientifically underdetermined.
Owner decision required.
```

The agent MUST NOT choose among examples such as:

```text
b / 99
b / 100
(b + 0.5) / 100
cell lower edge
cell center
cell upper edge
```

without explicit owner approval backed by a documented protocol decision.

Do not enable canonical `eval_where_sm`, `eval_where_mm`, or `eval_where_sed` until Gate B2 is resolved.

## Gate C — LL/IG tokenizer equivalence

Using the exact SemGaze checkpoint processor/tokenizer, audit these WHERE strings in full context:

```text
(00, 00, 000)<END_FIX>
(09, 09, 009)<END_FIX>
(10, 10, 010)<END_FIX>
(99, 99, 999)<END_FIX>
```

Record:

```text
characters
token strings
token IDs
offsets
predictor positions
```

Prove whether the four coordinate digits:

```text
x tens
x ones
y tens
y ones
```

can be scored exactly from predictor states already produced by the current teacher-forced WHERE loss forward.

The preflight MUST explicitly select one outcome:

```text
Outcome A — exact reuse of existing WHERE loss forward
Outcome B — dedicated probability-scoring forward/probing path
```

No implicit fallback is allowed.

## Gate D — CIDEr-R and center-bias provenance

Before integration, locate or freeze:

```text
CIDEr-R authors' implementation
CIDEr-R PTB tokenizer path
canonical COCO-Search18 center-bias asset root
center-bias asset hashes
```

Canonical IG MUST fail closed if a required center-bias asset is missing.

Synthetic Gaussian center bias is noncanonical and MUST NOT populate `eval_where_ig`.

---

# 3. Additional protocol locks added after audit

These locks override any ambiguous interpretation elsewhere.

## LOCK 1 — predicted-bin inverse cannot be inferred from GT parity

The prediction-side `0..99 -> 512x320` representative is an explicit scientific protocol choice if no historical canonical inverse is found.

Executable GT parity alone is insufficient evidence.

In that situation:

```text
STOP
DO NOT IMPLEMENT A GUESS
REQUEST OWNER DECISION
```

## LOCK 2 — GT duration source

For canonical ISP-comparable metrics, GT duration MUST come from the raw COCO-Search18 dwell-duration field used by SemGaze:

```text
source field: T
semantics: dwell duration
unit: milliseconds
conversion: identity
```

GT duration MUST NOT be round-tripped through SemGaze's serialized `DDD` field and MUST NOT inherit the model-facing `0..999 ms` clipping unless the audited ISP preprocessing explicitly proves the same clipping was part of the reference data.

If the audited ISP preprocessing contradicts this lock, STOP and report the contradiction before changing the contract.

Prediction duration is the parsed generated `DDD` value in milliseconds.

Adaptation:

```text
MultiMatch duration = duration_ms / 1000.0 seconds
ScanMatch-with-duration = duration_ms
```

## LOCK 3 — Outcome B preserves exact SemGaze conditioning context

If LL/IG require Outcome B, DeepGaze defines the **probability-scoring rule**, not a replacement SemGaze prompt.

Every LL/IG probability must be obtained under the exact SemGaze evaluation context for that logical episode:

```text
same query
same subject
same K
same draw_id
same support IDs and ordering
same query image/task/context
same GT preceding fixations
same SemGaze WHERE serialization/prefix semantics
same tokenizer/model/checkpoint
```

The dedicated probability path MUST NOT construct a DeepGaze-style alternative prompt if that prompt differs from SemGaze's current causal context.

Only the digit-normalized scoring rule is imported from DeepGaze.

## LOCK 4 — key-based joins only

Any values produced in different traversals, including:

```text
loss/probability statistics
prediction records
metric records
```

MUST be joined by the explicit identity key:

```text
(query_id, K, draw_id)
```

Required assertions before metric reduction:

```text
key sets are equal
keys are unique in each source
no duplicate logical episode
no missing logical episode
```

Never join by iteration order.

Never `zip(loss_rows, pred_rows)` unless key equality is separately asserted and the final lookup is key-based.

A mismatch is a hard evaluation error.

## LOCK 5 — MultiMatch numerical-degeneracy policy

GT integrity failures remain hard errors.

For a model prediction that:

```text
has at least one valid recovered fixation
has finite coordinates/durations
passes required coordinate conversion
uses the exact ISP short-path padding
```

but the audited MultiMatch reference still returns a `NaN`, `-1`, or undefined component because of a model-side degenerate trajectory:

```text
all five MM component contributions for that query = 0.0
query remains in the denominator
increment mm_numeric_failure_count
record the raw failure reason
```

This is the explicit:

```text
SemGaze invalid-generation / numerical-degeneracy extension
```

It is not claimed to be ISP behavior for valid reference paths.

Do not silently delete the query.

If failure indicates malformed GT, impossible dimensions, nonfinite GT, or reference-code/configuration corruption, raise a hard error instead of assigning zero.

## LOCK 6 — unique semantic-unit IDs

Every semantic metric unit MUST have a unique deterministic scorer identity.

Use identities equivalent to:

```text
<query_id>::WHAT::<fixation_index>
<query_id>::WHY::<group_index>
<query_id>::HOW::0
```

Indices are zero-based or one-based consistently, but the choice MUST be fixed and tested.

The identity is scorer bookkeeping only and MUST NOT alter text, pairing, or corpus semantics.

This prevents overwriting multiple WHAT/WHY units that belong to the same query.

## LOCK 7 — undefined LL/IG queries and draws

DeepGaze-style LL/IG score transitions only for fixation indices:

```text
1..N-1
```

A query with fewer than two GT fixations has zero defined transitions.

Such a query:

```text
is NOT a model failure
is excluded from LL/IG query-mean denominator only
remains present for all other metrics
increments probability_undefined_query_count
```

Persist at least:

```text
ll_defined_query_count
ig_defined_query_count
probability_undefined_query_count
transition_count
```

If a complete `(K, draw_id)` has zero defined LL/IG queries:

```text
canonical LL = null
canonical IG = null
```

Do not serialize JSON `NaN`.

Do not convert undefined LL/IG to zero.

K-level mean for LL/IG is computed over defined draw values only and MUST persist:

```text
defined_draw_count
undefined_draw_count
```

If all draws for a K are undefined, the K-level scalar is `null`.

## LOCK 8 — canonical reporting split

Canonical paper-facing evaluation metrics are produced for:

```text
split = test
```

Training prediction artifacts may continue to exist, but any metrics calculated for train-side diagnostics MUST be namespaced as diagnostics and MUST NOT overwrite canonical `eval_*` test metrics.

`metrics.json` MUST explicitly persist its split.

## LOCK 9 — `log_z` is not a tuning knob

The canonical DeepGaze-fast LL/IG formula in this contract has no free calibration parameter named `log_z`.

Do not expose `evaluation.metrics.probability.log_z` as a tunable scientific option.

If legacy code requires a field for compatibility, it MUST be validated as:

```text
0.0 only
```

and documented as a no-op compatibility field.

Prefer removing it from the new canonical metrics config if no current code requires it.

---

# 4. Locked metric definitions

## 4.1 SM — ScanMatch harmonic mean

Reference configuration:

```python
ScanMatch(
    Xres=512,
    Yres=320,
    Xbin=16,
    Ybin=12,
    Offset=(0, 0),
    TempBin=50,
    Threshold=3.5,
)
```

Also construct a second ScanMatch object with the same spatial settings and no `TempBin`.

Within one complete `(K, draw_id)`:

```text
sm_no_duration = arithmetic mean of same-subject query ScanMatch scores
sm_with_duration = arithmetic mean of same-subject query duration-ScanMatch scores

SM = scipy.stats.hmean([sm_no_duration, sm_with_duration])
```

Canonical metric name:

```text
eval_where_sm
```

Diagnostics may include:

```text
eval_diag_where_scanmatch_no_duration
eval_diag_where_scanmatch_with_duration
```

Do not multiply by 100.

## 4.2 MM — MultiMatch

Reference dependency:

```text
multimatch-gaze==0.1.3
```

or a proven-equivalent vendored copy.

Reference call:

```python
multimatch.docomparison(gt, pred, screensize=[512, 320])
```

A valid scanpath shorter than 3 fixations is padded to length 3 with:

```text
(1.0, 1.0, 0.001)
```

where duration is seconds.

Five components:

```text
vector
direction
length
position
duration
```

Within a draw:

```text
mean each component across query pairs
MM = arithmetic mean of the five component means
```

Canonical metric name:

```text
eval_where_mm
```

Do not drop duration.

Do not flatten or change the reduction hierarchy even if a flattened mean happens to coincide numerically.

## 4.3 SED — ISP/VAME String Edit Distance

Use the audited ISP `string_edit_distance` behavior.

Reference spatial partition:

```text
height = 320
width  = 512
n      = 5
height_step = 64
width_step  = 102
```

For integer coordinate `(x, y)`:

```text
symbol_index = (x // 102) + (y // 64) * 5
symbol = chr(97 + symbol_index)
```

Then compute raw edit distance over the symbol strings.

Canonical draw score:

```text
SED = arithmetic mean of same-subject raw SED values
```

Canonical name:

```text
eval_where_sed
```

Do not normalize by path length.

Do not substitute semantic/text edit distance.

Preserve the reference integer-boundary behavior exactly.

## 4.4 LL — DeepGaze fast spatial GT probability

For GT coordinate:

```text
x = 10*x1 + x2
y = 10*y1 + y2
```

Score four causal phases:

```text
log P(x1 | prefix)
log P(x2 | prefix, x1)
log P(y1 | prefix, x1, x2, separator)
log P(y2 | prefix, x1, x2, separator, y1)
```

Each phase normalizes over digit alternatives:

```text
0,1,2,3,4,5,6,7,8,9
```

When exact HF logits are available:

```text
digit_log_prob = target_logit - logsumexp(logits[digit_ids])
```

Coordinate fixation score:

```text
LL_fix = sum(four digit log probabilities)
```

Use natural logarithms.

Do not score fixation index 0.

Duration is not part of canonical LL.

Per query:

```text
LL_query = mean(LL_fix for indices 1..N-1)
```

Per draw:

```text
LL_draw = mean(LL_query over defined queries)
```

Per K:

```text
mean of defined complete draw scores
```

Canonical name:

```text
eval_where_ll
```

## 4.5 IG — Information Gain

Use canonical data-driven center bias over the `100x100` reduced grid.

For each scored fixation:

```text
IG_fix = (LL_fix - log P_centerbias(x_gt, y_gt)) / ln(2)
```

Unit:

```text
bits / fixation
```

Center-bias loader behavior must match the audited DeepGaze path:

```text
read data["centerbias"]
exp(centerbias - max)
bilinear resize using scipy.ndimage.zoom(order=1)
normalize to sum 1
clip density at 1e-10
log
```

No synthetic Gaussian fallback for canonical metrics.

Missing canonical center bias is a hard error.

Canonical name:

```text
eval_where_ig
```

## 4.6 BERTScore-F1

Freeze:

```yaml
package: bert-score
version: 0.3.13
lang: en
model_type: roberta-large
num_layers: 17
idf: false
rescale_with_baseline: false
use_fast_tokenizer: false
```

Use one reusable `BERTScorer` per evaluation event/process.

Text normalization only:

```python
" ".join(text.strip().split())
```

Do not:

```text
lowercase manually
remove punctuation
stem
lemmatize
repair grammar
expand abbreviations
paraphrase
inject GT wording
```

Persist:

```text
bert-score version
transformers version
model
layer
official scorer hash
idf setting
rescale setting
fast/slow tokenizer setting
```

Canonical metric is F1 only.

Canonical names:

```text
eval_sem_what_bertscore_f1
eval_sem_why_bertscore_f1
eval_sem_how_bertscore_f1
```

## 4.7 CIDEr-R

Use the authors' CIDEr-R implementation associated with:

```text
CIDEr-R: Robust Consensus-based Image Description Evaluation
Gabriel Oliveira dos Santos et al., 2021
```

Expected source family:

```text
gabrielsantosrv/coco-caption---My-changes
pycocoevalcap/ciderR/ciderR.py
pycocoevalcap/ciderR/ciderR_scorer.py
pycocoevalcap/tokenizer/ptbtokenizer.py
```

Freeze:

```text
n = 4
k_r = 0.8
length coefficient = 0.2
final scale = x10 as implemented by authors
```

Canonical pipeline:

```text
parsed SemGaze semantic text
  -> whitespace canonicalization
  -> authors' PTBTokenizer
  -> authors' CIDEr-R scorer
```

Document frequency MUST be constructed from the complete reference corpus for one:

```text
(K, draw_id, semantic_branch)
```

Never compute CIDEr-R independently per physical minibatch.

Never pool WHAT/WHY/HOW into one corpus.

Canonical names:

```text
eval_sem_what_cider_r
eval_sem_why_cider_r
eval_sem_how_cider_r
```

---

# 5. Ground-truth pairing and semantic units

Canonical scanpath metrics compare:

```text
prediction for subject S
vs
GT scanpath for subject S
```

Never compare to another subject.

Never average all human paths for the same image.

Identity MUST preserve:

```text
query_id / record_id
subject
task
stimulus/image
condition
split
K
draw_id
support_ids
```

Semantic GT belongs to the exact same normalized query record.

Semantic units:

```text
WHAT = one expected unit per GT fixation
WHY  = one expected unit per GT WHY group
HOW  = one expected unit per query
```

Missing units remain in denominators under the failure policy below.

---

# 6. Aggregation hierarchy

Canonical hierarchy:

```text
K
  -> support draw
       -> complete eligible unseen test-query set
```

For every configured `K`:

```text
for each persisted draw_id:
    score every eligible unseen test query exactly once
```

Do not hardcode:

```text
K=1
query count
407
```

The current canonical persisted protocol may contain 10 draws for each of:

```text
K=1
K=5
K=10
```

but runtime execution is driven by:

```text
config["evaluation"]["k_values"]
```

For each `(K, draw_id)`, compute one complete metric block.

K-level canonical score:

```text
arithmetic mean of complete draw-level values
```

except undefined LL/IG draws follow LOCK 7.

Persist both:

```text
per-draw values
K-level draw mean
```

Never pool different K values into one unnamed overall metric.

No subject macro-average.

---

# 7. Evaluation architecture

Required logical architecture:

```text
EVALUATION TRIGGER
|
+-- [EVAL][LOSS]
|    |
|    +-- teacher-forced WHERE
|    |     -> existing WHERE loss
|    |     -> optional LL/IG sufficient statistics if Outcome A
|    |
|    +-- teacher-forced semantic
|          -> existing flat loss
|
+-- [EVAL][PRED]
|    |
|    +-- free-running WHERE generation
|    +-- GT-WHERE-conditioned semantic generation
|    +-- canonical prediction records
|
+-- [EVAL][PROB]  # Outcome B only
|    |
|    +-- exact SemGaze-context GT probability scoring
|    +-- no WHERE generation
|    +-- no semantic generation
|
+-- [EVAL][METRIC][WHERE]
|    |
|    +-- existing generated WHERE -> SM/MM/SED
|    +-- LL/IG sufficient statistics
|
+-- [EVAL][METRIC][SEM]
     |
     +-- existing parsed semantic generation -> BERTScore/CIDEr-R
```

For fixed:

```text
(query_id, K, draw_id)
```

new metric code MUST NOT call `model.generate(...)`.

---

# 8. LL/IG implementation outcomes

## Outcome A — reuse existing WHERE loss forward

Choose only if Gate C proves exact equivalence.

Use:

```text
existing teacher-forced WHERE forward
  -> selected digit predictor hidden states
  -> LM output head at only required positions
  -> exact 10-digit normalization
  -> per-fixation LL
  -> center-bias subtraction
  -> IG
```

Requirements:

```text
no extra backbone forward
no retained [B,L,V] tensor
no logits retained across queries
immediately reduce to scalar sufficient statistics
```

Do not change existing loss arithmetic.

## Outcome B — dedicated probability pass

Use only if Outcome A cannot be proven exact.

The pass MUST:

```text
use exact SemGaze episode context
use GT prefixes
reproduce four digit phases exactly
batch prompts where safe
run under torch.inference_mode()
reuse preprocessing/features only when mathematically identical
not call semantic generation
not call WHERE generation
not write another prediction file
reduce immediately to LL/IG scalars
run once per logical (query,K,draw)
```

Log distinctly as:

```text
[EVAL][PROB]
```

---

# 9. Failure policies

## 9.1 General rule

Failed model generations MUST NOT be silently dropped.

Every expected unit contributes either:

```text
a valid metric input
or
a deterministic failure contribution
```

with a recorded failure counter.

Never repair model output using GT.

## 9.2 WHERE — malformed text with recovered fixations

If the current parser recovers one or more complete fixation triples:

```text
score the recovered parsed trajectory
count the format failure
```

Do not discard the query.

## 9.3 WHERE — empty prediction

Use the SemGaze invalid-generation extension:

```text
SM contribution = 0
MM contribution = 0
SED contribution = len(GT spatial symbol string)
```

Do not pad an entirely empty prediction with GT-dependent information.

## 9.4 Dataset-invalid GT

If GT is empty, malformed, missing required duration, nonfinite, or dimensionally inconsistent:

```text
raise dataset-integrity error
```

Do not treat invalid GT as model failure.

## 9.5 Semantic missing unit

For an expected WHAT/WHY/HOW unit that is absent or invalid:

```text
candidate = missing
```

Required deterministic contribution:

```text
BERTScore F1 = 0
CIDEr-R sample contribution = 0
```

The GT reference MUST remain in the CIDEr-R reference corpus used to construct document frequency.

## 9.6 Extra semantic lines

Score only the deterministic expected-unit mapping recovered by the current parser.

Do not create extra GT/reference pairs from extra model text.

Count the parse failure.

## 9.7 MultiMatch numerical failure

Apply LOCK 5 exactly.

---

# 10. Canonical records

Prefer extending current prediction rows additively.

If a helper structure is needed, a logical canonical record should contain only what metrics require:

```text
query_id
subject
stimulus_id
image_name
task
condition
split
K
draw_id
support_ids
annotation_width
annotation_height

where:
  gt_raw_pixels
  gt_binned
  gt_duration_ms_raw
  pred_text
  pred_binned
  pred_duration_ms
  requested_count
  accepted_count
  under_generated
  canonical_format_valid

semantic:
  what:
    gt[]
    pred[]
  why:
    gt[]
    pred[]
  how:
    gt
    pred
  parser_errors
  flat_format_valid

probability:
  ll_query
  ig_query
  transition_count
  centerbias_source
  defined
```

Do not store:

```text
full image tensors
full hidden-state tensors
full-vocabulary logits
```

---

# 11. File-by-file implementation plan

## `semgaze/evaluation/metrics_scanpath.py` — NEW

Create only after Gates A and B pass.

Responsibilities:

```text
verified ISP coordinate adapter
ScanMatch wrapper
MultiMatch wrapper
VAME SED wrapper
duration adaptation
short-path padding
failure policy
draw-level aggregation
```

Suggested public API:

```python
score_scanpath_pair(...)
aggregate_scanpath_draw(...)
```

Must not load the VLM or generate outputs.

## `semgaze/evaluation/metrics_probability.py` — NEW

Responsibilities:

```text
LL/IG tokenizer audit helpers
exact digit-normalized log probability
transition scoring
center-bias loader
probability batching if Outcome B
query/draw aggregation
undefined handling
```

Suggested components:

```python
audit_digit_tokenization(...)
digit_logprob(...)
score_gt_transition(...)
score_probability_batch(...)
load_canonical_centerbias(...)
aggregate_probability_draw(...)
```

## `semgaze/evaluation/metrics_semantic.py` — NEW

Responsibilities:

```text
unique semantic unit extraction
BERTScore scorer lifecycle
CIDEr-R corpus construction
branch-level scoring
failure accounting
provenance
```

Suggested components:

```python
collect_semantic_units(records, branch)
build_bertscorer(...)
score_bertscore_branch(...)
score_cider_r_branch(...)
aggregate_semantic_draw(...)
```

## `semgaze/evaluation/records.py` — NEW only if justified

Create only if it materially removes alignment duplication.

Do not create a second independent prediction representation if current rows can be extended safely.

## `semgaze/evaluation/predictions.py`

Required additive changes:

```text
preserve existing generated outputs
expose structured parsed WHERE/semantic fields
attach condition and annotation dimensions if missing
preserve support/query identity
collect complete records by (K,draw)
compute generation-based metrics only after prediction rows already exist
do not call prediction_batches twice
preserve existing train.jsonl/test.jsonl compatibility
```

## `semgaze/evaluation/test.py`

Touch only as required for LL/IG.

Outcome A:

```text
capture/reduce optional probability sufficient statistics
leave existing loss values unchanged
return additive probability diagnostics
```

Outcome B:

```text
leave batch_losses unchanged
```

Existing keys such as:

```text
test_where
test_what
test_why
test_how
test_flat
test_total
```

must remain numerically unchanged.

## `semgaze/where/forward.py`

Modify only if Outcome A requires exposure of selected predictor states.

Any return field MUST be additive.

Do not:

```text
change loss
change state positions
detach states
change projector inputs
change cache semantics for existing path
retain [B,L,V] logits
```

## `semgaze/model/selected_loss.py`

Prefer no change.

If unavoidable for Outcome A, only add an optional interface exposing selected predictor states/metadata or accepting a scalar-reduction callback.

Default behavior must be identical.

## `semgaze/training/loop.py`

Integrate metric summaries into the current evaluation trigger.

Use distinct logs:

```text
[EVAL][LOSS]
[EVAL][PRED]
[EVAL][PROB]            # Outcome B only
[EVAL][METRIC][WHERE]
[EVAL][METRIC][SEM]
```

Persist metric events into `trainer_log.jsonl`.

Do not call `predict_epoch()` twice.

Preserve checkpoint ordering.

## `evaluate_flat.py`

Use the same metric implementations.

Frozen prediction artifacts must be sufficient to recompute:

```text
SM
MM
SED
BERTScore-F1
CIDEr-R
```

without loading the VLM.

LL/IG require model probabilities unless sufficient statistics are already stored.

Do not regenerate WHERE or semantic outputs for LL/IG.

## `configs/defaults.yaml`

Add explicit validated metric defaults.

Do not depend on package defaults for scientific settings.

## `configs/flat_single.yaml`

Add explicit canonical metric configuration without changing:

```text
evaluation.k_values
support sampling
prediction budget
loss settings
```

## `pyproject.toml`

Expected additions as needed:

```text
bert-score==0.3.13
multimatch-gaze==0.1.3
```

Do not let BERTScore installation silently replace the project's Transformers version.

## `third_party/cider_r/` — NEW unless already verified locally

Vendor the minimum authors' CIDEr-R implementation and PTB tokenizer.

Include:

```text
README_PROVENANCE.md
upstream repository
commit if recoverable
file SHA256 hashes
license
list of copied files
```

Do not import ordinary `pycocoevalcap.cider` and call it CIDEr-R.

---

# 12. Recommended canonical configuration

Use an explicit structure equivalent to:

```yaml
evaluation:
  metrics:
    enabled: true

    where:
      scanmatch: true
      multimatch: true
      sed: true

      isp_reference:
        width: 512
        height: 320
        scanmatch_xbins: 16
        scanmatch_ybins: 12
        scanmatch_temp_bin_ms: 50
        scanmatch_threshold: 3.5
        sed_grid_n: 5

      # Must remain unresolved/disabled until Gate B2 passes.
      coordinate_adapter: null

    probability:
      ll: true
      ig: true
      mode: deepgaze_fast
      resolution: 100
      normalize_digits: true

      centerbias:
        source: canonical_data
        root: null
        allow_synthetic_fallback: false

    semantic:
      bertscore:
        enabled: true
        package_version: "0.3.13"
        lang: en
        model_type: roberta-large
        num_layers: 17
        idf: false
        rescale_with_baseline: false
        use_fast_tokenizer: false
        batch_size: 64

      cider_r:
        enabled: true
        implementation: vendored_authors
        n: 4
        k_r: 0.8
        tokenizer: stanford_ptb_reference
```

BERTScore `batch_size` is engineering-only and must pass batch-size invariance.

Canonical IG config MUST reject missing `centerbias.root` or missing per-image center-bias assets.

---

# 13. Logging and artifacts

## Console

Examples:

```text
[EVAL][LOSS] K=1 | draw=1/10 | ...
[EVAL][PRED] K=1 | draw=1/10 | ...
[EVAL][PROB] K=1 | draw=1/10 | ...    # Outcome B only
```

WHERE metric summary:

```text
[EVAL][METRIC][WHERE]
K=1 draw=0
SM=...
MM=...
SED=...
LL=...
IG=...
valid=...
failed=...
mm_numeric_failures=...
probability_undefined_queries=...
```

Semantic metric summary:

```text
[EVAL][METRIC][SEM]
K=1 draw=0
WHAT_BERTScore_F1=...
WHAT_CIDEr_R=...
WHY_BERTScore_F1=...
WHY_CIDEr_R=...
HOW_BERTScore_F1=...
HOW_CIDEr_R=...
parse_failures=...
```

Do not print one canonical metric line per query outside explicit debug mode.

## Persistent artifact

Integrated evaluation:

```text
predictions/<tag>/metrics.json
```

Standalone evaluation:

```text
<output-dir>/metrics.json
```

Required high-level structure:

```json
{
  "schema_version": 1,
  "baseline_commit": "b84e9c752124457244ee34b93e70f72be5690914",
  "checkpoint": "...",
  "split_manifest_identity": "...",
  "config": "...",
  "split": "test",
  "metric_provenance": {},
  "by_k": {
    "1": {
      "draws": {
        "0": {},
        "1": {}
      },
      "draw_mean": {}
    }
  }
}
```

Persist sufficient provenance to reproduce every metric:

```text
SemGaze baseline commit
actual implementation HEAD
checkpoint
resolved config path/hash
split manifest identity
split
K
draw_id
query count

ISP repository/commit/file hashes
ScanMatch source/hash
VAME source/hash
multimatch-gaze version
coordinate conversion contract/version

DeepGaze source/hash
probability outcome A/B
probability scoring mode
digit normalization
tokenizer identity/revision
center-bias source/hash

bert-score version
Transformers version
BERTScore model/layer/hash/settings

CIDEr-R upstream source/hashes
n
k_r
PTB tokenizer provenance

total/valid/failure counts
under-generation count
format-invalid count
mm_numeric_failure_count
semantic parse-failure count
LL/IG defined/undefined counts
```

---

# 14. Required tests

## 14.1 ISP coordinate parity — blocking

Create a fixture equivalent to:

```text
tests/fixtures/isp_coordinate_pairs.json
```

Include real matched records with:

```text
subject
task
image
condition
raw SemGaze GT coordinates
ISP reference coordinates
image dimensions
```

Assert exact/tolerance-matched GT conversion.

Do not enable SM/MM/SED until this passes and Gate B2 is resolved.

## 14.2 ScanMatch parity

Test:

```text
normal path
1-fixation path
2-fixation path
edge coordinates
duration near TempBin boundaries
```

Compare:

```text
no-duration score
duration score
final harmonic-mean SM
```

Target tolerance:

```text
abs(new - ref) <= 1e-6
```

unless the audited dependency proves a justified larger tolerance.

## 14.3 MultiMatch parity

Compare all five components and final MM.

Include 1-, 2-, and >=3-fixation paths to validate padding.

Target tolerance:

```text
<= 1e-6
```

Also test LOCK 5 numerical-degeneracy handling.

## 14.4 SED parity

Fixtures:

```text
identical
substitution
insertion
deletion
edge coordinates
5x5 boundaries
```

Require exact raw integer equality per pair.

## 14.5 LL tokenizer preflight

Use real checkpoint processor/tokenizer and coordinates:

```text
00
09
10
99
```

The test MUST resolve Outcome A or Outcome B explicitly.

## 14.6 DeepGaze LL parity

Use deterministic phase logits for all ten digit alternatives.

Test leading-zero cases.

Tolerance:

```text
<= 1e-6
```

## 14.7 DeepGaze IG parity

Test a fixed center-bias map for:

```text
center
corners
x/y ordering
100x100 edges
```

Compare:

```text
(LL - logCB) / ln(2)
```

## 14.8 Center-bias loader parity

Compare SemGaze loader to audited DeepGaze behavior after:

```text
exp
resize
normalize
clip
log
```

## 14.9 BERTScore parity

Test:

```text
identical text
paraphrase
different text
punctuation
multiple spaces/newlines
empty candidate extension
```

Assert official F1 under frozen settings and record scorer hash.

## 14.10 CIDEr-R parity

Use a fixed corpus with:

```text
single refs
multiple refs if supported
repeated words
length differences
punctuation
case differences
```

Compare per-sample and corpus score against authors' reference pipeline including PTB tokenization.

## 14.11 Existing-loss regression

Before/after metric integration must preserve:

```text
test_where
test_what
test_why
test_how
test_flat
test_total
```

exactly where deterministic or within an appropriately tiny tolerance.

## 14.12 Existing-generation regression

Freeze checkpoint/config/seed and compare:

```text
WHERE.PRED
SEMANTIC.PRED
where_generation diagnostics
semantic_generation diagnostics
```

They must remain unchanged.

## 14.13 Generate-call-count regression

Instrument `bundle.model.generate`.

Enabling any metric MUST NOT increase generation call count:

```text
SM
MM
SED
BERTScore
CIDEr-R
LL
IG
```

Outcome B may add ordinary forward/probing calls only.

## 14.14 Query coverage and key-join integrity

For each `(K,draw)`:

```text
metric query IDs == expected unseen test query IDs
no duplicates
no omissions
```

For cross-traversal joins:

```text
loss/prob key set == prediction key set
```

using exact key:

```text
(query_id, K, draw_id)
```

## 14.15 Personalized pairing

Use two subjects on the same stimulus with intentionally different GT paths.

Assert each subject is scored against its own GT.

## 14.16 Semantic-unit alignment and identity

Require:

```text
WHAT count = N expected fixations
WHY count = M GT WHY groups
HOW count = 1
```

Assert all semantic-unit IDs are unique and deterministic.

## 14.17 Batch-size invariance

Using frozen predictions:

```text
batch_size=1
batch_size=N
```

must produce equal:

```text
SM
MM
SED
BERTScore-F1
CIDEr-R
```

within deterministic tolerance.

Probability batch-size changes must also preserve LL/IG within floating-point tolerance.

## 14.18 Draw isolation

CIDEr-R document frequency MUST be constructed independently for each complete `(K,draw,branch)` corpus.

## 14.19 K isolation

When `evaluation.k_values=[1,5,10]`, output independent blocks for all K values with no pooled canonical score.

## 14.20 LL/IG undefined handling

Test:

```text
query with 0 fixation
query with 1 fixation
query with >=2 fixations
draw with zero defined probability queries
K with all draws undefined
```

Assert JSON uses `null`, never `NaN`, and counters follow LOCK 7.

## 14.21 Smoke before full run

Before the full configured query universe:

```text
run tiny deterministic subset
verify record counts
verify generation call counts
verify Outcome A/B
verify provenance
verify JSON serialization
verify key joins
verify failure counters
```

Only then run full evaluation.

---

# 15. Performance constraints

Metric integration must add:

```text
0 new autoregressive VLM generation passes
```

Preferred cost:

```text
SM/MM/SED -> CPU from existing predictions
BERTScore -> separate encoder, after VLM tensors are releasable
CIDEr-R -> CPU
LL/IG -> existing WHERE loss forward if Outcome A; otherwise one explicit probability path
```

Do not retain full `[B,L,V]` logits across the dataset.

For LL/IG:

```text
select required predictor positions
select digit logits
reduce to scalar
release tensors
```

Reuse one BERTScorer per evaluation event.

Reuse ScanMatch objects across queries.

Do not broaden this task into DDP/multi-GPU work.

---

# 16. Implementation order

The coding agent MUST work in this order.

## Step 1 — verify pinned baseline

Confirm exact starting commit and clean/known worktree state.

## Step 2 — provenance preflight

Resolve:

```text
ISP source
CIDEr-R source
center-bias assets
source hashes
```

## Step 3 — coordinate parity

Resolve raw GT -> ISP transform.

Search for canonical predicted-bin inverse.

If none exists, STOP SM/MM/SED and report owner blocker.

## Step 4 — scanpath wrappers

Only after Gate B2 passes:

```text
SM
MM
SED
```

Pass standalone reference parity tests before integration.

## Step 5 — semantic references

Freeze/vendor CIDEr-R and BERTScore.

Pass standalone parity tests.

## Step 6 — LL/IG tokenizer audit

Explicitly choose Outcome A or Outcome B.

## Step 7 — LL/IG implementation

Implement exact DeepGaze scoring while preserving exact SemGaze conditioning context.

Pass probability fixtures.

## Step 8 — canonical record extension

Add only required fields and exact identity keys.

## Step 9 — integrate generation-based metrics

Compute SM/MM/SED and BERTScore/CIDEr-R from already-generated records.

Verify zero additional generation calls.

## Step 10 — integrate LL/IG

Reuse Outcome A statistics or add explicit Outcome B pass.

Use key-based joins only.

## Step 11 — logging and `metrics.json`

Add provenance, per-draw values, K means, and failure counters.

## Step 12 — invariance/regression smoke

Verify:

```text
loss unchanged
generation unchanged
support IDs unchanged
query coverage unchanged
key joins exact
generate call count unchanged
batch-size invariance
```

## Step 13 — full evaluation

Only after all applicable gates and tests pass.

---

# 17. Acceptance criteria

Implementation is accepted only if all applicable items below are true.

1. Starting baseline is verified as `b84e9c752124457244ee34b93e70f72be5690914`.
2. Existing evaluation losses remain numerically unchanged.
3. Existing WHERE prediction text remains unchanged.
4. Existing semantic prediction text remains unchanged.
5. Existing support sampling remains unchanged.
6. Existing K/draw semantics remain unchanged.
7. Query membership remains unchanged.
8. Semantic conditioning remains teacher-forced GT XYD + gold WHY groups.
9. SM matches audited ISP-SENet COCO-Search18 behavior.
10. Canonical SM is the harmonic mean of aggregated ScanMatch without-duration and with-duration.
11. ScanMatch uses 512x320, 16x12, TempBin=50 ms, Threshold=3.5, Offset=(0,0).
12. MM uses `multimatch-gaze==0.1.3` or proven-equivalent source.
13. MM uses all five components including duration.
14. MM preserves short-path padding.
15. MM numerical degeneracy follows LOCK 5 and never silently drops a query.
16. SED is the exact ISP/VAME spatial String Edit Distance.
17. SED uses the exact 5x5 partition and raw distance.
18. Scanpath metrics pair exact same-subject prediction/GT.
19. Raw GT -> ISP transform is proven by executable parity.
20. Predicted-bin -> ISP mapping has historical provenance or explicit owner approval.
21. No guessed inverse mapping is silently introduced.
22. GT duration uses raw verified dwell duration rather than model serialization clipping.
23. LL uses GT conditional probabilities, not generated coordinates.
24. LL follows exact digit-normalized DeepGaze-fast spatial scoring.
25. LL/IG preserve exact SemGaze episode conditioning.
26. Initial fixation is not scored by LL/IG.
27. Duration is excluded from canonical spatial LL.
28. LL is natural-log probability.
29. IG uses `(LL - center-bias log probability) / ln(2)`.
30. IG is reported in bits.
31. Canonical IG uses real center-bias data and fails closed when missing.
32. Tokenizer preflight explicitly records Outcome A or B.
33. Outcome A adds no VLM backbone forward.
34. Outcome B is logged as `[EVAL][PROB]`.
35. Outcome B performs no WHERE generation.
36. Outcome B performs no semantic generation.
37. Cross-traversal data are joined by `(query_id,K,draw_id)` with equality assertions.
38. BERTScore uses `bert-score==0.3.13`, `roberta-large`, layer 17, `idf=false`, no baseline rescaling, slow tokenizer.
39. BERTScore official hash is persisted.
40. BERTScore F1 is the canonical semantic BERT metric.
41. CIDEr-R matches the authors' implementation.
42. CIDEr/CIDEr-D are not substituted.
43. CIDEr-R uses the authors' PTB tokenizer path.
44. CIDEr-R uses the complete branch corpus per `(K,draw)`.
45. CIDEr-R is not computed independently per minibatch.
46. WHAT/WHY/HOW unit identities are unique and deterministic.
47. WHAT = one unit per expected fixation.
48. WHY = one unit per GT WHY group.
49. HOW = one unit per query.
50. WHAT/WHY/HOW are not concatenated for canonical metrics.
51. Invalid WHERE generations are counted.
52. Missing semantic units are counted.
53. Parse failures are counted.
54. Failed model outputs are never silently removed from denominators except LL/IG structural undefined queries per LOCK 7.
55. LL/IG undefined values serialize as `null`, never `NaN` or arbitrary zero.
56. Predictions are never repaired with GT content.
57. Metric computation adds zero generation traversals.
58. Enabling metrics leaves `model.generate()` call count unchanged.
59. Implementation remains valid for K=1/5/10 and other configured valid K values.
60. Different K values are not pooled.
61. Different draws remain independently reproducible.
62. K-level metrics are means of complete draw-level metrics.
63. Metric provenance is sufficient to reproduce values.
64. Standalone `evaluate_flat.py` uses the same wrappers.
65. Frozen predictions can be rescored for all non-probability metrics without VLM load.
66. Canonical paper-facing metrics are test-split metrics.
67. `log_z` is absent or validated as a fixed no-op `0.0` compatibility field.
68. Tiny deterministic smoke passes before full evaluation.

---

# 18. Forbidden changes

The implementing agent MUST NOT:

```text
implement SM from memory
implement MM from memory
implement SED from memory
substitute generic metric libraries with matching names
replace ISP ScanMatch without parity
use semantic/text edit distance as scanpath SED
treat SemGaze 0..99 bins as 512x320 pixels
invent a prediction-bin inverse
infer prediction inverse solely from GT parity
round-trip raw GT through binned coordinates unnecessarily
use clipped DDD as GT duration without reference proof
omit MultiMatch duration
change MultiMatch padding
silently drop NaN/-1 MM model predictions
cross subjects
average all human scanpaths for an image
silently drop scanpath failures
silently drop semantic failures
repair model output with GT
use CIDEr or CIDEr-D instead of CIDEr-R
compute CIDEr-R per minibatch
estimate CIDEr-R DF from candidates instead of the reference corpus
silently change BERTScore model/layer/settings
omit BERTScore hash
compute LL from generated x/y
compute LL from ordinary WHERE CE without equivalence proof
score fixation 0 in LL/IG
include duration in canonical spatial LL
replace the SemGaze causal context with a DeepGaze prompt for Outcome B
use synthetic Gaussian center bias for canonical IG
call model.generate() for LL/IG
call model.generate() solely for any metric
rerun free-running WHERE for metrics
rerun semantic generation for metrics
join loss/prediction rows by array position without key assertions
write JSON NaN for undefined LL/IG
convert undefined LL/IG to zero
retain full-vocabulary logits for the evaluation set
change WHERE generation length semantics
change semantic conditioning
change gold WHY semantics
change training loss weights
change optimization behavior
change model architecture
change support sampling
change persisted support draws
change split protocol
change unseen-subject protocol
hardcode K=1
hardcode query count
add DDP/multi-GPU behavior
rewrite the entire evaluator when additive metric integration is sufficient
```

---

# 19. Mandatory stop conditions

The agent MUST stop the affected implementation and report a blocker instead of guessing when any of these occur:

```text
HEAD baseline mismatch
unverified ISP source/provenance
no scientifically justified predicted-bin -> ISP inverse
GT/ISP coordinate parity failure
CIDEr-R authors' source cannot be verified
canonical center-bias data are missing for IG
tokenizer audit cannot establish exact Outcome A or a valid Outcome B
Outcome B would require changing SemGaze conditioning context
cross-traversal key sets differ
reference parity tests fail beyond justified tolerance
existing loss or generation changes after metric integration
metric enablement increases model.generate call count
```

A partial implementation is acceptable when blocked metrics remain explicitly disabled and the blocker is documented.

Do not bypass a stop condition to make the pipeline run.

---

# 20. Final scientific locks

```text
SM:
  ISP-SENet COCO-Search18 ScanMatch
  mean without duration
  mean with duration
  harmonic mean of those two means

MM:
  ISP-SENet MultiMatch
  multimatch-gaze 0.1.3 behavior
  vector + direction + length + position + duration
  mean per component then mean of five components
  deterministic zero extension for model-side numerical degeneracy

SED:
  ISP-SENet VAME String Edit Distance
  5x5 spatial string
  raw edit distance
  same-subject pairing

LL:
  DeepGaze fast spatial GT probability definition
  exact SemGaze episode conditioning context
  four digit phases
  per-phase digit normalization
  natural log
  skip first fixation
  query mean -> draw mean -> K draw mean

IG:
  (LL - canonical center-bias log probability) / ln(2)
  bits/fixation
  same transition/query/draw hierarchy
  no synthetic canonical fallback

BERTScore-F1:
  bert-score 0.3.13
  roberta-large
  layer 17
  English
  idf=false
  rescale=false
  slow tokenizer
  whitespace normalization only
  official hash recorded

CIDEr-R:
  authors' implementation
  authors' PTB tokenizer path
  n=4
  k_r=0.8
  reference-corpus DF
  complete branch corpus per K/draw
  never per minibatch

Semantic units:
  WHAT = per expected fixation
  WHY  = per GT WHY group
  HOW  = per query
  unique deterministic unit IDs

Aggregation:
  logical query/unit
    -> complete draw
    -> draw mean within K

Identity join:
  (query_id, K, draw_id)

Canonical split:
  test

Never pool K.
Never cross subjects.
Never silently remove model failures.
Never regenerate outputs for metrics.
Never invent unresolved scientific mappings.
```

Implementation is scientifically complete only when every enabled metric has passed its required provenance, parity, regression, and invariance gates.
