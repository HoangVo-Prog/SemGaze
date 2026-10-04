# SemGaze — Few-Shot Personalization Specification

## 1. Scope

This file is the canonical owner for the current SemGaze few-shot data protocol on **COCO-Search18 only**.

The current executable benchmark scope is:

```text
dataset: COCO-Search18
runtime variant: all
unseen subjects: 7, 8, 9
K: 1, 5, 10
split ratio: 81 / 9 / 10 by unique stimulus image
final support draws: 10 exclusive draws per K
```

**Air-D is out of scope for the current implementation and experiments.** No Air-D preprocessing, split, manifest, support-sampling, training, validation, or evaluation behavior is defined by this version of the specification.

The authoritative curated source is:

```text
data/COCO_Search18/COCOSearch-18.json
```

The root-level split builder is:

```text
prepare_cocosearch18_splits.py
```

The authoritative runtime split for training, validation, and final evaluation is:

```text
data/COCO_Search18/split/all/
```

The split builder may also materialize `tp_only` and `ta_only` artifacts, but those variants are **not consumed by the current SemGaze training/evaluation runtime**. Runtime code must not silently switch away from `all`.

The canonical data flow is:

```text
raw/original COCO-Search18
    -> upstream correctness filtering
    -> generate WHAT / WHY / HOW only for retained records
    -> validate curated semantic records
    -> data/COCO_Search18/COCOSearch-18.json
    -> prepare_cocosearch18_splits.py
    -> data/COCO_Search18/split/all/{train,validation,test}.json
    -> few-shot episode construction
    -> training / validation / final evaluation
```

The upstream correctness-filtering and semantic-annotation procedure is an offline dataset-creation stage. This file does **not** recreate that filter from the final semantic JSON. In particular, the split preprocessor must never infer correctness from `answer == condition`, generated WHAT/WHY/HOW text, or the final fixation text.

This file does **not** own:

- the upstream criterion/code used to determine whether a raw trial is correct before semantic annotation;
- the offline prompt/procedure used to generate WHAT/WHY/HOW;
- support/query chat-role rendering, WHERE prompt text, or scanpath serialization -> `02_WHERE_SPEC.md`;
- low-level InternVL multi-image tensor packing and batch construction -> implementation adapter / `05_TRAINING_SPEC.md`;
- fixation-state extraction -> `03_STATE_SPEC.md`;
- primary multibranch semantic annotation/branch contract -> `multi/04_SEMANTIC_MULTI_SPEC.md`;
- flat single-output semantic contract -> `flat/04_SEMANTIC_SINGLE_SPEC.md`;
- optimizer/backward/batch-RNG details -> `05_TRAINING_SPEC.md`;
- evaluation metrics and final aggregation -> `06_EVALUATION_SPEC.md`.

The purpose of this file is to make benchmark membership, split materialization, support/query sampling, ordering, reuse, and failure behavior deterministic enough that a coding agent never has to invent a data-construction rule.

Persisted split files and frozen support manifests are authoritative runtime inputs. Dataloaders do not split the merged dataset on the fly.

---

## 2. Core definitions

### 2.1 Subject

A **subject** is the observer whose previous gaze demonstrations are used to personalize predictions for that same observer.

Subject identity is used only by the data pipeline for:

- support/query construction;
- seen/unseen subject partitioning;
- subject-disjoint few-shot evaluation.

Subject identity is **not** exposed to the VLM as a token or learned representation.

There is:

- no learned `<USER>` token;
- no subject-ID token;
- no persistent user embedding;
- no subject-specific encoder;
- no per-user adapter;
- no per-user fine-tuning or optimizer step at inference.

### 2.2 Seen and unseen subjects

The current COCO-Search18 protocol fixes:

$$
\mathcal U_{\mathrm{test}} = \{7,8,9\}.
$$

Every other subject ID present in the curated `all` variant belongs to:

$$
\mathcal U_{\mathrm{train}}.
$$

The two sets are disjoint:

$$
\mathcal U_{\mathrm{train}} \cap \mathcal U_{\mathrm{test}} = \varnothing.
$$

Subject partitioning and image partitioning are independent axes:

- seen/unseen determines whether a subject may contribute optimization queries;
- train/validation/test determines the role of a stimulus image.

### 2.3 Canonical record

A canonical gaze record contains at least:

$$
(u, I, Q, S),
$$

where:

- $u$ is the raw subject identity;
- $I$ is the stimulus image;
- $Q$ is the visual-search task;
- $S$ is the observed scanpath.

The current curated COCO record must include at least:

```text
name
subject
task
condition
X
Y
T
prediction
```

The condition must satisfy:

```text
condition in {"present", "absent"}
```

The current runtime variant is `all`, so both conditions are retained. There is no TP/TA filtering during runtime loading.

Canonical bookkeeping fields are:

```text
record_id   = f"coco_semgaze::{subject}::{task}::{name}"
stimulus_id = name
trial_key   = f"{task}::{name}"
```

`record_id` must be globally unique inside the curated artifact. A collision is a preprocessing hard failure.

### 2.4 Support demonstration

A support demonstration is:

$$
d_j^{\mathrm{sup}}=(I_j,Q_j,S_j).
$$

The primary support contract exposes only:

$$
(I,Q,S)_{\mathrm{sup}}.
$$

Support-side semantic labels are hidden:

$$
(W_j,\mathcal G_j,Y_j,H_j) \notin \mathcal C_u^K.
$$

Support trajectories use the serializer owned by `02_WHERE_SPEC.md`.

### 2.5 Query

A query contains:

$$
(I_q,Q_q).
$$

During training, the query additionally has the ground-truth targets required by the owner components. Their serialization is not defined here.

### 2.6 Shot count

The supported few-shot settings are:

$$
K\in\{1,5,10\}.
$$

At train time:

$$
P(K=1)=P(K=5)=P(K=10)=\frac13.
$$

At validation and final evaluation, all three values are evaluated separately.

### 2.7 Curated artifact and structural validation

The split builder consumes:

```text
data/COCO_Search18/COCOSearch-18.json
```

Membership in this curated artifact is the upstream curation decision. The split builder does **not** reproduce correctness filtering.

Before splitting, every record must pass structural validation:

```text
name is non-empty
subject is present
task is non-empty
condition in {"present", "absent"}
len(X) == len(Y) == len(T) > 0
prediction.fixations exists
len(prediction.fixations) == len(X)
prediction.regions exists
prediction.how is non-empty
semantic annotation invariants from `multi/04_SEMANTIC_MULTI_SPEC.md` pass
```

The split preprocessor preserves the original gaze and semantic payload exactly apart from deterministic bookkeeping fields such as:

```text
record_id
stimulus_id
trial_key
split
variant
```

For the canonical runtime variant:

```text
variant = "all"
```

---

## 3. Personalization interface

### 3.1 Conditioning mechanism

Personalization is produced by placing raw demonstrations from the same subject before the query in one causal multimodal context:

$$
\mathcal C_u^K
\rightarrow
\text{native multimodal causal context}
\rightarrow
\text{query prediction}.
$$

A KV cache may be used for efficiency, but it is only a runtime cache. It is not a learned or persistent subject representation.

At Transformer layer $\ell$, query-side tokens can directly attend to support-prefix tokens. Therefore the query computation may condition on support images, tasks, coordinates, durations, and fixation boundaries without compressing the support set into a user vector:

$$
\mathcal C_u^K\not\rightarrow U_u.
$$

### 3.2 Ordered support interface

This file outputs an **ordered support sequence**, not a finished chat transcript.

For one personalized episode, let:

$$
\mathcal C_u^K = \{d_1^{\mathrm{sup}},\ldots,d_K^{\mathrm{sup}}\}.
$$

After applying the training permutation or reading the frozen evaluation order:

$$
\widetilde{\mathcal C}_u^K
=
(d_{\pi(1)}^{\mathrm{sup}},\ldots,d_{\pi(K)}^{\mathrm{sup}}).
$$

The order is part of the model input and must be preserved exactly when handed to WHERE.

The interface to WHERE is:

```text
ordered_supports = [
  (I_1, Q_1, S_1),
  ...,
  (I_K, Q_K, S_K)
]
query = (I_q, Q_q)
```

### 3.3 VLM conversation realization

The canonical VLM realization is owned by `02_WHERE_SPEC.md`.

The ordered supports are rendered as a multi-turn multimodal conversation:

```text
user:      [support image 1] + support prompt 1
assistant: serialized support scanpath 1

user:      [support image 2] + support prompt 2
assistant: serialized support scanpath 2

...

user:      [support image K] + support prompt K
assistant: serialized support scanpath K

user:      [query image] + query prompt
assistant: <query generation starts here>
```

This is a multi-turn chat representation, not a requirement for $K$ separate model forwards. The selected InternVL chat template may serialize the full history into one packed causal model input.

The following are invariants at the few-shot/WHERE boundary:

- support turn $j$ corresponds exactly to ordered support record $j$;
- no support may be inserted, dropped, duplicated, or reordered during rendering;
- every support assistant turn contains the observed scanpath for that same record;
- the query appears only after all $K$ support pairs;
- subject identity is not added as a model token;
- this spec does not add a custom system prompt.

### 3.4 Native multi-image handoff and context-fit policy

Support and query images are handed to the VLM as **separate raw images** in the same order as their corresponding turns. They must not be stitched into one image or replaced by a pooled subject representation.

The exact InternVL-3.5 image preprocessing, visual patch packing, tokenizer/template calls, and token budget are owned by the WHERE/training stack.

Failure policy:

- never silently drop a support because of context length;
- never reorder supports to satisfy batching constraints;
- never silently reduce $K$;
- never silently truncate a support scanpath;
- never resample frozen validation/final supports because of implementation overflow.

For **training**, if the complete causal input exceeds the allowed context budget, reject only that sampled training episode and sample a new episode from the same normative distribution.

For **validation/final evaluation**, all frozen support/query contexts must be preflight-checked. Any invalid context is a hard failure requiring an explicit configuration change; frozen membership/order must not be mutated.

---

## 4. Train-time episodic construction

Training uses only:

```text
variant == "all"
AND subject in seen_subject_ids
AND split == "train"
```

Unseen-subject records in the `all/train.json` split are reserved for few-shot final-evaluation support and never update model parameters.

### 4.1 Training support image identity

Train-time $K$ counts **distinct stimulus images**, not arbitrary trajectory rows.

For COCO-Search18:

```text
support image identity = name = stimulus_id
```

For a seen subject $u$ and candidate query $q$, let $\mathcal I_{u,q}^{\mathrm{sup}}$ be the set of distinct eligible train-split image identities for subject $u$ after removing the query image.

A selected support image must resolve to at least one eligible record from the same subject. If an image has multiple eligible records, one record is sampled uniformly among records for that selected image.

Therefore every train episode has:

- exactly $K$ support records;
- exactly $K$ distinct support images;
- no support image equal to the query image.

### 4.2 Precomputed valid-query pools

For every seen subject $u$ and $K\in\{1,5,10\}$, precompute:

$$
\mathcal Q_{u,K}^{\mathrm{valid}}
=
\{q: |\mathcal I_{u,q}^{\mathrm{sup}}|\ge K\}.
$$

Only queries in this valid pool may be sampled for `(subject, K)`.

Before training starts, assert:

$$
|\mathcal Q_{u,K}^{\mathrm{valid}}|>0
$$

for every seen subject and supported $K$.

If any required pool is empty, fail. Runtime must not repair this by changing subject distribution, lowering $K$, allowing same-image support/query pairs, or sampling support images with replacement.

### 4.3 Sampling distribution

Each train episode is constructed as follows:

1. sample subject uniformly:

$$
u\sim\operatorname{Uniform}(\mathcal U_{\mathrm{train}});
$$

2. sample $K$ uniformly from $\{1,5,10\}$;
3. sample query uniformly from $\mathcal Q_{u,K}^{\mathrm{valid}}$;
4. construct same-subject train-split support-image candidates excluding the query image;
5. sample $K$ distinct support images uniformly without replacement;
6. for each selected image, sample one eligible same-subject record uniformly if multiple records resolve to that image;
7. uniformly randomize the resulting $K$ support records before handing them to WHERE.

The query record itself can never appear in support, no support may share the query image, and no two train supports may share an image.

### 4.4 Training support order

For sampled support demonstrations, draw:

$$
\pi\sim\operatorname{Uniform}(S_K).
$$

Support order is randomized independently for training episodes so sequence position is not a stable subject cue.

Worker seeding and distributed-RNG derivation are owned by `05_TRAINING_SPEC.md`; the probability law above is normative here.

---

## 5. Validation and final evaluation

### 5.1 Seen-subject validation for model selection

Validation mirrors the train-support / held-out-query structure without using unseen-subject gaze.

For each seen subject $u$ and each $K\in\{1,5,10\}$:

- support comes only from eligible `all/train.json` records of subject $u$;
- queries are the full eligible `all/validation.json` query set of subject $u$;
- exactly one ordered validation support set is frozen for `(subject, K)`;
- that support set is reused for every validation query of that subject;
- support identity is `trial_key = (task, image_name)`;
- validation support images must be distinct within one $K$-shot set;
- support order is frozen and never reshuffled per validation query;
- unseen-subject gaze must not affect checkpoint selection, early stopping, hyperparameter selection, or threshold selection.

Deterministic validation support construction:

```python
units = sorted(candidate_validation_support_trial_keys)
rng = random.Random(1000 + K)
rng.shuffle(units)

# Scan in shuffled order and keep a trial only when its image_name
# has not already been selected for this support set.
ordered_validation_supports = take_first_K_distinct_images(units, K)
```

Require at least $K$ valid support units and at least one validation query for every seen subject included in validation.

All three $K$ values are evaluated for model selection. Metric definition and aggregation are owned by `06_EVALUATION_SPEC.md`.

### 5.2 Unseen-subject final evaluation

For unseen subject:

$$
u^\star\in\{7,8,9\},
$$

final evaluation uses disjoint stimulus pools:

```text
support source -> data/COCO_Search18/split/all/train.json
query source   -> data/COCO_Search18/split/all/test.json
```

For fixed `(K, draw_id)`, one frozen ordered support set is reused for every final-test query of that subject. Support is never resampled per query.

### 5.3 Ten exclusive support draws

For each $K\in\{1,5,10\}$ independently, construct exactly 10 final support draws.

The candidate unit is:

```text
trial_key = f"{task}::{image_name}"
```

A candidate is eligible only when:

- its image is in the `all` train split;
- exactly one matching curated record exists for each unseen subject 7, 8, and 9.

For each $K$:

```python
units = sorted(candidate_support_trial_keys)
rng = random.Random(K)
rng.shuffle(units)

# Strong exclusivity across the complete K-family:
# retain a trial only if its image has not previously been retained.
retained = first_(10*K)_units_with_distinct_images(units)
draws = [retained[r*K:(r+1)*K] for r in range(10)]
```

Required properties:

- exactly 10 blocks for each $K$;
- each block contains exactly $K$ support trials;
- no support trial appears in two draw IDs of the same $K$ family;
- no image is reused anywhere across the 10 draws of the same $K$ family;
- the stored order inside each block is the causal support order;
- $K=1$, $K=5$, and $K=10$ are generated independently;
- no nesting or prefix relationship is required across different $K$ values.

At least 100 eligible shared train images are required to materialize the 10-shot family.

### 5.4 Shared support identity across unseen subjects

For a fixed `(K, draw_id)`, subjects 7, 8, and 9 receive the same ordered `(task, image_name)` trial keys.

Each trial key is resolved to the corresponding subject-specific scanpath record:

```text
resolved_record_id_by_subject = {
  "7":  "coco_semgaze::7::<task>::<image_name>",
  "8":  "coco_semgaze::8::<task>::<image_name>",
  "9":  "coco_semgaze::9::<task>::<image_name>"
}
```

This keeps the visual-search trial identity paired while preserving subject-specific gaze.

### 5.5 Paired comparison

For any direct SemGaze-vs-baseline comparison on this curated `all` benchmark, compared methods must receive supports derived from the same frozen support units for the same:

```text
(dataset="COCO-Search18", variant="all", K, draw_id)
```

and must evaluate the same generated test records.

The representation consumed by a baseline may differ, but the underlying selected support trials and query membership may not.

---

## 6. COCO-Search18 `all` split realization

### 6.1 Protocol identity

SemGaze does **not** use the original COCO-Search18 train/validation/test split files at runtime.

The curated source is:

```text
data/COCO_Search18/COCOSearch-18.json
```

The current runtime benchmark is the `all` variant:

```text
all := every curated record, including both present and absent conditions
```

The split builder is:

```text
prepare_cocosearch18_splits.py
```

The builder may create `tp_only`, `ta_only`, and `all`, but this specification treats only:

```text
data/COCO_Search18/split/all/
```

as authoritative runtime input.

### 6.2 Deterministic image-level split

Split by unique:

```text
stimulus_id = name
```

Never split trajectory rows independently.

All records sharing one image inherit the same split.

Frozen protocol version:

```text
protocol_version = "cocosearch18_semgaze_81910_v1"
variant = "all"
image_ratio = [0.81, 0.09, 0.10]
```

For integer `attempt_id >= 0` and image name `x`, rank images by:

```text
SHA256(f"{protocol_version}|all|{attempt_id}|image::{x}")
```

For $N_I$ unique images:

```text
n_train = floor(0.81 * N_I)
n_val   = floor(0.09 * N_I)
n_test  = N_I - n_train - n_val
```

Assign:

```text
first n_train images -> train
next n_val images    -> validation
remaining images     -> test
```

Start from `attempt_id = 0` and choose the first attempt satisfying every coverage requirement below. Store the chosen `attempt_id` in the manifest. Runtime never searches for another split.

### 6.3 Coverage requirements

A candidate `all` split is valid only if all conditions hold.

For every seen subject:

```text
>= 11 distinct train stimulus_ids
>= 1 validation query record
```

The 11-image minimum guarantees that a training query can still have ten different support images after excluding its own image.

For every unseen subject in `{7,8,9}`:

```text
>= 1 test query record
```

For shared final support:

```text
>= 100 eligible shared trial keys with 100 distinct image names
```

If a candidate attempt fails any rule, try the next attempt. Do not manually move records/images between splits.

### 6.4 Persisted split artifacts

Canonical output directory:

```text
data/COCO_Search18/split/all/
```

Required files:

```text
data/COCO_Search18/split/all/train.json
data/COCO_Search18/split/all/validation.json
data/COCO_Search18/split/all/test.json
data/COCO_Search18/split/all/split_manifest.json
data/COCO_Search18/split/all/preprocess_report.json
```

Each split file is a JSON list of complete curated records assigned to that split.

Each persisted record must include at least:

```text
record_id
stimulus_id
trial_key
split
variant
name
subject
task
condition
X
Y
T
answer
prediction
```

For every runtime record:

```text
variant == "all"
```

The semantic payload under `prediction` is copied from the curated artifact; split preprocessing does not regenerate or rewrite it.

`split_manifest.json` is authoritative for membership and frozen supports.

`preprocess_report.json` is diagnostic only and must not be used as a runtime membership source.

### 6.5 Optimization and validation usage

Optimization:

```text
subject not in {7,8,9}
AND record in data/COCO_Search18/split/all/train.json
```

Validation:

```text
support -> same seen subject, all/train.json
query   -> same seen subject, all/validation.json
```

Final evaluation:

```text
support -> unseen subject, all/train.json
query   -> same unseen subject, all/test.json
```

No unseen-subject record may influence optimization or model selection.

### 6.6 Runtime variant selection

The current training/evaluation stack must explicitly bind to:

```text
variant = "all"
```

The following are forbidden:

- auto-detecting a variant directory;
- selecting the first available split folder;
- using `tp_only` because it matches an ISP-SENet target-present setting;
- using `ta_only` for any current training/evaluation path;
- merging the three generated variant folders at runtime.

If `data/COCO_Search18/split/all/` is missing or invalid, runtime must fail loudly.

---

## 7. Frozen split/draw manifest

Canonical manifest:

```text
data/COCO_Search18/split/all/split_manifest.json
```

Minimum schema:

```json
{
  "protocol_version": "cocosearch18_semgaze_81910_v1",
  "dataset": "COCO-Search18",
  "benchmark_kind": "SemGaze-curated",
  "variant": "all",
  "condition_filter": "all curated records",
  "curated_source_file": "data/COCO_Search18/COCOSearch-18.json",
  "curated_source_sha256": null,
  "split_generation": {
    "algorithm": "SHA256 ranking over unique stimulus_id=name",
    "attempt_id": 0,
    "image_ratio": [0.81, 0.09, 0.10],
    "split_unit": "stimulus_id=name",
    "variant_split_independently": true
  },
  "split_files": {
    "train": "data/COCO_Search18/split/all/train.json",
    "validation": "data/COCO_Search18/split/all/validation.json",
    "test": "data/COCO_Search18/split/all/test.json"
  },
  "split_file_sha256": {
    "train": null,
    "validation": null,
    "test": null
  },
  "train_stimulus_ids": [],
  "validation_stimulus_ids": [],
  "test_stimulus_ids": [],
  "seen_subject_ids": [],
  "unseen_subject_ids": [7, 8, 9],
  "validation_supports": {},
  "support_sampling": {
    "unit": "trial_key=(task,image_name)",
    "same_trial_keys_for_unseen_subjects": true,
    "distinct_image_within_block": true,
    "exclusive_across_draws_within_k": true,
    "num_draws_per_k": 10,
    "python_random_seed_by_k": {
      "1": 1,
      "5": 5,
      "10": 10
    },
    "nested_k": false
  },
  "support_draws": {
    "1": [],
    "5": [],
    "10": []
  }
}
```

The actual `attempt_id`, concrete image membership, subject IDs, validation supports, support blocks, source checksum, and split-file checksums stored in the generated manifest are frozen benchmark metadata, not experiment hyperparameters.

Each final support entry stores enough information to resolve exactly:

```text
task
image_name
trial_key
resolved_record_id_by_subject
```

Runtime must read the manifest rather than reconstructing frozen validation or final supports.

---

## 8. Runtime assertions and failure policy

### 8.1 Preprocessing-time hard failures

Split materialization must fail loudly if:

- `data/COCO_Search18/COCOSearch-18.json` is missing;
- a required field is missing;
- `condition` is not `present` or `absent`;
- a canonical `record_id` collides;
- `stimulus_id` is missing;
- scanpath arrays are misaligned;
- semantic annotations violate `multi/04_SEMANTIC_MULTI_SPEC.md`;
- one image would be assigned to more than one split;
- no attempted deterministic split satisfies coverage requirements;
- a shared `(task, image_name)` trial does not resolve uniquely for each unseen subject 7, 8, and 9;
- fewer than `10K` exclusive support images exist for a required final-evaluation $K$;
- train/validation/test do not exactly partition the `all` variant records.

Do not silently drop malformed curated records to make preprocessing succeed.

### 8.2 Runtime hard failures

Training/evaluation must fail loudly if:

- `data/COCO_Search18/split/all/` is missing;
- a required split file is missing;
- the loaded manifest has `variant != "all"`;
- the loaded manifest has unexpected unseen subject IDs;
- a split-file checksum differs from the manifest;
- the curated-source checksum is inconsistent with the materialized benchmark version;
- a required frozen support list is missing;
- any required $\mathcal Q_{u,K}^{\mathrm{valid}}$ is empty before training;
- a supposedly valid support unit cannot resolve to the expected same-subject record;
- a frozen support record is not in `all/train.json`;
- a validation query is not in `all/validation.json`;
- a final query is not in `all/test.json`;
- two records with the same `stimulus_id` appear in different split files;
- final support blocks overlap where exclusivity is required;
- a validation/final causal input exceeds the allowed model context;
- WHERE rendering changes support membership or support order.

Training context overflow is handled only by rejecting the sampled episode and drawing another episode from the same valid training distribution.

### 8.3 Forbidden runtime behavior

No runtime path may:

- load the original/raw COCO-Search18 split files as replacement membership;
- use `tp_only` or `ta_only` in the current experiment pipeline;
- re-filter by `condition` after loading the `all` split;
- recompute correctness filtering;
- regenerate WHAT/WHY/HOW;
- recompute train/validation/test membership;
- move a record/image between split files;
- resample validation/final supports;
- lower $K$;
- sample train support images with replacement;
- permit same-image train support/query leakage;
- reorder a frozen validation/evaluation support list;
- drop a support because of batching or `drop_last`;
- truncate support content;
- replace independent K-specific draw families with nested/prefix supports.

---

## 9. Non-negotiable invariants

- Current dataset scope is COCO-Search18 only.
- Current runtime variant is `all` only.
- Air-D is not part of the current implementation/evaluation scope.
- Both target-present and target-absent curated records are retained in `all`.
- Runtime data comes from persisted `data/COCO_Search18/split/all/*` files.
- Correctness filtering occurs upstream before WHAT/WHY/HOW generation.
- Split preprocessing does not infer correctness from final semantic JSON.
- Train/validation/test splitting is by `stimulus_id=name`, never by trajectory row.
- Split ratio is 81/9/10 by unique image using floor/floor/remainder.
- Every record sharing one image belongs to exactly one split.
- Split membership is deterministic, checksummed, and frozen.
- Unseen subjects are exactly 7, 8, and 9.
- Support and query within one personalized episode belong to the same subject.
- Support content exposed to WHERE is image + task + observed scanpath only.
- Support semantic labels are not replayed to WHERE.
- Subject identity is never supplied as a model token.
- $K\in\{1,5,10\}$.
- Train-time K is sampled uniformly over the three supported values.
- Train subjects are sampled uniformly before query/support sampling.
- Every `(seen_subject, K)` valid-query pool is precomputed and non-empty.
- Train supports use $K$ distinct images sampled without replacement.
- No train support shares the query image.
- Training support order is randomized per episode.
- Validation uses seen subjects only: support from `all/train`, query from `all/validation`.
- Unseen-subject gaze is never used for model selection.
- Final evaluation support comes only from `all/train`.
- Final evaluation query comes only from `all/test`.
- Each $K$ independently uses ten frozen ordered final support blocks.
- The ten blocks within one $K$ family are exclusive by support image.
- There is no required nesting across K=1,5,10.
- A frozen support set is reused across the corresponding full query set.
- COCO unseen-subject supports are frozen by `(task, image_name)` and resolved to subject-specific records.
- Raw support/query images remain separate multimodal inputs.
- Frozen validation/final contexts must fit without truncating, dropping, reordering, or resampling supports.

---

## 10. Canonical persisted artifacts

The canonical few-shot data contract consists of the following persisted artifacts:

```text
data/COCO_Search18/COCOSearch-18.json
prepare_cocosearch18_splits.py

data/COCO_Search18/split/all/train.json
data/COCO_Search18/split/all/validation.json
data/COCO_Search18/split/all/test.json
data/COCO_Search18/split/all/split_manifest.json
data/COCO_Search18/split/all/preprocess_report.json
```

The runtime system additionally depends on:

- `02_WHERE_SPEC.md` for multi-turn support/query rendering and scanpath serialization;
- `03_STATE_SPEC.md` for `<END_FIX>` state extraction;
- `multi/04_SEMANTIC_MULTI_SPEC.md` for the canonical semantic annotation invariants and primary branch contracts;
- `flat/04_SEMANTIC_SINGLE_SPEC.md` for the flat single-output comparison contract;
- `05_TRAINING_SPEC.md` for low-level packing, RNG, batching, and optimization;
- `06_EVALUATION_SPEC.md` for metrics and final aggregation.

If the canonical `all` split files or manifest are absent, implementation is incomplete and must stop. Runtime must never fall back to original COCO-Search18 membership or to another generated variant.
