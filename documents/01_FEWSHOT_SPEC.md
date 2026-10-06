# SemGaze — Few-Shot Personalization Specification

## 1. Scope

This file is the canonical owner for the current SemGaze few-shot data protocol on **COCO-Search18 only**.

The current executable benchmark scope is:

```text
dataset: COCO-Search18
runtime variant: all
unseen subjects: 7, 8, 9
train K: uniformly sampled from 1..10 (not a coverage axis)
evaluation K: 1, 5, 10
split ratio: 95 / 5 (train / test; no validation) by unique stimulus image
final support draws: 10 exclusive draws per K
```

**Air-D is out of scope for the current implementation and experiments.** No Air-D preprocessing, split, manifest, support-sampling, training, validation, or evaluation behavior is defined by this version of the specification.

The authoritative curated source is:

```text
data/COCO_Search18/COCOSearch-18.json
```

The root-level split builder is:

```text
prepare_cocosearch18_splits_95_5.py
```

The authoritative runtime split for training and test evaluation is:

```text
data/COCO_Search18/split_95_5/all/
```

The split builder may also materialize `tp_only` and `ta_only` artifacts, but those variants are **not consumed by the current SemGaze training/evaluation runtime**. Runtime code must not silently switch away from `all`.

The canonical data flow is:

```text
raw/original COCO-Search18
    -> upstream correctness filtering
    -> generate WHAT / WHY / HOW only for retained records
    -> validate curated semantic records
    -> data/COCO_Search18/COCOSearch-18.json
    -> prepare_cocosearch18_splits_95_5.py
    -> data/COCO_Search18/split_95_5/all/{train,test}.json
    -> few-shot episode construction
    -> training / test evaluation
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
- train/test determines the role of a stimulus image.

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

### 2.6 Support cardinality and evaluation shot settings

SemGaze separates **train-time support cardinality** from **evaluation shot settings**.

At train time:

$$
K_{\mathrm{train}}\sim\operatorname{Uniform}\{1,\ldots,10\}.
$$

Train-time $K$ is stochastic context cardinality only. It is **not** a coverage dimension, is not stratified, and does not determine query membership. A query is considered covered when it has appeared as an optimization query, regardless of the sampled $K$ for that occurrence.

At test evaluation, the canonical reported shot settings remain:

$$
K_{\mathrm{eval}}\in\{1,5,10\}.
$$

All three evaluation values are evaluated separately under the frozen support protocol in Section 5. The current maximum train-time cardinality is:

$$
K_{\max}^{\mathrm{train}}=10.
$$

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
- never resample frozen test supports because of implementation overflow.

For **training**, context overflow must never replace or skip the current epoch query. Keep the query and sampled $K$ fixed, resample only the same-subject support realization/order, and retry. If no valid realization is found within the configured finite retry budget, fail loudly. Do not lower or resample $K$ to repair overflow.

For **test evaluation**, all frozen support/query contexts must be preflight-checked. Any invalid context is a hard failure requiring an explicit configuration change; frozen membership/order must not be mutated.

---

## 4. Train-time query-coverage construction

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

For a seen-subject query $q$ with subject $u$, let $\mathcal I_{u,q}^{\mathrm{sup}}$ be the set of distinct eligible train-split image identities for subject $u$ after removing the query image.

A selected support image must resolve to at least one eligible record from the same subject. If an image has multiple eligible records, one record is sampled uniformly among records for that selected image.

For a sampled train-time cardinality $K$, every training episode has:

- exactly $K$ support records;
- exactly $K$ distinct support images;
- no support image equal to the query image.

### 4.2 Optimization-query universe

Define the complete optimization-query universe:

$$
\mathcal Q_{\mathrm{train}}
=
\{q:\ q\in \texttt{all/train.json},\ \operatorname{subject}(q)\in\mathcal U_{\mathrm{train}}\}.
$$

Query membership is independent of $K$. There are no `(subject, K)` valid-query pools in the canonical training protocol.

Before training starts, assert for every $q\in\mathcal Q_{\mathrm{train}}$:

$$
|\mathcal I_{u(q),q}^{\mathrm{sup}}|\ge K_{\max}^{\mathrm{train}}=10.
$$

The current split construction already requires at least 11 distinct train stimulus images for every seen subject, so excluding the query image leaves at least ten candidate support images.

If any optimization query fails this requirement, fail loudly. Runtime must not repair it by dropping the query, changing subject distribution, lowering $K$, allowing same-image support/query pairs, or sampling support images with replacement.

### 4.3 Query-coverage epoch

A **training epoch** is exactly one shuffled-without-replacement traversal of $\mathcal Q_{\mathrm{train}}$.

For epoch $e$, draw one permutation:

$$
\Pi_e
=
\operatorname{Shuffle}(\mathcal Q_{\mathrm{train}};\,\mathrm{seed}_e)
=
(q_1,\ldots,q_{N_Q}),
$$

where:

$$
N_Q=|\mathcal Q_{\mathrm{train}}|.
$$

Therefore every optimization query appears **exactly once per completed epoch**. Query sampling with replacement inside an epoch is forbidden. A new permutation is created for each new epoch.

Subject frequency is induced by the query set itself:

$$
P_e(u)=\frac{|\{q\in\mathcal Q_{\mathrm{train}}:\operatorname{subject}(q)=u\}|}{N_Q}.
$$

Subjects are not sampled uniformly before queries in the canonical training protocol.

### 4.4 Per-query stochastic support construction

For each query $q_i$ in the current epoch permutation:

1. keep $q_i$ fixed as the optimization query;
2. sample:

$$
K_i\sim\operatorname{Uniform}\{1,\ldots,10\};
$$

3. construct same-subject train-split support-image candidates excluding the query image;
4. sample $K_i$ distinct support images uniformly without replacement;
5. for each selected image, sample one eligible same-subject record uniformly if multiple records resolve to that image;
6. uniformly randomize the resulting $K_i$ support records before handing them to WHERE.

The query record itself can never appear in support, no support may share the query image, and no two train supports may share an image.

The only guaranteed train-time coverage axis is **query identity**. The protocol does not guarantee query-by-K coverage, K-frequency equality within one finite epoch, support-combination coverage, or support-record coverage. Those quantities arise stochastically from repeated epochs.

### 4.5 Training support order

For sampled support demonstrations, draw:

$$
\pi\sim\operatorname{Uniform}(S_K).
$$

Support order is randomized independently for training episodes so sequence position is not a stable subject cue.

Worker seeding and distributed-RNG derivation are owned by `05_TRAINING_SPEC.md`; they must preserve the epoch-level no-replacement query traversal and the stochastic support law above.

---

## 5. Test evaluation

### 5.1 Epoch-end evaluation

After every completed query-coverage epoch, evaluate the unseen-subject test
queries using the same frozen protocol as final evaluation below. There is no
validation split, no seen-subject validation support table, and no implicit
best-checkpoint selection. Test metrics are logged with the `test` namespace.

### 5.2 Unseen-subject final evaluation

For unseen subject:

$$
u^\star\in\{7,8,9\},
$$

final evaluation uses disjoint stimulus pools:

```text
support source -> data/COCO_Search18/split_95_5/all/train.json
query source   -> data/COCO_Search18/split_95_5/all/test.json
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

SemGaze does **not** use the original COCO-Search18 train/test split files at runtime.

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
prepare_cocosearch18_splits_95_5.py
```

The builder may create `tp_only`, `ta_only`, and `all`, but this specification treats only:

```text
data/COCO_Search18/split_95_5/all/
```

as authoritative runtime input.

### 6.2 Shared master image partition

Protocol version is `cocosearch18_semgaze_master_955_v1`. The builder searches
deterministically over master image partitions with 95% train and 5% test,
checks coverage, and selects the valid candidate with the best distribution
score. All variants are filters of the selected master partition; they are
never independently split. The existing artifacts are authoritative and must
not be regenerated during training.

Current master: 4,048 train images and 213 test images; 26,125 train records and
1,375 test records in `all`. The split unit is `stimulus_id = name`.

### 6.3 Coverage requirements

Optimization queries are all eligible seen-subject `all/train.json` records.
They must have enough same-subject distinct train images for any K_train in
1..10 after excluding their own image; failure must not filter query membership.
Unseen train records are evaluation support only. Test records never become
optimization queries or support examples.

### 6.4 Persisted artifacts and runtime variants

```text
data/COCO_Search18/split_95_5/master_split_manifest.json
data/COCO_Search18/split_95_5/split_index.json
data/COCO_Search18/split_95_5/{all,tp_only,ta_only}/train.json
data/COCO_Search18/split_95_5/{all,tp_only,ta_only}/test.json
data/COCO_Search18/split_95_5/{all,tp_only,ta_only}/split_manifest.json
```

Training requires `all`; the loader and evaluation can inspect the other variants
under the same master partition. There is no `validation.json` artifact.

## 7. Frozen split/draw manifest

Each variant manifest records the protocol version, curated-source checksum,
train/test file checksums and stimulus IDs, seen/unseen subject IDs, and
`support_draws`. The shared index verifies variant manifest checksums and counts;
the master manifest owns image membership. Runtime checks all three metadata
files and includes their combined identity in checkpoints.

For each K_eval in 1/5/10, `support_draws[str(K)]` contains ten ordered blocks.
Each entry contains `image_name`, `task`, `trial_key`, and
`resolved_record_id_by_subject` for subjects 7/8/9. All entries resolve to train
records. Images are exclusive across the ten draws within a K family. One block
is reused unchanged across its full test query set. There is no
`validation_supports` field.

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
- train/test do not exactly partition the `all` variant records.

Do not silently drop malformed curated records to make preprocessing succeed.

### 8.2 Runtime hard failures

Training/evaluation must fail loudly if:

- `data/COCO_Search18/split_95_5/all/` is missing;
- a required split file is missing;
- the loaded manifest has `variant != "all"`;
- the loaded manifest has unexpected unseen subject IDs;
- a split-file checksum differs from the manifest;
- the curated-source checksum is inconsistent with the materialized benchmark version;
- a required frozen support list is missing;
- any optimization query has fewer than $K_{\max}^{\mathrm{train}}=10$ eligible same-subject support images after excluding its query image;
- a supposedly valid support unit cannot resolve to the expected same-subject record;
- a frozen support record is not in `all/train.json`;
- a test query is not in `all/test.json`;
- a final query is not in `all/test.json`;
- two records with the same `stimulus_id` appear in different split files;
- final support blocks overlap where exclusivity is required;
- a test causal input exceeds the allowed model context;
- WHERE rendering changes support membership or support order.

Training context overflow must preserve the current epoch query and sampled $K$: resample only the support realization/order. If the configured finite retry budget is exhausted, fail loudly rather than skipping the query, replacing the query, or changing $K$.

### 8.3 Forbidden runtime behavior

No runtime path may:

- load the original/raw COCO-Search18 split files as replacement membership;
- use `tp_only` or `ta_only` in the current experiment pipeline;
- re-filter by `condition` after loading the `all` split;
- recompute correctness filtering;
- regenerate WHAT/WHY/HOW;
- recompute train/test membership;
- move a record/image between split files;
- resample test supports;
- lower $K$;
- sample train support images with replacement;
- permit same-image train support/query leakage;
- reorder a frozen test evaluation support list;
- drop a support because of batching or `drop_last`;
- drop or duplicate an optimization query because of batching, sharding, `drop_last`, or context-overflow recovery;
- truncate support content;
- replace independent K-specific draw families with nested/prefix supports.

---

## 9. Non-negotiable invariants

- Current dataset scope is COCO-Search18 only.
- Current runtime variant is `all` only.
- Air-D is not part of the current implementation/evaluation scope.
- Both target-present and target-absent curated records are retained in `all`.
- Runtime data comes from persisted `data/COCO_Search18/split_95_5/all/*` files.
- Correctness filtering occurs upstream before WHAT/WHY/HOW generation.
- Split preprocessing does not infer correctness from final semantic JSON.
- Train/test splitting is by `stimulus_id=name`, never by trajectory row.
- Split ratio is 95/5 by unique image, shared across all variants.
- Every record sharing one image belongs to exactly one split.
- Split membership is deterministic, checksummed, and frozen.
- Unseen subjects are exactly 7, 8, and 9.
- Support and query within one personalized episode belong to the same subject.
- Support content exposed to WHERE is image + task + observed scanpath only.
- Support semantic labels are not replayed to WHERE.
- Subject identity is never supplied as a model token.
- Train-time support cardinality satisfies $K_{\mathrm{train}}\sim\operatorname{Uniform}\{1,\ldots,10\}$.
- Canonical test evaluation shot settings are $K_{\mathrm{eval}}\in\{1,5,10\}$.
- Train-time K is stochastic context cardinality, not a coverage axis.
- The optimization-query universe is independent of K.
- One completed training epoch visits every optimization query exactly once in shuffled order.
- Training queries are shuffled without replacement within each epoch.
- Subject frequency during training is induced by query membership; subjects are not sampled uniformly first.
- Every optimization query has at least ten eligible same-subject support images after excluding its query image.
- Train supports use $K$ distinct images sampled without replacement.
- No train support shares the query image.
- Training support order is randomized per episode.
- Epoch-end evaluation uses unseen subjects: support from `all/train`, query from `all/test`.
- Unseen-subject gaze is never used for model selection.
- Final evaluation support comes only from `all/train`.
- Final evaluation query comes only from `all/test`.
- Each $K$ independently uses ten frozen ordered final support blocks.
- The ten blocks within one $K$ family are exclusive by support image.
- There is no required nesting across K=1,5,10.
- A frozen support set is reused across the corresponding full query set.
- COCO unseen-subject supports are frozen by `(task, image_name)` and resolved to subject-specific records.
- Raw support/query images remain separate multimodal inputs.
- Frozen test contexts must fit without truncating, dropping, reordering, or resampling supports.

---

## 10. Canonical persisted artifacts

The canonical few-shot data contract consists of the following persisted artifacts:

```text
data/COCO_Search18/COCOSearch-18.json
prepare_cocosearch18_splits_95_5.py

data/COCO_Search18/split_95_5/all/train.json
data/COCO_Search18/split_95_5/all/test.json
data/COCO_Search18/split_95_5/all/split_manifest.json
data/COCO_Search18/split_95_5/all/preprocess_report.json
```

The runtime system additionally depends on:

- `02_WHERE_SPEC.md` for multi-turn support/query rendering and scanpath serialization;
- `03_STATE_SPEC.md` for `<END_FIX>` state extraction;
- `multi/04_SEMANTIC_MULTI_SPEC.md` for the canonical semantic annotation invariants and primary branch contracts;
- `flat/04_SEMANTIC_SINGLE_SPEC.md` for the flat single-output comparison contract;
- `05_TRAINING_SPEC.md` for low-level packing, RNG, batching, and optimization;
- `06_EVALUATION_SPEC.md` for metrics and final aggregation.

If the canonical `all` split files or manifest are absent, implementation is incomplete and must stop. Runtime must never fall back to original COCO-Search18 membership or to another generated variant.
