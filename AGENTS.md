# AGENTS.md

## Scope
These instructions apply to the SemGaze repository, with emphasis on the COCO-Search18 data, training-sampling, and evaluation pipeline.

The current COCO-Search18 protocol combines two changes that must be implemented together:

1. the dataset split is now **95% train / 5% test**, with no validation split;
2. training is now **query-coverage epoch based**, not random-with-replacement episode sampling.

Do not implement only one of these changes while leaving assumptions from the old protocol elsewhere in the repository.

## COCO-Search18 Split Protocol
COCO-Search18 no longer uses `train / validation / test` for the SemGaze split.

Use only:

- `train`: 95%
- `test`: 5%
- no validation split

The split unit is the image/stimulus (`stimulus_id = image name`). Train and test images must be strictly disjoint.

Current master split:

- train images: 4048
- test images: 213
- train records: 26125
- test records: 1375

Protocol version:

```text
cocosearch18_semgaze_master_955_v1
```

## New COCO-Search18 Paths
Use:

```text
data/COCO_Search18/split_95_5/
```

Main `all` variant:

```text
data/COCO_Search18/split_95_5/all/train.json
data/COCO_Search18/split_95_5/all/test.json
```

Other variants:

```text
data/COCO_Search18/split_95_5/tp_only/train.json
data/COCO_Search18/split_95_5/tp_only/test.json

data/COCO_Search18/split_95_5/ta_only/train.json
data/COCO_Search18/split_95_5/ta_only/test.json
```

Shared metadata:

```text
data/COCO_Search18/split_95_5/master_split_manifest.json
data/COCO_Search18/split_95_5/split_index.json
```

Do not use the old root:

```text
data/COCO_Search18/split/
```

Do not expect or open `validation.json` for the new COCO-Search18 protocol.

## Optimization Query Universe
For COCO-Search18, define the optimization-query universe once from the persisted `all/train.json` split:

```text
Q_train =
    every eligible record in data/COCO_Search18/split_95_5/all/train.json
    whose subject is a seen/training subject
```

The unseen subjects remain:

```text
{7, 8, 9}
```

Records from unseen subjects whose images are in the train split may be used as few-shot support for final evaluation, but they must never be optimization queries.

Query membership must **not** depend on `K`.

Do not create or use separate optimization pools such as:

```text
Q_valid[subject, K]
Q_train_K1
Q_train_K5
Q_train_K10
```

unless they are diagnostics only and do not affect membership or sampling.

The training coverage target is:

```text
query coverage only
```

`K`, support identity, and support order are stochastic context variables, not coverage axes.

## Query-Coverage Epoch Sampling
One training epoch is defined as exactly one shuffled-without-replacement traversal of `Q_train`.

At the beginning of epoch `e`:

```text
epoch_queries = shuffled permutation of Q_train
```

Then process every query exactly once:

```text
for q in epoch_queries:
    sample K_train
    sample same-subject train supports
    randomize support order
    emit one training episode for q
```

Required invariant:

```text
each q in Q_train appears exactly once in every completed epoch
```

Therefore:

- do not sample optimization queries with replacement;
- do not uniformly sample a subject first and then sample a query;
- do not use `query x K` as the coverage target;
- do not repeat a query merely to fill a preferred batch shape;
- do not omit a query because its batch is incomplete;
- do not use `drop_last=True` in a way that drops optimization queries.

Subject frequency during optimization is induced by the query universe:

```text
P(subject=u) = number_of_queries_for_u / |Q_train|
```

There is no separate uniform-subject sampling requirement.

Query order must be reshuffled between epochs using reproducible RNG/seeding.

## Train-Time K Sampling
Training and evaluation use different roles for `K`.

### Training

Train-time support cardinality is sampled independently for each query:

```text
K_train ~ Uniform({1, 2, ..., 10})
```

Equivalent mathematical form:

```text
P(K_train = k) = 1/10,  k in {1, ..., 10}
```

Important:

- `K_train` is **not** a coverage dimension;
- do not guarantee every query is paired with every K;
- do not stratify or enumerate `query x K`;
- do not create a larger epoch because there are multiple K values;
- adding more possible train-time K values must not change the definition of query coverage.

A complete epoch still contains exactly:

```text
|Q_train| training episodes
```

before accounting for any hard-failure condition.

### Evaluation

Canonical evaluation shot settings remain:

```text
K_eval = {1, 5, 10}
```

These values remain benchmark/reporting settings. Do not change them because train-time K is variable.

## Train-Time Support Sampling
For each current query `q`:

1. keep the query fixed;
2. sample `K_train`;
3. construct support candidates from the **same subject** and **train split only**;
4. exclude the query stimulus image;
5. sample exactly `K_train` distinct support images without replacement;
6. resolve one eligible same-subject record per selected support image;
7. uniformly randomize the support-record order;
8. pass the ordered support list unchanged to WHERE.

Required support invariants:

```text
support subject == query subject
support split == train
support stimulus_id != query stimulus_id
all support stimulus_ids are distinct within the episode
support count == K_train
```

Support examples are random context augmentation. There is no requirement to cover every support sample or support combination within one epoch.

The current 95/5 split has enough train support coverage for the configured maximum `K_train=10`. Do not weaken distinct-image or same-subject support constraints.

## Context-Overflow Handling
Training context overflow must **not** destroy query coverage.

If the current query with a sampled `K_train` produces an over-budget context:

```text
keep the same query
keep the same sampled K_train
resample only the support realization
```

Do not:

- replace the query with another query;
- mark the query as covered without training on it;
- silently lower `K_train`;
- drop a support;
- truncate a support scanpath;
- reorder a frozen evaluation support block.

If no valid support realization can satisfy the configured context budget for that query and sampled `K_train`, fail clearly rather than silently changing the training distribution.

Validation/final-evaluation support blocks remain frozen and must never be resampled to repair overflow.

## Same-K Batching and Physical Batching
Same-K batching is an implementation optimization only. It is **not** the sampling law.

Do not alter query coverage or K sampling merely to build same-K batches.

In particular:

- never resample a query to obtain a desired K;
- never skip a query because its K does not match nearby episodes;
- never force K to `{1,5,10}` only to simplify batching;
- never duplicate a query to fill a same-K batch.

If physical batch size is 1, no K-grouping is required.

If physical batch size is greater than 1, bucketing/reordering pending episodes is allowed only if it preserves:

```text
one occurrence of every query per epoch
the configured K_train distribution
the sampled support membership of each emitted episode
```

Batching strategy, gradient checkpointing, and gradient accumulation remain implementation concerns and must not redefine epoch membership.

## Epoch as the Primary Training Unit
For COCO-Search18, the canonical training budget is epoch based.

Define:

```text
1 epoch = 1 complete traversal of Q_train
```

Prefer user-facing configuration such as:

```text
num_train_epochs
```

over arbitrary semantics such as:

```text
steps_per_epoch = 10000
```

when that value does not correspond to full query coverage.

Keep the following concepts separate:

```text
data unit      = one query-conditioned training episode
coverage unit  = one epoch
optimizer unit = one optimizer update after physical batching / gradient accumulation
```

An epoch is **not** one optimizer step.

If:

```text
N_q = |Q_train|
B_global = global physical episode batch size
G = gradient accumulation factor
```

then the nominal optimizer updates per epoch are derived from the coverage epoch, approximately:

```text
ceil(N_q / (B_global * G))
```

subject to the framework's exact handling of the final partial accumulation group.

Never drop tail queries merely to keep optimizer-step sizes uniform.

Scheduler length and warmup should be derived from the configured number of epochs and the resulting optimizer-update count, rather than from an unrelated fixed `max_steps`, unless `max_steps` is explicitly used only as a debugging/smoke-test override.

## Training and Epoch-End Evaluation
For COCO-Search18:

```text
complete one query-coverage epoch
-> evaluate on test split
-> log test metrics
-> continue training
```

There is no validation dataset for this protocol.

Where COCO-Search18-specific code currently uses concepts such as:

- `validation_dataset`
- `validation_loader`
- `val_dataset`
- `val_loader`
- `validation_file`
- `val_file`
- `val/*` metric names

migrate them to the corresponding `test` concepts.

Prefer real semantic cleanup instead of aliases such as:

```python
validation_dataset = test_dataset
```

or configs that keep a `validation_file` field pointing to `test.json`.

If an external framework strictly requires a validation-named field, keep the compatibility layer minimal and document it clearly.

Epoch-end evaluation must run only after the current epoch's query traversal is complete. Do not define an epoch boundary by an arbitrary optimizer-step count that can leave optimization queries unseen.

## Few-Shot Evaluation Protocol
Preserve the canonical evaluation design:

```text
unseen subjects = {7, 8, 9}
K_eval = {1, 5, 10}
10 support draws per K_eval
```

Rules:

- support examples come only from `train`;
- test queries come only from `test`;
- never use test examples as support;
- preserve train/test stimulus disjointness;
- preserve existing frozen/exclusive support-draw semantics;
- one frozen ordered support block is reused across the corresponding full test-query set;
- do not derive evaluation supports from the stochastic train-time support sampler.

The train-time `K_train ~ Uniform({1,...,10})` rule does not change the frozen evaluation settings above.

## Checkpoint and Resume Semantics
A resumable checkpoint must preserve enough sampler state to continue the current query-coverage epoch without duplication or omission.

Save/restore at least:

```text
current epoch index
current query cursor within the epoch
epoch shuffle seed or exact permutation state
K/support RNG state required for deterministic continuation
optimizer state
scheduler state
```

On resume in the middle of an epoch:

```text
continue from the next not-yet-consumed query
```

Do not reshuffle the current epoch and restart it from the beginning unless the run is explicitly restarted as a new training run.

The guarantee remains:

```text
every completed epoch covers every optimization query exactly once
```

## Distributed Training Semantics
Distributed execution must preserve the same global query-coverage epoch.

Conceptually:

```text
1. create one global shuffled epoch permutation
2. partition it across ranks without overlap
3. process the complete union across ranks
```

Require:

```text
union(rank_query_sets) == Q_train
intersection(rank_i, rank_j) == empty for i != j
```

Do not use distributed padding behavior that silently duplicates optimization queries to equalize rank lengths.

If uneven final rank/batch sizes must be supported, handle them explicitly rather than duplicating or dropping queries.

## Master Split Semantics
`all`, `tp_only`, and `ta_only` derive from the same master train/test image partition.

Do not independently resplit the variants.

## What to Remove or Update
Audit the repository for COCO-Search18-specific assumptions involving:

```text
train/validation/test
validation.json
81/9/10
81910
validation_supports
old split paths
val metric namespaces
validation loaders/datasets

random-with-replacement query sampling
uniform subject-first train sampling
query x K coverage
(seen_subject, K) valid-query pools
train-time K restricted to {1,5,10}
steps_per_epoch used as an arbitrary optimizer-step count
max_steps used as the primary training-budget definition
samplers that duplicate/drop queries at epoch boundaries
resume logic that restarts or reshuffles a partially consumed epoch
distributed samplers that pad by duplicating queries
```

Update affected:

- configs
- dataset builders
- dataloaders
- samplers
- training loop
- epoch construction
- epoch-end evaluation
- checkpoint/resume state
- scheduler-step accounting
- few-shot support loading
- standalone evaluation/benchmark scripts
- manifest parsing
- logging
- comments/docs/tests

Do not blindly replace generic functions named `validate` when they perform schema/input validation rather than dataset validation.

## Dataset Scope
These instructions are specifically for the new SemGaze COCO-Search18 protocol.

Do not remove validation splits from AiR-D or other datasets unless their own specification explicitly requires it.

If shared abstractions assume every dataset has three splits, refactor them so datasets can expose different available split sets.

If shared samplers assume random-with-replacement episode generation, refactor them so COCO-Search18 can expose an epoch-sized deterministic query traversal with stochastic support construction.

Do not force other datasets to adopt the COCO-Search18 query-coverage sampler unless their own specification requires it.

## Do Not Change Unrelated Model Behavior
Do not modify, unless required by the split/sampling migration:

- model architecture
- InternVL configuration
- LoRA settings
- WHERE tokenization
- WHAT/WHY/HOW representation
- loss definitions
- teacher forcing
- generation settings
- gradient checkpointing
- unseen subject IDs
- number of evaluation support draws
- canonical evaluation K values `{1,5,10}`
- train-time maximum support count `K_train_max=10`

Batching may be refactored only as needed to preserve the new query-coverage semantics. Do not treat batching convenience as permission to change the sampling law.

## Required Safety Checks
Preserve or add checks for:

```text
train image set ∩ test image set = empty
support images ⊆ train images
test query images ⊆ test images

Q_train contains seen-subject train records only
every completed epoch consumes every Q_train query exactly once
no query is duplicated within a completed epoch
no query is dropped because of batching/drop_last
K_train is always in [1,10]
support subject == query subject
support images are distinct within an episode
query image is not used as support
```

COCO-Search18 runtime code should fail clearly if configured with obsolete `validation.json` or the old `data/COCO_Search18/split/` root.

Training should also fail clearly if query-coverage guarantees cannot be maintained because of sampler, distributed, resume, or overflow behavior.

## Verification Before Finishing
Before considering the migration complete, verify:

- COCO-Search18 uses only train/test;
- no COCO-Search18 runtime path requires `validation.json`;
- train/test JSON paths point to `split_95_5`;
- `Q_train` is constructed from seen-subject `all/train.json` records;
- one epoch is one full shuffled-without-replacement traversal of `Q_train`;
- each optimization query appears exactly once per completed epoch;
- subject-first random sampling is removed from COCO-Search18 training;
- query x K coverage logic is removed;
- train-time K is sampled from `1..10` and does not affect query membership;
- evaluation still uses K=`1/5/10`;
- support is random same-subject train-only context;
- support images are distinct and exclude the query image;
- overflow retries preserve the current query and sampled K;
- epoch-end evaluation uses test;
- COCO-Search18 metric/log naming uses `test`, not `val`, where applicable;
- checkpoint/resume preserves the current epoch permutation/cursor;
- distributed execution neither duplicates nor drops optimization queries;
- 10 support draws per evaluation K remain supported;
- all/tp_only/ta_only share one master partition;
- no train/test image leakage exists;
- unrelated datasets are not accidentally changed.

Run relevant syntax/import/unit/smoke checks after editing.

## Short Execution Prompt
Use this after placing this file at the repository root:

```text
Read AGENTS.md first, then audit and update the repository to fully migrate the SemGaze COCO-Search18 pipeline to the specified 95/5 train/test protocol and query-coverage epoch sampler.

For COCO-Search18 training, build Q_train from seen-subject records in split_95_5/all/train.json. Define one epoch as one shuffled-without-replacement traversal of Q_train. Each query must appear exactly once per completed epoch. For each query, sample K_train uniformly from 1..10, then sample distinct same-subject train supports excluding the query image and randomize support order. K and support identity are stochastic context variables, not coverage axes. Do not use subject-first random sampling, query x K coverage, or (subject,K) query pools.

Preserve evaluation K={1,5,10}, 10 frozen support draws per K, unseen subjects {7,8,9}, train-only supports, test-only queries, and the 95/5 image-disjoint split. There is no validation split. Evaluate on test after each completed query-coverage epoch.

Apply the changes consistently across configs, data loading, samplers, training, scheduler accounting, checkpoint/resume, epoch-end evaluation, few-shot support handling, manifests, benchmarks, logging, and tests. Do not change unrelated datasets or model behavior. After editing, run appropriate checks and summarize changed files, exact paths now used, sampler/epoch semantics, and any remaining compatibility aliases or risks.
```
