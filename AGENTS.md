# AGENTS.md — Add AiR-D Training to SE-Net User Embedding

## Goal

Implement **AiR-D support in `isp-senet/SE-Net` only** so the existing SE-Net can train and export **user embeddings** from AiR-D.

Target flow:

```text
AiR-D RGB image + full observed scanpath + question embedding
    -> existing SE-Net fixation tokenization
    -> existing UserEmbeddingNet
    -> user_emb + subject logits
    -> existing subject CE + triplet loss
    -> existing evaluate_user_siamese export
```

This task is **not** scanpath prediction.

---

## Scope

Only edit:

```text
isp-senet/SE-Net/**
```

Expected edits:

```text
isp-senet/SE-Net/
├── configs/
│   └── air_useremb.json          # new
├── src/
│   └── builder.py                # AiR-D loading / loader wiring
├── common/
│   ├── dataset.py                # AiR-D processing adapter
│   ├── data.py                   # only if sample construction needs AiR handling
│   └── utils.py                  # only if subject/support helpers need AiR handling
└── src/models.py                 # avoid unless required to activate existing task branch
```

Prefer leaving these unchanged:

```text
train.py
src/eval_user.py
```

Do **not** modify:

```text
isp-senet/ISP/**
gazeformer-isp/**
```

Do not implement:

```text
Gazeformer / predictor changes
scanpath CE or RL
joint SE-Net + predictor training
new personalization architecture
new metric-learning loss
hard-negative mining
WHAT / WHY / HOW
unrelated bug fixes
```

---

## Source Priority

When behavior is ambiguous:

1. actual source code;
2. for **AiR-D data semantics**, use:
   ```text
   gazeformer-isp/AiR/GazeformerISP/
   ```
3. for **SE-Net architecture/training/export**, use:
   ```text
   isp-senet/SE-Net/
   ```
4. use existing OSIE/COCO SE-Net code only as structural reference.

Do not copy AiR semantics from `ChenLSTMISP` when `AiR/GazeformerISP` already defines them.

Before editing, inspect:

```text
gazeformer-isp/AiR/GazeformerISP/src/dataset/dataset.py
gazeformer-isp/AiR/GazeformerISP/src/preprocess/preprocess_fixations.py
gazeformer-isp/AiR/GazeformerISP/src/preprocess/feature_extractor.py

isp-senet/SE-Net/src/builder.py
isp-senet/SE-Net/common/dataset.py
isp-senet/SE-Net/common/data.py
isp-senet/SE-Net/common/utils.py
isp-senet/SE-Net/src/models.py
isp-senet/SE-Net/src/eval_user.py
isp-senet/SE-Net/train.py
```

Do not guess image naming, embedding keys, or coordinate conventions when the source can answer them.

---

## Preserve the Existing SE-Net Contract

Reuse the existing model:

```python
UserEmbeddingNet.forward(
    img,
    tgt_seq,
    tgt_padding_mask,
    tgt_seq_high,
    duration,
    task_emb,
)
```

Preserve:

```text
full observed scanpath as one encoder example
anchor / positive / negative triplets
positive = same subject
negative = different subject
subject classification CE
triplet margin loss
existing optimizer / scheduler / checkpoint flow
existing image encoder
existing fixation tokenization
existing evaluate_user_siamese export
existing embedding dimension unless config explicitly changes it
```

Do not create a second user-embedding model.

Do not convert AiR-D trajectories into next-fixation training examples.

---

## AiR-D Source Contract

Prefer the AiR contract already implemented under:

```text
gazeformer-isp/AiR/GazeformerISP/
```

Source split files:

```text
AiR_fixations_train.json
AiR_fixations_validation.json
AiR_fixations_test.json
```

Relevant fields:

```text
question_id
image_id
subject_idx
height
width
X
Y
T_start
T_end
length
subject_answer
answer
```

Interpretation:

```text
subject_idx  -> subject identity
question_id  -> question/task condition
image_id     -> stimulus image identity
height,width -> source dimensions for this record
X,Y          -> fixation coordinates
```

Construct fixation duration as:

```python
duration[i] = T_end[i] - T_start[i]
```

Requirements:

```text
preserve fixation order
use the complete scanpath
truncate only at Data.max_traj_length
do not subtract 1 from AiR X/Y
use record-specific source width/height
```

Validate before training:

```text
len(X) == len(Y)
len(T_start) == len(T_end)
length is compatible with the fixation arrays
subject_idx is valid
question_id has an embedding
image_id resolves to an RGB image
```

If source files contain an explicit edge-case rule, follow the source rather than inventing a new one.

---

## Split Handling

For AiR-D:

```text
train      -> AiR_fixations_train.json
validation -> AiR_fixations_validation.json
test       -> AiR_fixations_test.json
```

Training must use:

```text
train loader -> train
valid loader -> validation
```

Do not reuse the current SE-Net convention where some datasets validate on a `test` split.

Do not merge AiR-D `test` into training or validation.

If the current builder assumes one combined annotation file, add an AiR-specific loader path instead of renaming source splits.

---

## Recommended Adapter Design

Prefer one explicit AiR-D adapter:

```text
src/builder.py
    -> load AiR split JSONs
    -> common/dataset.py::process_air_data(...)
    -> canonical SE-Net samples
    -> Siamese_Triplet_Gaze
```

Avoid scattering many AiR-specific branches across the codebase.

A canonical processed record should expose only what SE-Net needs, for example:

```python
{
    "subject_id": ...,
    "image_id": ...,
    "img_name": ...,
    "question_id": ...,
    "fixations": ...,
    "duration": ...,
    "task_emb": ...,
    "scanpath_length": ...,
}
```

Use key names that minimize changes to existing `Siamese_Triplet_Gaze`.

Do not rewrite the source AiR JSON files.

---

## RGB Image Resolution

SE-Net consumes RGB images; the AiR Gazeformer path consumes cached features.

Therefore, add deterministic image resolution for AiR-D.

Inspect the AiR source/data layout first, then implement one helper such as:

```python
resolve_air_image_path(image_id, record, image_root) -> str
```

Do not guess:

```text
extension
zero padding
COCO prefix
train/val folder
whether image_id is already a filename
```

Raise a descriptive `FileNotFoundError` when resolution fails.

Do not recursively search the filesystem inside every `__getitem__`.

---

## Coordinate Processing

AiR coordinates use per-record dimensions.

Before SE-Net normalization/tokenization, convert them into the resized RGB frame using the AiR convention.

Conceptually:

```python
x_resized = x * Data.im_w / record["width"]
y_resized = y * Data.im_h / record["height"]
```

but verify the exact source implementation before committing the formula.

Critical rule:

```text
NO OSIE/COCO-style (X - 1, Y - 1) correction for AiR-D.
```

After the AiR-specific resize step, reuse the current SE-Net:

```text
normalized_fixations
-> transform_fixations(...)
-> coarse + high-resolution fixation tokens
```

Do not create another tokenization pipeline unless unavoidable.

---

## Duration Processing

AiR provides:

```text
T_start
T_end
```

Create:

```python
T = [end - start for start, end in zip(T_start, T_end)]
```

Keep the unit expected by the existing SE-Net path.

Do **not** divide by `1000` just because the downstream Gazeformer predictor does so.

Do not modify the SE-Net duration architecture as part of this task.

---

## Subject Mapping

Use `subject_idx` as the authoritative AiR identity.

Requirements:

```text
one deterministic mapping shared by train/validation/test
classifier target is an integer in [0, Data.num_subjects)
do not remap each split independently
do not infer identity from sample order
```

Preferred:

```text
if subject_idx is already contiguous/stable -> use directly
otherwise -> build one deterministic mapping and reuse it for all splits
```

The mapped AiR branch uses 20 subjects; verify the actual source/annotations before relying on that value.

If exclusion/few-shot helpers are active, preserve the existing SE-Net ID contract.

---

## Question Embeddings

AiR-D is question-conditioned.

Use the question embedding producer/lookup semantics from:

```text
gazeformer-isp/AiR/GazeformerISP/src/preprocess/feature_extractor.py
gazeformer-isp/AiR/GazeformerISP/src/dataset/dataset.py
```

For every AiR sample:

```python
task_emb = question_embedding_dict[question_id]
```

Do **not**:

```text
take the first dictionary value
use one free-viewing vector
average all questions
ignore question_id
```

Normalize string/int keys only as required by the actual saved embedding dictionary.

Add a config path if needed, e.g.:

```json
"task_embedding_path": "processed/embeddings.npy"
```

Keep the expected width compatible with the existing SE-Net task projection, normally 768-D.

---

## Activate Existing Task Conditioning

The current `UserEmbeddingNet` already contains a task-conditioning path, but it is only active under its existing constructor condition.

AiR-D must activate that path.

Before changing `src/models.py`, inspect every use of `ntask`.

Preferred behavior:

```text
if ntask is only a gate for task conditioning:
    activate the existing branch from builder.py
    do not create one head per question
else:
    add the smallest explicit use_task_embedding flag needed
```

Do not redesign `UserEmbeddingNet`.

Do not make the number of unique AiR questions define classifier dimensions.

---

## Triplet Sampling

Keep the current SE-Net semantics:

```text
anchor   -> one AiR trial
positive -> another trial from the same subject
negative -> a trial from a different subject
```

Do not add constraints on:

```text
same question
same image
same answer
same task difficulty
```

Do not add hard-negative mining.

For every train anchor, validate that:

```text
at least one same-subject candidate exists
at least one different-subject candidate exists
```

Fail early with a descriptive error instead of a later `random.choice` failure.

---

## Full-Scanpath Rule

SE-Net must receive the **complete observed AiR scanpath** for one trial, truncated only by `Data.max_traj_length`.

Do not train on prefixes.

If the generic SE-Net preprocessing expands scanpaths into prefixes and later keeps only the terminal sample:

```text
verify exact equivalence for AiR-D
```

If equivalence is not obvious, implement a dedicated `process_air_data` path that directly emits one full-scanpath sample per AiR trial.

Prefer clarity over reusing a pipeline with hidden dataset-specific behavior.

---

## New Config

Add:

```text
isp-senet/SE-Net/configs/air_useremb.json
```

Use existing SE-Net JSONs as the structural template.

Review/set at minimum:

```text
Data.name
AiR split paths or fix_path root
Data.image_path
Data.task_embedding_path
Data.num_subjects
Data.max_traj_length
Data.im_h
Data.im_w
Data.pad_idx

Model.embedding_dim
Data.backbone_config
Data.pixel_decoder

Train.*
```

Use canonical dataset name:

```text
AiR-D
```

unless the current config parser makes another spelling substantially cleaner.

For dataset-specific values such as subject count, image size, trajectory length, and embedding path, inspect AiR source first.

Do not invent values from memory.

---

## Builder

Extend `src/builder.py` with a localized AiR-D branch.

Conceptually:

```python
if hparams.Data.name == "AiR-D":
    train_records = load_air_train(...)
    valid_records = load_air_validation(...)

    subject_mapping = build_or_validate_subject_mapping(...)
    question_embeddings = load_question_embeddings(...)

    train_data = process_air_data(
        train_records,
        subject_mapping,
        question_embeddings,
        ...
    )
    valid_data = process_air_data(
        valid_records,
        subject_mapping,
        question_embeddings,
        ...
    )

    train_dataset = Siamese_Triplet_Gaze(train_data, ...)
    valid_dataset = Siamese_Triplet_Gaze(valid_data, ...)
else:
    # preserve existing behavior
```

Keep the existing `build(...)` return contract.

Do not refactor OSIE/COCO paths unless strictly required.

---

## Training

Do not add a new training loop.

AiR-D must use:

```text
train.py
-> build(...)
-> train_iter
-> compute_loss
-> compute_output
-> UserEmbeddingNet
```

Keep:

```text
subject CE + triplet margin loss
```

A normal command should remain structurally:

```bash
cd isp-senet/SE-Net

python train.py \
  --hparams configs/air_useremb.json \
  --dataset-root /path/to/air-d
```

Use the actual existing CLI flags from `train.py`.

---

## Evaluation / Export

Keep:

```text
train.py
-> run_evaluation
-> src/eval_user.py::evaluate_user_siamese
-> per-subject aggregation
-> saved user-embedding tensor
```

Do not run the downstream predictor.

Do not save a Gazeformer checkpoint.

The final artifact is still the SE-Net subject embedding table.

Prefer preserving the current tensor format.

A dataset-specific filename is allowed if useful, e.g.:

```text
air_train_user_embedding.pt
```

---

## Do Not Fix Unrelated Legacy Behavior

Do not opportunistically change:

```text
SE-Net duration positional encoding behavior
one-shot export aggregation
drop_last
existing support sampling
OSIE/COCO path quirks
predictor checkpoint logic
predictor duration sampling
stale downstream fine-tuning code
```

If one of these blocks AiR-D execution, make the smallest localized fix and report it explicitly.

---

## Validation

Add clear AiR-D validation before training.

Check:

```text
train non-empty
validation non-empty
question_id present
image_id present
subject_idx present
width > 0
height > 0
X/Y compatible
T_start/T_end compatible
durations non-negative
RGB image exists
question embedding exists
embedding width matches model input
subject IDs fit Data.num_subjects
at least two train subjects exist
every train anchor has positive and negative candidates
```

Errors should mention the relevant:

```text
question_id
image_id
subject_idx
```

when possible.

---

## Smoke Tests

### 1. Adapter test

For one real or synthetic AiR record verify:

```text
subject_id == source subject_idx
fixation order preserved
duration[i] == T_end[i] - T_start[i]
no X/Y -1 offset
task_emb selected by question_id
```

### 2. Split test

Verify:

```text
train loader -> AiR train
valid loader -> AiR validation
test is not silently used for validation
```

### 3. Triplet test

For:

```python
sample = dataset[0]
```

verify existing structure:

```text
anchor
positive
negative
```

and:

```text
anchor.subject_id == positive.subject_id
anchor.subject_id != negative.subject_id
```

### 4. Batch test

Load one DataLoader batch and verify fields expected by `compute_output`:

```text
true_state
normalized_fixations
is_padding
duration
task_emb
subject_id
```

### 5. Question-conditioning test

Select two records with different `question_id` and different saved question embeddings.

Verify their `task_emb` tensors differ.

This test must catch accidental "first dictionary value" reuse.

### 6. Forward/loss smoke test

If CUDA, external backbone config/weights, and custom extensions are available:

```text
run one compute_output forward
check user_emb
check pred_subject_id
check user_emb.shape[-1] == Model.embedding_dim
compute CE + triplet
assert finite loss
```

If external runtime assets are unavailable, report exactly what blocked the forward test. Do not claim full runtime success.

---

## Regression Rules

Existing SE-Net datasets must keep their current behavior.

Final diff must not modify:

```text
isp-senet/ISP/**
gazeformer-isp/**
```

Existing configs must still parse.

Avoid global renames of existing dataset fields.

---

## Definition of Done

- [ ] `configs/air_useremb.json` exists.
- [ ] `src/builder.py` recognizes AiR-D.
- [ ] train uses `AiR_fixations_train.json`.
- [ ] validation uses `AiR_fixations_validation.json`.
- [ ] `subject_idx` is the SE-Net user identity.
- [ ] `image_id` resolves to the correct RGB image.
- [ ] full AiR scanpaths are used.
- [ ] no AiR coordinate `-1` shift is introduced.
- [ ] duration is `T_end - T_start`.
- [ ] question embedding is selected by `question_id`.
- [ ] existing SE-Net task-conditioning branch is active.
- [ ] existing triplet construction is preserved.
- [ ] existing CE + triplet loss is preserved.
- [ ] existing embedding export path works.
- [ ] no downstream predictor code is modified.
- [ ] adapter/split/triplet/batch tests pass.
- [ ] forward/loss smoke test passes when required external runtime assets are available.
- [ ] final report lists changed files and unresolved external assumptions.

---

## Final Report Format

After implementation report:

```markdown
## Changed files
- `path`: change

## AiR-D contract
- split handling:
- subject mapping:
- image resolution:
- coordinate handling:
- duration handling:
- question embedding lookup:
- task-conditioning activation:

## Verification
- adapter:
- triplet:
- DataLoader:
- forward/loss:

## Remaining external assumptions
- ...
```

---

## Design Rule

> **Take AiR-D data semantics from `gazeformer-isp/AiR/GazeformerISP`, but keep the model, objective, triplet construction, training loop, and embedding export semantics of `isp-senet/SE-Net`. Do not touch the downstream predictor.**
