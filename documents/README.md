# Joint AiR + COCO-Search18 patch for SemGaze `no-attn`

This patch kit targets the **existing** `no-attn` source reviewed on 2026-10-08.
The current hosted GitHub connection does not have write access, so the patch
is provided for application on the owner's checkout. The patch script **does
not alter the repo** unless `--apply` is supplied.

## Apply

```bash
# Work from a clean checkout of no-attn.
cd ~/hoang/SemGaze
git switch no-attn
git status --short

# Unzip this kit, then dry run first:
python /path/to/joint_air_coco/apply_joint_patch.py --repo .
python /path/to/joint_air_coco/apply_joint_patch.py --repo . --apply

# Review and validate.
git diff --check
python -m compileall -q semgaze train_flat.py evaluate_flat.py
python -m pytest tests/test_joint_air_coco.py -q -rs
```

The patcher checks that all source anchors exist **before** writing and
compiles patched Python modules. It will refuse a mismatched code revision.
Do not apply to a dirty checkout without saving unrelated work. Bundled new files (`semgaze/data/joint.py` and `tests/test_joint_air_coco.py`)
are accepted when they **already exist with identical bytes** (including when the
patch kit was extracted into the repository root). If an existing file differs,
the script aborts without overwriting. Existing source transformations still
expect the unpatched target revision; this is not a fully idempotent migration.

## Scope

- Reads **all** persisted records in `data/split/all` instead of selecting only
  the COCO portion; validates AiR against the dataset-specific source manifest
  and 10 precomputed support draws per K.
- Adds an AiR image/record adapter, with namespaced AiR subjects and stimuli.
  AiR is a VQA task; the original `qid` is used for its frozen support groups.
- Updates both train and test episode samplers so support and query share the
  same dataset, subject and train-image membership.
- Selects `dataset: all` for `configs/flat_single.yaml`, while the defaults
  continue to permit older COCO-only experiments.
- Keeps legacy aggregate loss for compatibility **and** adds
  `test_by_dataset` and `test_by_dataset_k` outputs to evaluation summaries.
- Dataset-splits semantic/scanpath metric artifacts into separate directories
  under `predictions/<epoch-or-step>/AiR/` and `.../COCO_Search18/` and writes
  a top-level `metrics_by_dataset.json`. Does not introduce a fictitious joint
  metric. Probability LL/IG is COCO-only until AiR is independently validated.
- Includes CPU-only regression fixtures for record typing, split combination,
  same-subject K-shot sampling and qid-based AiR support selection.

## Dataset layout assumed

```text
data/
  split/
    all/{train,test,test_seen,split_manifest}.json
    AiR/split/all/split_manifest.json
    COCO_Search18/split/all/split_manifest.json
  images/
    AiR/                     # unique AiR filenames inside this subtree
    COCO_Search18/           # or existing COCO recognized aliases
```

The AiR image adapter assumes a unique image basename within the AiR subtree.
If the actual image layout differs, adjust `AirAdapter.image_info` in
`semgaze/data/joint.py` rather than disabling integrity checks.

## Verify exact counts (including unseen support-only training rows)

```bash
python - <<'PY'
from collections import Counter
from pathlib import Path
from semgaze.model.config import load_config
from semgaze.data.cocosearch18 import read_persisted_splits
from semgaze.data.joint import JointAdapter, unseen_subject_ids
c = load_config(Path('configs/flat_single.yaml'))
d = c['data']
raw, manifest, identity = read_persisted_splits(Path(d['split_root']), data_config=d)
for split, rows in raw.items():
    print(split, Counter(r['dataset'] for r in rows))
adapter = JointAdapter(Path(d['images_root']), annotation_frame=d['annotation_frame'], duration_field=d['duration']['source_field'])
records = [adapter(r) for r in raw['train']]
unseen = unseen_subject_ids(d, manifest)
print('eligible train queries:', Counter(r.dataset for r in records if r.subject not in unseen))
PY
```

Expected loader counts from the user's current persisted files:
`train: COCO-Search18=24766, AiR=9677` and
`test: COCO-Search18=806, AiR=339`. These are *all persisted records*,
including unseen-subject support-only rows in train; actual gradient queries
are the **eligible seen-subject subset**.

## New run required

**Do not resume COCO-only checkpoints to claim joint training.** A model
already trained only on COCO has not seen AiR. Use a fresh run directory and
train from a new joint config. The run's split identity and query universe are
different, so resume of a COCO-only training run is intentionally unsupported.

Example:

```bash
python train_flat.py --config configs/flat_single.yaml --output-dir runs/joint-air-coco \
  --semantic-max-new-tokens 128
```

Be aware that the total train duration, train step count, evaluation workload
and memory footprint can increase. Eval now includes 1,145 unseen test
records per (K, draw), instead of 806. At K=1/draw=1 this is +42.1% test
queries. K=1/5/10 and draw=10 are far more expensive; keep the chosen
`evaluation.k_values` and `evaluation.draw` deliberate.

## Validation status / limitations

- Source anchors were compared to the remote `no-attn` code.
- The new joint reader passed an isolated test with mocked dependencies.
- Python syntax was checked for all added patch-kit source files.
- **Full repository unit tests, real manifest/image loading and GPU integration
  tests cannot be run here** because the original repo and image datasets are
  not mounted in the execution container. Run the commands above before
  launching a long training job.
- AiR scanpath scoring reuses the COCO/ISP coordinate transform but its
  canonical AiR comparability requires a separate dataset-specific check;
  do **not** present the metrics as a single cross-dataset benchmark score.
- Evaluation with `ll=true` limits probability scoring to COCO. AiR
  probability-metric implementation is not claimed in this patch.

## Patch kit v1.1 — duplicate subject-anchor fix

The original script failed during dry-run on `semgaze/evaluation/predictions.py`
with `prediction string subject 1: expected exactly one anchor, found 2`.
The patcher now requires **exactly two** occurrences and updates both in one
operation (WHERE and semantic consistency checks). The dry-run remains
fail-closed and atomic: source files are written only when `--apply` is used.

For a standalone sanity check before applying:

```bash
python /path/to/joint_air_coco/test_patcher_anchor.py
python /path/to/joint_air_coco/apply_joint_patch.py --repo .
```

The regression checks that two exact anchors are replaced and any unexpected
third anchor is rejected, and compiles the generated Python source.


## Patch kit v1.2 — bundled files already present

The patcher now accepts `semgaze/data/joint.py` and `tests/test_joint_air_coco.py`
when their bytes exactly match the bundled files. It prints `EXISTS IDENTICAL`
during dry-run and `UNCHANGED (identical)` during apply. Any differing file is
rejected before writing; no force-overwrite is supported.
