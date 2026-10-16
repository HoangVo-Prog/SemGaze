# Independent prediction K and TA/TP/VQA scoring

Patch built from `fixation-ss-lwce` commit `30c0ab737e5f686d2eeea57bbc43b92600a8a6a1`.

The new `evaluation.predictions.where_k_values` and
`evaluation.predictions.semantic_k_values` choose *prediction* support K
independently. Either may be `null` to inherit the legacy
`evaluation.k_values`. The latter continues to control teacher-forced loss.
Unlike the previous shared-K behavior, WHERE and semantic K lists may have
no intersection, overlap, or be identical.

Example evaluation-only YAML (passed to `evaluate_flat.py --config`):

```yaml
evaluation:
  compute_loss: false
  predictions:
    where: true
    semantic: true
    where_k_values: [1, 5, 10]
    semantic_k_values: [1]
    semantic_max_new_tokens: 512
```

At a given K and draw, the evaluator executes only scheduled branches; if
both are scheduled it can reuse that episode and store a single combined
record. `semantic_use_draws: false` still runs semantic once (draw 0) for
every semantic K. Semantic generation remains conditioned on **GT** WHERE
states, not autoregressive WHERE predictions.

Metric output now has `by_group` (`TA`, `TP`, `VQA` when joint), `overall`, and
legacy `by_dataset`. On COCO-only runs, `by_group` has `TA` and `TP`, with an
overall for both. `overall` is recomputed from the pooled raw records using
the original scanpath and semantic scorer, not averaged group results; where
and semantic K sets can differ. The original `metrics.json` contains pooled
per-K, per-draw results, and `metrics_by_group.json` indexes the group files.
The old `metrics_by_dataset.json` remains available for joint runs.
Teacher-forced loss also has `test_by_group` and `test_by_group_k`, in addition
to the original overall and `test_by_dataset` results.

The historical one-off regression scripts were retired during cleanup.
Use the production evaluator to validate these settings on the intended checkpoint.

A full model inference run is required to validate GPU memory consumption and
the external metric packages (BERTScore, CIDEr-R, METEOR), which are unchanged.
