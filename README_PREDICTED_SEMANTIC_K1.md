# InterGAZE / SemGaze: K=1 semantic evaluation from frozen WHERE predictions

## Placement

Copy `evaluate_predicted_semantic.py` to the **repository root** beside
`evaluate_flat.py`, and copy `tests/test_predicted_semantic_k1.py` to `tests/`.
Run from the repository root on branch `clean` (audited 2026-11-06).
The baseline evaluator remains unchanged. The `--gold-why-groups` switch selects
two distinct semantic prompts. The new script reuses its checkpoint,
splits, episode construction, hidden-state extractor, projector, and existing
semantic scoring code.

## Commands

Set `CKPT` to the **actual** trained InterGAZE checkpoint directory.
The checkpoint path and input JSONL files were not present in the supplied
attachment; the following paths are examples to replace on the running machine.

```bash
CKPT=/absolute/path/to/your/checkpoints/checkpoint-name
mkdir -p /path/to/where_inputs
# Put share_state(5).jsonl and share_state(6).jsonl in /path/to/where_inputs

python evaluate_predicted_semantic.py \
  --checkpoint "$CKPT" \
  --where-predictions '/path/to/where_inputs/share_state(5).jsonl' \
  --output-dir runs/pred_sem_unseen_k1 \
  --subject-scope unseen \
  --dry-run

python evaluate_predicted_semantic.py \
  --checkpoint "$CKPT" \
  --where-predictions '/path/to/where_inputs/share_state(6).jsonl' \
  --output-dir runs/pred_sem_seen_k1 \
  --subject-scope seen \
  --dry-run

python evaluate_predicted_semantic.py \
  --checkpoint "$CKPT" \
  --where-predictions '/path/to/where_inputs/share_state(5).jsonl' \
  --output-dir runs/pred_sem_unseen_k1 \
  --subject-scope unseen \
  --semantic-max-new-tokens 512 \
  --gold-why-groups off

python evaluate_predicted_semantic.py \
  --checkpoint "$CKPT" \
  --where-predictions '/path/to/where_inputs/share_state(6).jsonl' \
  --output-dir runs/pred_sem_seen_k1 \
  --subject-scope seen \
  --semantic-max-new-tokens 512 \
  --gold-why-groups off
```

## Select gold WHY-group conditioning

The option defaults to **off**, preserving the original predicted-file patch.
Run each setting in a **different output directory**, so that their semantic
predictions and metrics cannot be confused:

```bash
# No gold group information (new dedicated prompt; DEFAULT)
python evaluate_predicted_semantic.py \
  --checkpoint "$CKPT" \
  --where-predictions '/path/to/where_inputs/share_state(5).jsonl' \
  --output-dir runs/sem_unseen_nogold_k1 \
  --subject-scope unseen \
  --gold-why-groups off \
  --semantic-max-new-tokens 512

# Original gold WHY-group membership in prompt, PREDICTED XYD hidden states
python evaluate_predicted_semantic.py \
  --checkpoint "$CKPT" \
  --where-predictions '/path/to/where_inputs/share_state(5).jsonl' \
  --output-dir runs/sem_unseen_gold_k1 \
  --subject-scope unseen \
  --gold-why-groups on \
  --semantic-max-new-tokens 512
```

The switch can also be configured using an evaluation override YAML:

```yaml
evaluation:
  predicted_semantic:
    gold_why_groups: true  # false by default
```

CLI `--gold-why-groups on|off` takes precedence over this YAML value.
The chosen mode is saved to resolved config, every output record,
`validation_summary.json`, and `evaluation_summary.json`.

**Mode off (`no_gold_why_groups`):** The dedicated prompt provides the viewing
task and predicted chronological fixation states (or PRED XYD text for the
explicit ablation). It requests exactly N WHAT statements, **one WHY for the
entire predicted trajectory**, and one HOW statement. It provides **no GT
WHY group count or memberships**, and does not pretend the single WHY is a
ground-truth group. A one-group placeholder exists internally solely to stay
compatible with the existing parser; it is never shown as a grouping label.
An output with one WHY does **not** predict GT WHY segmentation.

**Mode on (`gold_why_groups`):** Use the repository's canonical `build_flat_prompt`
with the **actual GT WHY-group membership and group count** (not the semantic
reference text). The hidden states still come exclusively from replayed
`WHERE.PRED`. This mode intentionally exposes GT segmentation and should
be reported as gold-group-conditioned semantic inference, *not* a fully
GT-free predicted-trajectory condition. If the predicted fixation count differs
from the GT count, the sample fails with
`gold_groups_require_equal_fixation_counts`; do not pad, crop, or remap either
sequence. These failures remain in the overall metric denominator as empty
semantic outputs.

The old GT-conditioned evaluator is unaffected by both settings.

Add `--config configs/evaluation/eval.yaml` only when suitable: the checkpoint
remains authoritative for model/data/annotation settings; runtime/evaluation/test
overrides are read using the existing `evaluate_flat._load_evaluation_config`.
Use `--branch-metrics` for additional field-level diagnostic metrics; beware
that in `--gold-why-groups off`, WHY units may not align with GT groups.

## Provenance and scope

- Strictly filter `K == k == 1, draw_id == 0` **before** inference.
- Match `query_id`, condition, observer, image, task, support IDs,
  split-manifest identity, and canonical JSONL GT information to the official
  test population before loading the 8B model.
- The WHERE-generation function is never called. `WHERE.PRED` is parsed in full;
  it is not truncated to match the reference fixation count. Canonical text
  consistency is checked against legacy `where_generation.fixations`.
- Construct a **shadow query** containing predicted XYD only (inverse mapped
  from quantized coordinates to pixel coordinates, then checked by exact
  reserialization). In mode off, a placeholder group is retained internally
  for parser compatibility; no group membership is supplied to the model.
  In mode on, only GT WHY membership is attached to the prompt.
- `forward_where_batch` teacher-forces **stored PRED tokens**, NOT GT tokens,
  to reconstruct the exact causal readout at each predicted `<END_FIX>`.
  This is a replay forward pass, **not a WHERE generation**.
- Feed the projected states to the semantic prompt and decode WHAT/WHY/HOW.
  For the explicit-scanpath ablation, the same PRED sequence is supplied as text.
  Both group modes use the same PRED trajectory for semantic conditioning.
- Use the existing semantic metric implementation. Missing or malformed WHERE
  predictions receive empty semantic output and therefore zero contribution
  to the overall semantic text score, not a substituted GT scanpath.
- The validator permits *missing* input query rows, but not unexpected rows.
  Missing rows are explicitly emitted as missing records and included in
  full-manifest scoring. For branch `clean`, `test` has 1,145 queries;
  `test_seen` has 2,659, so the claimed 2,656 seen records imply 3 missing.
- Fail the run (preserving `.incomplete` output) for unanticipated model/runtime
  errors rather than quietly claiming every sample was processed.

## Outputs

Each run produces `predictions.jsonl`, `validation_summary.json`,
`evaluation_config.json` (this evaluator), `resolved_config.json`
(existing checkpoint helper), `metrics_by_group.json` and subdirectories when scoring is enabled,
and `evaluation_summary.json`.

Each JSONL record includes original WHERE GT/PRED if present, semantic GT/PRED,
query and support metadata, `scanpath_source: predicted_file`,
`state_source: replayed_precomputed_PRED_XYD`, `gold_groups_supplied`,
`gold_groups_requested`, `semantic_prompt_mode`, WHY-group provenance, and
invalid/skipped reasons. Missing input queries contain `WHERE.GT = null` and
`WHERE.PRED = null` rather than invented predictions.

## Scientific limitations

**Off:** WHY is a single *trajectory-level* rationale, whereas the GT semantic
reference may contain multiple groups. Whole-text similarity to the unchanged
GT WHAT/WHY/HOW reference therefore measures a different output structure.
Field-level WHY scores must be treated as exploratory; WHAT scores for unequal
fixation counts also have alignment limitations. No grouping accuracy is claimed.

**On:** The predicted trajectory's hidden states are isolated from GT scanpath
tokens, but GT WHY-group **membership and count** are supplied. Where predicted
and GT fixation lengths match, this is a better controlled comparison with
the existing GT-conditioned semantic protocol, but it is not a GT-free
end-to-end semantic evaluation. Short/long predicted trajectories are explicitly
counted as failed outputs; report the number of such failures alongside scores.

Regardless of the toggle, whole-text BERTScore, CIDEr-R, BLEU and ROUGE do
not alone establish fixation-level grounding or faithfulness. Replay inherits
the original oracle-length-conditioned WHERE generation *protocol* (already
baked into frozen WHERE.PRED), but does not add GT query fixation tokens.
Support demonstrations legitimately contain support scanpaths.

## Testing status

`python -m py_compile` passed for the evaluator and the test module.
14 focused unit tests passed in a local **isolated harness with stubbed SemGaze
imports** (K filtering, predicted-only replay provenance interception,
no-gold prompt invariance, gold-group membership, mismatched-length rejection).
This is **not** a substitute for testing against the actual SemGaze checkout.
Full repository integration tests and actual GPU inference **were not run**
because the checkpoint, image assets, and prediction JSONLs were not available
in this environment.

Inside the repository execute:

```bash
pytest -q tests/test_predicted_semantic_k1.py
```
