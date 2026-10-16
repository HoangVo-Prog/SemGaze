# Controlled GT-prefix intervention (SFT checkpoint, frozen test)

**Source audited:** `HoangVo-Prog/SemGaze`, branch `no-attn`, commit `051a022ba5607fc74303d976ae3d43bc5c740904` (2026-10-13).

**Scope:** Evaluation-only. No SFT retraining, no changes to the model, frozen subject split, support selection, images, tokenization, greedy decoding, or oracle-length WHERE instruction. No WHAT/WHY/HOW generation during this experiment.

## Running the intervention

Use the checked-in `evaluate_gt_prefix.py` and `gt_prefix_core.py` scripts.
The historical patch installer and its test harness were retired.

## Exact test-set run

```bash
cd /home/cvpr2027/hoang/SemGaze
conda activate semgaze-flat  # replace with the actual environment used for this checkpoint

python evaluate_gt_prefix.py \
  --checkpoint runs/2e/checkpoint-6050 \
  --output-dir runs/gt_prefix_sft_test \
  --k-values 1 5 10 --draw 1 \
  --prefixes 1 3 5 --batch-size 1 \
  --dry-run

python evaluate_gt_prefix.py \
  --checkpoint runs/2e/checkpoint-6050 \
  --output-dir runs/gt_prefix_sft_test \
  --k-values 1 5 10 --draw 1 \
  --prefixes 1 3 5 --batch-size 1
```

To smoke-test GPU generation (separate empty output directory):

```bash
python evaluate_gt_prefix.py --checkpoint runs/2e/checkpoint-6050 \
  --output-dir runs/gt_prefix_smoke --k-values 1 \
  --draw 1 --prefixes 1 3 5 --max-queries 8 --batch-size 1
```

**Note:** `--max-queries` deliberately limits coverage and **must be omitted** from the final test run. If you used `--where-support-prompt-style full` for the original checkpoint evaluation, pass it to this experiment as well. Default is checkpoint-resolved style. `--config` is optional for evaluation-only config overrides, as with `evaluate_flat.py`; never change the checkpoint-derived model/data/training properties.

## Conditions

For a GT scanpath of N fixations, the query prompt is the **same** in both conditions and still explicitly asks for exactly N fixations.

- `free`: native `generate_where_batch`, unchanged, greedy autoregressive WHERE.
- `gt_prefix_p`: query assistant response consists of the first p **GT XYD fixations**, ending exactly at the p-th supervised `<END_FIX>` token; the model greedily generates the remaining suffix from that context.

Support turns, query image, query/task text, K, draw ID, processor, adapter/checkpoint, and model weights remain identical. Prefixes are cut from the **native teacher-forced token IDs** to prevent BPE mismatch, EOS injection, GT-suffix leakage, or re-tokenized multimodal prompts. The *only* model-input change is the presence of known query GT fixation tokens prior to continuation.

`gt_prefix_p` is eligible only when `GT length > p`. All comparisons are aligned to the **same** query, K, draw ID, and suffix indices `t > p` (1-indexed). This avoids counting forced GT tokens as model successes.

Generation token budget after inserting p fixations is original oracle-length budget minus the inserted token count, preserving the original total assistant-generation length ceiling (after accounting for prompt length). No EOS is inserted after the forced prefix.

## Artifacts

Under `--output-dir`:

- `protocol.json`: checkpoint, split identity, K, draws, prompts, source commit, flags.
- `interventions.jsonl`: one full-free record plus every eligible GT-prefix condition; contains GT, prediction and generation diagnostics. Each GT-prefix prediction includes forced prefix **for traceability**; do **not** evaluate its first p positions as predicted.
- `paired_queries.csv`: paired suffix-only spatial errors, free-prefix divergence, 3-identical-XY collapse, absorbing collapse, movement and completeness for each `(query_id,K,draw_id,p)`.
- `paired_fixations.csv`: free/prefix errors and XY-repeat indicators for each unforced fixation index t>p; missing predictions remain blank, not imputed as correct.
- `summary.json`: dataset × K × p, paired spatial-error delta (`GT-prefix − free`, **negative is improvement**), complete-generation rates, collapse recovery and cluster-bootstrap 95% CI (resample query IDs).

A 3-identical-XY **suffix collapse** requires the three identical fixation XY positions to lie **entirely inside the generated suffix**. No forced GT step is allowed to contribute to collapse detection. A *free* collapse occurring before the intervention boundary is flagged independently as `free_collapse_before_prefix`; interpret p=5 carefully because injection at p=5 can overwrite an existing early collapse. P=1 and P=3 are informative for earlier intervention.

## Interpretation

- Stronger support for exposure bias: `gt_prefix_p` markedly reduces **suffix-only** late-fixation collapse and/or paired spatial error relative to free, for the **same** queries. If one GT step is enough to rescue the subsequent sequence, this suggests damaging dependence on self-generated context.
- If both conditions collapse: consistent with learned repetition/mode collapse or decoder pathology even with GT prefix, not specifically exposure bias.
- If only p=5 helps after free has already collapsed at t≤5: this is a useful *rescue* demonstration, but not evidence that the initial mistake caused the failure. Report time of collapse before the intervention.
- This experiment does not estimate next-step teacher-forced NLL or autonomous stopping: WHERE remains **oracle-length-conditioned**. A separate one-step GT-prefix test at every t, or an unconditioned-length control, would further isolate mechanisms.

## Limitations of validation here

Only Python syntax checks and CPU unit tests were run when authoring this patch. No SemGaze model, checkpoint weights, split manifest, GPU or image files were available in the authoring container, so **no true test-set intervention predictions are claimed**. Validate the dry run and run on the machine that holds the SFT checkpoint.
