# SemGaze: fair standalone WHAT / WHY / HOW baselines

This is an **additive** implementation for the `no-attn` branch. It never edits the SemGaze WHERE predictor, state-insertion pathway, original tokenizer, `FlatEpisode`, or canonical semantic scorers.

## Input / output contract

Both methods read a single **normalized query record** from the repository's *persisted* split and take exactly one image (one native InternVL 448x448 image tile), its original task string, every GT fixation `(X,Y,D)` with 1-based chronological index, the total fixation count N, gold WHY-group **membership only** (`group.members`) and the total WHY-group count M. X and Y use the native `coordinate_bin` function (00..99), durations use `duration_bin` (000..999), preserving original Python rounding/clamping. No GT WHAT/WHY/HOW answers are included in user prompts. Both styles share the *same factual prompt prefix*, character-for-character.

No support record, K, few-shot prompt, subject adaptation, predicted WHERE, `<END_FIX>` token, gaze-state extraction, or state projector is used. The model emits the original flat contract:

```
WHAT 1: ...
...
WHAT N: ...
WHY 1: ...
...
WHY M: ...
HOW: ...
```

The original `build_flat_target()` is the supervised SFT target. Every generated text goes through `parse_flat_output()` unchanged; malformed/missing units remain missing in scoring.

### Full origin prompt template

The **identical input facts prefix** of both styles is:

```
Task:
{task}
Fixation count: {N}
Exploration states (ground-truth fixation coordinates and dwell durations):
Fixation 1: X={x:02d}, Y={y:02d}, D={d:03d}
...
Fixation N: X={x:02d}, Y={y:02d}, D={d:03d}

Gold WHY group count: {M}
Gold WHY groups:
Group 1: fixations {1-based members}
...
Group M: fixations {1-based members}
```

`origin` appends the **unmodified** `INSTRUCTION.format(N=N,M=M)` string from `semgaze/semantic/flat/prompt.py`:

```
Describe the visual exploration represented by these states.

For every fixation, provide one concise WHAT description of the specific visible object or image region being inspected.

For every provided gold WHY group, provide one concise shared task-relevant search or exploration rationale for that group.

Finally, provide one concise HOW description of the overall visual search or exploration strategy across the complete trajectory.

Return only the following flat line format:
WHAT 1: ...
...
WHAT {N}: ...
WHY 1: ...
...
WHY {M}: ...
HOW: ...

Rules:
- Return exactly {N} WHAT lines, then exactly {M} WHY lines, then exactly one HOW line.
- Keep WHAT indices in fixation order 1..{N}.
- Keep WHY indices in gold-group order 1..{M}.
- Use the provided gold WHY groups; do not predict, rewrite, merge, split, or output group membership.
- Keep every WHAT, WHY, and HOW value on one line.
- Do not output JSON, Python objects, lists, bullets, code fences, or extra explanation.
```

The unavoidable difference from the latent-state original is that each `Fixation i:` slot is filled with *textual GT XYD*, not the projected latent embedding. The original instruction is not rewritten.

### Full concise instruction

This is appended after that same factual prefix:

```
Given the image, task, ground-truth fixations and gold WHY groups, describe the visual exploration.
Write one concise WHAT about the visible region at each fixation, one task-relevant WHY for each provided gold group, and one HOW for the overall trajectory.

Return only this flat format:
WHAT 1: ...
...
WHAT {N}: ...
WHY 1: ...
...
WHY {M}: ...
HOW: ...

Exactly {N} WHAT, then {M} WHY, then one HOW. Follow chronological fixation order and listed gold-group order. Do not predict, change, merge, split, or print group membership. Keep each value on one line. No JSON, bullets, lists, fences, or extra text.
```

## Methods

* `semantic_zero_shot`: frozen, unmodified `OpenGVLab/InternVL3_5-8B-HF`, native HF processor/conversation and native 448x448 preprocessing, greedy generation. The base tokenizer is never extended.
* `semantic_sft`: same backbone; LoRA rank 8, alpha 16, dropout 0.05 on matching language-model projection/MLP modules only. Vision, native multimodal projector, and original LM weights are frozen; there are no trainable vocabulary rows. Native `collate_native()` creates labels supervising **assistant response tokens plus EOS only**, not prompt or image tokens. AdamW, cosine warmup, gradient accumulation, checkpointing. Train on *seen-subject train queries* only; the original SemGaze training sampler is not used. Each of `origin` and `concise` needs a separately trained checkpoint for a matched-prompt result.

The LoRA checkpoint folder holds PEFT `adapter_model.safetensors` / `adapter_config.json`, `semantic_sft_checkpoint.json`, and a `training_state.pt` snapshot. The base model/processor are retrieved from the original HF model ID at loading. `training_state.pt` is retained for inspection, not yet for exact interruption/resume.

## Commands

All commands run from the root of the SemGaze checkout with dependencies installed and the **persisted train/test splits and source images present**. By default the data path/settings come from `configs/flat_single.yaml` (override with `--semgaze-config`). Override `--images-root /absolute/image/path` and `--split-root /absolute/split/path` as needed. Output directories in the examples must be distinct.

```bash
# Frozen / origin
python evaluate_semantic_baseline.py --config configs/semantic_zero_shot.yaml \
  --method semantic_zero_shot --prompt-style origin \
  --output-dir runs/semzero_origin --batch-size 1 --max-new-tokens 512

# Frozen / concise
python evaluate_semantic_baseline.py --config configs/semantic_zero_shot.yaml \
  --method semantic_zero_shot --prompt-style concise \
  --output-dir runs/semzero_concise --batch-size 1 --max-new-tokens 512

# Independent SFT / origin
python train_semantic_sft.py --config configs/semantic_sft.yaml \
  --prompt-style origin --output-dir runs/semsft_origin \
  --epochs 2 --batch-size 1 --effective-batch-size 8 --learning-rate 1e-4

# Independent SFT / concise
python train_semantic_sft.py --config configs/semantic_sft.yaml \
  --prompt-style concise --output-dir runs/semsft_concise \
  --epochs 2 --batch-size 1 --effective-batch-size 8 --learning-rate 1e-4

# SFT checkpoint / origin
python evaluate_semantic_baseline.py --config configs/semantic_sft.yaml \
  --method semantic_sft --prompt-style origin \
  --checkpoint runs/semsft_origin/final --output-dir runs/semeval_origin \
  --max-new-tokens 512

# SFT checkpoint / concise
python evaluate_semantic_baseline.py --config configs/semantic_sft.yaml \
  --method semantic_sft --prompt-style concise \
  --checkpoint runs/semsft_concise/final --output-dir runs/semeval_concise \
  --max-new-tokens 512

# Same held-out queries / refs / scorers, existing SemGaze K=1 draw=0
python compare_semantic_baselines.py \
  --semgaze-predictions runs/semgaze_eval/predictions.jsonl \
  --baseline runs/semzero_origin runs/semzero_concise \
             runs/semeval_origin runs/semeval_concise \
  --reference-k 1 --reference-draw 0 --max-new-tokens 512 \
  --semgaze-metrics-root runs/semgaze_eval \
  --output-dir runs/semantic_comparison

# Incremental resume of an interrupted baseline evaluation (only identical settings)
python evaluate_semantic_baseline.py --config configs/semantic_zero_shot.yaml \
  --prompt-style origin --output-dir runs/semzero_origin --resume
```

## Metrics and fairness

Standalone evaluation outputs `predictions.jsonl`, `metrics.json` (per dataset), `metrics_by_subset.json` (TP/TA/VQA), `validity_summary.json`, `resolved_config.json`, `run_metadata.json` (plus `evaluation_provenance.json`). The comparison writes `comparison_metrics.json`, `comparison_table.csv`, `comparison_protocol.json`. Both baselines use the *same* metric calls as the original: `score_prediction_semantics` -> canonical corpus BLEU-4, CIDEr-R, ROUGE-L, Java METEOR 1.5 and optional BERTScore-F1. Missing units count as missing; BLEU and CIDEr are evaluated on **whole subsets**, not averaged per-query proxies. All artifacts are scored independently for TP, TA, VQA and for COCO/AiR datasets (no blended headline scalar).

Existing SemGaze predictions are read *without regenerating them* and filtered to one complete K/draw semantic traversal. The query ID, subject, task, stimulus, raw GT XYD, image dimensions, and GT target are checked against the same normalized held-out source. Duplicate/missing/extra queries fail. In comparison mode each candidate is re-scored on the identical query population. When `--semgaze-metrics-root` points to the original evaluator directory, the tool also checks common dataset-level metrics against that evaluator at numeric tolerance `--parity-atol`; otherwise it explicitly records parity as **not checked**. Generation settings must match `--max-new-tokens` (or supply `--allow-generation-budget-mismatch` and **document the deviation**, not strict generation parity).

**Limit of fairness:** SemGaze consumes **projected teacher-forced GT WHERE hidden states**, which can include support conditioning. These baselines consume **explicit textual GT XYD**, with **no support**. Even with matched annotation-level oracle WHERE, gold WHY membership, decoded budget, task/image, query identity, format and scoring implementations, the internal representations and available support context are *not equivalent*. Do not present this as strict input-modality parity.

## Dependencies and memory

Use the repository's installed native InternVL transformers/PEFT versions, PyTorch with CUDA, Pillow, YAML, and the pinned metric dependencies. CIDEr-R requires its vendored implementation and Java; METEOR needs `third_party/llada_meteor/meteor-1.5.jar` and Java, with its required paraphrase data; BERTScore needs `bert-score==0.3.13`, `roberta-large` weights, and compatible transformers. Set metric options to `false` **in the config** if dependencies are unavailable, and record these exclusions; never substitute NLTK METEOR or sentence BLEU. Enable BERTScore explicitly if desired.

For InternVL-3.5-8B, CPU is suitable only for contract tests, not practical generation. Use GPU VRAM sufficient for 8B BF16 and multimodal prefill (a single A100-40GB may require batch size 1, gradient checkpointing, and shorter contexts; RTX PRO 6000 96GB is more comfortable). `max_new_tokens` controls response length; if output is truncated it is still counted as a parse error. Numeric generation throughput/VRAM has **not** been benchmarked here.

## Validation

```
python -m compileall -q semgaze/semantic/baselines train_semantic_sft.py evaluate_semantic_baseline.py compare_semantic_baselines.py
pytest -q tests/test_semantic_baselines.py
python train_semantic_sft.py --help
python evaluate_semantic_baseline.py --help
python compare_semantic_baselines.py --help
```

GPU smoke should run a small test fixture or a short evaluation on the target machine, then confirm a real generated `predictions.jsonl` and corpus artifacts. Never claim zero-shot or SFT numeric results without actually executing the 8B model. SFT interrupted-run *optimizer resume* and full GPU parity with existing evaluator need additional machine-side verification.
