# SemGaze K10 missing opening syntax: implementation / execution report

## A. Source-level prompt findings

At `no-attn` HEAD `315a43a55d4a5cf1aceb1e3ccffb21619c0fb234`, `build_where_conversation` emits one full WHERE instruction for each support user turn (K=10), each followed by an assistant ground-truth scanpath, and one final full query user turn. `collate_native` ends at `add_generation_prompt=True` with no extra assistant response opening. Support images and GT responses are paired in original episode order. Evaluation frozen supports are manifest-based and K-specific.

## B. Runtime environment / reproduction

The supplied prior audit, not this implementation, reported that **12/16** historical missing-prefix cases were reproduced across serial, batch and visual-cache settings. Its raw continuation evidence attributed the missing `[(` to generated token IDs, not production slicing.

The present environment is **PyTorch 2.10.0 CPU-only**, with **no CUDA GPU, training checkpoint, COCO-Search18 images or original evaluation JSONL**. It cannot execute 8B model ablations. The GPU experiments are **NOT EXECUTED**; no query-level empirical causal effects can be claimed.

## C. Controlled ablation table

| Condition | Queries GPU-run | Historical failures corrected | Previously valid damaged | Validity | Mean prompt tokens |
|---|---:|---:|---:|---|---:|
| A: K10-nested K scaling | NOT EXECUTED | — | — | — | — |
| B: concise/minimal support user instructions | NOT EXECUTED | — | — | — | — |
| C: deterministic support order | NOT EXECUTED | — | — | — | — |
| D: real assistant prefill `[(` | NOT EXECUTED | — | — | — | — |
| E: conditional prefix token probabilities | NOT EXECUTED | — | — | — | — |

## D. Query-level transition matrix

Not available without the original evaluation JSONL and checkpoint on the GPU node. `audit_where_k10_root_cause.py` writes paired rows to `diagnostics.jsonl` and report tables after model execution.

## E. Raw token evidence

Historical diagnostic: the continuation itself starts numerically in the affected cases. The new diagnostic records raw generated token IDs for all A/B/C/D runs, and first-step/top-token/conditional chain scores in experiment E; these outputs have not been produced for the 8B checkpoint here.

## F. Hypothesis verdicts

| Hypothesis | Verdict based on NEW controlled runtime evidence |
|---|---|
| H1 context length / K-only training | NOT EXECUTED |
| H2 instruction repetition | NOT EXECUTED |
| H3 support content / order | NOT EXECUTED |
| H4 response opening / assistant prefill | NOT EXECUTED |

## G. Root-cause conclusion

The currently supported conclusion is **the model generated a malformed continuation for tested K10 episodes**, while its *underlying cause* among long context, support content, repeated instruction and learned response-opening behavior remains undetermined. The K1-only training setup is a credible distribution-shift hypothesis but cannot be shown causal from one checkpoint.

## H. Recommended intervention

Do **not** change production inference or retrain yet. Run the provided stage 1 (A/B/D); then stage 2 (C/E) on the same query selection; then stage 3 on all 16. Only rank interventions after paired format validity, accepted fixation counts and tuple changes have been observed.

## QA performed here

- Script compile: PASS
- CLI help / argument parsing: PASS
- Five CPU-only mocked / pure unit tests: PASS
- Live model, original supports, GPU generation and E probabilities: NOT EXECUTED

No production WHERE generation, annotations, official evaluation metric, existing predictions or checkpoint weights were changed.
