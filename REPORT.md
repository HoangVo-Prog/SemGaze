# WHERE output-prefix diagnostic — evidence available so far

## A. Execution environment

- Source inspected: `HoangVo-Prog/SemGaze`, branch `no-attn`, commit `5adafa2ec9c7831e19eb771b11fad83fa203a9e9` (GitHub).
- Workspace: Python 3, `torch 2.10.0+cpu`, `pytest 9.0.2`.
- `nvidia-smi`: unavailable. CUDA-capable GPU: unavailable.
- SemGaze model checkpoint, tokenizer, original `predictions.jsonl` and persisted train/test split: **not provided in this execution environment**.
- `transformers`, `peft`: unavailable in this execution environment.

## B. Reproduction results

| Test | Execution | Query count | Missing-prefix reproduced | Prefix alignment |
|---|---|---:|---:|---|
| A | NOT EXECUTED — model unavailable | — | — | — |
| B | NOT EXECUTED — model unavailable | — | — | — |
| C | NOT EXECUTED — model unavailable | — | — | — |
| D | NOT EXECUTED — model unavailable | — | — | — |
| E | NOT EXECUTED — model unavailable | — | — | — |

Synthetic unit tests: **8 passed**, but are not runtime evidence for InternVL.

## C. Raw token evidence

**NOT EXECUTED.** No real-model raw `generate()` token sequence could be captured. Three real examples cannot be shown without running the checkpoint.

## D. Hypothesis verdicts

| Hypothesis | Verdict | Basis |
|---|---|---|
| H1 Incorrect output slicing | NOT EXECUTED | No actual HF output prefix observed |
| H2 Tokenizer decoding corruption | NOT EXECUTED | No real tokenizer/output token IDs |
| H3 Left-padding/batched generation | NOT EXECUTED | No matched serial/batched model runs |
| H4 Visual-cache / inputs_embeds interaction | NOT EXECUTED | No matched cached/uncached model runs |
| H5 Model-generated malformed prefix | NOT EXECUTED | No raw continuation from real checkpoint |

## E. Root-cause conclusion

**Unknown:** the current source slices `output[row, length:]`, with `length` the padded prompt width, but the current session cannot establish whether the returned token sequence echoes exactly that prefix under the installed InternVL/HF version. Therefore the missing `[(` is neither proven to have been generated nor proven to have been dropped.

## F. Recommended fix

**No production patch justified yet.** Run the provided observer on the same checkpoint and frozen original episodes. Retain raw sequence evidence before changing prompt structure, retraining or introducing `[(` repair. Preserve production metrics and predictions.

## Static source findings

- `semgaze/where/generation.py` groups samples by `where_generation_budget` even within a physical batch, then slices by `inputs['input_ids'].shape[1]` after left-padding.
- `semgaze/where/collator.py` packs right-padded text before generation conversion to left-padding, and preserves support/query ordering.
- `semgaze/model/visual_cache.py` fuses image features into `inputs_embeds` while `generate_where_batch` passes both `inputs_embeds` and `input_ids` to the model.
- `semgaze/evaluation/predictions.py` sorts within evaluation windows when `bucket_by_length` is configured, and uses `physical_size()` from `semgaze/evaluation/batching.py`. The configured batch size need not equal each actual generation call's size.
- `semgaze/where/serialization.py` defines canonical text `[(xx, yy, ddd)<END_FIX>, ...]`, and parsing can recover some tuples from invalid strings without marking canonical format valid.

These are **static facts, not confirmation of a particular failure mechanism**.
