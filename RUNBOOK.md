# SemGaze K=10 WHERE opening-prefix root cause diagnostic

## Source snapshot inspected

- Repository: `HoangVo-Prog/SemGaze`
- Branch: `no-attn`
- HEAD verified through GitHub: `315a43a55d4a5cf1aceb1e3ccffb21619c0fb234`
- Reference: `tools/audit_where_prefix_runtime.py` (preexisting baseline diagnostic)

### Static findings

1. `semgaze/where/conversation.py` constructs one support user message **with `build_where_prompt(...)` for every support**, one corresponding assistant GT message, and a final query user message with the full instruction. K=10 therefore presents 11 copies of the full WHERE instruction.
2. `semgaze/where/prompt.py` defines this full instruction, including a target/task introduction, fixation count, tuple syntax, duration units and an example opening with `[(`.
3. `semgaze/where/collator.py` calls the native processor with `apply_chat_template(..., add_generation_prompt=True)`; `collate_native(..., target=None)` ends at the assistant generation prompt with no generated `[(` explicitly in the context.
4. `semgaze/where/generation.py` uses greedily decoded HF `model.generate` and slices returned sequences after the full prompt length. The prior runtime audit found that raw generated continuations themselves omit `[(` in affected K=10 cases.
5. `semgaze/data/fewshot.py` resolves separately persisted frozen K=1, K=5 and K=10 draws. Official K=1/K=5 support sets must **not** be treated as nested subsets of K=10 without checking identities.
6. `semgaze/evaluation/batching.py::physical_size` uses configured per-K overrides, generic evaluation batch size and execution mode. This diagnostic does **not** use physical grouping (always one episode per `generate`).
7. `semgaze/model/visual_cache.py` owns separate preprocessing and visual-feature caches. This diagnostic passes neither cache for its causal experiments.

## Installation

Copy this package's two Python files into the checkout, preserving directories:

```bash
cp tools/audit_where_k10_root_cause.py /path/to/SemGaze/tools/
cp tests/test_audit_where_k10_root_cause.py /path/to/SemGaze/tests/
cd /path/to/SemGaze
git switch no-attn
git rev-parse HEAD
python -m unittest discover -s tests -p 'test_audit_where_k10_root_cause.py' -v
```

If the branch has moved since the inspected commit, recheck the source integration.

## Commands for staged A100 experiments

Use a real checkpoint and the original historical K=10 JSONL. All outputs must use fresh directories.

```bash
export CHECKPOINT=runs/2e/checkpoint-6050
export ORIGINAL=runs/eval_metrics-1/predictions.jsonl

# Stage 1: 4 historic prefix failures + 2 valid controls, A / B / D
python tools/audit_where_k10_root_cause.py \
  --checkpoint "$CHECKPOINT" \
  --predictions-jsonl "$ORIGINAL" --draw-id 0 \
  --stage pilot --output-dir runs/k10_rootcause/pilot

# Stage 2: same selected query strategy, C / E
python tools/audit_where_k10_root_cause.py \
  --checkpoint "$CHECKPOINT" \
  --predictions-jsonl "$ORIGINAL" --draw-id 0 \
  --stage disambiguation --output-dir runs/k10_rootcause/disambiguation

# Stage 3: 12 historic prefix failures + 4 valid controls
python tools/audit_where_k10_root_cause.py \
  --checkpoint "$CHECKPOINT" \
  --predictions-jsonl "$ORIGINAL" --draw-id 0 \
  --stage confirmation --output-dir runs/k10_rootcause/confirmation
```

Use `--stage all` for all five experiments, or `--experiments A B C D E` together with `--stage confirmation` for the full matrix, *only if GPU budget permits*. Use repeated `--query-id ...` for explicit selection. `--seed` controls the deterministic order permutations. `--continue-on-oom` skips OOM cases and records the failure; do not interpret them as negative results.

### Experimental conditions

- **A:** K10 nested first1/last1/first5/last5/all10, plus the *separate* official K1/K5 persisted support episodes where available.
- **B:** Full query instruction and original assistant GTs unchanged; support user text either concise (`B2`) or task-only (`B3`). These are text-reduction interventions and necessarily change prompt token length.
- **C:** Original/reverse/two seeded permutations of complete support pairs.
- **D:** `[(` included in the assistant-side text **before processing/tokenization/generation**; *new* model-generated tokens are recorded separately. `max_new_tokens` is unchanged to avoid penalizing the prefilled condition.
- **E:** Actual first-step greedy scores; top candidates and conditional sequence log probabilities for native tokenizer IDs of `[(` and a numeric competing opening when available. Extra E scoring forwards are separate from production generation.

### Artifacts written by each run

- `diagnostics.jsonl` — per-variant input hashes, complete ordered support IDs, prompt metadata, exact raw tokens, parser results, fixation tuples.
- `first_token_scores.jsonl` — conditional prefix-token logprobs and first-step top candidates (only with E).
- `prompts/<sha256>.txt` — processor-expanded native rendered prompt; harmless content hash deduplication.
- `environment.json` — model and dataset provenance, training-K, setup and baseline reproduction counts.
- `summary.json`, `summary.csv` — condition-level paired results.
- `REPORT.md` — automatically generated query-level comparison and hypothesis verdicts.

## Test execution status

**Executed locally:** 5 CPU-only unit tests passed: K10 subset identity, deterministic support ordering, message role/content preservation, real processor-side prefill and generated-token slicing with a mocked model. Compilation passed.

**Not executed:** Full InternVL 8B runtime ablations, true first-token probabilities, any metric outcomes at the user checkpoint. The current environment has PyTorch CPU only and no checkpoint/images/predictions file available. The historical 12/16 reproduction rate remains previous evidence, not a result of this diagnostic.

**Interpretation:** Do not claim instruction repetition, support order or K1-only training as a root cause until `REPORT.md` contains actual paired K10 runtime observations.
