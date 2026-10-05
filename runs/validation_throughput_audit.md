# Validation throughput audit — 2026-10-05

Written before validation behavior changes. Source: post-training same-K implementation.

## 1. Current call graph and serialization
`training.loop` triggers `validation.validate_epoch`: configured K order -> persisted query order -> `frozen_episode` -> `_episode_losses` -> `forward_where` -> projector -> `prepare_flat_inputs` -> native model forward -> `flat_component_nll` -> Python episode sums -> progress. WHERE and semantic are full-model serial execution. The projector is also called separately per episode. Existing episode means are correct. The semantic model creates full logits/native CE; diagnostics compute a second selected CE. WHERE already delegates its singleton to the optimized shared batch path.

## 2. Shared training contracts
| Location / symbol | Input -> output | Training dependency / inference change |
|---|---|---|
| training/batching.py `sample_optimizer_batches` | sampler -> homogeneous `WhereBatch` groups | random K/retry policy training-only; replace scheduling only |
| where/collator.py `NativeBatch`, `WhereBatch`, `collate_where`, `pack_where_batch` | native episodes -> right-padded IDs/masks/labels, sample-major image tiles, metadata | safe; reuse directly; pass validation-lifetime image cache |
| where/forward.py `forward_where_batch` | episodes and optional WhereBatch -> per-episode CE, chronological query states, query features | safe; force no KV cache for validation; optional frozen feature cache |
| model/selected_loss.py `forward_backbone`, `compute_selected_causal_nll` | final hidden [B,L,D], host labels -> one full-vocabulary CE on supervised predictors, NLL[M], losses[B] | safe unchanged, no native outputs.loss required |
| state/extractor.py `extract_query_states` | final states, query-supervised END_FIX mask -> states[B,Nmax,D], mask, positions | already batched; excludes supports and padding |
| bundle.projector (state/projector.py) | concatenated valid states -> projected states split by query N | safe, eval mode required |
| semantic/flat/forward.py `prepare_semantic_batch`, `forward_flat_batch` | queries, states, optional WHERE result -> batched semantic NLL, insertion positions, metadata | safe; expose response offsets for diagnostic decomposition; generalize preparation for generation |
| state/insertion.py `insert_states_batch` | per-sample embeddings/masks/boundaries/states -> right-padded fused inputs | safe; metadata loop calls no model |

Training executor itself performs train(), backward and gradient assertions: do not call it in validation. Reuse its forward components, not its optimizer wrapper.

## 3. Singleton findings
`forward_where` wrapper indexing [0] is a public singleton API, already fixed by shared batching. `collate_native` [0]/unsqueeze and `insert_states` singleton assembly are harmless preprocessing/embedding loops underneath shared packing. `flat_component_nll` B==1 restriction is diagnostic only, but its duplicate CE is wasteful. `validate_epoch` query loop and prediction query loop perform full-model serial execution and need replacement.

## 4. Exact metrics and decomposition
Preserve eval_where/flat/what/why/how/total. Total uses configured lambda_where/lambda_sem. Each component averages its own response tokens within one episode, then episodes equally. WHAT and WHY text lengths determine character sections; crossing BPE token belongs to its first character; prefixes/newlines belong to their line and EOS to HOW. Return response offsets from shared semantic preparation and partition the existing unreduced NLL once. Preserve old native semantic singleton as explicit server reference only.

## 5. Deterministic coverage and fixed support work
Canonical order is configured K, then validation file query order (no explicit subject sort). `frozen_episode` resolves manifest support IDs in persisted order on every query; FlatEpisode validates subject and query exclusion. Schedule bounded stable windows within K, retain K/query stable IDs and canonical output order, check duplicates/counts. Same subject is not required for physical packing. Manifest/config determine authoritative counts, never hard-code 1751/5253.

## 6. Support/query reuse
`collate_native` currently opens/caches decoded PIL images only within a training optimizer group; native preprocessing and support prompt serialization repeat per query. `forward_where_batch` returns projected query image features; semantic preparation already reuses them when native vision tower/projector are frozen and deterministic, verifying pixels and placeholders. Reuse this mechanism. Add a bounded validation-lifetime preprocessing/feature cache, retaining native processor expansion and native get_image_features, scoped to one checkpoint/eval invocation. Do not retain language hidden states or LM KV prefixes. Clear caches before training resumes.

## 7. Same-K padding evidence
Shared WHERE metadata already exposes where_length, fixation_count, image indices and token counts; semantic metadata exposes semantic_length including inserted states. Padding slots are B*max(length) per actual physical batch. Current B=1 validation has padding efficiency 1 by construction, with no evidence of B>1 efficiency. Real tokenizer/model runtime is required for production token distributions; locally only synthetic/fixture checks are authoritative. Report real/padded totals and canonical-versus-bucketed slots in new tooling rather than presume bucketing helps.

## 8. Prediction audit
`predict_epoch` calls `evaluate_where_episode` -> `generate_where` (greedy, KV cache, oracle query fixation count, count-dependent budget) and `evaluate_flat_episode` (GT teacher-forced WHERE states, gold WHY groups, explicit semantic budget, greedy semantic generation). Both serialize full-model calls. Keep these scientific paths; same-K WHERE generation must preserve each count-dependent budget, requiring budget subgroups or per-row stopping. Left-pad generation prompts, reuse shared image packing and semantic insertion. Keep parsers and file order unchanged.

## 9. Patch order
1. Deterministic same-K scheduler and exact accumulator; shared teacher-forced path and NLL partition.
2. Independent validation config, bounded optional length windows, metadata and coarse logging.
3. Native preprocessing/frozen feature reuse with cache lifetime and frozen checks.
4. CPU correctness, isolation and training regressions.
5. Server serial/batched/parity/sweep/full report tooling reusing benchmark_a100 utilities.
6. Same-K autoregressive generation and parity tests; no prefix KV work without server evidence.

## 10. Server-only conclusions
A100 throughput, allocated/reserved memory, OOM limits, useful B1/B5/B10, full wall time, kernel numerical/generation parity, cache benefit and remaining bottlenecks require measured reports. Local Python currently has CPU Torch 2.11; production weights/CUDA are not available to establish these conclusions. No performance claim or production batch-size selection is made by this audit.

## Post-implementation local verification appendix
The persisted validation JSON contains 2452 records before subject filtering. The configured seen-subject filter leaves 1751 queries, hence 5253 episodes at K=1,5,10. The final CPU suite passed 134 tests with no skips. Variable-length native token padding is exposed in benchmark reports and fixture tests; no production-token distribution or A100 timing is inferred from these CPU checks. See `documents/VALIDATION_THROUGHPUT.md` for file-by-file changes and executable server acceptance commands.
