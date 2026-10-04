# SemGaze Agent Guide

Keep this file short. It is a repository map + stable guardrails, not the scientific specification.

## Current target

Implement only:

```text
semantic_mode = "flat_single_output"
```

First milestone:

```text
same-subject support(s)
-> teacher-forced XYD WHERE
-> query <END_FIX> states F
-> shared P_E -> R
-> one flat WHAT/WHY/HOW semantic forward
-> L_WHERE + L_FLAT
-> one backward with gradient assertions
```

Do not implement/refactor toward `multibranch` unless explicitly requested.

## Read order

1. `IMPLEMENTATION_PLAN_FLAT.md` — active execution plan, phases, files, gates.
2. `documents/00_METHOD_IDENTITY.md`
3. `documents/01_FEWSHOT_SPEC.md`
4. `documents/02_WHERE_SPEC.md`
5. `documents/03_STATE_SPEC.md`
6. `documents/flat/04_SEMANTIC_SINGLE_SPEC.md`
7. `documents/05_TRAINING_SPEC.md`
8. `documents/06_EVALUATION_SPEC.md`
9. `DeepGaze-VL/CODEBASE_MAP.md` — verified upstream behavior/reuse map.

Read `documents/multi/04_SEMANTIC_MULTI_SPEC.md` only for the shared normalized semantic annotation schema/validation. Do not implement its execution path.

## Authority

```text
explicit user instruction
> canonical documents/* specs
> IMPLEMENTATION_PLAN_FLAT.md for implementation decomposition
> DeepGaze-VL/CODEBASE_MAP.md for verified upstream facts
> DeepGaze-VL historical source
> assumptions
```

`DeepGaze-VL/` is reference/baseline code. Do not turn that checkout into the SemGaze training package.

If a scientific behavior is not frozen, do not invent it. Record an implementation gate in the plan and surface the ambiguity.

## Newly frozen decisions

These supersede stale "unresolved" wording until Phase 0 synchronizes the canonical docs.

### COCO-Search18 duration

```text
source field      = T[t]
semantics         = fixation dwell duration
source unit       = milliseconds
conversion to ms  = identity
```

WHERE still rounds non-integer ms if encountered and clips serialized duration to `000..999` per `02_WHERE_SPEC.md`.

### WHERE initialization

```text
base              = OpenGVLab/InternVL3_5-8B-HF
initial adapter   = DeepGaze-VL/model/visual_search_adapter
```

Load the released visual-search LoRA as **trainable initialization**. Do not create a fresh random LoRA and do not merge-and-unload it for training.

## Non-negotiable constraints

- Runtime dataset: `data/COCO_Search18/split/all/`.
- Unseen subjects: `7,8,9`; unseen records never update parameters.
- K is `{1,5,10}`; preserve same-subject support membership/order exactly.
- No user token, user embedding, per-user adapter, or test-time user tuning.
- Primary WHERE is XYD with exact `02_WHERE_SPEC.md` serializer.
- `<END_FIX>` is one atomic trainable token after every fixation tuple.
- Support assistant labels are `-100`; only final query WHERE response is supervised.
- Extract final-layer states only at **query** `<END_FIX>` positions.
- Exactly one shared `P_E = Linear(d,d) + LayerNorm(d)`.
- Never detach `F` or `R` before semantic supervision.
- Flat semantic forward uses query image/task + full chronological `R` + gold WHY groups.
- No semantic state token, semantic `<END_FIX>`, GROUP prediction, or JSON output.
- `L_FLAT` is one response-token mean NLL; no branch balancing inside flat mode.
- Default joint objective: `L_total = L_WHERE + L_FLAT`.
- Vision tower/native multimodal projector stay frozen.
- Trainable set: released LM LoRA + shared `P_E` + required `<END_FIX>` vocabulary row(s).

## Upstream reuse rule

Consult `DeepGaze-VL/CODEBASE_MAP.md` before porting behavior.

Reuse/port concepts for coordinate formatting, search prompt inheritance, native chat ordering, multi-image support/query context, and released LoRA metadata.

Do **not** use vLLM/NumPy scoring, merged-model lifecycle, legacy GT-based subject lookup, image-group support reuse, or the permissive legacy parser as the training core. Training requires a retained differentiable HF + PEFT path.

## Implementation surface

Follow `IMPLEMENTATION_PLAN_FLAT.md`:

```text
semgaze/{data,where,state,semantic/flat,model,training,evaluation}/
train_flat.py
evaluate_flat.py
scripts/smoke_flat.py
tests/
```

Avoid duplicate utilities outside these ownership boundaries.

## Config + fixture

- Executable template: `configs/flat_single.yaml`.
- Deterministic normalized fixture: `tests/fixtures/flat_episode.json`.
- Golden outputs: `expected_where.txt`, `expected_flat_target.txt`.
- The fixture is post-adapter normalized data, not a raw COCO storage example.
- Do not change golden outputs merely to make failing code pass.

`null` config values are intentionally unresolved engineering knobs. Do not convert them into hidden scientific defaults. Save the fully resolved config for every run.

## Validation workflow

For each plan phase:

1. implement the smallest dependency-complete slice;
2. run targeted tests;
3. run `python scripts/smoke_flat.py --contract-only`;
4. once HF/PEFT path exists, run `python scripts/smoke_flat.py --with-model`;
5. record command/result in the plan's verification log.

Do not start full training until the model smoke proves finite losses and intended gradients.

## First milestone definition of done

```text
[OK] released visual_search_adapter loaded trainably
[OK] END_FIX atomic
[OK] support WHERE labels masked
[OK] supervised query END_FIX count == N
[OK] extracted query END_FIX state count == N
[OK] P_E output == [N,d_model]
[OK] state insertion masks/lengths aligned
[OK] finite L_WHERE, L_FLAT, L_total
[OK] finite gradient reaches released LoRA
[OK] finite nonzero gradient reaches P_E
[OK] finite gradient reaches END_FIX input row
[OK] finite gradient reaches END_FIX output row when untied
```
