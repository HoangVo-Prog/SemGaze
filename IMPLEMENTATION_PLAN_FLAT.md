# SemGaze Flat Single-Output Baseline — Implementation Plan

## Status

Configuration update (2026-10-05): explicit user direction makes experiment settings
configurable, including fresh or continued LoRA initialization. Earlier fixed experiment
values in this historical plan are reference recommendations, not Python-enforced
requirements. Scientific graph/data integrity remains intact. See Section 28 and
`documents/CONFIGURATION_AUDIT.md` for field-level behavior and implementation limits.

Active execution plan for the first implementation target:

```text
semantic_mode = "flat_single_output"
```

Implementation status (2026-10-04): the flat package, training/checkpoint entrypoints,
and separate evaluation prediction paths are implemented. **The released 8B
vertical-slice gate is still blocked**, not complete. Pure contracts, the actual
base processor, and a small real HF InternVL + PEFT model have runtime coverage.
See Sections 24–25 for exact evidence and remaining gates. No full training ran.

This is an implementation plan, not a replacement for the canonical scientific specs in `documents/`.

Maintain this file as a living plan: mark completed gates, record exact verification commands, and document deviations. If implementation reveals a scientific ambiguity, do not silently resolve it here; surface it for a canonical spec decision.

---

## 0. Goal and first vertical slice

The first deliverable is **not** full training. It is one end-to-end differentiable episode:

```text
ordered same-subject support(s)
        +
query image/task + GT XYD scanpath + normalized semantics
        |
        v
native InternVL multi-turn causal WHERE context
        |
        v
teacher-forced WHERE forward
        |----------------------> L_WHERE
        |
        +-> query <END_FIX> final-layer states F=[f1..fN]
                    |
                    v
              shared P_E
                    |
                    v
              R=[r1..rN]
                    |
                    v
query image/task + full R + gold WHY groups
                    |
                    v
one flat semantic InternVL forward
                    |
                    v
                  L_FLAT
                    |
                    v
          L_total=L_WHERE+L_FLAT
                    |
                    v
              one backward
```

The vertical slice is complete only when the required gradient assertions pass.

---

## 1. Source-of-truth map

Read before implementation:

| Need | Canonical owner |
|---|---|
| Method identity / no user embedding | `documents/00_METHOD_IDENTITY.md` |
| Split, seen/unseen subjects, K sampling, support membership/order | `documents/01_FEWSHOT_SPEC.md` |
| XYD, coordinate conversion, duration serializer, prompt, `<END_FIX>`, WHERE labels | `documents/02_WHERE_SPEC.md` |
| `<END_FIX>` readout, `P_E`, continuous state insertion | `documents/03_STATE_SPEC.md` |
| Flat prompt, group bookkeeping, target, parser, `L_FLAT` | `documents/flat/04_SEMANTIC_SINGLE_SPEC.md` |
| Shared initialization, PEFT/freeze/trainable set, optimizer/checkpoints | `documents/05_TRAINING_SPEC.md` |
| Decoding and evaluation | `documents/06_EVALUATION_SPEC.md` |
| Shared normalized semantic annotation schema only | `documents/multi/04_SEMANTIC_MULTI_SPEC.md` §12 |
| Verified DeepGaze source behavior and symbol map | `DeepGaze-VL/CODEBASE_MAP.md` |

Do not use `DeepGaze-VL/` as the training implementation. Its release path is vLLM inference/evaluation and has no executable trainer/autograd state extraction.

---

## 2. Frozen implementation decisions

### 2.1 COCO-Search18 duration declaration

Human-frozen for this implementation:

```text
duration_source_field = T
duration_semantics    = dwell_duration
source_unit           = ms
conversion_to_ms      = identity
```

Therefore dataset adaptation outputs one duration in milliseconds per fixation directly from `T[t]`.

WHERE serialization still obeys `02_WHERE_SPEC.md`:

```text
round non-integer ms to nearest integer if encountered
clip to [0,999]
format as exactly 3 digits
```

### 2.2 Initialization checkpoint

Human-frozen for this implementation:

```text
base = OpenGVLab/InternVL3_5-8B-HF
initial LoRA = DeepGaze-VL/model/visual_search_adapter
```

Rationale: the SemGaze WHERE path extends released DeepGaze visual-search scanpath prediction.

Required training interpretation:

1. load the base HF InternVL model/processor/tokenizer;
2. add/verify atomic `<END_FIX>` and resize embeddings;
3. load the released visual-search PEFT adapter **as trainable initialization**;
4. keep its existing rank/alpha/dropout/targets;
5. add trainable `<END_FIX>` row mechanism;
6. add trainable shared `P_E`;
7. freeze the rest according to `05_TRAINING_SPEC.md`.

Do not create a fresh random rank-8 adapter. Do not merge-and-unload the released adapter for training.

### 2.3 Canonical-doc cleanup gate

Before canonical training, patch stale unresolved wording in:

- `documents/02_WHERE_SPEC.md`: replace COCO-Search18 duration unresolved status with the declaration above.
- `documents/00_METHOD_IDENTITY.md`: update the sentence that says the COCO duration mapping is not frozen.
- `documents/05_TRAINING_SPEC.md`: record released `visual_search_adapter` as the primary flat/WHERE initialization instead of leaving initialization checkpoint implicit.

This is documentation synchronization, not a method redesign.

---

## 3. Proposed implementation layout

Create only what the flat vertical slice needs:

```text
semgaze/
├── data/
│   ├── __init__.py
│   ├── schema.py
│   ├── cocosearch18.py
│   ├── semantic.py
│   └── fewshot.py
├── where/
│   ├── __init__.py
│   ├── coordinates.py
│   ├── duration.py
│   ├── serialization.py
│   ├── prompt.py
│   ├── conversation.py
│   ├── collator.py
│   ├── forward.py
│   └── generation.py
├── state/
│   ├── __init__.py
│   ├── extractor.py
│   ├── projector.py
│   └── insertion.py
├── semantic/
│   ├── __init__.py
│   └── flat/
│       ├── __init__.py
│       ├── prompt.py
│       ├── target.py
│       ├── parser.py
│       └── forward.py
├── model/
│   ├── __init__.py
│   ├── build.py
│   ├── trainable_tokens.py
│   └── checkpoint.py
├── training/
│   ├── __init__.py
│   └── flat_step.py
└── evaluation/
    ├── __init__.py
    ├── where.py
    └── flat.py

train_flat.py
evaluate_flat.py
```

Do not create `semantic/multi/` during this implementation unless a later task explicitly requests the primary branch.

---

## 4. Stable runtime contracts

### 4.1 Normalized record

The runtime data adapter should expose a typed normalized record independent of the raw persisted JSON's exact nested storage:

```python
@dataclass(frozen=True)
class NormalizedRecord:
    record_id: str
    stimulus_id: str
    subject: int
    image_path: str
    image_width: int
    image_height: int
    task: str
    condition: Literal["present", "absent"]
    x_px: tuple[float, ...]
    y_px: tuple[float, ...]
    duration_ms: tuple[float, ...]
    semantic: NormalizedSemantic
```

Required invariants:

```text
N = len(x_px) = len(y_px) = len(duration_ms) > 0
len(semantic.what) == N
semantic.why_groups partition exactly 1..N
semantic.how non-empty
```

The fixture in `tests/fixtures/flat_episode.json` is already in this normalized implementation shape. It is not a raw COCO storage example.

### 4.2 Normalized semantic annotation

```python
@dataclass(frozen=True)
class WhyGroup:
    members: tuple[int, ...]  # 1-based
    why: str

@dataclass(frozen=True)
class NormalizedSemantic:
    what: tuple[str, ...]
    why_groups: tuple[WhyGroup, ...]
    how: str
```

Validation must hard-fail rather than repair.

### 4.3 Episode

```python
@dataclass(frozen=True)
class FlatEpisode:
    supports: tuple[NormalizedRecord, ...]
    query: NormalizedRecord
```

Episode invariants come from `01_FEWSHOT_SPEC.md`:

```text
same subject
K in {1,5,10}
distinct support stimulus images
query image excluded from supports
support order final and preserved
support semantics never exposed to model input
```

---

## 5. Phase 0 — environment and upstream preflight

### Purpose

Prove the intended HF/PEFT path is viable before building SemGaze around an unverified model interface.

### Inputs

```text
DeepGaze-VL/model/visual_search_adapter/
DeepGaze-VL/configs/internvl3_5_8b_visual_search.yaml
DeepGaze-VL/CODEBASE_MAP.md
OpenGVLab/InternVL3_5-8B-HF
configs/flat_single.yaml
```

### Checks

1. Record installed versions of `torch`, `transformers`, `peft`, `accelerate`, `safetensors`.
2. Verify adapter config reports:
   - base model `OpenGVLab/InternVL3_5-8B-HF`;
   - rank 8;
   - alpha 16;
   - dropout 0.05;
   - LM target modules `q/k/v/o/gate/up/down`.
3. Hash/record the adapter config and safetensors file for run provenance.
4. Load base HF processor/tokenizer/model without vLLM.
5. Compare base tokenizer's existing added-token IDs/chat-template essentials with adapter-local tokenizer metadata; hard-fail on an incompatible mapping rather than assuming equivalence.
6. Add `<END_FIX>` if absent and assert isolation tokenization length is exactly one.
7. Resize embeddings before loading the released trainable adapter.
8. Load `visual_search_adapter` with trainability enabled; assert non-zero trainable LoRA parameters.
9. Verify the concrete model forward can return final hidden states with `output_hidden_states=True`, `use_cache=False`.
10. Inspect whether embeddings are tied and record the result.
11. Verify the native InternVL processor's multimodal expansion for one image and for two images.
12. Verify the one-tile policy needed by `02_WHERE_SPEC.md`; do not assume DeepGaze release metadata establishes it.

### Gate

Do not proceed to model-dependent phases until all checks above are explicitly observed.

### Expected implementation note

The release's merge-and-vLLM path is historical inference machinery. SemGaze training needs a retained differentiable HF + PEFT model.

---

## 6. Phase 1 — dataset adapter and semantic validation

### Files

```text
semgaze/data/schema.py
semgaze/data/cocosearch18.py
semgaze/data/semantic.py
```

### Responsibilities

`schema.py`
- define normalized immutable record/semantic types;
- expose `normalized_episode_from_dict(payload) -> FlatEpisode` for fixture/contract tests and other already-normalized mappings;
- no dataset-specific parsing logic.

`cocosearch18.py`
- read only persisted `data/COCO_Search18/split/all/{train,validation,test}.json` at runtime;
- map raw fields into `NormalizedRecord`;
- set `duration_ms = T` directly under the frozen declaration;
- resolve image paths under `data/COCO_Search18/images/`;
- obtain original annotation image width/height deterministically;
- never recreate split membership/correctness filtering.

`semantic.py`
- convert the dataset's semantic payload to normalized `what/why_groups/how`;
- validate exact partition and non-empty text invariants;
- raise a dedicated validation error with `sample_id`, invariant, offending value.

### Tests

- fixture loads successfully;
- mismatched X/Y/T lengths fail;
- invalid WHY partition fails;
- empty WHAT/WHY/HOW fails;
- unseen subject record can load but training-query eligibility rejects it.

### Gate

```text
fixture -> NormalizedRecord(s) -> validated episode-ready objects
```

---

## 7. Phase 2 — few-shot episode construction

### File

```text
semgaze/data/fewshot.py
```

### Training behavior

Implement only the probability law already frozen by `01_FEWSHOT_SPEC.md`:

```text
K in {1,5,10}, each with probability 1/3
seen subject query only
supports from same subject
support stimuli distinct
query stimulus excluded
support order sampled as specified
```

Do not port legacy DeepGaze's query-GT subject resolver or image-group support reuse.

### Evaluation behavior

Use persisted/frozen support manifests from `01_FEWSHOT_SPEC.md`; do not resample inside the model/evaluator.

### Gate assertions

```text
len(supports) == K
all support.subject == query.subject
all support stimulus_id unique
query.stimulus_id not in support stimulus IDs
input support order preserved after handoff
```

---

## 8. Phase 3 — pure WHERE codec

### Files

```text
semgaze/where/coordinates.py
semgaze/where/duration.py
semgaze/where/serialization.py
```

### `coordinates.py`

Implement exactly the two-stage Python-compatible rounding from `02_WHERE_SPEC.md`:

```text
r = round(100 * value / original_size, 1)
b = clip(int(round(r)), 0, 99)
```

Do not replace with floor or one-stage rounding.

### `duration.py`

Input contract for COCO-Search18:

```text
T[t] already means dwell duration in ms
```

Canonicalization:

```text
nearest integer ms if non-integer
clip to 0..999 for serialized WHERE value
```

Keep raw canonical ms separate from serialized clipped duration if later evaluation needs unclipped source values.

### `serialization.py`

Primary XYD fixation:

```text
({x:02d}, {y:02d}, {d:03d})<END_FIX>
```

Full path:

```text
"[" + ", ".join(fixations) + "]"
```

Examples must exactly match `tests/fixtures/expected_where.txt`.

### Required API

The first smoke skeleton expects:

```python
serialize_xyd_record(record) -> str
```

The implementation may internally expose lower-level helpers.

### Unit tests

- exact fixture support serialization;
- exact fixture query serialization;
- x/y border clipping;
- duration 0, 999, >999;
- Python banker-rounding edge cases;
- `<END_FIX>` count equals fixation count.

---

## 9. Phase 4 — WHERE prompt and causal conversation

### Files

```text
semgaze/where/prompt.py
semgaze/where/conversation.py
```

### `prompt.py`

Port the canonical DeepGaze-style search prompt wording as frozen by `02_WHERE_SPEC.md`, including:

```text
requested N
x/y prompt wording inherited from DeepGaze
xyd duration instruction
list-style wording
<END_FIX> requirement
```

No SemGaze personalization system prompt.

### `conversation.py`

Build structured native chat messages:

```text
support user(image + Prompt)
support assistant(serialized observed scanpath)
...
query user(image + Prompt)
```

For teacher forcing, append query assistant GT scanpath through the training collator/template path.

Do not stitch images together.

### Gate

Rendered turn order and image order must be one-to-one and deterministic.

---

## 10. Phase 5 — HF model + released adapter + `<END_FIX>` trainability

### Files

```text
semgaze/model/build.py
semgaze/model/trainable_tokens.py
```

### Initialization sequence

Normative order:

```text
1. load base processor/tokenizer/HF InternVL
2. verify tokenizer compatibility with released adapter artifacts
3. add atomic <END_FIX> if absent
4. resize token embeddings
5. record END_FIX_TOKEN_ID
6. determine tied/untied input/output vocabulary weights
7. load released visual_search_adapter as trainable PEFT initialization
8. configure row-selective END_FIX trainability
9. instantiate shared P_E
10. assert frozen/trainable parameter sets
```

### Critical rules

- No `merge_and_unload()` in the training path.
- No fresh random LoRA replacement.
- No LoRA on vision tower/native multimodal projector.
- All old vocabulary rows remain frozen.
- If PEFT row-selective token training uses a version-specific API, isolate that implementation in `trainable_tokens.py` and test save/load.

### Gate

Print/assert:

```text
released adapter loaded
LoRA trainable count > 0
END_FIX atomic
END_FIX input row trainable
END_FIX output row trainable if untied
vision frozen
native multimodal projector frozen
P_E trainable
```

---

## 11. Phase 6 — WHERE collator and query-only supervision

### File

```text
semgaze/where/collator.py
```

### Responsibilities

1. preserve full support/query causal history;
2. use native processor/chat template;
3. preserve K+1 image ordering;
4. enforce one-tile-per-image observable policy;
5. enforce effective total sequence length <= 8192;
6. build token labels with only final query assistant response supervised;
7. expose exact token spans/indices needed for query `<END_FIX>` extraction.

### Label contract

```text
support user             -> -100
support assistant        -> -100
query user               -> -100
image/context/control    -> -100
query assistant XYD      -> target token IDs
query assistant END_FIX  -> END_FIX_TOKEN_ID
native assistant EOS     -> supervised if part of target/template contract
```

### Context overflow

Training: reject/resample the complete episode. Never reduce K or truncate a support.

### Required assertions

```text
number of supervised query END_FIX labels == query N
all support END_FIX labels == -100
query assistant target decodes to canonical serialized query scanpath
```

---

## 12. Phase 7 — teacher-forced WHERE forward and state extraction

### Files

```text
semgaze/where/forward.py
semgaze/state/extractor.py
semgaze/state/projector.py
```

### Forward

Call the retained model with:

```text
labels = query-only labels
output_hidden_states = True
use_cache = False
```

Return at least:

```python
WhereOutput(
    loss_where,
    last_hidden_state,
    query_end_fix_positions,
)
```

### Extraction

For final-query assistant positions only:

```text
F = final layer hidden states at input positions whose token is END_FIX
```

Use the hidden state **at** the `<END_FIX>` position (post-boundary-token causal representation), not the preceding `)` position and not the logit position predicting `<END_FIX>`.

### Projector

Exactly:

```python
P_E = nn.Sequential(
    nn.Linear(d_model, d_model),
    nn.LayerNorm(d_model),
)
```

No branch-specific projector.

### Gate

```text
F.shape == [N, d_model]
R.shape == [N, d_model]
F.requires_grad
R.requires_grad
```

No detach/CPU/NumPy/text conversion.

---

## 13. Phase 8 — flat target and parser

### Files

```text
semgaze/semantic/flat/target.py
semgaze/semantic/flat/parser.py
```

### `target.py`

Implement:

```python
flatten_text(s) = " ".join(s.strip().split())
```

Target:

```text
WHAT 1: ...
...
WHAT N: ...
WHY 1: ...
...
WHY M: ...
HOW: ...
```

The smoke skeleton expects:

```python
build_flat_target(semantic) -> str
```

The fixture output must exactly match `tests/fixtures/expected_flat_target.txt`.

### `parser.py`

Implement the deterministic regex parser from `flat/04_SEMANTIC_SINGLE_SPEC.md`.

No LLM repair.

### Tests

- exact golden target;
- flatten multiline whitespace;
- duplicate WHAT/WHY invalid;
- missing line invalid;
- wrong ordering invalid;
- extra prose invalid;
- recognized diagnostic fields may still be returned when invalid.

---

## 14. Phase 9 — semantic prompt and direct state insertion

### Files

```text
semgaze/semantic/flat/prompt.py
semgaze/state/insertion.py
semgaze/semantic/flat/forward.py
```

### Logical prompt

Follow the exact flat spec:

```text
<query image>

Task:
{Q_q}

Exploration states:
Fixation 1:
<r_1 direct vector>
...
Fixation N:
<r_N direct vector>

Gold WHY groups:
Group 1: fixations ...
...

<literal flat semantic instruction>
```

Raw few-shot support turns are not replayed in this semantic forward.

### State insertion primitive

Each `r_t` becomes exactly one LM sequence position:

```text
attention_mask = 1
label = -100
```

No token ID exists for that position.

The implementation must preserve:

```text
inputs_embeds length == attention_mask length == labels length
```

and any model-required positional/multimodal alignment.

### Critical HF interface gate

Before finalizing insertion, verify the concrete `InternVLForConditionalGeneration` path can preserve image conditioning while using a modified LM embedding stream. Do not assume `pixel_values + inputs_embeds` semantics from another model family.

If the top-level InternVL forward cannot accept the required combination directly, implement the smallest verified InternVL-native path that:

1. obtains the normal image-conditioned LM input embedding stream using the model's own vision/projector logic;
2. inserts `R` into that stream without detaching image features or states;
3. calls the shared language model with the resulting embeddings;
4. preserves response-only labels and gradient flow.

Document the concrete verified route in this plan once observed.

### Gate

First test with fake trainable `R`, then real WHERE-produced `R`.

---

## 15. Phase 10 — flat semantic forward and loss

### File

```text
semgaze/semantic/flat/forward.py
```

Exactly one semantic example per query.

Labels:

```text
user text               -> -100
image/context           -> -100
inserted R positions    -> -100
assistant flat target   -> supervised
```

Loss:

```text
L_FLAT = mean NLL over supervised assistant response tokens only
```

No branch fan-out, no `L_WHAT/L_WHY/L_HOW`, no `L_STRUCT`, no scale balancing.

---

## 16. Phase 11 — one joint training step

### File

```text
semgaze/training/flat_step.py
```

### Required API

The smoke skeleton expects a model-dependent entry point equivalent to:

```python
run_flat_training_step(model_bundle, episode, config) -> diagnostics
```

Exact internal dataclass names may differ, but diagnostics must expose the smoke assertions.

### Order

```text
1. collate one WHERE episode
2. teacher-force WHERE
3. compute L_WHERE
4. extract F
5. project F -> R
6. construct one flat semantic conversation
7. compute L_FLAT
8. L_total = lambda_where * L_WHERE + lambda_sem * L_FLAT
9. zero existing grads
10. backward once on L_total
11. inspect gradients
12. optional clip
13. optimizer step only outside diagnostic no-step mode
```

For `scripts/smoke_flat.py`, default `--with-model` should run **no optimizer step** unless `--step` is explicitly passed. The goal is graph validation, not weight mutation.

### Gradient assertions

Require finite gradients reaching:

- at least one released visual-search LoRA parameter;
- `P_E`, with at least one non-zero finite element;
- `<END_FIX>` input row;
- `<END_FIX>` output row if untied and query `<END_FIX>` is supervised.

Do not require every individual element to be non-zero.

---

## 17. Phase 12 — checkpoint/resume

### File

```text
semgaze/model/checkpoint.py
```

Checkpoint must preserve at least:

```text
processor/tokenizer containing END_FIX
END_FIX_TOKEN_ID
released+continued PEFT adapter state
trainable END_FIX row state
P_E state_dict
resolved config
split/manifest identity or hash
protocol version
optimizer/scheduler for resumable training
```

Round-trip test:

```text
save -> fresh load -> same END_FIX ID -> same trainable set -> same fixture forward shapes
```

Never save only ordinary LoRA matrices and call the checkpoint complete.

---

## 18. Phase 13 — free-running WHERE and primary semantic evaluation

This phase starts only after the training vertical slice is stable.

Keep two evaluation paths conceptually separate:

### A. Free-running WHERE

```text
frozen supports + query image/task
-> autoregressive XYD generation
-> strict SemGaze parser
-> scanpath metrics
```

Do not reuse permissive legacy regex behavior as the canonical parser.

### B. Primary flat semantic evaluation

```text
frozen supports + query + GT query XYD
-> teacher-force WHERE
-> GT-conditioned R
-> gold WHY groups
-> one flat semantic generation
-> deterministic flat parser
-> semantic metrics
```

Do not transfer GT groups to a free-running predicted trajectory unless a new method decision explicitly defines alignment.

---

## 19. Config contract

`configs/flat_single.yaml` is the only initial flat runtime config template.

Sections:

```text
experiment
data
model
where
state
semantic
training
checkpoint
smoke
```

Frozen scientific decisions are explicit values.

Hardware/runtime knobs that are not method identity are marked as engineering defaults or `null`.

Every run must write the fully resolved config to its output/checkpoint directory.

---

## 20. Fixture contract

Files:

```text
tests/fixtures/flat_episode.json
tests/fixtures/expected_where.txt
tests/fixtures/expected_flat_target.txt
tests/fixtures/images/support_1.jpg
tests/fixtures/images/query.jpg
```

The JSON fixture is normalized post-adapter data.

It intentionally uses:

```text
K = 1
query N = 4
M = 2
non-contiguous gold groups {1,3} and {2,4}
a duration >999 ms to verify serialization clipping
a multiline semantic string to verify flatten_text
```

Golden WHERE file format:

```text
SUPPORT=<exact support scanpath>
QUERY=<exact query scanpath>
```

Golden flat target contains only the exact assistant response text.

---

## 21. Smoke harness contract

`scripts/smoke_flat.py` has two layers.

### `--contract-only`

No GPU/model download required. It should:

1. load the fixture;
2. import implementation symbols when they exist;
3. validate exact support/query serialization against `expected_where.txt`;
4. validate exact flat target against `expected_flat_target.txt`;
5. validate flat parser round-trip;
6. validate fixture structural invariants.

Before modules exist, the skeleton exits with a clear list of missing symbols rather than pretending success.

### `--with-model`

Requires prepared model environment/GPU. It should:

1. load base HF InternVL;
2. add atomic `<END_FIX>`;
3. load released visual-search adapter trainably;
4. build K=1 fixture WHERE context;
5. teacher-force query XYD;
6. extract exactly 4 query states;
7. project to `R`;
8. execute one flat semantic forward;
9. compute finite `L_WHERE`, `L_FLAT`, `L_total`;
10. backward once;
11. print the gradient assertions.

The smoke does not claim scientific quality on synthetic images.

---

## 22. Do-not-do list

Do not:

- edit `DeepGaze-VL/` to turn it into SemGaze;
- train through vLLM;
- merge the released adapter before training;
- create a fresh LoRA instead of continuing released visual-search LoRA;
- use raw/original benchmark splits at runtime;
- infer duration units from `T` at runtime;
- floor coordinates;
- create learned `<X_*>`, `<Y_*>`, or `<D_*>` tokens;
- use `)` as the fixation boundary;
- mask support turns out of the causal input;
- extract support `<END_FIX>` states as query states;
- detach `F` or `R`;
- serialize `R` values as text;
- add a semantic state token;
- predict WHY groups in the flat baseline;
- replay supports in the flat semantic forward;
- split flat output into N+M+1 semantic forwards;
- use JSON as flat semantic model output;
- add an LLM repair parser;
- silently truncate supports/states/semantic targets on overflow;
- start full training before gradient smoke passes.

---

## 23. Progress checklist

### Phase 0 — preflight
- [x] canonical stale duration/init wording patched
- [x] environment versions recorded
- [x] adapter metadata verified
- [x] base/adapter vocabulary IDs and native role/image template checked against materialized vocab/added-token/template artifacts
- [x] atomic `<END_FIX>` verified on actual base tokenizer
- [ ] resized-base + trainable released adapter load verified
- [x] HF hidden-state forward verified on small actual HF architecture (released 8B pending)
- [x] tied/untied behavior tested and recorded for small models (released 8B pending)
- [x] one-image/two-image actual base processor expansion measured
- [x] one-tile policy verified with actual base processor

### Phase 1 — data
- [x] normalized record implemented
- [x] COCO adapter implemented; real-image loading awaits images and coordinate-frame declaration
- [x] duration `T -> ms` identity implemented
- [x] semantic validation implemented
- [x] fixture passes

### Phase 2 — episodes
- [x] K sampling implemented
- [x] same-subject/exclusion invariants tested
- [x] support order preserved

### Phase 3 — WHERE codec
- [x] coordinate codec exact
- [x] duration codec exact
- [x] XYD serializer exact
- [x] golden WHERE passes

### Phase 4 — conversation
- [x] search prompt exact (literal canonical-doc comparison test)
- [x] native multi-turn messages exact
- [x] image order tested

### Phase 5 — model init
- [ ] released adapter trainable
- [x] END_FIX rows trainable in real HF small-model tests, tied and untied
- [x] P_E trainable in real HF small-model tests
- [x] frozen parameter assertions pass in small-model tests; old rows unchanged after AdamW

### Phase 6 — collator
- [x] query-only labels exact (actual base processor)
- [x] support END_FIX masked
- [x] query END_FIX supervised
- [x] <=8192 enforcement

### Phase 7 — states
- [x] query END_FIX positions exact
- [x] F count=N (small HF)
- [x] R count=N (small HF)
- [x] no detach (semantic-only backward reaches WHERE states and input row)

### Phase 8 — flat target/parser
- [x] golden target passes
- [x] strict parser tests pass

### Phase 9 — insertion
- [x] fake-R image-conditioned forward works on small HF InternVL
- [x] attention/labels aligned
- [x] fake-R receives grad
- [x] real WHERE-produced R receives grad on small HF InternVL

### Phase 10/11 — joint step
- [x] L_WHERE finite (small HF; released 8B pending)
- [x] L_FLAT finite (small HF; released 8B pending)
- [x] L_total finite (small HF; released 8B pending)
- [x] LoRA grad finite (test adapter, not released weights)
- [x] P_E grad finite/non-zero (small HF)
- [x] END_FIX input grad finite (small HF)
- [x] END_FIX output grad finite when applicable (small HF)

### Phase 12 — checkpoint
- [x] save/load round trip for tied/untied small HF models, optimizer/scheduler and trainable set
- [ ] released-base checkpoint round trip

### Phase 13 — eval
- [x] free-running WHERE separate from GT-conditioned semantic path; native HF generation tested on small model
- [ ] released-model generation and canonical metric implementation/settings validated

---

## 24. Verification log

Append actual commands/results here as implementation progresses. Do not mark a gate complete from code inspection alone when the gate requires runtime evidence.

```text
Date:
Commit:
Command:
Result:
Evidence/notes:
```

### 2026-10-04 — actual implementation verification

Working tree was already dirty (`AGENTS.md`, plan/config/fixture/smoke scaffold
untracked; `DeepGaze-VL/CODEBASE_MAP.md` modified). No commit was created, no golden
fixture was changed, and no file under `DeepGaze-VL/` was edited.

Environment: Windows, Python 3.13, PyTorch `2.11.0+cpu`, Transformers `5.8.0`,
safetensors `0.7.0`. Created `.venv` with system site packages and installed PEFT
`0.18.1`, Accelerate `1.12.0`, Pillow `12.2.0`, torchvision `0.26.0` there. The
system environment was not changed. `torch.cuda.is_available()` is false;
`nvidia-smi` is unavailable. The tested HF/PEFT/torch versions are pinned in
`pyproject.toml` because native multimodal/generation APIs are version-sensitive.

| Phase/evidence | Command | Result |
|---|---|---|
| 1–4, 8 contracts | `python -m pytest tests/test_contracts.py -q` | Initial 20 tests passed: validation, exact goldens, codecs, strict parser, sampling/RNG and unseen exclusion. |
| Contract smoke at independent slice boundaries | `python scripts/smoke_flat.py --contract-only` | Passed after initial contracts and after subsequent model/checkpoint/evaluation slices; final pass unchanged goldens. |
| Native architecture gradient slice | `.venv/Scripts/python -m pytest tests/test_model_path.py -x -q` | Initial 5 model tests passed after fixes; later suite includes checkpoint and generation. |
| Final full scoped suite | `.venv/Scripts/python -m pytest tests -x -q` | **45 passed**, including real base-processor test; 30 third-party NumPy/Pillow conversion deprecation warnings. |
| Released initialization/joint backward gate | `.venv/Scripts/python scripts/smoke_flat.py --with-model` | **Blocked at preflight**, exit 1: released weights and tokenizer JSON are LFS pointers, CUDA unavailable. No model backward or optimizer update claimed. |
| Entry points | `.venv/Scripts/python train_flat.py --help` and `.venv/Scripts/python evaluate_flat.py --help` | Passed. |
| Syntax/import compilation | `.venv/Scripts/python -m compileall -q semgaze train_flat.py evaluate_flat.py scripts/smoke_flat.py` | Passed. |
| Whitespace | `git diff --check` | Passed; only Git CRLF conversion notices. |
| Persisted benchmark audit | `.venv/Scripts/python -c "from semgaze.data.cocosearch18 import read_persisted_splits; s,m,h=read_persisted_splits(); print({k:len(v) for k,v in s.items()}); print('manifest_sha256='+h)"` | Passed source/split checksums, semantic validation, frozen supports, subject and image membership: train 22439, validation 2452, test 2609. |

Manifest SHA-256:
`d40e795788bf725d5d252ace25410b9c899b92f143f2dbe720b64781172f3a14`.

Latest released-model preflight evidence:
`runs/flat-20261004T112344Z-e684be/{resolved_config.json,preflight.json}`.
Preflight records installed versions, adapter metadata and file hashes even when
the payload/device checks block execution. The adapter metadata has the expected
base, rank 8, alpha 16, dropout 0.05, and LM projection targets. Current local
`adapter_model.safetensors` and `tokenizer.json` are each 133-byte LFS pointers;
the historical codebase-map materialization observation does not hold in this
checkout. The pointer for released adapter weights identifies payload SHA-256
`1e865cd7a46d915a5fdd45a8b5ce9ac40be76f90b4e495e4d3370ee30ba6add1`.

### Observed native HF route

Downloaded only the base processor/tokenizer assets (not 8B weights) and saved a
test cache at `.cache/semgaze-base-processor`. The actual base vocabulary equals
the adapter's materialized `vocab.json + added_tokens.json`, and native role/image
rendering equals its saved chat template. END_FIX becomes one atomic token.

`crop_to_patches=False, size={height:448,width:448}` observably yields one tile
per image. Fixture WHERE has `[2,3,448,448]` pixels, 512 image-context positions,
1081 native tokens and 61 supervised query-response tokens. Semantic collation
has `[1,3,448,448]` pixels and 256 image-context positions. Historical assistant
tokens and prompt/example END_FIX occurrences remain ignored labels.

On Transformers 5.8.0, the top-level HF InternVL forward accepts `inputs_embeds`
plus `pixel_values`. Its native `get_placeholder_mask` recognizes unchanged image
embeddings and applies native image replacement. SemGaze uses this route directly;
no custom chat model or manual vision/projector implementation was needed. Host
character boundaries split text tokenization at continuous-state positions (BPE
can otherwise merge `:\n` across a boundary). These are metadata only. Each R
adds exactly one embedding position to that prepared native text stream, with
attention 1 and label -100. Position handling remains native HF.

Small actual InternVL/Qwen3 + PEFT tests observed finite losses, one joint backward,
semantic-only gradients through R/P_E/F to WHERE and END_FIX, preserved image
conditioning, tied/untied row isolation, non-reentrant checkpointing, native
`inputs_embeds` generation, and fresh PEFT checkpoint reload. These tests use
explicitly test-only random small-model adapters and **do not replace released
initialization or satisfy the full-size milestone**.

## 25. Remaining implementation/runtime gates

1. **Released milestone / Phase 0 production gate:** materialized released adapter
   and tokenizer payloads, base 8B weights, and a suitable model device are needed.
   Re-run `--with-model`; only a successful released-model backward closes this
   gate. Full training is guarded by that fixture backward in `train_flat.py`.
2. **Real images:** `data/COCO_Search18/images/tp` and `ta` are empty. Runtime must
   fail rather than substitute fixture images or silently skip records.
3. **Original annotation frame:** persisted records have no width/height. Configure
   `data.annotation_frame` as verified `{width,height}` or explicitly
   `original_image` when the installed files are the original annotation images.
   It remains null in the template; no 1680x1050 or other guessed frame is inserted.
4. **Canonical evaluation:** SM/MM/SED implementations/settings and invalid-output
   treatment, semantic text metrics, semantic generation budget, and the
   model-selection scalar are not frozen. The evaluator exports frozen-episode
   predictions and separate validity diagnostics; it does not invent metric
   defaults, claim a canonical score table, or choose a best checkpoint.

Dependency-order qualification: because released Phase 0 cannot complete locally,
later model-dependent code is implemented provisionally against the observed
native HF interfaces and small-model tests. Production gates stay open. This
isolates the environmental blocker while completing independent code and tests,
as requested; it is not authorization to bypass the released-model training gate.

## 26. Epoch-end validation and trainer history (2026-10-04)

Implemented the requested epoch-level validation without changing the flat
training objective, support sampling law, token supervision, or state path.

- The with-replacement sampler has no natural dataset-pass epoch. Its epoch length
  is now explicitly `training.steps_per_epoch` or `--steps-per-epoch`; the template
  leaves it null until selected. `--max-steps` must be a positive multiple, so each
  scheduled epoch is complete. Counts are optimizer steps, after accumulation.
- `evaluation.strategy: epoch`, `k_values: [1,5,10]`, and
  `loss_aggregation: episode_mean` are explicit in `configs/flat_single.yaml`.
- At every epoch boundary, run all seen-subject persisted validation queries with
  their frozen same-subject train supports for every K. No unseen query enters
  validation/model selection. Frozen context overflow fails without resampling.
- Validation is under `no_grad` with the model and P_E in eval mode. Previous
  module modes are restored even on failure. No backward, optimizer, scheduler,
  parameter, or existing gradient mutation occurs during validation.
- Each episode still uses one teacher-forced WHERE and one flat semantic forward.
  `eval_where` and `eval_flat` use their existing response-token mean NLL;
  `eval_total = lambda_where * eval_where + lambda_sem * eval_flat`.
- `eval_what`, `eval_why`, `eval_how` are **diagnostic partitions of that same flat
  response's shifted token NLL**, not multibranch objectives or text-quality
  metrics. Include each section's line prefixes and trailing newlines; native EOS
  belongs to HOW. Any BPE token spanning sections is attributed by its starting
  character, without retokenizing/changing the target. Every supervised token is
  counted once. Token-weighted component values reconstruct the episode flat NLL.
- Epoch losses average per-episode losses, matching the existing trainer's episode
  mean. Per-K losses and episode counts are also saved in `eval_by_k`. Nothing
  averages WHAT/WHY/HOW to form the flat loss. Generation-quality metrics remain
  explicitly unconfigured under the outstanding Section 25 scientific gates.
- Print a human-readable epoch summary, append step/validation events to
  `trainer_log.jsonl`, and save the full history in each checkpoint's
  `trainer_state.json`. Epoch checkpoints are written after validation, independent
  of the ordinary step save interval. Resume preserves history and absolute epoch
  boundaries, including mid-epoch checkpoints, without repeating completed epochs.

Modified/added files: `train_flat.py`, `configs/flat_single.yaml`,
`semgaze/training/loop.py`, `semgaze/evaluation/validation.py`,
`semgaze/where/collator.py` (response-offset metadata only),
`semgaze/model/{build,config,checkpoint}.py`, `tests/test_epoch_validation.py`,
`README_FLAT.md`, and this plan. No scientific spec, golden fixture, or upstream
reference file was modified.

Verification:

| Command | Result |
|---|---|
| `.venv/Scripts/python -m pytest tests/test_epoch_validation.py -x -q` | 9 passed: diagnostic loss identity, two forwards/no-grad, unchanged parameters/gradients/RNG, restored modes, frozen all-K coverage, leakage rejection, schedule configuration, epoch timing, JSONL/checkpoint history and mid-epoch resume. |
| `.venv/Scripts/python -m pytest tests -q` | **54 passed**, 50 existing third-party NumPy conversion deprecation warnings. |
| `python scripts/smoke_flat.py --contract-only` | Passed; original goldens unchanged. |
| `.venv/Scripts/python scripts/smoke_flat.py --with-model` | Blocked at existing released-model preflight: adapter/tokenizer LFS pointers and unavailable CUDA. Evidence: `runs/flat-20261004T162856Z-5a4b09/`. |
| `.venv/Scripts/python train_flat.py --help` | Passed; explicit `--steps-per-epoch` option exposed. |
| `.venv/Scripts/python -m compileall -q semgaze train_flat.py` | Passed. |
| `git diff --check` | Passed (only CRLF conversion notices). |
| `.venv/Scripts/python -m pytest tests/test_epoch_validation.py::test_real_small_hf_epoch_validation -q -s` | Passed; observed the example below using the actual small HF/PEFT fixture and all three K values. |

Example from the small random HF fixture, **not a released-model benchmark**:

```text
Epoch 1 | step 2 | validation: 1 seen queries x K=1,5,10 (3 episodes)
  response NLL (episode mean): eval_where=5.994777 | eval_what=5.977595 | eval_why=5.968741 | eval_how=5.997607 | eval_flat=5.979805 | eval_total=11.974583
  WHAT/WHY/HOW: diagnostic sections of one flat response; generation-quality metrics not configured
```

## 27. Epoch autoregressive prediction exports (2026-10-04)

Added generation after the existing epoch validation-loss event/console summary
and before checkpoint saving. The loss reductions, diagnostics, summary format,
training objective, and supervision remain unchanged.

- Train prediction consumes exactly one batch of `per_device_train_batch_size`
  accepted episodes: the first batch from the final optimizer step of the epoch.
  Later accumulation batches are excluded, and inference never samples additional
  training episodes. This also works after a mid-epoch resume without new sampler
  state or replaying earlier epochs.
- Validation prediction covers every seen validation query at every K=1/5/10,
  preserving frozen same-subject support membership/order. Shared eligibility
  checks were extracted from the loss validator without changing its behavior.
- Reuse `evaluate_where_episode` and `evaluate_flat_episode`, both invoking native
  HF autoregressive generation. WHERE receives no query GT scanpath. Semantic
  generation receives full chronological states from teacher-forced GT WHERE and
  gold WHY groups, as required by Section 18 / `06_EVALUATION_SPEC.md`. The semantic
  assistant response is not teacher-forced and predicted WHERE is not transferred
  to semantic inference.
- `evaluation.predictions` fixes train batches to 1 and validation scope to
  `all_seen`. `semantic_max_new_tokens` remains null in the template and must be
  declared explicitly there or with `--semantic-max-new-tokens` before training.
  It is saved in resolved configuration and checked on resume. No canonical
  generation budget or quality metric is invented.
- Stream `predictions/epoch-NNNN/train.jsonl` and `validation.jsonl`, with query/K,
  ordered support IDs, split identity, exact `WHERE: {GT, PRED}` and
  `SEMANTIC: {GT, PRED}`, and existing parser/conditioning metadata. Preserve raw
  malformed output. Partial files retain `.partial` on failure; no successful
  epoch-prediction event/checkpoint is written for incomplete generation.
- Console summary prints counts, paths, and conditioning. `epoch_predictions`
  events are retained in JSONL trainer logs and checkpoint `log_history` alongside
  the unchanged loss events. Files stay in the original run directory and their
  absolute paths remain in resumed history.
- No gradient/update occurs; exact module modes and Python/Torch CPU/CUDA RNG are
  preserved, including when generation fails. Existing gradients are untouched.

Files changed for this addition: `semgaze/evaluation/predictions.py` (new),
`semgaze/evaluation/validation.py`, `semgaze/training/loop.py`,
`semgaze/model/config.py`, `train_flat.py`, `configs/flat_single.yaml`,
`tests/test_epoch_predictions.py` (new), `tests/test_epoch_validation.py`,
`README_FLAT.md`, and this plan. No canonical spec, golden, or upstream file changed.

Verification:

| Command | Result |
|---|---|
| `.venv/Scripts/python -m pytest tests/test_epoch_predictions.py tests/test_epoch_validation.py -x -q` | 17 passed: full seen/all-K coverage, raw GT/PRED preservation, exactly first train batch with accumulation, separate real HF generation calls, state conditioning, no parameter/gradient/RNG changes, failure restoration, checkpoint history and mid-epoch resume. |
| `.venv/Scripts/python -m pytest tests -q` | **62 passed**, 68 existing third-party NumPy conversion deprecation warnings. |
| `.venv/Scripts/python -m pytest tests/test_epoch_predictions.py::test_real_hf_predictions_generate_separately_and_preserve_training -q -s` | Passed again after adding explicit zero semantic response-length/all-masked native input assertions; printed the summary below using real small HF/PEFT generation. |
| `python scripts/smoke_flat.py --contract-only` | Passed, unchanged goldens. |
| `.venv/Scripts/python scripts/smoke_flat.py --with-model` | Existing preflight blocker: adapter weights/tokenizer are LFS pointers and CUDA unavailable. Evidence: `runs/flat-20261004T164331Z-a28ace/`. |
| `.venv/Scripts/python train_flat.py --help` | Passed; explicit semantic budget option exposed. |
| `.venv/Scripts/python -m compileall -q semgaze train_flat.py tests/test_epoch_predictions.py` | Passed. |
| `git diff --check` | Passed (CRLF conversion notices only). |

Small random HF fixture output (paths shortened; not benchmark quality evidence),
printed after the unchanged loss summary:

```text
Epoch 1 | step 2 | autoregressive predictions:
  train: 1 batch (1 queries) -> .../predictions/epoch-0001/train.jsonl
  validation: 1 seen queries x K=1,5,10 (3 episodes) -> .../predictions/epoch-0001/validation.jsonl
  Each record: WHERE GT/PRED; SEMANTIC GT/PRED. WHERE is free-running; semantic uses GT WHERE states + gold WHY groups.
```

Section 25 production gates remain open: released weights/prepared device, real
images and verified annotation frame. No full training or released-model
prediction run is claimed by these small-model tests.


## 28. Configuration-driven runtime (2026-10-05)

Explicit user request: make YAML experiment choices effective end to end; preserve real
runtime/data validation and the existing flat architecture/objective. This supersedes
old initialization/default-enforcement instructions in this plan, without implementing
multibranch semantics or modifying reference baseline checkouts.

Completed: missing-field-only YAML defaults; CLI-before-construction resolution; fresh
and continued real PEFT initialization; configurable LM LoRA, token, context and device;
config-driven persisted data roots, duration field, subjects and weighted K sampling;
optimizer/scheduler/precision/horizons/batching; epoch/step/disabled evaluation and K
subsets; prediction count/scope/budget; logging/output/checkpoint policies; resumable
optimizer hyperparameters and precision. Explicit null is never filled silently.
All reference fields have a consumer or a documented concrete implementation limit in
`documents/CONFIGURATION_AUDIT.md`. The existing golden files were not changed.

Verification:

| Command | Result |
|---|---|
| `.venv/Scripts/python -m pytest tests/test_config_plumbing.py -q --basetemp=.cache/pytest-config-3 --tb=short` | 14 passed at that stage: actual fresh/continued HF+PEFT consumers, precision/gradients, optimizer/scheduler/sampling, CLI epochs, relocated dataset and checkpoint integrity. Two further evaluation/cache tests added afterward. |
| `.venv/Scripts/python -m pytest tests -q --basetemp=.cache/pytest-config-final --tb=short` | **76 passed**, including **16 configuration-plumbing cases**; 75 existing third-party NumPy conversion warnings. No model downloads. |
| `python scripts/smoke_flat.py --contract-only` | Passed unchanged fixture/golden/parser contracts. |
| `.venv/Scripts/python scripts/smoke_flat.py --with-model` | Contract portion passed; configured fresh 8B model stopped at CUDA preflight on CPU-only host. Evidence: `runs/semgaze_flat_single-20261005T072342Z-0b621a/`. No full training ran. |
| `.venv/Scripts/python train_flat.py --help` and `.venv/Scripts/python evaluate_flat.py --help` | Passed. |
| `.venv/Scripts/python -m compileall -q semgaze train_flat.py evaluate_flat.py scripts/smoke_flat.py` | Passed. |
| `git -c safe.directory=D:/Programming/Python/SemGaze diff --check` | Passed. |

The first pytest invocation hit a permission error in the host user's temporary pytest
folder; subsequent verification uses an isolated workspace `.cache/` basetemp. A new
evaluation entrypoint test found YAML parsing of JSON scientific notation as strings;
JSON checkpoint configs now use the JSON parser, and the complete suite passes.

Remaining production gates are unchanged: prepared device/full base weights and real
images/annotation frame. Compatible tiny real HF/PEFT tests prove mechanics, not 8B
benchmark quality. Unsupported scientific/backend combinations and incomplete checkpoint
exports fail explicitly, with the reasons and full field inventory in the audit report.
