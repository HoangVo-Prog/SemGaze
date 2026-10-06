# SemGaze — WHERE / Scanpath Specification

## 1. Scope

This file is the canonical owner for the SemGaze WHERE branch: scanpath definition, coordinate and duration representation, scanpath serialization, the `<END_FIX>` boundary token, DeepGaze3.5-VL-style prompt construction, **rendering an ordered few-shot support list into the VLM chat conversation**, autoregressive factorization, teacher-forced WHERE loss, and free-running WHERE inference.

`01_FEWSHOT_SPEC.md` owns support-set membership, subject partitions, support sampling, support ordering, support reuse, and evaluation draws. It hands WHERE an **already ordered** sequence of `(image, task, observed scanpath)` support records plus the query. WHERE must preserve that membership and order exactly while rendering the multi-turn multimodal conversation. Hidden-state extraction at `<END_FIX>` is owned by `03_STATE_SPEC.md`. Optimizer/backward details, low-level batch packing, and any whole-system multi-forward schedule are owned by `05_TRAINING_SPEC.md`.

### 1.1 Alignment contract with DeepGaze3.5-VL

The WHERE branch follows DeepGaze3.5-VL in the aspects that define scanpath prediction as an autoregressive VLM problem:

- image-conditioned autoregressive text generation;
- coordinates represented on a 100-by-100 discrete grid;
- zero-padded two-digit coordinate text;
- optional per-fixation duration represented in milliseconds;
- Python-list-style scanpath grammar;
- explicit conditioning on the requested number of fixations;
- standard next-token prediction / teacher forcing rather than a separate geometric regression head;
- native multimodal chat context for few-shot demonstrations;
- no custom system prompt added by WHERE;
- free-running generation from the same autoregressive decoder used during training.

SemGaze intentionally differs from DeepGaze3.5-VL in two places:

1. `<END_FIX>` is inserted after every fixation tuple so another SemGaze component can read a fixation-level hidden state.
2. Personalization is provided by same-observer few-shot demonstrations in the causal multimodal context rather than by a subject-ID token.

These are intentional SemGaze extensions. They must not silently alter the remaining WHERE representation or generation rules.

DeepGaze3.5-VL references used to freeze this contract:

- Paper: <https://arxiv.org/pdf/2607.02083>
- Released inference prompt: `predict_scanpath.py`
- Released coordinate/duration formatter and few-shot builder: `evaluate_vllm_unified.py`
- Repository: <https://github.com/Susmit-A/DeepGaze3.5-VL>
- InternVL3.5 paper for the underlying VLM/chat stack: <https://arxiv.org/pdf/2508.18265>

When the DeepGaze paper prose and released implementation differ on an implementation detail, SemGaze follows the released DeepGaze code/configuration for reproduction-oriented WHERE behavior unless this specification explicitly states a SemGaze override.

---

## 2. Scanpath definition

The current primary SemGaze method design uses the **spatial + duration (XYD)** WHERE mode. The spatial-only grammar is retained only as a non-primary reference/control contract; it is not a second current primary runtime mode.

### 2.1 Spatial-only mode (non-primary reference/control)

For a trajectory with `N` fixations:

$$
S^{xy}=(s_1,\ldots,s_N),
$$

with

$$
s_t=(x_t,y_t).
$$

### 2.2 Spatial + duration mode (current primary mode)

When duration modeling is enabled:

$$
S^{xyd}=(s_1,\ldots,s_N),
$$

with

$$
s_t=(x_t,y_t,\delta_t),
$$

where:

- $x_t,y_t$ are fixation coordinates;
- $\delta_t$ is fixation duration in milliseconds;
- $N$ is the number of fixations requested for the trajectory.

A dataset adapter may expose aligned raw fixation fields such as `X`, `Y`, and a duration-related field. WHERE must **not** infer duration semantics from a field name such as `T`. Before XYD mode is enabled for a dataset, its adapter must explicitly declare:

```text
duration_source_field
duration_semantics = dwell_duration
source_unit
conversion_to_ms
```

After this adapter-level verification, the adapter outputs one duration value in milliseconds per fixation, aligned one-to-one with `X` and `Y`:

$$
|X|=|Y|=|D_{ms}|=N.
$$

If the official dataset documentation/source code does not establish that the source quantity is fixation dwell duration (or does not provide enough information to derive dwell duration from verified start/end timestamps), XYD mode is unsupported for that dataset. WHERE must not guess or synthesize durations.

**Frozen COCO-Search18 declaration:** `duration_source_field = T`, `duration_semantics = dwell_duration`, `source_unit = ms`, `conversion_to_ms = identity`. Thus `T[t]` is fixation dwell duration in milliseconds. Preserve source milliseconds in the normalized record; round non-integer milliseconds and clip to `000..999` only for WHERE serialization.

The method design uses XYD globally once that dataset-duration declaration is frozen. A deliberately named non-primary spatial-only control may use XY globally. A single assistant scanpath response must never mix `(x, y)` and `(x, y, d)` tuples.

---

## 3. Coordinate representation

### 3.1 Continuous-to-grid conversion

Let the raw fixation be `(x_t, y_t)` in image-pixel coordinates and let the image width and height be `W_I` and `H_I`. `W_I` and `H_I` are the dimensions of the **original annotation coordinate frame** used by the raw gaze data, before any InternVL resizing, dynamic tiling, thumbnail creation, or other visual preprocessing.

To match the released DeepGaze3.5-VL conversion path, SemGaze uses the same two-stage rounding rule rather than floor quantization:

$$
r_t^x=
\operatorname{round}
\left(
100\frac{x_t}{W_I},
1
\right),
$$

$$
r_t^y=
\operatorname{round}
\left(
100\frac{y_t}{H_I},
1
\right),
$$

followed by

$$
b_t^x=
\operatorname{clip}
\left(
\operatorname{int}(\operatorname{round}(r_t^x)),
0,
99
\right),
$$

$$
b_t^y=
\operatorname{clip}
\left(
\operatorname{int}(\operatorname{round}(r_t^y)),
0,
99
\right).
$$

Implementation rule: use Python-compatible `round` semantics for both stages so the conversion matches the released DeepGaze3.5-VL code path. Do not replace this with `floor`, truncation, or a one-stage rounding implementation.

### 3.2 Coordinate text

Each grid coordinate is serialized as ordinary zero-padded two-digit decimal text:

```text
00 01 ... 09 10 ... 99
```

Formally:

$$
\operatorname{FmtXY}(b)=\texttt{f"\{b:02d\}"}.
$$

No learned coordinate vocabulary such as `<X_52>` or `<Y_48>` is used.

The valid serialized coordinate domain is therefore exactly `00 ... 99`.

### 3.3 Prompt wording versus actual coordinate domain

The released DeepGaze3.5-VL prompt literally describes coordinates as `0-100`, while the paper and released formatter use valid discrete values `00 ... 99` and clamp at `99`. SemGaze preserves the DeepGaze prompt wording for prompt inheritance, but its serializer/parser and all mathematical definitions use the canonical domain `0 ... 99`.

---

## 4. Duration representation

Duration mode follows the DeepGaze3.5-VL `(x, y, d)` formulation.

### 4.1 Unit and quantization

Durations are represented in milliseconds and discretized into 1-ms bins.

If a dataset already provides integer milliseconds, use that value directly. If the dataset adapter produces a non-integer millisecond value, SemGaze rounds it to the nearest integer millisecond before serialization.

Let the resulting integer duration be $d_t^{\mathrm{ms}}$. The canonical WHERE duration is

$$
b_t^d=
\operatorname{clip}
\left(
d_t^{\mathrm{ms}},
0,
999
\right).
$$

### 4.2 Duration text

Duration is serialized as a zero-padded three-digit decimal string:

```text
000 ... 999
```

Formally:

$$
\operatorname{FmtD}(b)=\texttt{f"\{b:03d\}"}.
$$

Examples:

```text
096
183
521
```

The policy is global and frozen:

- unit: milliseconds;
- bin width: 1 ms;
- minimum: `000`;
- maximum: `999`;
- width: exactly 3 decimal digits;
- values below 0 are clipped to 0;
- values above 999 ms are clipped to 999;
- no learned duration token vocabulary is introduced.

---

## 5. Canonical scanpath serialization

DeepGaze3.5-VL uses a Python-list-style sequence of fixation tuples. SemGaze preserves that outer-list and tuple grammar and adds exactly one structural token: `<END_FIX>`.

### 5.1 Spatial-only fixation

A single fixation is serialized as:

```text
(52, 48)<END_FIX>
```

A full scanpath is serialized on one logical response sequence as:

```text
[(52, 48)<END_FIX>, (47, 56)<END_FIX>, (22, 48)<END_FIX>]
```

### 5.2 Spatial + duration fixation

A single fixation is serialized as:

```text
(52, 48, 231)<END_FIX>
```

A full scanpath is:

```text
[(52, 48, 231)<END_FIX>, (47, 56, 183)<END_FIX>, (22, 48, 096)<END_FIX>]
```

### 5.3 Exact grammar

For spatial-only mode:

```text
"[" + ", ".join(f"({x:02d}, {y:02d})<END_FIX>" for each fixation) + "]"
```

For duration mode:

```text
"[" + ", ".join(f"({x:02d}, {y:02d}, {d:03d})<END_FIX>" for each fixation) + "]"
```

The following are invariants:

- one outer `[` ... `]` pair per scanpath;
- comma + single-space separator between fixation items;
- comma + single-space separator inside each tuple;
- `x` and `y` are always two digits;
- `d` is always three digits when duration mode is enabled;
- `<END_FIX>` appears immediately after the closing `)` of every fixation tuple, including the last fixation;
- no newline is semantically required between fixations;
- no extra prose is allowed in the assistant scanpath response.

`<END_FIX>` makes the resulting string no longer a literal valid Python list. Therefore SemGaze prompts say **list-style sequence of tuples** rather than claiming that the response is syntactically valid Python.

The exact hidden-state readout at `<END_FIX>` is owned by `03_STATE_SPEC.md` and must not change the serialization above.

### 5.4 `<END_FIX>` tokenizer/model contract

`<END_FIX>` is a SemGaze-specific **atomic vocabulary token**, not an arbitrary text substring. The following contract is frozen:

- tokenizing the literal string `<END_FIX>` in isolation must produce exactly one token ID;
- the token is added to the tokenizer vocabulary once before training if it is not already present;
- model token embeddings are resized consistently with the tokenizer vocabulary;
- the parameter(s) required to learn the new input embedding and output-logit row for `<END_FIX>` must remain trainable even when the pretrained backbone is otherwise LoRA-frozen; if input/output embeddings are tied, this refers to the shared parameter;
- generation text used by the WHERE parser must preserve `<END_FIX>` rather than silently dropping it through `skip_special_tokens=True` or an equivalent decoding option;
- every teacher-forced target uses this exact single token after every fixation tuple.

A startup assertion must verify the atomic-token invariant before any training/evaluation run. If the invariant fails, the run must stop rather than falling back to a multi-token spelling.

Using an existing punctuation token such as `)` as the fixation boundary is outside the primary specification and may only be studied as an explicit ablation.

---

## 6. Support scanpaths and multimodal causal context

### 6.1 Input contract from `01_FEWSHOT_SPEC.md`

`01_FEWSHOT_SPEC.md` supplies WHERE with an **ordered** support sequence

$$
\widetilde{\mathcal C}_u^K
=
\left(
d_1^{\mathrm{sup}},
\ldots,
d_K^{\mathrm{sup}}
\right),
$$

where

$$
d_j^{\mathrm{sup}}=(I_j,Q_j,S_j),
$$

plus the query

$$
(I_q,Q_q).
$$

The order in $\widetilde{\mathcal C}_u^K$ is already final:

- during training, it is the sampled random support order from `01_FEWSHOT_SPEC.md`;
- during evaluation, it is the frozen manifest order from `01_FEWSHOT_SPEC.md`.

WHERE must not resample, reshuffle, sort, drop, duplicate, or replace supports.

### 6.2 Support response

Each support assistant response is the observed support scanpath serialized with exactly the same active WHERE mode and serializer used for the query target:

$$
Z_j^S
=
\operatorname{Tok}
\left(
\operatorname{Serialize}(S_j)
\right).
$$

If the run is spatial-only, every support response uses the canonical `(x, y)<END_FIX>` grammar.

If the run is duration-enabled, every support response uses the canonical `(x, y, d)<END_FIX>` grammar.

The support response contains **only** the serialized scanpath. It does not contain a subject ID, an explanation of personalization, or additional prose.

### 6.3 Frozen personalized conversation design

The canonical personalized WHERE input is a **multi-turn multimodal conversation**:

```text
user:      [support image 1] + PROMPT(Q_1, N_1)
assistant: SERIALIZE(S_1)

user:      [support image 2] + PROMPT(Q_2, N_2)
assistant: SERIALIZE(S_2)

...

user:      [support image K] + PROMPT(Q_K, N_K)
assistant: SERIALIZE(S_K)

user:      [query image] + PROMPT(Q_q, N_q)
assistant: <query generation starts here>
```

This is the exact high-level chat-role design for the primary personalized baseline.

In particular, the implementation must **not** replace it with:

- one giant user message containing all demonstrations;
- a custom `"Demonstration 1 / Observed scanpath"` transcript;
- a learned `fewshot_tensor`;
- a persistent user embedding;
- a subject-ID token;
- a custom system instruction such as `"these demonstrations were produced by the same observer"`.

The same-subject property is guaranteed by `01_FEWSHOT_SPEC.md`; it is not restated in prompt text.

### 6.4 Multi-turn does not mean multi-forward

The support/query structure above is **multi-turn at the chat-template level**, not a requirement to run one VLM forward per turn.

For the WHERE branch, the selected InternVL chat template serializes the complete history into one causal context:

$$
[U_1,A_1,U_2,A_2,\ldots,U_K,A_K,U_q]
\rightarrow A_q.
$$

Thus the query can causally attend to all preceding support-image, support-prompt, and support-scanpath representations.

Any larger whole-system forward schedule outside this packed WHERE computation is owned by `05_TRAINING_SPEC.md`.

### 6.5 Personalized prompt rule

There is **no separate personalized prompt string**.

For a given output mode, the text used in each support user turn and in the query user turn comes from the same canonical prompt function:

$$
\operatorname{Prompt}(Q,N,m),
\qquad
m\in\{xy,xyd\}.
$$

Therefore:

```text
support user turn j = [I_j] + Prompt(Q_j, N_j, m)
query user turn     = [I_q] + Prompt(Q_q, N_q, m)
```

Personalization is expressed **only** by the presence of the ordered same-observer support turns before the query.

Do not add phrases such as:

```text
for this observer
for the same observer
based on this person's previous scanpaths
by subject X
```

to the support or query prompt.

### 6.6 Native InternVL chat/multi-image pathway

The canonical WHERE backbone is frozen to the Hugging Face InternVL3.5 stack used by the released DeepGaze3.5-VL implementation:

```text
BASE_MODEL = OpenGVLab/InternVL3_5-8B-HF
processor/tokenizer are loaded from the same checkpoint
chat formatting uses the checkpoint's native Hugging Face chat template
```

The implementation must not replace this baseline with the separate custom `model.chat(...)` API path or with a hand-written SemGaze chat template. A representative rendering call is:

```python
processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

For a non-personalized query, `messages` contains one query user turn followed by the generation prompt.

For a personalized `K`-shot query, `messages` contains:

```text
K x (support user turn, support assistant turn)
1 x query user turn
generation prompt
```

Support and query images remain **separate multimodal inputs** aligned with their corresponding user turns. They must not be stitched into one composite image.

The native chat template may add its own model-required role separators, image placeholders, and assistant-generation markers. SemGaze must not inject an additional personalization system message and must not manually rewrite those native template tokens.

### 6.7 Context-budget contract for the primary baseline

Few-shot personalization creates up to `K+1` images in one causal context. This differs materially from the single-image DeepGaze training setup, so SemGaze freezes an explicit baseline budget rather than copying DeepGaze's single-image sequence limit.

For the primary implementation:

```text
maximum supported K = 10
maximum images per episode = 11
visual budget = at most one 448x448 visual tile per support/query image
maximum total model sequence length = 8192 tokens
```

The exact processor arguments used to realize the one-tile policy are adapter-specific, but the observable behavior above is mandatory. In particular:

- preserve every support record selected by `01_FEWSHOT_SPEC.md`;
- preserve support order exactly;
- preserve one image association per user turn;
- preserve all support assistant scanpath text;
- never silently drop, reorder, or text-truncate an individual support turn to satisfy the token budget; the fixed one-tile visual preprocessing policy is the only allowed primary-baseline visual reduction;
- never silently reduce `K`;
- do not use DeepGaze's single-image `2048/4096` context limits as the SemGaze few-shot default.

After native chat rendering and multimodal preprocessing, the full episode must fit the 8192-token limit. Overflow handling is split-specific:

- training: preserve the current epoch query and its already sampled $K$; resample only the same-subject support realization/order under `01_FEWSHOT_SPEC.md`. If the configured finite retry budget is exhausted, fail rather than replacing/skipping the query or changing $K$;
- test evaluation: fail the run because frozen episode membership/order must not be altered or resampled.

Low-level patch packing, tensor collation, and the exact processor argument names remain owned by the model/training adapter / `05_TRAINING_SPEC.md`; they may not change the behavioral contract above.

---

## 7. Task rendering

DeepGaze3.5-VL demonstrates that task conditioning is changed through prompt text rather than a task-specific prediction head. SemGaze adopts the same principle.

Let `Q` be the task/query associated with an image. WHERE renders the task into the first and third lines of the DeepGaze-style prompt.

### 7.1 Free-viewing task

When the dataset condition is free viewing:

```text
INTRO(Q) = "during free viewing for {VIEW_SECONDS} seconds."
CONSIDER(Q) = "Consider visual saliency, semantic importance, and how attention naturally flows across a scene."
```

`VIEW_SECONDS` is supplied by verified dataset/trial metadata from the dataset adapter. It must not be globally hard-coded to DeepGaze's standalone 3-second example. If the free-viewing exposure duration cannot be verified for a dataset/trial, this renderer is unsupported for that sample rather than guessed.

### 7.2 Target-directed visual search

When `Q` is a search target, use the released DeepGaze wording:

```text
INTRO(Q) = "while searching for a {Q}."
CONSIDER(Q) = "Consider the search target, visual saliency, and how attention naturally flows during visual search."
```

For COCO-Search18, `{Q}` is the target category such as `bottle`, `clock`, or `tv`.

### 7.3 Generic task/query

For datasets in which `Q` is a general task or natural-language query rather than a search-target label, use the SemGaze extension:

```text
INTRO(Q) = "while performing the following task: {Q}."
CONSIDER(Q) = "Consider the task, visual saliency, semantic importance, and how attention naturally flows while performing the task."
```

This generic renderer is a SemGaze extension; it is not claimed to be a verbatim DeepGaze3.5-VL prompt. The remaining scanpath definition, coordinate description, requested trajectory length, and output grammar stay identical to the DeepGaze-derived templates below.

---

## 8. WHERE prompt contracts

Personalization is defined **only** by whether the ordered same-observer support turns from `01_FEWSHOT_SPEC.md` precede the query. There is no subject-ID token, no personalized system message, and no special personalized query wording.

Therefore the personalized and non-personalized prompt **functions are literally identical** for the same output mode. The only difference is conversation history.

The four SemGaze variants below are a factorial combination of two independent switches: support context (`K=0` versus same-observer `K>0`) and output arity (`xy` versus `xyd`). DeepGaze3.5-VL directly demonstrates the spatial prompt, duration formulation/prompt examples, subject conditioning, and few-shot chat mechanism separately; this four-way combination is the frozen SemGaze composition. The only deliberate prompt-level structural addition is the SemGaze `<END_FIX>` requirement.

Let:

- `{INTRO(Q)}` be selected by Section 7;
- `{CONSIDER(Q)}` be selected by Section 7;
- `{N}` be the required number of fixations.

The literal wording below is frozen apart from those placeholders.

### 8.1 Variant A — non-personalized, no duration (non-primary control)

Conversation structure:

```text
user: [query image] + PROMPT_XY(Q_q, N_q)
assistant: <generate query scanpath>
```

`PROMPT_XY(Q, N)`:

```text
Analyze this image and predict a human eye movement scanpath {INTRO(Q)}
A scanpath is the temporal sequence of fixation points showing where a person looks over time.
{CONSIDER(Q)}

Generate a scanpath of exactly {N} fixation points in temporal order as a list-style sequence of tuples: (x, y)
- x: horizontal position (0-100, 0=left, 100=right)
- y: vertical position (0-100, 0=top, 100=bottom)
- Points should be ordered from first fixation to last fixation.
- Append <END_FIX> immediately after every fixation tuple.

Output ONLY the list-style scanpath sequence:
[(51, 46)<END_FIX>, (38, 28)<END_FIX>, ...]
```

No support turns are present.

### 8.2 Variant B — personalized, no duration (non-primary control)

Let the ordered support sequence from `01_FEWSHOT_SPEC.md` be

$$
\widetilde{\mathcal C}_u^K
=
\left(
(I_1,Q_1,S_1),
\ldots,
(I_K,Q_K,S_K)
\right).
$$

The exact conversation structure is:

```text
user:      [support image 1] + PROMPT_XY(Q_1, N_1)
assistant: Serialize_XY(S_1)

user:      [support image 2] + PROMPT_XY(Q_2, N_2)
assistant: Serialize_XY(S_2)

...

user:      [support image K] + PROMPT_XY(Q_K, N_K)
assistant: Serialize_XY(S_K)

user:      [query image] + PROMPT_XY(Q_q, N_q)
assistant: <generate query scanpath>
```

`PROMPT_XY` is **exactly** the same literal function defined in Variant A.

There is:

- no custom system prompt;
- no `"same observer"` sentence;
- no subject-ID token;
- no extra demonstration wrapper text;
- no change to the query wording because the episode is personalized.

The support order above must be exactly the order supplied by `01_FEWSHOT_SPEC.md`.


### 8.3 Variant C — non-personalized, with duration (duration-enabled control)

Conversation structure:

```text
user: [query image] + PROMPT_XYD(Q_q, N_q)
assistant: <generate query scanpath with duration>
```

`PROMPT_XYD(Q, N)`:

```text
Analyze this image and predict a human eye movement scanpath {INTRO(Q)}
A scanpath is the temporal sequence of fixation points showing where a person looks over time.
{CONSIDER(Q)}

Generate a scanpath of exactly {N} fixation points in temporal order, with fixation duration, as a list-style sequence of tuples: (x, y, t)
- x: horizontal position (0-100, 0=left, 100=right)
- y: vertical position (0-100, 0=top, 100=bottom)
- t: fixation duration in milliseconds (0-999)
- Points should be ordered from first fixation to last fixation.
- Append <END_FIX> immediately after every fixation tuple.

Output ONLY the list-style scanpath sequence:
[(51, 46, 221)<END_FIX>, (38, 28, 754)<END_FIX>, ...]
```

No support turns are present.

### 8.4 Variant D — personalized, with duration (current primary personalized path)

Let the ordered support sequence from `01_FEWSHOT_SPEC.md` be

$$
\widetilde{\mathcal C}_u^K
=
\left(
(I_1,Q_1,S_1),
\ldots,
(I_K,Q_K,S_K)
\right).
$$

The exact conversation structure is:

```text
user:      [support image 1] + PROMPT_XYD(Q_1, N_1)
assistant: Serialize_XYD(S_1)

user:      [support image 2] + PROMPT_XYD(Q_2, N_2)
assistant: Serialize_XYD(S_2)

...

user:      [support image K] + PROMPT_XYD(Q_K, N_K)
assistant: Serialize_XYD(S_K)

user:      [query image] + PROMPT_XYD(Q_q, N_q)
assistant: <generate query scanpath with duration>
```

`PROMPT_XYD` is **exactly** the same literal function defined in Variant C.

There is:

- no custom system prompt;
- no `"same observer"` sentence;
- no subject-ID token;
- no extra demonstration wrapper text;
- no change to the query wording because the episode is personalized.

The support order above must be exactly the order supplied by `01_FEWSHOT_SPEC.md`.


### 8.5 Mode consistency rule

For one episode, support and query must use the same output mode:

```text
XY episode  -> all support responses are XY and query target is XY
XYD episode -> all support responses are XYD and query target is XYD
```

Mixing duration-enabled and duration-free support responses inside one canonical episode is outside the primary specification.

---

## 9. Formal WHERE input

Let the ordered support sequence supplied by `01_FEWSHOT_SPEC.md` be

$$
\widetilde{\mathcal C}_u^K
=
\left(
(I_1,Q_1,S_1),
\ldots,
(I_K,Q_K,S_K)
\right).
$$

For mode

$$
m\in\{xy,xyd\},
$$

define support turn pair $j$ as

$$
B_j^{\mathrm{sup}}
=
\left[
U(I_j,\operatorname{Prompt}(Q_j,N_j,m)),
A(\operatorname{Serialize}(S_j,m))
\right].
$$

Define the query user turn as

$$
B_q^W
=
U(I_q,\operatorname{Prompt}(Q_q,N_q,m)).
$$

### 9.1 Non-personalized context

$$
X_{\mathrm{WHERE}}^{0}
=
[B_q^W].
$$

### 9.2 Personalized context

$$
X_{\mathrm{WHERE}}^{K}
=
[
B_1^{\mathrm{sup}},
\ldots,
B_K^{\mathrm{sup}},
B_q^W
].
$$

The index order $1,\ldots,K$ is the order already fixed by `01_FEWSHOT_SPEC.md`.

There is no SemGaze-authored personalization system prompt:

$$
P_{\mathrm{sys}}^{\mathrm{personalize}}=\varnothing.
$$

At implementation time, the native InternVL chat template is responsible only for model-specific role separators, generation markers, and image placeholders. It must not alter support membership, support order, or support/query role assignment.

During training, the GT query assistant response is appended after $B_q^W$ for teacher forcing. During free-running inference, generation begins after the query user turn.


---

## 10. Autoregressive WHERE probability

Let

$$
Z_q^S=(z_1,\ldots,z_L)
$$

be the tokenized serialized query scanpath, including tuple punctuation, coordinate/duration digits, `<END_FIX>` tokens, list punctuation, and the final response terminator as required by the model template.

Then

$$
P_\theta
\left(
Z_q^S
\mid
X_{\mathrm{WHERE}}
\right)
=
\prod_{\ell=1}^{L}
P_\theta
\left(
z_\ell
\mid
X_{\mathrm{WHERE}},
z_{<\ell}
\right).
$$

WHERE is therefore a standard autoregressive scanpath language model over structured text. There is no separate coordinate decoder head or duration decoder head.

---

## 11. Teacher forcing and WHERE loss

During training, the GT query scanpath response is teacher-forced under the standard causal language-model objective.

Let the GT query response tokens be

$$
Z_q^{S,*}=(z_1^*,\ldots,z_L^*).
$$

The WHERE loss is

$$
\mathcal L_{\mathrm{WHERE}}
=
-\frac{1}{|\Omega_q|}
\sum_{\ell\in\Omega_q}
\log
P_\theta
\left(
z_\ell^*
\mid
X_{\mathrm{WHERE}},
z_{<\ell}^*
\right),
$$

where $\Omega_q$ is the set of supervised query assistant-response token positions.

### 11.1 Training conversation assembly and loss masking

During training, the complete ordered conversation is rendered with the GT query assistant response appended as the final assistant turn. The native InternVL chat template is therefore applied to:

```text
[U_1, A_1, ..., U_K, A_K, U_q, A_q^*]
```

with training-time chat rendering equivalent to `add_generation_prompt=False` because the final assistant response is already present.

The semantic label mask is frozen as follows:

```text
labels = -100 for:
  all image/visual tokens
  all support user-turn tokens
  all support assistant-turn tokens
  all query user-turn tokens
  native chat separators belonging only to context

labels = token_id for:
  the final query assistant serialized scanpath
  its native assistant-response terminator / EOS token(s), if the template emits them
```

Thus:

- support images are context only;
- support user prompts are context only;
- support assistant scanpaths are context only;
- query user/image tokens are context only;
- only the final query assistant response span contributes to `L_WHERE`.

`05_TRAINING_SPEC.md` may choose the low-level mechanism for deriving the final assistant span (for example, a native assistant-token mask or a deterministic role-span reconstruction), but it must satisfy this exact semantic mask. A unit test must verify that no support-assistant token receives a non-ignored label.

Supervising all support assistant responses is outside the primary objective and may only be used as an explicitly named alternative/ablation.

### 11.2 No auxiliary geometric loss

The canonical WHERE branch has no separate:

- coordinate regression loss;
- duration regression loss;
- saccade loss;
- stopping loss;
- subject-ID classification loss.

This preserves the DeepGaze3.5-VL principle that scanpath prediction is learned through native next-token prediction.

---

## 12. Free-running WHERE inference

### 12.1 Non-personalized inference

$$
(I_q,Q_q,N_q,m)
\xrightarrow[\text{free run}]{F_\theta}
\hat S_q.
$$

### 12.2 Personalized inference

For an unseen subject $u^\star$, let
$\widetilde{\mathcal C}_{u^\star}^{K}$ denote the frozen ordered support sequence from `01_FEWSHOT_SPEC.md`:

$$
(\widetilde{\mathcal C}_{u^\star}^{K},I_q,Q_q,N_q,m)
\xrightarrow[\text{free run}]{F_\theta}
\hat S_q.
$$

The query GT scanpath **content** is not part of the model input during free-running evaluation/inference. The evaluation harness may access GT only to derive the scalar oracle length `N_q` defined in Section 13 and to compute metrics after generation.

The same prompt family and scanpath serializer used during training must be used at inference.

### 12.3 Decoding policy

The canonical deterministic inference setting follows the released DeepGaze principle of greedy autoregressive decoding:

```text
temperature = 0.0
do_sample = False
```

Because SemGaze adds `<END_FIX>` and supports both XY and XYD outputs, it does **not** copy DeepGaze's fixed `16*N+16` token heuristic literally. Instead, the response budget is computed from the active tokenizer and canonical SemGaze grammar:

```text
dummy_XY(N)  = Serialize_XY([(99, 99)] repeated N times)
dummy_XYD(N) = Serialize_XYD([(99, 99, 999)] repeated N times)
response_budget = token_count(dummy_active_mode(N)) + 16
max_new_tokens = max(64, response_budget)
```

The token count must be measured using the active model tokenizer after `<END_FIX>` has been added. Generation terminates on the model/template's native EOS/assistant-response terminator or at `max_new_tokens`, whichever occurs first. The primary baseline does not add a custom stopping criterion at the `N`-th `<END_FIX>`.

If stochastic scanpath sampling is used for an experiment, its temperature, seed, and number of samples must be reported explicitly. It is an evaluation choice, not a change to the WHERE representation.

### 12.4 Output parser and validation

Canonical training targets must obey the exact serializer in Section 5. Evaluation parsing is deliberately tolerant to harmless formatting errors but remains boundary-aware.

For XY mode, an accepted fixation is any match equivalent to:

```regex
\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)\s*<END_FIX>
```

For XYD mode:

```regex
\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)\s*<END_FIX>
```

Parser rules:

1. scan left-to-right and accept only tuples of the active arity that are immediately followed by the literal `<END_FIX>` after optional whitespace;
2. outer brackets, zero-padding, and unrelated surrounding prose are **not** required for fallback coordinate extraction;
3. convert accepted integer fields to integers;
4. clip parsed `x,y` to `0...99` only as a defensive fallback; canonical targets are already in range;
5. clip parsed duration to `0...999` only as a defensive fallback;
6. retain the first `N_q` accepted fixations if generation over-produces;
7. if fewer than `N_q` fixations are accepted, flag the trajectory as invalid/failed according to `06_EVALUATION_SPEC.md`; never fabricate missing fixations.

Separately, implementations should expose a boolean `canonical_format_valid` that checks whether the **assistant content span**, excluding any native chat-template response terminator/EOS, exactly matches the zero-padded Section-5 serializer. This diagnostic must not replace the tolerant parser used for scanpath metrics.

---

## 13. Trajectory-length conditioning and termination

The requested trajectory length is explicit conditioning, following DeepGaze3.5-VL prompt construction. The primary SemGaze benchmark uses **oracle trajectory-length conditioning** rather than autonomous stopping.

### 13.1 Training length

For every training query:

```text
N_q = number of GT query fixations = len(S_q^GT)
```

Each support prompt similarly uses its observed support length:

```text
N_j = len(S_j)
```

### 13.2 Evaluation length

For canonical benchmark evaluation:

```text
N_q = len(S_q^GT)
```

Only the scalar trajectory length is exposed to the model. GT fixation coordinates, durations, and semantic annotations remain hidden from the query input. The method and paper must describe this setting explicitly as **oracle-length conditioned scanpath prediction**.

`06_EVALUATION_SPEC.md` owns the fairness check that compared baselines/metrics are evaluated under a compatible length protocol; WHERE itself does not change `N_q` to imitate another model's stopping mechanism.

### 13.3 No autonomous stopping in the primary baseline

- `N_q` appears literally in the query prompt: `Generate a scanpath of exactly {N_q} fixation points ...`.
- The primary SemGaze WHERE baseline does not learn autonomous stopping.
- There is no stopping head and no separate stopping loss.
- Search-trial termination is reflected through the oracle dataset trajectory length rather than predicted by WHERE.

Fixed-length prompting or learned EOS/stopping may be studied later as explicit alternatives, but they are outside the primary specification. Claims that the primary baseline predicts scanpath termination autonomously are therefore out of scope.

---

## 14. Relationship to shared fixation representation

`<END_FIX>` is the only SemGaze-specific modification to the DeepGaze-style scanpath language.

For each predicted or teacher-forced fixation tuple, the token sequence reaches a unique boundary token:

$$
\cdots (x_t,y_t[,d_t])\;\texttt{<END_FIX>}\cdots
$$

The Transformer hidden state associated with that boundary may be consumed as a shared fixation representation by downstream SemGaze components.

WHERE does not define:

- which Transformer layer is read;
- whether the pre- or post-token hidden state is used;
- any projection applied to that state;
- semantic heads consuming the state.

Those decisions are owned by `03_STATE_SPEC.md` and later component specs. They must not alter the canonical WHERE output grammar.

---

## 15. DeepGaze3.5-VL inheritance versus SemGaze extensions

| Component | SemGaze rule | Status relative to DeepGaze3.5-VL |
|---|---|---|
| Autoregressive VLM scanpath generation | Standard causal next-token generation | Inherited |
| Coordinate grid | `0...99`, 2-digit text | Inherited |
| Pixel-to-grid conversion | DeepGaze two-stage rounding + clip | Inherited from released code |
| Duration | 1-ms bins, `000...999` | Inherited |
| Tuple/list grammar | `(x, y)` or `(x, y, d)` in outer list | Inherited |
| `<END_FIX>` | atomic trainable token after every tuple | **SemGaze extension** |
| Requested fixation count | explicit `N` in prompt | Inherited |
| Task conditioning | natural-language prompt | Inherited principle |
| COCO visual-search wording | `while searching for a {target}` | Inherited |
| Generic arbitrary-query wording | `while performing the following task: {Q}` | **SemGaze extension** |
| Custom system prompt | none | Matches released builder |
| Few-shot construction | ordered alternating multimodal support user/assistant turns, then query user turn | Inherited chat mechanism; support membership/order are SemGaze few-shot policy |
| Same-observer few-shot personalization | same-subject ordered supports; no extra personalization text | **SemGaze personalization policy** |
| Subject-ID token | not used | Differs intentionally from DeepGaze subject-conditioning experiment |
| Separate geometric/duration head | none | Inherited principle |
| Query loss | token-level causal LM loss on final query assistant only | Inherited AR principle; **SemGaze multi-turn masking policy** |
| Oracle trajectory length | `N_q=len(S_q^GT)` at benchmark train/eval | **SemGaze benchmark policy**; DeepGaze supplies external `N` principle |
| Context budget | one tile/image, max 8192 tokens, no support truncation | **SemGaze few-shot policy** |
| Greedy standalone generation | `temperature=0.0`, `do_sample=False` | Matches released inference principle |
| Output parser | tolerant integer tuple parser + mandatory `<END_FIX>` boundary | Adapted from DeepGaze parsing; **SemGaze boundary extension** |

---

## 16. Invariants

The following conditions must hold in every primary WHERE implementation:

- Pixel coordinates use the DeepGaze two-stage rounding conversion; floor quantization is forbidden.
- Serialized spatial coordinates are zero-padded `00 ... 99`.
- Duration mode uses 1-ms values serialized as zero-padded `000 ... 999`.
- A scanpath keeps the DeepGaze-style outer list and tuple structure.
- `<END_FIX>` occurs immediately after every fixation tuple.
- `<END_FIX>` tokenizes to exactly one trainable vocabulary token, and decoding for WHERE parsing preserves it.
- Support and query use the same active output mode within one episode.
- `N_q` is explicit in the query prompt and equals the GT query trajectory length in the canonical benchmark protocol.
- Support prompt `N_j` equals the observed support scanpath length.
- WHERE receives support membership and support order from `01_FEWSHOT_SPEC.md` and preserves both exactly.
- Personalized support is rendered as alternating multimodal `user` support turns and `assistant` observed-scanpath turns, followed by one query `user` turn.
- Multi-turn chat structure does not imply one model forward per turn; the WHERE history is serialized as one causal context.
- Personalized and non-personalized prompt functions are identical for the same output mode; personalization adds context, not special wording.
- WHERE adds no custom `"same observer"` system message or subject-description text.
- WHERE uses `OpenGVLab/InternVL3_5-8B-HF` with the checkpoint's native Hugging Face chat template rather than a custom SemGaze personalization `P_sys`.
- Personalized input uses raw same-observer demonstrations, not a subject-ID token, user embedding, stitched support image, or pooled support representation.
- WHERE is generated autoregressively by the VLM decoder.
- Primary few-shot episodes use at most one 448x448 visual tile per image and an 8192-token maximum context; support membership must never be silently reduced to fit the budget. During training, overflow recovery must also preserve the current epoch query and sampled K.
- The default training loss is token-level causal LM loss on the final query assistant scanpath response only; support assistant responses are context and receive ignored labels.
- No separate coordinate, duration, saccade, personalization, or stopping loss is part of the canonical WHERE branch.
- The primary baseline uses oracle trajectory-length conditioning rather than predicting autonomous termination.
- Greedy inference uses `temperature=0`, `do_sample=False`, and a tokenizer-measured response budget rather than blindly copying DeepGaze's fixed per-fixation token heuristic.
- Evaluation parsing requires `<END_FIX>` after every accepted fixation; exact zero-padding/outer-list validity is tracked separately from tolerant coordinate extraction.

---

## 17. Cross-file ownership boundaries

The behavioral WHERE contract is frozen by this document. The following contracts are owned by other files:

- exact support/query membership, sampling, ordering, reuse, and evaluation draws: `01_FEWSHOT_SPEC.md`;
- low-level InternVL processor arguments that realize the frozen one-tile/image policy, multimodal tensor packing/collation, exact label-mask construction code, optimizer/backward schedule, and LoRA configuration: implementation adapter / `05_TRAINING_SPEC.md`;
- exact hidden-state layer/readout convention at the already-atomic `<END_FIX>` token: `03_STATE_SPEC.md`;
- primary multibranch semantic supervision: `multi/04_SEMANTIC_MULTI_SPEC.md`;
- flat single-output semantic baseline: `flat/04_SEMANTIC_SINGLE_SPEC.md`;
- fairness/reporting rules for oracle trajectory length and malformed/under-generated trajectories: `06_EVALUATION_SPEC.md`.

These ownership boundaries do not alter the WHERE definitions above.
