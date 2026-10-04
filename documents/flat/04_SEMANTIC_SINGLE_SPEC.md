# SemGaze — Single-Output Semantic Baseline Specification

## 1. Scope

This file is the canonical owner for the **single-output semantic VLM baseline** for **WHAT, WHY, and HOW**.

The baseline is intentionally simple:

> one query -> one semantic InternVL conversation -> one assistant response containing all WHAT, all WHY, and one HOW.

It defines:

- the shared semantic target units WHAT / WHY / HOW;
- the gold/oracle WHY-group contract used by the single-output baseline;
- the full chronological exploration-state input consumed from `03_STATE_SPEC.md`;
- the canonical single-turn InternVL input layout;
- the exact flat semantic prompt;
- direct use of projected exploration states $r_t$ with **no semantic state token**;
- deterministic gold-group rendering in the prompt;
- the flat assistant target serializer;
- the flat output parser and format-validity contract;
- response-only autoregressive semantic loss $\mathcal L_{\rm FLAT}$;
- differentiable coupling back to the WHERE `<END_FIX>` readout states;
- annotation invariants and normalized schema;
- invalid-annotation behavior;
- context-overflow behavior;
- semantic evaluation behavior;
- implementation-time fail-fast assertions.

It does **not** own:

- dataset creation or offline WHAT/WHY/HOW generation — `01_FEWSHOT_SPEC.md`;
- few-shot support/query sampling — `01_FEWSHOT_SPEC.md`;
- WHERE serialization, duration representation, or `<END_FIX>` placement — `02_WHERE_SPEC.md`;
- `<END_FIX>` state extraction, projection, or low-level direct continuous-vector insertion — `03_STATE_SPEC.md`;
- model initialization, LoRA/freeze policy, optimizer, checkpointing, or general training schedule — `05_TRAINING_SPEC.md`;
- semantic metric definitions, decoding hyperparameters, or reporting — `06_EVALUATION_SPEC.md`.

This baseline is a comparison method for the multi-branch semantic method defined separately in `04_SEMANTIC_MULTI_SPEC.md`.

The comparison boundary is intentionally narrow:

```text
SINGLE baseline:
    full R + gold groups
    -> one semantic forward
    -> one flat WHAT/WHY/HOW response

MULTI method:
    branch-selected R
    -> N WHAT + M WHY + 1 HOW semantic examples
```

Both use the same semantic annotations, gold groups, shared exploration states, backbone family, and training infrastructure unless an experiment explicitly states otherwise.

The normalized WHAT/WHY/HOW annotation invariants are shared with the primary method; `multi/04_SEMANTIC_MULTI_SPEC.md` is the canonical owner of that shared logical annotation contract. This file repeats only the mode-specific flat-input/output requirements needed to consume those annotations.

There is no GROUP-prediction branch in this baseline.

---

## 2. InternVL3.5-compatible formulation

The canonical semantic implementation uses the Hugging Face-format InternVL3.5 path selected by the rest of SemGaze:

```text
OpenGVLab/InternVL3_5-8B-HF
architecture: InternVLForConditionalGeneration
```

The single-output baseline follows the checkpoint's ordinary conversation objective:

1. use the checkpoint's native Hugging Face processor/chat template and native/default system behavior;
2. construct exactly one single-turn multimodal semantic conversation per query;
3. provide the query image and original task text;
4. provide the complete projected exploration-state sequence $R=[r_1,\ldots,r_N]$;
5. render the gold WHY groups as ordinary input-side bookkeeping text;
6. insert each $r_t$ directly into the LLM embedding sequence according to `03_STATE_SPEC.md`;
7. generate one flat natural-language assistant response containing all WHAT, WHY, and HOW targets;
8. train with standard response-only autoregressive next-token prediction;
9. preserve gradient flow through $R\rightarrow P_E\rightarrow F\rightarrow h_{\texttt{<END\_FIX>}}$.

The baseline adds no:

- semantic classifier;
- second decoder;
- branch-specific decoder;
- branch-specific LM head;
- branch-specific projector;
- grouping head;
- grouping router;
- structured decoder;
- constrained JSON decoder;
- `<WHAT>`, `<WHY>`, or `<HOW>` token;
- semantic state token;
- semantic `<END_FIX>`;
- chain-of-thought target.

`WHAT`, `WHY`, and `HOW` in the assistant response are ordinary text prefixes, not newly added vocabulary tokens.

---

## 3. Semantic targets

For a query trajectory of $N$ fixations, `03_STATE_SPEC.md` provides:

$$
R=[r_1,\ldots,r_N].
$$

The semantic annotation provides:

$$
W=(w_1,\ldots,w_N),
$$

$$
\mathcal G^{GT}=(G_1^{GT},\ldots,G_M^{GT}),
$$

$$
Y=(y_1,\ldots,y_M),
$$

and:

$$
H=h.
$$

### 3.1 WHAT — fixation level

There is exactly one WHAT target per fixation:

$$
\boxed{
\text{fixation }t\rightarrow w_t
}
$$

Each $w_t$ is a concise task-grounded description of the visible object or image region inspected at fixation $t$.

### 3.2 Gold WHY groups

WHY uses an oracle partition:

$$
\mathcal G^{GT}
=
(G_1^{GT},\ldots,G_M^{GT}).
$$

Every group is non-empty, groups are pairwise disjoint, and:

$$
\bigcup_{m=1}^{M}G_m^{GT}
=
\{1,\ldots,N\}.
$$

Groups may be noncontiguous. Example:

$$
G_1^{GT}=\{1,3,5\}.
$$

Within each group, member indices are strictly increasing.

For deterministic storage/rendering only:

$$
\min G_1^{GT}<\min G_2^{GT}<\cdots<\min G_M^{GT}.
$$

Group numbering is bookkeeping only and carries no learned semantic meaning.

### 3.3 WHY — oracle-group level

There is exactly one WHY target per oracle group:

$$
\boxed{
G_m^{GT}\rightarrow y_m
}
$$

Each $y_m$ is a concise task-grounded search/exploration rationale associated with the autoregressive exploration states indexed by $G_m^{GT}$.

The baseline does **not** predict group membership.

### 3.4 HOW — trajectory level

There is exactly one HOW target:

$$
\boxed{
\text{full trajectory}\rightarrow h
}
$$

HOW describes the overall visual search/exploration strategy across the complete ordered trajectory.

### 3.5 Complete semantic text target

The semantic text target is:

$$
A_{SEM}=(W,Y,H).
$$

while:

$$
\mathcal G^{GT}\notin A_{SEM}.
$$

Gold groups are input-side conditioning/alignment information. They are not generated targets.

---

## 4. Single semantic-example contract

### 4.1 Exactly one semantic conversation per query

For each query, construct exactly one native InternVL single-turn conversation:

```text
USER:
<query image>
<query task>
<full chronological exploration-state sequence>
<gold WHY-group bookkeeping>
<single flat-output instruction>

ASSISTANT:
<all WHAT lines>
<all WHY lines>
<one HOW line>
```

The raw few-shot support conversation is **not replayed** in this semantic forward.

Personalization reaches the semantic baseline through the support-conditioned WHERE states:

$$
\mathcal C_u^K
\rightarrow
\text{WHERE}
\rightarrow
F
\rightarrow
P_E
\rightarrow
R
\rightarrow
\text{single semantic forward}.
$$

### 4.2 Full-state input rule

The single-output baseline always consumes the complete chronological state sequence:

$$
R=[r_1,\ldots,r_N].
$$

There is no branch-level state selection in this baseline.

Every fixation state appears exactly once in the semantic prompt.

### 4.3 Direct continuous-state insertion

The model-visible embedding stream contains ordinary fixation labels interleaved with direct vectors:

$$
[
E(\text{Fixation 1:}),r_1,
E(\text{Fixation 2:}),r_2,
\ldots,
E(\text{Fixation N:}),r_N
].
$$

Each $r_t$ occupies exactly one LLM sequence position and has no token ID.

There is:

```text
NO <FIX_STATE>
NO semantic <END_FIX>
NO textual serialization of r_t
NO numeric dump of hidden vectors
NO state pooling
```

Every inserted state position has:

```text
attention_mask = 1
label          = -100
```

Direct insertion must preserve autograd to:

$$
R\rightarrow P_E\rightarrow F\rightarrow h_{\texttt{<END\_FIX>}}.
$$

Exact low-level embedding construction is owned by `03_STATE_SPEC.md`.

### 4.4 Query-side ordering

The logical user-side ordering is frozen as:

```text
<image>

Task:
{Q_q}

Exploration states:
Fixation 1:
<r_1 inserted directly>
Fixation 2:
<r_2 inserted directly>
...
Fixation N:
<r_N inserted directly>

Gold WHY groups:
Group 1: fixations {...}
Group 2: fixations {...}
...
Group M: fixations {...}

<flat semantic instruction>
```

Normative rules:

- query image appears exactly once;
- original query/task text is preserved verbatim;
- fixation labels use normalized **1-based trajectory indices**;
- fixation labels are chronological `1..N`;
- each $r_t$ is inserted immediately after its matching `Fixation {t}:` label;
- all states appear before the gold-group bookkeeping block;
- group members are rendered using the same normalized 1-based indices;
- groups are rendered in canonical group order;
- no coordinates, duration values, hidden-state values, or `<END_FIX>` tokens are serialized into semantic text;
- no raw support turn is replayed.

---

## 5. Gold-group input contract

### 5.1 Gold groups are supplied at training and semantic evaluation

The single-output baseline receives the same oracle partition at semantic training and primary semantic inference/evaluation:

$$
\mathcal G^{GT}
\rightarrow
X_{SEM}^{single}.
$$

The baseline never predicts:

$$
R\rightarrow\hat{\mathcal G}.
$$

### 5.2 Canonical group rendering

For:

$$
G_m^{GT}=\{t_1<\cdots<t_{K_m}\},
$$

render exactly one ordinary-text line:

```text
Group {m}: fixations {t_1}, {t_2}, ..., {t_Km}
```

Examples:

```text
Group 1: fixations 1, 3, 5
Group 2: fixations 2, 4
```

Rules:

- group number `m` is deterministic bookkeeping only;
- member indices are increasing;
- no brackets, JSON, Python lists, or nested object syntax are required;
- group membership is input-side metadata;
- group members are not repeated in the assistant target;
- the model must not be asked to rewrite, merge, split, or predict groups.

This rendering gives the single-output baseline an explicit deterministic mapping between `WHY m` and the corresponding oracle group without adding a grouping task.

### 5.3 No grouping machinery

The baseline contains no:

- GROUP prediction branch;
- grouping loss;
- grouping classifier;
- learned group ID;
- Hungarian matching;
- group-boundary decoder;
- contiguity constraint;
- post-hoc group repair.

Groups may be noncontiguous.

---

## 6. Canonical single-output prompt

### 6.1 Literal semantic instruction

After the query image, task, full state sequence, and gold-group bookkeeping are rendered, append the following instruction:

```text
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

The placeholders `{N}` and `{M}` are filled from the validated semantic annotation.

### 6.2 Complete logical prompt

The complete logical user message is therefore:

```text
<image>

Task:
{Q_q}

Exploration states:
Fixation 1:
<r_1 inserted directly>
Fixation 2:
<r_2 inserted directly>
...
Fixation {N}:
<r_N inserted directly>

Gold WHY groups:
Group 1: fixations {members of G_1^GT}
Group 2: fixations {members of G_2^GT}
...
Group {M}: fixations {members of G_M^GT}

<instruction from Section 6.1>
```

Use the checkpoint's native Hugging Face chat template and native/default system behavior.

Do not add a SemGaze-specific system prompt.

### 6.3 Interpretation constraint

Because each $r_t$ is read from an autoregressive WHERE hidden state, it may encode trajectory history preceding fixation $t$.

Therefore use the interpretation:

> WHY is generated from the full autoregressive exploration-state sequence together with explicit oracle-group membership.

Do not claim that WHY has access only to the raw fixation observations inside its group.

WHY is a task-grounded semantic rationale, not a verified private cognitive cause.

---

## 7. Flat assistant target

### 7.1 Line-safety normalization

The single-output target requires exactly one semantic item per output line.

Define deterministic whitespace flattening:

```python
def flatten_text(s: str) -> str:
    return " ".join(s.strip().split())
```

This normalization:

- removes leading/trailing whitespace;
- collapses internal whitespace/newlines to single spaces;
- does not rewrite the persisted annotation artifact;
- is applied only when constructing the single-output target and exact comparison text.

An empty result after normalization is invalid.

### 7.2 Canonical assistant target

The assistant target is exactly:

```text
WHAT 1: {flatten_text(w_1)}
WHAT 2: {flatten_text(w_2)}
...
WHAT N: {flatten_text(w_N)}
WHY 1: {flatten_text(y_1)}
WHY 2: {flatten_text(y_2)}
...
WHY M: {flatten_text(y_M)}
HOW: {flatten_text(h)}
```

Formally:

$$
A_{flat}
=
\operatorname{Join}_{\texttt{"\\n"}}
\left(
[\text{WHAT lines}],
[\text{WHY lines}],
[\text{HOW line}]
\right).
$$

There are exactly:

```text
N WHAT lines
M WHY lines
1 HOW line
```

for a total of:

$$
N+M+1
$$

output lines.

### 7.3 Output restrictions

The assistant target contains no:

- group member list;
- JSON object;
- JSON array;
- Python list/dictionary;
- nested semantic object;
- Markdown bullet list;
- code fence;
- chain-of-thought text;
- additional commentary.

No semantic-specific end token is added.

Generation terminates with the checkpoint's native assistant EOS/terminator.

---

## 8. Flat output parser

### 8.1 Parser inputs

The parser receives:

```text
generated_text
N = expected number of WHAT items
M = expected number of WHY items
```

`N` and `M` come from the GT trajectory and validated gold-group annotation in the primary semantic evaluation setting.

### 8.2 Recognized lines

Recognize lines equivalent to:

```regex
^WHAT\s+(\d+)\s*:\s*(.+?)\s*$
^WHY\s+(\d+)\s*:\s*(.+?)\s*$
^HOW\s*:\s*(.+?)\s*$
```

Canonical generation is case-sensitive.

The parser may tolerate surrounding whitespace around separators as shown above.

### 8.3 Parsing algorithm

1. normalize line endings to `\n`;
2. split into lines;
3. scan top-to-bottom;
4. accept `WHAT t` only if `1 <= t <= N`;
5. accept `WHY m` only if `1 <= m <= M`;
6. accept at most one value for each WHAT index;
7. accept at most one value for each WHY index;
8. accept at most one HOW value;
9. trim recognized value text;
10. never fabricate, renumber, merge, or infer missing values.

### 8.4 Format validity

Set:

```text
flat_format_valid = true
```

if and only if all of the following hold:

- exactly `N` WHAT lines are present;
- WHAT indices are exactly `1..N` in order;
- exactly `M` WHY lines are present;
- WHY indices are exactly `1..M` in order;
- exactly one HOW line is present after all WHY lines;
- every extracted text is non-empty;
- there are no duplicate WHAT/WHY indices;
- there are no out-of-range indices;
- there is no extra unrecognized prose or additional non-empty line.

Otherwise:

```text
flat_format_valid = false
```

Recognized fields may still be returned for diagnostic semantic scoring where permitted by `06_EVALUATION_SPEC.md`.

### 8.5 Parser output

Expose at least:

```text
flat_format_valid: bool
what: dict[int, str]
why: dict[int, str]
how: str | null
errors: list[str]
```

The parser must be deterministic and rule-based.

Do not use another VLM/LLM to repair or interpret malformed output.

---

## 9. Semantic training objective

### 9.1 Response-only autoregressive NTP

The single-output baseline trains on one assistant response per query.

Let the flat assistant token sequence be:

$$
a^{flat}=(a_1,\ldots,a_L).
$$

Let $\Omega_{flat}$ contain only assistant-response target positions.

The semantic baseline loss is the mean response-token negative log-likelihood:

$$
\boxed{
\mathcal L_{FLAT}
=
-\frac{1}{|\Omega_{flat}|}
\sum_{\ell\in\Omega_{flat}}
\log p_\theta
\left(
a_\ell
\mid
X_{SEM}^{single},a_{<\ell}
\right)
}
$$

The following positions use `label=-100`:

- user-side text;
- image/context positions;
- direct exploration-state positions;
- native chat-template prefix/control positions not belonging to the assistant target.

### 9.2 No semantic scale balancing inside the baseline

The single-output baseline intentionally uses an ordinary one-response VLM objective.

There is no separate:

```text
L_WHAT
L_WHY
L_HOW
branch averaging
branch balancing
scale-token weighting
L_STRUCT
L_GROUP
```

WHAT/WHY/HOW receive whatever token weighting naturally follows from their occurrence inside the one flattened assistant target.

This is part of the baseline definition, not an implementation accident.

### 9.3 Joint objective handoff

For a joint WHERE + single-output baseline run:

$$
\boxed{
\mathcal L_{total}^{single}
=
\lambda_{WHERE}\mathcal L_{WHERE}
+
\lambda_{SEM}\mathcal L_{FLAT}
}
$$

Use the same top-level `lambda_WHERE` and `lambda_SEM` values as the compared multi-branch experiment unless an experiment explicitly changes them.

Global optimization scheduling, optimizer construction, and backward execution are owned by `05_TRAINING_SPEC.md`.

### 9.4 Semantic gradient path

$\mathcal L_{FLAT}$ must remain differentiable through:

$$
\mathcal L_{FLAT}
\rightarrow
R
\rightarrow
P_E
\rightarrow
F
\rightarrow
h_{\texttt{<END\_FIX>}}
\rightarrow
\text{WHERE language-model computation}.
$$

Do not detach `F` or `R` before semantic supervision.

The baseline adds no separate boundary loss. `<END_FIX>` generation remains supervised by WHERE.

---

## 10. Semantic execution contract

For one query with $N$ fixations and $M$ oracle groups, construct exactly:

```text
1 semantic example
1 semantic assistant target
1 semantic forward
1 L_FLAT
```

Conceptually:

```text
query image + task
        +
full R = [r_1, ..., r_N]
        +
gold groups G^GT
        ↓
one semantic InternVL conversation
        ↓
one flat WHAT/WHY/HOW response
        ↓
L_FLAT
```

There is no semantic-example fan-out into fixation/group units.

All WHAT/WHY/HOW text is generated autoregressively inside the same assistant response.

This serialization ordering is an implementation format only. It is not a cognitive or causal hierarchy.

---

## 11. Context-window and overflow contract

Let:

$$
L_{ctx}=\text{configured semantic model context limit}.
$$

The configured runtime limit is authoritative.

The single-output baseline inserts exactly `N` exploration states.

Because each inserted state occupies one LLM sequence position:

$$
L_{final}
=
L_{native\ InternVL}+N.
$$

`L_native InternVL` is measured after native chat serialization and image-token expansion but before exploration-state insertion.

### 11.1 Training

Require:

$$
\boxed{
L_{final,prompt}+L_{target}
\leq
L_{ctx}
}
$$

where `L_target` is the token length of the complete flattened assistant target.

### 11.2 Generation/evaluation

Require:

$$
\boxed{
L_{final,prompt}+L_{gen\_budget}
\leq
L_{ctx}
}
$$

### 11.3 Overflow behavior

If the condition fails, raise:

```text
SemanticContextOverflowError
```

with at least:

```text
sample_id
semantic_mode = flat_single_output
native_prompt_length
number_of_inserted_states = N
number_of_gold_groups = M
final_prompt_length
target_length OR generation_budget
context_limit
```

Do **not** silently:

- remove states;
- keep only first/last states;
- mean-pool states;
- truncate the query/task;
- truncate the gold-group bookkeeping block;
- truncate semantic targets;
- remove WHAT/WHY lines;
- shorten `N` or `M`;
- drop the sample.

Any future compression policy is a separate method.

---

## 12. Primary semantic evaluation

The primary single-output semantic evaluation is **GT-trajectory-conditioned and gold-group-conditioned**.

Procedure:

1. teacher-force the GT query scanpath through WHERE;
2. extract one final-query `<END_FIX>` state per GT fixation;
3. project to:

$$
R^{GT}=[r_1^{GT},\ldots,r_N^{GT}];
$$

4. load the validated gold groups $\mathcal G^{GT}$;
5. construct one semantic generation prefix using the query image, task, full $R^{GT}$, and gold groups;
6. generate one assistant response;
7. parse WHAT 1..N, WHY 1..M, and HOW using Section 8;
8. score semantic text using metrics owned by `06_EVALUATION_SPEC.md`;
9. report flat-format validity separately where required.

There is no predicted-to-GT group matching because groups are supplied as oracle input.

There is no GROUP prediction metric.

### 12.1 No predicted-state semantic path

This file does not define mapping gold groups onto a free-running predicted trajectory.

Do not invent:

- nearest-fixation matching;
- interpolation;
- heuristic truncation;
- index transfer;
- predicted-to-GT alignment.

Such behavior would be a separate experiment.

---

## 13. Fair-comparison contract with the multi-branch method

The single-output baseline and `04_SEMANTIC_MULTI_SPEC.md` must use the same:

```text
curated dataset records
train / validation / test membership
few-shot support/query episodes
teacher-forced WHERE query trajectory for semantic training/evaluation
query image
query task text
projected exploration states R
shared projector P_E
gold/oracle WHY groups
WHAT annotation texts
WHY annotation texts
HOW annotation text
InternVL checkpoint
native image preprocessing
LoRA target modules
trainable <END_FIX> row policy
optimizer family
learning-rate schedule
semantic context limit
WHERE loss definition
semantic metric definitions
```

The intentional comparison difference is:

```text
SINGLE:
    full R + G^GT
    -> one semantic forward
    -> one flat response
    -> standard response-token L_FLAT

MULTI:
    branch-selected R
    -> N WHAT + M WHY + 1 HOW examples
    -> branch-balanced semantic objective
```

The single-output baseline must not receive:

- raw support turns in the semantic prompt;
- GT coordinates/durations serialized as semantic text;
- extra object labels;
- extra captions;
- extra annotations unavailable to the multi-branch method;
- extra model heads or structured decoders.

The baseline is allowed to see the complete `R` sequence and explicit gold-group bookkeeping because this is the minimum direct input needed to generate all WHAT/WHY/HOW in one semantic response without adding grouping prediction.

---

## 14. Shared semantic annotation contract

`flat_single_output` consumes the same normalized WHAT/WHY/HOW annotations and the same 1-based gold WHY groups as the primary multibranch method. The canonical logical schema, index convention, invariants, and hard-fail validation behavior are owned by `multi/04_SEMANTIC_MULTI_SPEC.md`.

The flat baseline does not add, remove, repair, reorder, merge, split, or reinterpret annotation fields. In particular, it requires exactly `N` non-empty WHAT texts, a complete disjoint gold-group partition of `{1, ..., N}` with one non-empty WHY text per group, and one non-empty HOW string.

## 15. Pseudo-label interpretation

Offline-generated WHAT, WHY, and HOW are task-grounded semantic annotations, not direct measurements of internal human cognition.

Preferred terminology:

- WHAT: semantic description of inspected content;
- WHY: exploration/search rationale associated with an oracle fixation group;
- HOW: overall exploration/search strategy.

Do not present WHY as a verified cognitive cause unless independently supported.

The line order:

```text
WHAT ...
WHY ...
HOW ...
```

is an output serialization convention, not a cognitive-causal claim. Because the baseline generates one autoregressive assistant response, later response tokens are computationally conditioned on the earlier assistant-response prefix: WHY tokens follow the WHAT prefix, and HOW tokens follow the WHAT/WHY prefix. This textual autoregressive dependency is part of the flat baseline and must not be reinterpreted as a human cognitive hierarchy.

It does not assert:

$$
WHAT\rightarrow WHY\rightarrow HOW
$$

or:

$$
WHERE\rightarrow WHAT\rightarrow WHY\rightarrow HOW
$$

as a cognitive-causal hierarchy.

---

## 16. Excluded structured-output machinery

The single-output baseline is deliberately **flat**, not structured/nested.

It does not use:

- JSON output;
- nested Python/list/dictionary output;
- semantic object schema decoding;
- `members` output;
- group-ID prediction;
- constrained decoding;
- JSON schema validation;
- JSON repair;
- post-hoc LLM output repair;
- a semantic parser learned by another model.

The ordinary line prefixes:

```text
WHAT 1:
WHY 1:
HOW:
```

exist only to make the one response deterministically parseable.

They are not intended as a learned structured reasoning hierarchy.

---

## 17. InternVL3.5 / HF-backbone compatibility note

The canonical implementation uses:

```text
OpenGVLab/InternVL3_5-8B-HF
architecture: InternVLForConditionalGeneration
```

Relevant compatibility contract:

1. use the checkpoint's native HF processor/chat template;
2. construct one ordinary single-turn semantic conversation per query;
3. use the top-level HF InternVL path with continuous-state-augmented `inputs_embeds`;
4. preserve native image-feature replacement/fusion;
5. do not depend on the separate custom `InternVLChatModel` API;
6. preserve autograd from $\mathcal L_{FLAT}$ through all inserted state vectors;
7. use the same shared InternVL/Qwen parameters and LM head as the rest of SemGaze.

SemGaze-specific pieces reused by the baseline are:

- trainable WHERE `<END_FIX>` boundary/readout convention;
- shared projector $P_E$;
- direct continuous-state insertion;
- gold-group semantic supervision.

The semantic baseline does not add vocabulary tokens.

References:

- InternVL3.5: `https://arxiv.org/pdf/2508.18265`
- HF InternVL implementation: `https://github.com/huggingface/transformers/blob/main/src/transformers/models/internvl/modeling_internvl.py`
- DeepGaze3.5-VL repository: `https://github.com/Susmit-A/DeepGaze3.5-VL`

---

## 18. Fail-fast assertions

Implementation must stop rather than guess or silently repair if any of the following occurs:

- semantic annotation validation fails;
- `len(R) != N`;
- the extracted/projected state order is not chronological;
- gold groups do not form the validated oracle partition;
- the number of rendered `Fixation {t}:` insertion boundaries differs from `N`;
- an insertion boundary receives the wrong $r_t$;
- a padded state is inserted as a real semantic sequence position;
- direct state insertion detaches `R`;
- the prompt renders fewer or more than `M` gold-group lines;
- the flat target serializer produces other than exactly `N + M + 1` semantic lines;
- an input annotation text becomes empty after `flatten_text` normalization;
- semantic context overflow occurs;
- raw few-shot support is replayed in the semantic conversation;
- coordinates/durations are serialized into the semantic prompt without an explicit alternative experiment;
- the baseline predicts or trains group membership;
- JSON/structured-output code is invoked for this baseline;
- more than one semantic conversation is constructed for one query;
- `L_FLAT` is replaced silently by per-branch or scale-balanced semantic loss.

---

## 19. Invariants

### 19.1 Shared semantic information

- exactly `N` WHAT targets;
- exactly `M` WHY targets;
- exactly one HOW target;
- gold groups form a complete disjoint partition;
- noncontiguous groups are allowed;
- gold grouping is supplied, not predicted;
- normalized fixation indices are 1-based.

### 19.2 Single-output execution

- exactly one semantic conversation per query;
- exactly one semantic model forward per query at the method level;
- full chronological `R` is inserted once;
- each $r_t$ occupies one direct embedding position;
- no state pooling;
- gold groups are rendered once as input-side bookkeeping text;
- exactly one flat assistant target/response is produced;
- line order is all WHAT, then all WHY, then HOW;
- no group members are generated;
- no JSON or nested structured output;
- no semantic special token.

### 19.3 Training

- semantic supervision uses response-only autoregressive NTP;
- non-assistant positions use `label=-100`;
- one mean response-token $\mathcal L_{FLAT}$ is computed;
- there is no WHAT/WHY/HOW branch balancing inside the baseline;
- semantic gradients flow through all inserted states to $P_E$ and the WHERE readout computation;
- there is no auxiliary semantic `<END_FIX>` loss.

### 19.4 Evaluation

- primary semantic evaluation is GT-trajectory-conditioned;
- gold groups are supplied during semantic evaluation;
- one assistant response is generated and parsed;
- no predicted-to-GT trajectory/group alignment is part of this baseline;
- no group prediction metric is required;
- flat-format validity is separate from semantic text quality.

---

## 20. Explicitly excluded alternatives

The following are outside this baseline unless introduced as separately named experiments:

- multi-branch semantic execution;
- JSON semantic output;
- nested Python/list/dictionary output;
- constrained schema decoding;
- post-hoc LLM format repair;
- predicted GROUP membership;
- grouping loss;
- semantic classifier heads;
- branch-specific LM heads;
- branch-specific projectors;
- cross-attention semantic memory;
- state mean pooling;
- raw support replay in semantic prompts;
- textual coordinate/duration replay in semantic prompts;
- predicted-to-GT trajectory alignment;
- semantic chain-of-thought targets.

The baseline is intentionally a plain single-response VLM semantic generation baseline.
