# SemGaze — Multi-Scale Semantic Branch Specification

## 1. Scope

This file is the canonical owner for semantic supervision of **WHAT, WHY, and HOW**.

It defines:

- the semantic target unit at each scale;
- the gold/oracle WHY-group contract;
- the exact branch state-selection indices consumed by `03_STATE_SPEC.md`;
- the canonical single-turn InternVL input layout;
- the exact WHAT/WHY/HOW prompt text;
- direct use of projected exploration states $r_t$ with **no semantic state token**;
- assistant target format and stopping contract;
- response-only autoregressive semantic objective and its differentiable coupling back to the WHERE `<END_FIX>` readout states;
- annotation invariants and normalized schema;
- invalid-annotation behavior;
- context-overflow behavior;
- offline annotation-generation prompts and validation.

It does **not** own:

- few-shot support/query sampling — `01_FEWSHOT_SPEC.md`;
- WHERE serialization or `<END_FIX>` placement — `02_WHERE_SPEC.md`;
- state extraction, projection, selected-state batching, or direct continuous-vector insertion — `03_STATE_SPEC.md`;
- branch loss weighting, optimizer behavior, LoRA/freeze policy, or training schedule — `05_TRAINING_SPEC.md`;
- semantic metrics, decoding hyperparameters, or reporting — `06_EVALUATION_SPEC.md`.

The primary semantic design is:

> three independent **single-turn natural-language generation tasks** sharing the same InternVL3.5/Qwen parameters and LM head.

There is no unified semantic JSON generation, no semantic chain `WHAT -> WHY -> HOW`, and no GROUP-prediction branch.

---

## 2. InternVL3.5-compatible formulation

InternVL3.5 uses a **ViT–MLP–LLM** architecture. Its multimodal language objective is autoregressive next-token prediction over multimodal context, and for conversation samples only response tokens contribute to the language-model loss. The SFT stage keeps the same objective and uses a 32K context window in the paper.

SemGaze therefore uses the simplest compatible semantic formulation:

1. use the checkpoint's native InternVL chat template and default system message;
2. construct one single-turn multimodal conversation per semantic target;
3. provide query image + original task/query + branch-selected projected exploration states;
4. insert the selected $r_t$ vectors directly into the LLM embedding sequence according to `03_STATE_SPEC.md`;
5. generate one direct natural-language assistant response;
6. train with response-only autoregressive next-token prediction;
7. keep the semantic path differentiable through $r_t \rightarrow P_E \rightarrow h_{\texttt{<END\_FIX>}}$ so semantic supervision shapes the WHERE readout representation.

The semantic branches do not add a separate `<END_FIX>` loss. `<END_FIX>` generation is supervised by the WHERE objective; semantic losses provide representation-level supervision through the hidden states read at those positions.

SemGaze adds no:

- semantic classifier;
- second decoder;
- branch-specific LM head;
- branch-specific state projector;
- cross-attention memory;
- `<WHAT>`, `<WHY>`, or `<HOW>` token;
- semantic state token;
- reused semantic `<END_FIX>`;
- chain-of-thought target;
- unified structured-output decoder.

The paper does **not** publish WHAT/WHY/HOW prompts. The exact prompts below are SemGaze-specific prompts designed to follow InternVL3.5's ordinary instruction/conversation SFT pattern: clear multimodal input, direct response, and no extra reasoning format.

Reference: InternVL3.5, arXiv:2508.18265, Sections 2.1–2.3.

---

## 3. Semantic targets

For a query trajectory of $N$ fixations, `03_STATE_SPEC.md` provides:

$$
R=[r_1,\ldots,r_N].
$$

### 3.1 WHAT — fixation level

There is exactly one WHAT target per fixation:

$$
W=(w_1,\ldots,w_N).
$$

Each $w_t$ describes the visible object or region inspected at fixation $t$.

$$
\boxed{
\text{one fixation}\rightarrow\text{one WHAT text}
}
$$

### 3.2 Gold WHY groups

WHY uses an oracle partition:

$$
\mathcal G^{GT}
=
(G_1^{GT},\ldots,G_M^{GT}).
$$

Each group is non-empty, groups are pairwise disjoint, and:

$$
\bigcup_{m=1}^{M}G_m^{GT}
=
\{1,\ldots,N\}.
$$

Groups may be noncontiguous. Example:

$$
G_1^{GT}=\{1,3,5\}.
$$

Within every group, fixation indices are stored and consumed in increasing chronological order.

For deterministic storage only:

$$
\min G_1^{GT}<\min G_2^{GT}<\cdots<\min G_M^{GT}.
$$

Group numbering is bookkeeping only and has no learned semantic meaning.

### 3.3 WHY — oracle-group level

There is exactly one WHY target per oracle group:

$$
Y=(y_1,\ldots,y_M).
$$

$$
\boxed{
G_m^{GT}\rightarrow\text{one WHY text }y_m
}
$$

WHY is a task-grounded description of the shared exploration/search rationale represented by that oracle group.

The model does **not** predict group membership.

### 3.4 HOW — trajectory level

There is exactly one HOW target:

$$
H=h.
$$

$$
\boxed{
\text{full trajectory}\rightarrow\text{one HOW text}
}
$$

HOW describes the overall visual exploration/search strategy across the trajectory.

### 3.5 Complete target

$$
\boxed{
A_{SEM}=(W,Y,H)
}
$$

while:

$$
\boxed{
\mathcal G^{GT}\notin A_{SEM}.
}
$$

Gold grouping is conditioning structure for WHY, not a prediction target.

---

## 4. Shared semantic-example contract

### 4.1 One target = one independent single-turn conversation

Each WHAT, WHY, or HOW target is one independent InternVL conversation:

```text
USER:
<query image>
<query task>
<branch-selected exploration-state conditioning>
<branch instruction>

ASSISTANT:
<one semantic target text>
```

Use the **native InternVL chat template and native/default system message** from the selected checkpoint.

Do not add a SemGaze-specific system prompt in the primary method.

The raw few-shot support prefix is not replayed in semantic branches. Personalization is already carried by the support-conditioned exploration states extracted from WHERE.

### 4.2 Direct-state input rule

The semantic branch consumes continuous vectors $r_t$ **directly**.

There is:

```text
NO <FIX_STATE>
NO semantic <END_FIX>
NO textual serialization of r_t
NO numeric dump of hidden vectors
```

For selected indices:

$$
J_e=(j_1,\ldots,j_{K_e}),
$$

the model-visible embedding stream contains ordinary text labels interleaved with direct vectors:

$$
[
E(\text{Fixation }j_1:),
r_{j_1},
\ldots,
E(\text{Fixation }j_{K_e}:),
r_{j_{K_e}}
].
$$

Each $r_{j_k}$ occupies one LLM sequence position and has no token ID.

Exact embedding construction is owned by `03_STATE_SPEC.md`.

### 4.3 Canonical user-side ordering

Every branch follows this logical order:

```text
<image>

Task:
{Q_q}

<ordinary exploration-state labels with direct r-vector insertions>

<branch instruction>
```

Normative rules:

- query image appears once;
- original query/task text is preserved verbatim;
- selected fixation indices use the original **1-based trajectory indices**;
- fixation labels follow selected-state chronological order;
- each continuous state is inserted immediately after its corresponding `Fixation {j}:` label;
- no group ID is shown;
- no coordinates, duration values, or hidden-state values are serialized into text;
- no semantic state special token is introduced.

### 4.4 Assistant target

The assistant response is exactly the stored semantic target text:

```text
{target_text}
```

Do not prepend or append:

```text
WHAT:
WHY:
HOW:
Answer:
JSON:
```

No semantic-specific end token is introduced.

Generation stops using the checkpoint's native assistant EOS/terminator.

Only leading/trailing whitespace may be normalized by the loader. Empty targets are invalid.

### 4.5 Context-window and overflow contract

InternVL3.5 uses a 32K-token context window during SFT in the paper, but the configured checkpoint/runtime limit is authoritative:

$$
L_{ctx}=\text{effective model context limit}.
$$

Because each direct state adds one LLM sequence position, define:

$$
L_{\mathrm{final}}
=
L_{\mathrm{native\ InternVL}}
+
K_e.
$$

`L_native InternVL` is measured after native chat serialization and image-token expansion but before exploration-state insertion.

For training, require:

$$
\boxed{
L_{\mathrm{final,prompt}}+L_{\mathrm{target}}
\leq
L_{ctx}.
}
$$

For generation/evaluation:

$$
\boxed{
L_{\mathrm{final,prompt}}+L_{\mathrm{gen\_budget}}
\leq
L_{ctx}.
}
$$

If the condition fails, raise:

```text
SemanticContextOverflowError
```

with at least:

```text
sample_id
branch = WHAT | WHY | HOW
native_prompt_length
number_of_selected_states
final_prompt_length
target_length OR generation_budget
context_limit
```

Do **not** silently:

- remove selected states;
- keep only first/last states;
- mean-pool states;
- truncate the query/task;
- truncate gold semantic targets;
- drop the sample.

Any future compression policy is a separate method.

---

## 5. WHAT branch

### 5.1 State selection

For fixation $t$:

$$
\boxed{
J_t^W=(t),
\qquad
R_t^W=[r_t].
}
$$

### 5.2 Canonical prompt

The literal user text is:

```text
<image>

Task:
{Q_q}

Exploration state:
Fixation {t}:

Describe the specific visible object or image region being inspected at this fixation. Return only one concise visual description. Do not explain why it is being inspected or describe the overall search strategy.
```

**Embedding rule:** insert $r_t$ directly into the LLM embedding sequence immediately after the ordinary text `Fixation {t}:` and before the following newline/instruction text.

The blank position in the literal text is **not** filled with a token or textual value.

### 5.3 Output

Generate exactly one WHAT response:

$$
\boxed{
P_\theta(w_t\mid I_q,Q_q,r_t).
}
$$

Generated WHAT text is not fed into WHERE, WHY, or HOW.

---

## 6. WHY branch

### 6.1 Oracle-group state selection

For:

$$
G_m^{GT}
=
\{t_1,\ldots,t_{K_m}\},
\qquad
t_1<\cdots<t_{K_m},
$$

define:

$$
\boxed{
J_m^Y=(t_1,\ldots,t_{K_m})
}
$$

and:

$$
\boxed{
R_m^Y=[r_{t_1},\ldots,r_{t_{K_m}}].
}
$$

The complete ordered selected sequence is preserved.

Do not mean-pool it.

### 6.2 Canonical prompt

For a group with members $t_1,\ldots,t_{K_m}$, the literal user text is:

```text
<image>

Task:
{Q_q}

Exploration states:
Fixation {t_1}:
Fixation {t_2}:
...
Fixation {t_K}:

Describe the single shared task-relevant search or exploration rationale represented by these selected fixation states. Return only one concise rationale. Do not list fixation indices, predict group membership, or describe the overall trajectory strategy.
```

**Embedding rule:** after each ordinary text label `Fixation {t_k}:`, insert the corresponding $r_{t_k}$ directly as one LLM embedding position before continuing to the next text span.

For example, if:

$$
G_m^{GT}=\{1,3,5\},
$$

the embedding sequence contains:

$$
[
E(\text{Fixation 1:}),r_1,
E(\text{Fixation 3:}),r_3,
E(\text{Fixation 5:}),r_5
].
$$

No group ID and no semantic state token are used.

### 6.3 Interpretation constraint

Because each $r_t$ comes from an autoregressive WHERE hidden state, it may encode preceding trajectory history outside its own oracle group.

Therefore use the wording:

> WHY is generated from the autoregressive exploration states indexed by the oracle group.

Do not claim WHY has access exclusively to the raw fixations in that group.

The WHY label is a task-grounded semantic rationale, not a verified private cognitive cause.

### 6.4 Output

Generate exactly one WHY response:

$$
\boxed{
P_\theta(y_m\mid I_q,Q_q,R_m^Y).
}
$$

The model does not output:

- group membership;
- group boundaries;
- group IDs;
- `members` lists;
- grouping JSON.

There is no grouping loss.

---

## 7. HOW branch

### 7.1 State selection

HOW selects the complete trajectory:

$$
\boxed{
J^H=(1,\ldots,N),
\qquad
R^H=[r_1,\ldots,r_N].
}
$$

Do not mean-pool the sequence.

HOW does not consume oracle WHY-group labels or generated WHAT/WHY text.

### 7.2 Canonical prompt

The literal user text is:

```text
<image>

Task:
{Q_q}

Exploration states:
Fixation 1:
Fixation 2:
...
Fixation {N}:

Describe the overall visual search or exploration strategy across the complete ordered trajectory. Return only one concise strategy description. Do not enumerate fixation-by-fixation descriptions or oracle-group memberships.
```

**Embedding rule:** insert each $r_t$ directly after its corresponding `Fixation {t}:` label, preserving chronological order.

### 7.3 Output

Generate exactly one HOW response:

$$
\boxed{
P_\theta(h\mid I_q,Q_q,R^H).
}
$$

---

## 8. Gold/oracle WHY-group contract

### 8.1 Primary path

Gold/oracle groups are supplied during semantic training and primary semantic evaluation:

$$
\boxed{
\mathcal G^{GT}
\rightarrow
\operatorname{Select}(R,\mathcal G^{GT})
\rightarrow
Y.
}
$$

Not used:

$$
R\rightarrow\hat{\mathcal G}\rightarrow Y.
$$

### 8.2 Excluded machinery

The primary method contains no:

- GROUP prediction branch;
- grouping loss;
- grouping router;
- Hungarian matching;
- group-boundary decoder;
- contiguity constraint;
- learned group-ID semantics.

Groups may be noncontiguous.

### 8.3 No predicted-state semantic path

The current semantic method operates on GT-trajectory-conditioned states from `03_STATE_SPEC.md`.

Do not map oracle groups onto a free-running predicted trajectory. A predicted trajectory may differ in length and fixation identity, so nearest-neighbor matching, interpolation, truncation, or heuristic transfer would be a new scientific decision.

---

## 9. Semantic training objective

### 9.1 Response-only autoregressive NTP

Each semantic example is trained as a standard conversation response.

For assistant target tokens:

$$
a_e=(a_{e,1},\ldots,a_{e,L_e}),
$$

use:

$$
\boxed{
\mathcal L_e^{SEM}
=
-\frac{1}{L_e}\sum_{\ell=1}^{L_e}
\log p_\theta
\left(
a_{e,\ell}
\mid
X_{SEM,e},a_{e,<\ell}
\right).
}
$$

Only assistant-response tokens contribute to this language-model loss. The scalar per-example loss uses the mean over supervised assistant-response positions, consistent with `05_TRAINING_SPEC.md`.

The following have label `-100`:

- user-side text;
- image-context positions;
- directly inserted exploration-state positions;
- native chat-template prefix/control positions not belonging to the assistant target.

This follows InternVL3.5's response-only NTP treatment for conversation samples.

### 9.2 Coupling to the trainable `<END_FIX>` readout

Semantic supervision must remain differentiable through the exploration states supplied by `03_STATE_SPEC.md`:

$$
\mathcal L_e^{SEM}
\rightarrow
R_e^{\mathrm{sel}}
\rightarrow
P_E
\rightarrow
F
\rightarrow
h_{\texttt{<END\_FIX>}}
\rightarrow
\text{WHERE language-model computation}.
$$

This means semantic supervision intentionally shapes the representation at the learned fixation boundary.

The division of responsibility is:

```text
WHERE next-token loss:
    learns to generate <END_FIX> at the correct fixation boundaries

WHAT / WHY / HOW losses:
    learn to make the hidden state at those <END_FIX> positions semantically useful
```

No additional boundary-classification loss, contrastive boundary loss, or auxiliary `L_END_FIX` is part of the primary method.

The semantic branches never emit `<END_FIX>` themselves and never receive it as a semantic prompt token. They receive only the projected continuous states derived from the WHERE `<END_FIX>` readouts.

### 9.3 Loss reduction ownership

This file defines the per-example response-token objective only.

`05_TRAINING_SPEC.md` owns:

- token reduction/normalization;
- whether to reproduce InternVL3.5 square averaging;
- WHAT/WHY/HOW aggregation;
- branch weights;
- combination with $\mathcal L_{\mathrm{WHERE}}$;
- optimizer/backward schedule.

Do not introduce classification, contrastive, group, RL, or other semantic objectives without an explicit alternative-method specification.

---

## 10. Semantic execution contract

For one query with $N$ fixations and $M$ oracle groups:

- $N$ WHAT examples;
- $M$ WHY examples;
- $1$ HOW example.

Thus:

$$
\boxed{
N_{SEM}=N+M+1.
}
$$

The state cardinality per semantic example is:

```text
WHAT -> 1 selected state
WHY  -> |G_m^GT| selected states
HOW  -> N selected states
```

WHAT, WHY, and HOW share the same InternVL3.5/Qwen parameters and LM head. They are distinct semantic examples, not distinct decoders. During joint training, all semantic examples consume non-detached states derived from the teacher-forced WHERE computation, preserving the semantic gradient path to the `<END_FIX>` readout states. Forward scheduling and any equivalent recomputation strategy are owned by `05_TRAINING_SPEC.md`.

---

## 11. Primary semantic evaluation

Primary semantic evaluation is **GT-trajectory-conditioned**:

1. teacher-force GT query scanpath through WHERE;
2. extract/project $R^{GT}$ using `03_STATE_SPEC.md`;
3. create WHAT, WHY, HOW semantic examples;
4. use oracle $G^{GT}$ only for WHY state selection;
5. generate one response per semantic example.

Semantic metrics and decoding hyperparameters are owned by `06_EVALUATION_SPEC.md`.

Generation ends at the checkpoint's native assistant EOS/terminator. No semantic stop token is added.

---

## 12. Normalized semantic annotation contract

### 12.1 Canonical logical schema

After dataset-specific adaptation:

```json
{
  "what": [
    "WHAT text for fixation 1",
    "WHAT text for fixation 2"
  ],
  "why_groups": [
    {
      "members": [1, 3, 5],
      "why": "WHY text for this oracle group"
    }
  ],
  "how": "HOW text for the complete trajectory"
}
```

This is a loader/data schema only. It is **not** the semantic model output format.

### 12.2 Index convention

Use **1-based fixation indices** in normalized annotations and prompt labels.

Python/tensor access must explicitly convert:

```text
semantic fixation index t -> tensor position t - 1
```

### 12.3 Required invariants

For trajectory length $N$:

```text
len(what) == N
```

Every WHAT text must be non-empty after trimming.

WHY groups must satisfy:

- every group non-empty;
- every member in `[1, N]`;
- member indices strictly increasing within a group;
- no duplicate member within a group;
- groups pairwise disjoint;
- union of group members exactly `{1, ..., N}`;
- groups ordered by increasing minimum member index;
- exactly one non-empty WHY text per group;
- exactly one non-empty HOW string.

### 12.4 Invalid annotations: hard fail

Validation is mandatory **before semantic-example construction**.

Any violation raises:

```text
SemanticAnnotationValidationError
```

with at least:

```text
sample_id
violated invariant
offending field/value
```

Do not silently:

- pad/truncate WHAT;
- invent missing text;
- delete empty groups;
- merge/split groups;
- sort/deduplicate invalid groups to make them appear valid;
- reassign missing/duplicated fixation indices;
- insert placeholder text;
- skip/drop the sample.

Dataset adapters may perform only deterministic documented format normalization, such as field renaming or documented 0-based -> 1-based conversion, before validation.

Semantic repair/regeneration must be an explicit offline data-preparation step.

---

## 13. Pseudo-label interpretation

Offline-generated WHAT, WHY, and HOW are task-grounded semantic annotations, not direct measurements of internal human cognition.

Preferred terminology:

- WHAT: semantic description of inspected content;
- WHY: exploration/search rationale associated with an oracle fixation group;
- HOW: overall exploration/search strategy.

Do not present WHY as a verified cognitive cause unless independently supported.

---

## 14. No unified semantic output and no causal semantic hierarchy

Primary method:

```text
WHAT:
    image + task + r_t
    -> one WHAT response

WHY:
    image + task + oracle-selected ordered r states
    -> one WHY response

HOW:
    image + task + full ordered R
    -> one HOW response
```

Not used:

```text
one semantic forward
    -> JSON
        -> WHAT
        -> predicted GROUP
        -> WHY
        -> HOW
```

There is no generated dependency:

$$
WHAT\rightarrow WHY\rightarrow HOW.
$$

The design also does not claim:

$$
WHERE\rightarrow WHAT\rightarrow WHY\rightarrow HOW
$$

as a cognitive-causal hierarchy.

The intended interpretation is:

> WHERE, WHAT, WHY, and HOW supervise a shared personalized exploration representation at different scales.

---

## 15. InternVL3.5 / HF-backbone compatibility note

The canonical implementation uses the Hugging Face-format checkpoint also selected by the DeepGaze-derived WHERE stack:

```text
OpenGVLab/InternVL3_5-8B-HF
architecture: InternVLForConditionalGeneration
```

Relevant compatibility contract:

1. WHAT/WHY/HOW use the checkpoint's native HF chat template and processor;
2. semantic examples remain ordinary response-only autoregressive conversations;
3. `03_STATE_SPEC.md` constructs continuous-state-augmented `inputs_embeds` and passes them through the top-level HF InternVL model so native image-feature replacement is preserved;
4. semantic code must not depend on the separate custom `InternVLChatModel` API;
5. semantic losses remain differentiable through the selected state vectors to the WHERE `<END_FIX>` hidden states.

SemGaze-specific additions are:

- the trainable WHERE `<END_FIX>` boundary/readout convention;
- shared projector $P_E$;
- direct insertion of projected $r_t$ vectors as LLM embedding positions;
- WHAT/WHY/HOW state-selection rules;
- oracle-group-conditioned WHY.

The semantic branches do not change the model vocabulary. `<END_FIX>` belongs to WHERE; there is no `<FIX_STATE>`, `<WHAT>`, `<WHY>`, or `<HOW>` token in the primary semantic interface.

References:

- InternVL3.5: `https://arxiv.org/pdf/2508.18265`
- HF InternVL implementation: `https://github.com/huggingface/transformers/blob/main/src/transformers/models/internvl/modeling_internvl.py`
- DeepGaze3.5-VL repository: `https://github.com/Susmit-A/DeepGaze3.5-VL`

---

## 16. Invariants

### Shared

- one independent single-turn conversation per semantic target;
- native InternVL chat template/default system message;
- same shared InternVL/Qwen parameters and LM head across branches;
- query image and original query/task present in every example;
- raw support not replayed;
- no semantic state token;
- no semantic `<END_FIX>`;
- branch-selected $r_t$ vectors are inserted directly;
- semantic losses backpropagate through $r_t$ to the WHERE `<END_FIX>` readout states;
- no separate semantic or auxiliary `L_END_FIX` is introduced;
- selected state positions have `label=-100`;
- assistant target is raw semantic text only;
- response-only autoregressive loss;
- no unified JSON model output;
- no `<WHAT>`, `<WHY>`, `<HOW>`;
- no WHAT -> WHY -> HOW generation chain;
- no chain-of-thought target.

### WHAT

- one example per fixation;
- $J_t^W=(t)$;
- one direct state position;
- one WHAT response.

### WHY

- one example per oracle group;
- oracle grouping supplied, not predicted;
- noncontiguous groups allowed;
- selected indices preserve global chronological order;
- no pooling;
- no group ID;
- no grouping loss;
- one WHY response per group.

### HOW

- one example per trajectory;
- $J^H=(1,\ldots,N)$;
- full ordered state sequence inserted directly;
- no pooling;
- WHY groups not supplied;
- one HOW response.

### Evaluation

- primary semantic evaluation is GT-trajectory-conditioned;
- oracle groups used only for WHY;
- no predicted-to-GT trajectory/group alignment.

---
