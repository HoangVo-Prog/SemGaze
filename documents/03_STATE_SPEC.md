# SemGaze — Shared Exploration State Specification

## 1. Scope

This file is the canonical owner for the low-level exploration-state interface shared by semantic modes: fixation-state extraction, `<END_FIX>` readout, chronological state sequence, gradient flow, the shared projector, and the **direct continuous-state insertion primitive for the InternVL language-model embedding stream**.

The scanpath serializer, `<END_FIX>` placement, and query-only WHERE target mask are owned by `02_WHERE_SPEC.md`. `multi/04_SEMANTIC_MULTI_SPEC.md` owns multibranch state selection, prompts, and targets. `flat/04_SEMANTIC_SINGLE_SPEC.md` owns full-trajectory consumption, flat serialization/parsing, and flat semantic loss. The full optimization schedule for the primary multibranch method, shared optimizer behavior, LoRA/freeze policy, checkpointing, and low-level batching/collation strategy are owned by `05_TRAINING_SPEC.md`. This file freezes only the **state-level requirement** that `<END_FIX>` is a learned WHERE vocabulary token whose query occurrences are supervised by the ordinary WHERE next-token objective and whose hidden states remain differentiable inputs to semantic execution.

The shared state path is:

$$
\text{teacher-forced WHERE query}
\rightarrow
F
\rightarrow
P_E
\rightarrow
R.
$$

`03_STATE_SPEC.md` owns this extraction/projection path and the low-level primitive for inserting an ordered state sequence into the semantic embedding stream. `multi/04_SEMANTIC_MULTI_SPEC.md` owns selected-subsequence consumption for `multibranch`; `flat/04_SEMANTIC_SINGLE_SPEC.md` owns full chronological `R` consumption for `flat_single_output`. Both modes use the same `P_E` and the same projected representation; no second projector or second state representation is introduced.

The key design rule is:

> `<END_FIX>` has two roles inside the WHERE computation: it is a **trainable autoregressive boundary token** and the **readout position** for one fixation state.  
> Semantic execution does not generate or replay `<END_FIX>`; it consumes projected readout vectors $r_t$ **directly**, with mode-specific consumption owned by the corresponding semantic spec.

Thus `<END_FIX>` is not frozen and is not merely a passive marker. Query `<END_FIX>` occurrences receive the ordinary WHERE next-token cross-entropy, while semantic losses backpropagate through the hidden states read at those positions. There is no semantic state token such as `<FIX_STATE>`, and `<END_FIX>` is not reused inside semantic prompts.

---

## 2. Fixation-state definition

During the teacher-forced WHERE pass, run the WHERE model with `output_hidden_states=True` and take the **final language-model layer** hidden state at every query `<END_FIX>` position:

$$
f_t
=
h_{\operatorname{END\_FIX}(t)}.
$$

Equivalently:

$$
f_t
=
F_\theta
\left(
X_{\mathrm{WHERE}},
S_{q,\leq t}^{GT}
\right)_{\operatorname{END\_FIX}(t)}.
$$

Collect the states in chronological fixation order:

$$
F=[f_1,\ldots,f_N].
$$

Each $f_t$ is a **demonstration-conditioned exploration state**, not a purely spatial feature. It is conditioned on:

- the ordered same-subject support context;
- the query image;
- the query task;
- the requested trajectory length;
- the teacher-forced query scanpath prefix through fixation $t$.

The primary training and semantic-evaluation path uses these **teacher-forced GT-query states**.

---

## 3. Exact `<END_FIX>` readout contract

Let the complete teacher-forced WHERE sequence for one sample contain both support assistant scanpaths and the final query assistant scanpath. Because support turns also contain `<END_FIX>`, extraction must be restricted to the **final query assistant span**.

Reuse the query-only supervision mask required by `02_WHERE_SPEC.md`:

```text
query_assistant_mask = (labels != -100)
```

For sample $b$, valid fixation positions are exactly:

```text
(input_ids[b] == END_FIX_TOKEN_ID)
AND query_assistant_mask[b]
```

sorted by increasing token position.

The implementation must assert:

```text
number of selected query <END_FIX> positions == N_b
```

where $N_b$ is the GT query fixation count used by WHERE.

Conceptually:

```python
H = outputs.hidden_states[-1]   # [B, L, d_model]
pos_b = query END_FIX positions for sample b
F_b = H[b, pos_b, :]            # [N_b, d_model]
```

Do not read support `<END_FIX>` states into $F_b`.

Do not use an earlier Transformer layer in the primary method.

After $f_t$ is extracted, `<END_FIX>` has completed its **model-visible** role for the semantic path. It is not serialized again for WHAT, WHY, or HOW; however, semantic gradients still flow backward through $f_t$ to the WHERE computation and therefore may update the trainable `<END_FIX>` input embedding together with the adapted language-model parameters.

### 3.1 `<END_FIX>` learning contract

`<END_FIX>` is learned through the ordinary WHERE language-model objective; there is **no separate** `L_END_FIX`.

For the final query assistant response, every ground-truth fixation contributes one supervised boundary token:

```text
(...fixation tuple...) <END_FIX>
```

Therefore, for a query with $N_b$ fixations:

```text
number of supervised query <END_FIX> labels == N_b
```

Historical/support assistant turns remain in the causal input as few-shot demonstrations but are context-only under the canonical query-only WHERE mask:

```text
support <END_FIX> positions: label = -100
query   <END_FIX> positions: label = END_FIX_TOKEN_ID
```

If the training framework realizes this policy with a `mask_history`-style option, that option may mask **historical assistant labels only**. It must not delete, truncate, reorder, or otherwise remove the historical support turns from `input_ids` / the multimodal causal context.

The learning signals are complementary:

$$
\mathcal L_{\mathrm{WHERE}}
\rightarrow
\text{probability of generating }\texttt{<END\_FIX>},
$$

while semantic supervision from the active mode backpropagates through the hidden state at the same readout position:

$$
h_{\texttt{<END\_FIX>}}
\rightarrow
\text{WHERE language-model computation}.
$$

For `multibranch`, the mode-specific semantic objective is the branch-balanced loss defined in `05_TRAINING_SPEC.md`; the flat-baseline semantic loss is owned by `flat/04_SEMANTIC_SINGLE_SPEC.md`.

Thus `<END_FIX>` is trained both as a predictable boundary symbol and as the location whose hidden state must support downstream semantic generation.

### 3.2 Trainable-token requirement under LoRA

`<END_FIX>` must remain trainable even when the pretrained backbone is otherwise frozen except for LoRA adapters. The state-level contract is:

- `<END_FIX>` is one atomic tokenizer/model vocabulary entry;
- its input embedding parameter is trainable;
- if input and output token weights are untied, its output/logit parameter is also trainable;
- if those weights are tied, the shared parameter is trainable;
- the required `<END_FIX>` parameter state is preserved by checkpoint save/restore;
- semantic gradients may reach the trainable `<END_FIX>` parameter through the hidden state read at its query position.

The concrete PEFT mechanism used to satisfy this contract is owned by `05_TRAINING_SPEC.md` and is not part of the state definition.

---

## 4. Full-trajectory batched state representation

Trajectory lengths may differ across samples. The canonical WHERE-to-state interface therefore returns a padded tensor plus a boolean mask.

For batch size $B$:

$$
N_{\max}=\max_b N_b.
$$

Return:

$$
F^{\mathrm{batch}}
\in
\mathbb R^{B\times N_{\max}\times d_{\mathrm{model}}},
$$

and:

$$
M^{\mathrm{fix}}
\in
\{0,1\}^{B\times N_{\max}}.
$$

For each sample $b$:

```text
F_batch[b, :N_b] = [f_1, ..., f_Nb]
M_fix[b, :N_b]   = true
M_fix[b, N_b:]   = false
```

Padding values are implementation-only and must never be selected as real semantic states.

Chronological fixation order is preserved exactly.

---

## 5. Gradient requirement

Do **not** detach $F$ or $R$ before semantic supervision.

The intended gradient requirement is that the active mode's semantic loss remains differentiable through the state sequence supplied to that mode, then through:

$$
R
\rightarrow
F
\rightarrow
\theta.
$$

For `multibranch`, the supplied sequence may be a selected subsequence of `R`; for `flat_single_output`, it is the full chronological `R`. The exact multibranch loss/consumption rules are owned by `multi/04_SEMANTIC_MULTI_SPEC.md` and `05_TRAINING_SPEC.md`; the flat loss/consumption rules are owned by `flat/04_SEMANTIC_SINGLE_SPEC.md`.

Therefore semantic supervision may update the same scanpath-generating representation that reads the few-shot support context. In particular, because $f_t$ is the hidden state **at the `<END_FIX>` input position**, semantic supervision may update the trainable `<END_FIX>` input embedding through this path. The `<END_FIX>` output/logit row is trained by $\mathcal L_{\mathrm{WHERE}}$, not by the semantic branch directly.

The state-selection operation and direct embedding insertion must remain differentiable. No `.detach()`, NumPy conversion, CPU round-trip, serialization to text, or other graph-breaking operation is permitted in the primary training path.

---

## 6. Shared projector

Every fixation state uses the same projector:

$$
r_t
=
P_E(f_t)
=
\operatorname{LN}(W_Ef_t+b_E).
$$

Let the InternVL language-model hidden size be:

$$
d=d_{\mathrm{model}}.
$$

Then:

$$
f_t\in\mathbb R^d,
\qquad
W_E\in\mathbb R^{d\times d},
\qquad
b_E\in\mathbb R^d,
\qquad
r_t\in\mathbb R^d.
$$

Implementation form:

```python
P_E = nn.Sequential(
    nn.Linear(d_model, d_model),
    nn.LayerNorm(d_model),
)
```

Use framework-default initialization for `nn.Linear` and `nn.LayerNorm`; no custom initialization is part of the primary method.

`P_E` is always trainable in the primary method and must be included in the optimizer even when pretrained InternVL parameters are otherwise frozen or LoRA-adapted.

There are no task-specific projectors such as $P_W$, $P_Y$, or $P_H$.

Collect:

$$
R=[r_1,\ldots,r_N].
$$

---

## 7. Semantic state-consumption interface

### 7.1 Generic selection contract

`03_STATE_SPEC.md` owns the **mechanism** for carrying an ordered sequence of projected states into a semantic forward.

The semantic specs own **which sequence is consumed by each semantic mode**. In `multibranch`, this may be a selected subsequence. In `flat_single_output`, the consumed sequence is the full chronological `R=[r_1,\ldots,r_N]`. This mode difference does not change the state representation or instantiate another projector.

For one semantic example $e$, let:

$$
J_e=(j_1,\ldots,j_{K_e}),
\qquad
1\leq j_k\leq N.
$$

The selected state sequence is:

$$
\boxed{
R_e^{\mathrm{sel}}
=
[r_{j_1},\ldots,r_{j_{K_e}}].
}
$$

For the primary `multibranch` mode, canonical selections from `multi/04_SEMANTIC_MULTI_SPEC.md` are:

```text
WHAT at fixation t:
    J = [t]
    K = 1

WHY for oracle group G_m^GT = {t_1 < ... < t_K}:
    J = [t_1, ..., t_K]
    K = |G_m^GT|

HOW:
    J = [1, ..., N]
    K = N
```

For `flat_single_output`, `flat/04_SEMANTIC_SINGLE_SPEC.md` supplies the full chronological `R` to the same insertion primitive. This file introduces no grouping rule, semantic router, branch-specific state encoder, second projector, second state representation, or pooling operation.

### 7.2 Batched selected-state representation

When semantic examples have different selected-state lengths:

$$
K_{\max}=\max_e K_e.
$$

The collator may materialize:

```text
R_sel_batch: [B_sem, K_max, d_model]
M_sel:       [B_sem, K_max]
```

with:

```text
M_sel[e, :K_e] = true
M_sel[e, K_e:] = false
```

Only valid states may be inserted into the LLM sequence. Selected-state padding never becomes an LLM sequence position.

### 7.3 Order rule

The order of `R_sel` must exactly match the selected fixation indices.

- WHAT: one state.
- WHY: oracle-group members in increasing chronological order, including noncontiguous groups.
- HOW: the complete chronological trajectory.

No mean pooling is used in the primary method.

---

## 8. Direct continuous-state insertion contract

### 8.1 No model-visible semantic state token

Both semantic modes use **no token placeholder** for an exploration state.

In particular:

- do not add `<FIX_STATE>` or any equivalent token to the tokenizer;
- do not reuse `<END_FIX>` in the semantic prompt;
- do not serialize $r_t$ as numbers or text;
- do not quantize $r_t$ into a semantic token ID.

Each $r_t\in\mathbb R^{d_{\mathrm{model}}}$ is inserted **directly as one embedding position** in the language-model input sequence.

Thus one selected state contributes one LLM sequence position, but zero tokenizer tokens.

### 8.2 Model-visible semantic layout

The active semantic spec defines ordinary text labels around the states. Conceptually, for:

$$
J_e=(j_1,\ldots,j_{K_e}),
$$

the semantic user-side embedding sequence contains:

$$
[
E(\text{Fixation }j_1:),
r_{j_1},
E(\text{Fixation }j_2:),
r_{j_2},
\ldots,
E(\text{Fixation }j_{K_e}:),
r_{j_{K_e}}
].
$$

`Fixation {j}:` is ordinary text. The vector following it is **not text and has no token ID**.

The fixation index is bookkeeping that makes noncontiguous selections explicit to the language model. It is not a learned group ID.

### 8.3 Canonical Hugging Face InternVL + state composition

The canonical backbone for the DeepGaze-derived implementation is the Hugging Face-format checkpoint:

```text
OpenGVLab/InternVL3_5-8B-HF
architecture: InternVLForConditionalGeneration
```

Do **not** implement the semantic state path against the separate custom `InternVLChatModel` API (`extract_feature`, `model.language_model`, `model.chat`, etc.). For this checkpoint, the native Transformers model already exposes multimodal fusion through `InternVLForConditionalGeneration` / `InternVLModel`.

The relevant HF structure is conceptually:

```text
InternVLForConditionalGeneration
├── model: InternVLModel
│   ├── vision_tower
│   ├── multi_modal_projector
│   └── language_model
└── lm_head
```

`InternVLModel.forward(...)` accepts either `input_ids` or `inputs_embeds`, can receive `pixel_values`, obtains projected image features, identifies the native image-token positions, replaces those positions in the embedding sequence, and then calls the underlying language model. SemGaze should reuse this behavior rather than reimplement image fusion.

For semantic example $e$:

1. Build the native single-turn chat content defined by the active semantic spec and obtain the standard HF processor outputs for the query image/text.
2. Preserve explicit **host-side insertion boundaries** immediately after every ordinary `Fixation {j_k}:` text span. These boundaries are metadata only and are never tokenized.
3. Obtain ordinary token embeddings from the HF text embedding table:

```python
base_embeds = model.model.language_model.get_input_embeddings()(input_ids)
```

4. Interleave the selected projected states at the recorded boundaries using differentiable tensor operations:

$$
Z_e=[Z_0,r_{j_1},Z_1,\ldots,r_{j_{K_e}},Z_{K_e}].
$$

At this stage the native image placeholder embeddings are still present in `Z_e`.

5. Expand `attention_mask`, `labels`, and any explicit `position_ids` to the new sequence length. Every inserted state position has `attention_mask=1` and `label=-100`.
6. Call the **top-level HF model**, not the custom InternVL wrapper:

```python
outputs = model(
    inputs_embeds=semantic_inputs_embeds,
    pixel_values=pixel_values,
    attention_mask=semantic_attention_mask,
    labels=semantic_labels,
    ...
)
```

The HF InternVL model then performs its native image-feature replacement on the image-placeholder embeddings and runs the language model. This preserves the checkpoint's multimodal path while adding only the SemGaze continuous state positions.

Do not manually call `extract_feature(...)`, and do not assume a top-level `model.language_model` attribute for the HF checkpoint.

### 8.4 Required sequence metadata after insertion

For every directly inserted state position:

```text
attention_mask = 1
label          = -100
```

The state position is conditioning context, not a prediction target.

For user-side text and image context:

```text
label = -100
```

For assistant response target tokens during semantic training:

```text
label = target_token_id
```

Padding positions use:

```text
attention_mask = 0
label          = -100
```

If explicit `position_ids` are supplied, recompute them over the **final sequence after state insertion**. If the model derives positions from the attention mask internally, do not introduce a second conflicting position rule.

### 8.5 Sequence-length accounting

Each selected continuous state occupies exactly one LLM sequence position:

$$
\boxed{
L_{\mathrm{final}}
=
L_{\mathrm{native\ InternVL}}
+
K_e.
}
$$

`L_native InternVL` means the normal HF multimodal sequence length before SemGaze state insertion. This rule is consumed by the context-overflow validation in the active semantic spec.

### 8.6 Autograd requirement

Direct insertion must preserve autograd back to:

$$
R_e^{\mathrm{sel}}
\rightarrow
P_E
\rightarrow
F
\rightarrow
\text{WHERE computation}.
$$

Use differentiable concatenation/stacking/index assembly. Do not copy through NumPy, construct detached tensors, or serialize vectors as text.

Mixed-precision casting is allowed only to match the destination embedding dtype/device while preserving autograd.

### 8.7 Training forward

Semantic teacher forcing uses the top-level HF `InternVLForConditionalGeneration` forward with the fused `inputs_embeds`, native `pixel_values`, the final attention mask, and semantic labels:

```python
outputs = model(
    inputs_embeds=semantic_inputs_embeds,
    pixel_values=pixel_values,
    attention_mask=semantic_attention_mask,
    labels=semantic_labels,
    use_cache=False,
    ...
)
semantic_loss = outputs.loss
```

Do not call a custom `InternVLChatModel.forward(...)`, `model.extract_feature(...)`, or `model.language_model(...)` as the canonical implementation path.

### 8.8 Generation forward

For semantic generation:

1. build the same query-image + text + direct-state embedding prefix;
2. end the prefix at the native assistant-generation boundary;
3. invoke the HF generation path with the fused `inputs_embeds`, native image inputs/features, and matching attention metadata;
4. use the native assistant EOS/terminator and the decoding limits defined by `06_EVALUATION_SPEC.md`.

The generation adapter must be covered by an integration test because Transformers generation behavior with `inputs_embeds` is version-sensitive. The test must confirm that the exact installed Transformers version accepts the fused prefix, preserves the native image conditioning, and generates assistant tokens normally. If that test fails, the implementation must add a thin HF-compatible generation adapter; it must **not** switch the training model to the custom `InternVLChatModel` silently.

No semantic state token is generated or decoded.

### 8.9 Fail-fast conditions

Raise an explicit implementation error if any of the following occurs:

- the loaded architecture is not the configured HF InternVL architecture expected by the run;
- selected-state dimensionality differs from `d_model`;
- the number of host-side state boundaries differs from $K_e$;
- selected-state order differs from fixation-label order;
- padded states would be inserted as real positions;
- direct insertion detaches $R$;
- native image placeholder count and native image features no longer align after insertion;
- final embedding, attention-mask, label, or explicit-position lengths differ;
- semantic generation with fused `inputs_embeds` is unsupported by the installed Transformers version and no tested HF-compatible adapter is provided.

Do not repair these conditions by dropping states, duplicating states, mean-pooling states, serializing vectors as text, substituting `<END_FIX>`, or silently switching model implementations.

---

## 9. Formal semantic-state interface

For one semantic example $e$:

$$
R_e^{\mathrm{sel}}
=
[r_{j_1},\ldots,r_{j_{K_e}}].
$$

Define:

$$
\Phi(R_e^{\mathrm{sel}})
=
[
E(\text{Fixation }j_1:),
r_{j_1},
\ldots,
E(\text{Fixation }j_{K_e}:),
r_{j_{K_e}}
].
$$

Then conceptually:

$$
X_{\mathrm{SEM},e}
=
[
V_q,
E(Q_q),
\Phi(R_e^{\mathrm{sel}}),
E(\operatorname{Instr}_{\mathrm{branch}})
].
$$

The important distinction is:

$$
\boxed{
\texttt{<END\_FIX>}
\rightarrow
f_t
\rightarrow
r_t
\rightarrow
\text{direct semantic embedding position}.
}
$$

`<END_FIX>` belongs to WHERE serialization/readout. $r_t$ belongs to semantic conditioning.

The intended personalization path is:

$$
\mathcal C_u^K
\rightarrow
F
\rightarrow
R
\rightarrow
R_e^{\mathrm{sel}}
\rightarrow
\text{semantic branch}.
$$

This is an information path, not a cognitive-causal claim.

---

## 10. Inputs and outputs

### 10.1 Inputs

The state interface consumes:

- teacher-forced WHERE computation from `02_WHERE_SPEC.md`;
- final-layer hidden states for the packed WHERE sequence;
- final query assistant span mask;
- atomic `<END_FIX>` token ID;
- per-sample GT fixation counts;
- branch-selection indices from `multi/04_SEMANTIC_MULTI_SPEC.md`;
- semantic text/image spans and insertion boundaries from the semantic-example builder.

### 10.2 Outputs

For each query:

$$
F=[f_1,\ldots,f_N],
\qquad
R=[r_1,\ldots,r_N].
$$

For batching:

```text
F_batch: [B, N_max, d_model]
R_batch: [B, N_max, d_model]
M_fix:   [B, N_max]
```

For a semantic batch, the collator may expose:

```text
R_sel_batch: [B_sem, K_max, d_model]
M_sel:       [B_sem, K_max]
```

The final semantic model input is the variable-length fused `inputs_embeds` sequence after direct state insertion.

---

## 11. Training behavior

The teacher-forced WHERE computation provides both:

- $\mathcal L_{\mathrm{WHERE}}$, which directly supervises query scanpath tokens including every query `<END_FIX>`;
- query fixation states $F$, read at those `<END_FIX>` positions.

The states are projected by the shared $P_E$ and consumed according to the active semantic spec: `multibranch` uses its defined selected subsequences, while `flat_single_output` uses the full chronological `R`. In both cases, the states are inserted directly into semantic LLM embedding sequences without detachment.

For `multibranch`, WHAT, WHY, and HOW use different selected-state sequences and instructions while sharing the same InternVL/Qwen language-model parameters. The exact forward scheduling is owned by `05_TRAINING_SPEC.md`. The detailed flat-baseline execution is owned by `flat/04_SEMANTIC_SINGLE_SPEC.md`.

This file does **not** require a literal total of exactly two model calls per optimization step.

The full execution schedule, loss combination, branch weighting, gradient accumulation, optimizer schedule, LoRA configuration, parameter freezing, and activation-memory strategy are owned by `05_TRAINING_SPEC.md`.

For `multibranch`, the required semantic gradient path is:

$$
\mathcal L_{\mathrm{SEM}}
\rightarrow
R_e^{\mathrm{sel}}
\rightarrow
P_E
\rightarrow
F
\rightarrow
h_{\texttt{<END\_FIX>}}
\rightarrow
\theta.
$$

The flat baseline must preserve the same downstream gradient path through its full chronological `R`; its semantic loss definition remains owned by `flat/04_SEMANTIC_SINGLE_SPEC.md`.

No extra boundary loss is introduced: `$\mathcal L_{\mathrm{WHERE}}$ + semantic branch losses` already provide the two required learning signals for `<END_FIX>`.

---

## 12. Evaluation behavior

### 12.1 Primary semantic evaluation

Primary semantic evaluation is **gold-trajectory-conditioned**.

Teacher-force the GT query trajectory through WHERE and extract:

$$
R^{GT}=[r_1^{GT},\ldots,r_N^{GT}].
$$

The active semantic spec owns how the mode consumes this shared `R^{GT}`. The primary `multibranch` path uses its branch-specific selected subsequences, while `flat_single_output` consumes the full chronological sequence. No additional state extraction or projection path is introduced for the baseline.

This evaluates semantic generation without conflating it with free-running WHERE trajectory errors.

### 12.2 Predicted-state semantic path is deferred

There is no predicted-state semantic path in the current primary method.

Do not invent a mapping from oracle GT fixation/group indices to a free-running predicted trajectory. Matching, interpolation, nearest-fixation transfer, or heuristic truncation would constitute a separate experiment.

---

## 13. Known component-level ablations

### 13.1 Fixation-state readout

Primary:

$$
f_t=h_{\operatorname{END\_FIX}(t)}.
$$

Ablation:

$$
f_t=h_{\operatorname{lastToken}(s_t)}.
$$

### 13.2 Shared projector

Primary:

$$
r_t=P_E(f_t).
$$

Ablation:

$$
r_t=f_t.
$$

These are ablations, not unresolved primary alternatives.

---

## 14. InternVL3.5 / DeepGaze backbone compatibility note

The canonical SemGaze implementation follows the same Hugging Face-format backbone family used by the DeepGaze3.5-VL code path:

```text
OpenGVLab/InternVL3_5-8B-HF
InternVLForConditionalGeneration
```

Relevant verified HF behavior:

1. the checkpoint uses the InternVL ViT–MLP–LLM design with a Qwen3 language model;
2. `InternVLForConditionalGeneration` owns `model: InternVLModel` and a language-model head;
3. `InternVLModel` contains `vision_tower`, `multi_modal_projector`, and `language_model`;
4. `InternVLModel.forward(...)` accepts either `input_ids` or `inputs_embeds` together with native image inputs/features;
5. when `inputs_embeds` are supplied, the HF InternVL path can still identify native image-placeholder embeddings, replace them with projected image features, and call the underlying language model;
6. autoregressive language modeling remains the native training objective.

SemGaze adds only:

- the trainable `<END_FIX>` boundary/readout convention in WHERE;
- extraction of $f_t$ from final-layer query `<END_FIX>` hidden states;
- one shared projector $P_E$;
- branch-specific state selection;
- direct insertion of selected $r_t$ vectors as additional LLM embedding positions.

Do not mix the canonical HF implementation with the separate custom `InternVLChatModel` API in one primary code path.

References:

- DeepGaze3.5-VL repository: `https://github.com/Susmit-A/DeepGaze3.5-VL`
- HF InternVL model implementation: `https://github.com/huggingface/transformers/blob/main/src/transformers/models/internvl/modeling_internvl.py`
- PEFT trainable-token documentation: `https://github.com/huggingface/peft/blob/main/docs/source/package_reference/lora.md`

---

## 15. Invariants

### WHERE-to-state

- `<END_FIX>` is an atomic **trainable** WHERE vocabulary token, never a frozen marker;
- query `<END_FIX>` tokens receive ordinary WHERE next-token supervision; support `<END_FIX>` tokens remain causal context but are label-masked;
- no separate `L_END_FIX` is introduced;
- under LoRA, the required `<END_FIX>` input/output vocabulary row(s) remain trainable and are saved/restored;
- one exploration state is collected per GT query fixation;
- only final-query-assistant `<END_FIX>` positions are read;
- final language-model layer is used;
- state order is chronological;
- extracted-state count equals GT query fixation count;
- variable-length trajectories use an explicit validity mask;
- $F$ and $R$ are never detached before semantic supervision;
- one shared trainable $P_E:d_{\mathrm{model}}\rightarrow d_{\mathrm{model}}$ is used.

### Mode-specific consumption

- `multi/04_SEMANTIC_MULTI_SPEC.md` owns multibranch consumption of `R`;
- `flat/04_SEMANTIC_SINGLE_SPEC.md` owns flat full-sequence consumption of `R`;
- in `multibranch`, WHAT selects one state, WHY selects the ordered oracle-group states, and HOW selects the complete ordered trajectory;
- in `flat_single_output`, the full chronological `R` is consumed;
- both modes reuse the same `R` and the same shared `P_E`;
- no branch-specific or baseline-specific projector;
- no second state representation;
- no pooling unless explicitly introduced as a separate ablation.

### Direct insertion

- no `<FIX_STATE>` or equivalent semantic state token exists;
- `<END_FIX>` is never used as a semantic insertion token;
- each selected $r_t$ is inserted directly as one LLM embedding position;
- selected state order matches ordinary `Fixation {j}:` labels;
- state positions have `attention_mask=1` and `label=-100`;
- padded states are never inserted;
- direct insertion preserves autograd;
- native InternVL image-feature fusion is preserved;
- semantic branches use the query image and projected states without replaying raw few-shot support.

### Evaluation

- primary semantic evaluation is teacher-forced and GT-trajectory-conditioned;
- no predicted-to-GT trajectory/group alignment is part of the current method.

---

## 16. Cross-file ownership boundaries

The state interface is frozen by this document.

The following contracts are owned by:

- `01_FEWSHOT_SPEC.md`: support/query construction;
- `02_WHERE_SPEC.md`: WHERE serialization, atomic `<END_FIX>` insertion, query-only label mask, query assistant span, GT fixation count;
- `multi/04_SEMANTIC_MULTI_SPEC.md`: primary multibranch target units, selected indices, prompts, and semantic contract;
- `flat/04_SEMANTIC_SINGLE_SPEC.md`: full chronological `R` consumption, flat target serialization/parser, and flat semantic loss;
- `05_TRAINING_SPEC.md`: row-selective `<END_FIX>` trainability / PEFT realization, checkpoint saving, concrete segmented-tokenization/collation implementation, branch scheduling, loss weighting, optimizer/backward policy, LoRA configuration, memory strategy;
- `06_EVALUATION_SPEC.md`: decoding settings, metrics, and aggregation.

No implementation agent may replace direct continuous-state insertion with textual coordinates/vectors, a semantic special token, reused `<END_FIX>`, cross-attention memory, per-task projectors, subject embeddings, raw support replay, mean pooling, or predicted-to-GT alignment unless a separate method specification explicitly changes that contract.

---

## 17. Resolved decisions

The current primary design is frozen as:

1. the primary `semantic_mode="multibranch"` uses three branch-specific semantic generation tasks; the separately defined `flat_single_output` baseline is owned by `flat/04_SEMANTIC_SINGLE_SPEC.md`;
2. `<END_FIX>` is a **trainable WHERE boundary token and fixation-state readout location**;
3. query WHERE cross-entropy trains `<END_FIX>` generation; semantic losses shape the hidden state read at `<END_FIX>`; no separate boundary loss is added;
4. few-shot support turns remain in the causal context while their labels, including support `<END_FIX>`, are masked from the query-only WHERE loss;
5. under LoRA, the required `<END_FIX>` vocabulary row(s) remain trainable;
6. $r_t=P_E(f_t)$ is the semantic conditioning object;
7. both semantic modes consume projected $r_t$ states **directly as continuous LLM embedding positions**, with selection/consumption owned by the corresponding semantic spec;
8. no `<FIX_STATE>` or equivalent model-visible placeholder token and no reuse of `<END_FIX>` inside semantic prompts;
9. the canonical model path uses `OpenGVLab/InternVL3_5-8B-HF` / `InternVLForConditionalGeneration`, not the custom `InternVLChatModel` API;
10. WHAT/WHY/HOW share the same InternVL/Qwen parameters and LM head;
11. primary semantic evaluation uses GT-trajectory-conditioned states; predicted-state semantic alignment is deferred;
12. no state-level scientific choice is intentionally left to the implementation agent.
