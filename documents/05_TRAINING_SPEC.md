# SemGaze — Training Specification

## 1. Scope

This file is the canonical owner for **training execution of the primary `semantic_mode="multibranch"` SemGaze method**, together with the shared initialization/optimization/checkpoint infrastructure reused by semantic modes.

It defines:

- the runtime training-data boundary;
- model/token initialization required before PEFT;
- the teacher-forced WHERE forward;
- query-only WHERE supervision in a few-shot causal context;
- `<END_FIX>` trainability under LoRA;
- extraction/projector handoff to the semantic branches;
- construction and execution of the WHAT, WHY, and HOW semantic forwards;
- semantic loss reduction and branch balancing;
- the total objective;
- gradient flow and backward/optimizer ordering;
- the canonical trainable parameter set;
- checkpoint/resume requirements;
- training-time fail-fast assertions.


### Semantic-mode scope note

Unless explicitly marked as shared, statements in this file about semantic fan-out, `L_WHAT` / `L_WHY` / `L_HOW`, branch balancing, and `L_SEM` refer only to:

```text
semantic_mode = "multibranch"
```

`semantic_mode="flat_single_output"` is a separately named comparison baseline defined by `flat/04_SEMANTIC_SINGLE_SPEC.md`. Its semantic execution contract remains canonical there, including one semantic example per query, one flat semantic forward, `L_FLAT`, flat target serialization, and parsing.

The flat baseline reuses the shared training infrastructure defined here: model initialization, tokenizer and `<END_FIX>` setup, LoRA targets, trainable vocabulary rows, the shared projector `P_E`, optimizer family and scheduler, checkpoint/resume policy, and the requirement that semantic gradients remain connected through the shared exploration states. This reuse does not make this file the owner of the flat baseline's semantic execution.

Upstream contracts are referenced rather than redefined:

- curated benchmark materialization, train/validation/test split generation, subject partition, and episodic support/query sampling -> `01_FEWSHOT_SPEC.md`;
- WHERE prompt, scanpath serializer, `<END_FIX>` placement, query-only WHERE target mask, and `L_WHERE` token definition -> `02_WHERE_SPEC.md`;
- `<END_FIX>` hidden-state readout, shared projector `P_E`, branch-selected state interface, and direct continuous-state insertion -> `03_STATE_SPEC.md`;
- primary multibranch WHAT/WHY/HOW target units, oracle WHY groups, branch prompts/targets, and semantic execution -> `multi/04_SEMANTIC_MULTI_SPEC.md`;
- complete `flat_single_output` semantic execution, serialization/parser, and `L_FLAT` -> `flat/04_SEMANTIC_SINGLE_SPEC.md`;
- metrics, decoding, model selection, and reporting -> `06_EVALUATION_SPEC.md`.

Within `semantic_mode="multibranch"`, the current primary method has **no unified semantic forward**, **no semantic JSON generation**, and **no `L_STRUCT`**. This does not prohibit the separately defined `flat_single_output` baseline from using one ordinary semantic forward; its target is the flat non-JSON format owned by `flat/04_SEMANTIC_SINGLE_SPEC.md`.

The high-level training path below is specifically the primary `multibranch` path:

```text
curated SemGaze train split
    -> same-subject K-shot episode
    -> teacher-forced WHERE forward
         -> L_WHERE
         -> query <END_FIX> hidden states F
    -> shared projector P_E
         -> R
    -> WHAT semantic forward(s)
    -> WHY semantic forward(s)
    -> HOW semantic forward
         -> L_WHAT, L_WHY, L_HOW
    -> branch-balanced L_SEM
    -> L_total
    -> one joint backward
    -> one optimizer step
```

---

## 2. Runtime training-data contract

### 2.1 Training never reads the raw/original benchmark distribution

The runtime training source is the **persisted SemGaze curated split artifact** generated under `01_FEWSHOT_SPEC.md`.

For COCO-Search18, the canonical flow is:

```text
raw COCO-Search18
    -> upstream correctness filtering
    -> WHAT / WHY / HOW generation only for retained trials
    -> curated merged COCOSearch-18.json
    -> deterministic stimulus-level SemGaze split
    -> data/COCO_Search18/split/all/train.json
    -> THIS TRAINING SPEC
```


Training code must **not**:

- read the original COCO-Search18 split files to assign membership;
- recreate train/validation/test membership at runtime;
- infer trial correctness from `answer`, `condition`, fixation text, WHAT text, or any other final annotation field;
- regenerate WHAT/WHY/HOW online;
- move records between splits to repair a coverage problem.

The generated SemGaze split manifest is the source of truth for split and subject membership.

### 2.2 Eligible optimization records

Optimization uses only records satisfying:

```text
record belongs to persisted SemGaze train split
AND subject is a seen/training subject
AND semantic annotation passed the `multi/04_SEMANTIC_MULTI_SPEC.md` validation contract
```

Unseen-subject records whose stimuli belong to the SemGaze train split are reserved for few-shot evaluation support and must never update model parameters.

### 2.3 Query versus support payload

For one training episode, `01_FEWSHOT_SPEC.md` supplies:

```text
ordered_supports = [
    (I_1, Q_1, S_1),
    ...,
    (I_K, Q_K, S_K),
]

query = (I_q, Q_q, S_q, semantic_q)
```

where:

- support and query belong to the same seen subject;
- `K in {1,5,10}` follows the sampling distribution in `01_FEWSHOT_SPEC.md`;
- supports use distinct stimulus images and exclude the query image;
- support order is already final when handed to WHERE;
- support semantic annotations are **not model inputs**;
- the query semantic annotation provides the GT WHAT/WHY/HOW targets used by the active semantic spec.

Training must preserve the exact support membership and order supplied by `01_FEWSHOT_SPEC.md`.

### 2.4 Training context overflow

If the complete canonical few-shot WHERE input exceeds the configured context budget, reject only that sampled training episode and resample according to `01_FEWSHOT_SPEC.md`.

Context overflow never permits:

- reducing `K`;
- dropping a support;
- reordering supports;
- truncating one support scanpath;
- substituting another support inside the already-constructed episode.

Semantic context overflow is governed by the active semantic spec and is a hard error for that record; semantic targets/states are never silently truncated or pooled.

---

## 3. Model and tokenizer initialization

### 3.1 Backbone

The primary backbone follows the released DeepGaze3.5-VL InternVL setup:

```text
OpenGVLab/InternVL3_5-8B-HF
```

Use the checkpoint's native Hugging Face tokenizer/processor/chat template and the HF `InternVLForConditionalGeneration` path required by `03_STATE_SPEC.md`.

Do not silently switch to the custom `InternVLChatModel` implementation.

### 3.2 `<END_FIX>` must be added before PEFT construction

Initialization order is normative:

```text
1. load base tokenizer / processor / HF InternVL model
2. add atomic <END_FIX> token if absent
3. verify tokenize("<END_FIX>", add_special_tokens=False) -> exactly one token ID
4. resize token embeddings to the final tokenizer size
5. record END_FIX_TOKEN_ID
6. determine whether input/output token weights are tied
7. construct LoRA + trainable-token configuration
8. construct shared projector P_E
9. construct optimizer over the final trainable parameter set
```

Do not add `<END_FIX>` after the PEFT adapter or optimizer has already been constructed.

Semantic execution adds no semantic vocabulary token in either current mode. In particular, the primary `multibranch` method has no:

```text
<FIX_STATE>
<WHAT>
<WHY>
<HOW>
```

### 3.3 `<END_FIX>` trainability

`<END_FIX>` is a learned WHERE vocabulary token, not a frozen marker.

The trainability contract is architectural rather than PEFT-API-specific:

- the input embedding parameter corresponding to `<END_FIX>` is trainable;
- when input/output embeddings are untied, the output/logit parameter corresponding to `<END_FIX>` is also trainable;
- when embeddings are tied, the shared parameter remains trainable;
- all other vocabulary parameters remain outside the primary trainable set.

A frozen `<END_FIX>` vocabulary parameter is invalid for the primary method. The concrete PEFT mechanism used to satisfy this contract is not part of the method definition.

### 3.4 DeepGaze-style LoRA targets

Keep the released DeepGaze3.5-VL language-model LoRA targets:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Primary LoRA hyperparameters:

```text
rank    = 8
alpha   = 16
dropout = 0.05
```

The primary method does not add LoRA to the vision tower or native multimodal projector.

The native vision tower and native InternVL multimodal projector remain frozen in the primary configuration. The project-specific shared state projector `P_E` from `03_STATE_SPEC.md` is trainable.

### 3.5 Shared projector

Instantiate exactly one shared projector:

```python
P_E = nn.Sequential(
    nn.Linear(d_model, d_model),
    nn.LayerNorm(d_model),
)
```

`P_E` is trainable and shared across semantic modes. Within `multibranch`, the same `P_E` is shared by WHAT, WHY, and HOW.

Do not create branch-specific projectors.

---

## 4. Training episode construction

Episode membership and stochastic sampling are owned by `01_FEWSHOT_SPEC.md`.

For each sampled episode:

1. select one seen subject;
2. sample `K in {1,5,10}` according to the frozen few-shot distribution;
3. sample one valid query from the persisted SemGaze train split;
4. sample `K` same-subject support records under the image-identity constraints of `01_FEWSHOT_SPEC.md`;
5. randomize support order according to `01_FEWSHOT_SPEC.md`;
6. pass the ordered records unchanged to WHERE.

The training adapter may batch multiple episodes only if the multimodal packing preserves each episode independently and all masks/state indices remain exact.

---

## 5. WHERE training forward

### 5.1 Multi-turn causal input

Render the complete episode using the canonical WHERE conversation from `02_WHERE_SPEC.md`:

```text
user:      support image 1 + WHERE prompt 1
assistant: observed support scanpath 1

...

user:      support image K + WHERE prompt K
assistant: observed support scanpath K

user:      query image + WHERE prompt q
assistant: GT query scanpath
```

This is one packed causal history, not `K+1` independent model forwards.

### 5.2 Query-only WHERE supervision

Support assistant turns are demonstrations and remain in `input_ids`, but their labels are ignored.

Canonical labels:

```text
all support user tokens       -> -100
all support assistant tokens  -> -100
all query user tokens         -> -100
image/context/control tokens  -> -100
final query assistant tokens  -> supervised token IDs
```

This includes every query `<END_FIX>` and the native assistant response terminator/EOS if emitted by the template.

Therefore:

```text
support <END_FIX> labels = -100
query   <END_FIX> labels = END_FIX_TOKEN_ID
```

If the training framework exposes a `mask_history=true`-style mechanism, it may be used only if an integration test verifies that it **masks historical assistant labels without removing historical turns from the causal input**.

`mask_history` must never mean history deletion, support truncation, or support resampling.

### 5.3 WHERE forward call

Run teacher forcing with hidden states enabled and cache disabled:

```text
output_hidden_states = True
use_cache            = False
```

Conceptually:

$$
X_{
m WHERE}+S_q^{GT}
\xrightarrow[\text{teacher forcing}]{F_\theta}
(\mathcal L_{\rm WHERE},F).
$$

The forward provides:

- `L_WHERE`: response-only query scanpath NTP loss;
- final-layer hidden states for the complete packed episode;
- `F=[f_1,...,f_N]` extracted only from final-query-assistant `<END_FIX>` positions under `03_STATE_SPEC.md`.

The implementation must assert:

```text
number of supervised query <END_FIX> labels == N
number of extracted query <END_FIX> states == N
all support <END_FIX> labels == -100
```

### 5.4 `<END_FIX>` has two learning signals

No separate `L_END_FIX` exists.

The ordinary WHERE next-token loss directly trains the model to generate `<END_FIX>`:

$$
\mathcal L_{\rm WHERE}
\rightarrow
p(\texttt{<END\_FIX>}\mid\text{preceding scanpath context}).
$$

The semantic branch losses shape the hidden representation at those boundary positions:

$$
\mathcal L_{\rm SEM}
\rightarrow
F
\rightarrow
h_{\texttt{<END\_FIX>}}
\rightarrow
\text{WHERE computation}.
$$

Because support `<END_FIX>` tokens remain in the causal context, query losses may also backpropagate through their **input-side** contextual use. They simply receive no direct support-response next-token target.

---

## 6. Shared exploration-state preparation

From the same teacher-forced WHERE computation:

$$
F=[f_1,\ldots,f_N].
$$

Project once with the shared trainable projector:

$$
R=P_E(F)=[r_1,\ldots,r_N].
$$

Do not detach `F` or `R`.

Do not move `F`/`R` through NumPy, CPU serialization, text serialization, or any other graph-breaking path.

The query semantic annotation is validated before constructing semantic examples. Invalid annotations hard-fail under the shared semantic annotation contract in `multi/04_SEMANTIC_MULTI_SPEC.md`; training does not repair or silently drop them.

---

## 7. Primary multibranch semantic forwards

This section applies only to `semantic_mode="multibranch"`. For a query with:

- `N` fixations;
- `M` oracle WHY groups;

construct exactly:

```text
N WHAT examples
M WHY examples
1 HOW example
```

Thus:

$$
N_{\rm SEM}=N+M+1.
$$

### 7.1 WHAT

For fixation `t`:

```text
selected state indices = [t]
selected states        = [r_t]
target                 = WHAT_t text
```

Build the canonical WHAT single-turn semantic conversation from `multi/04_SEMANTIC_MULTI_SPEC.md` and insert `r_t` directly into the LLM embedding stream under `03_STATE_SPEC.md`.

### 7.2 WHY

For oracle group:

$$
G_m^{GT}=\{t_1<\ldots<t_{K_m}\},
$$

use:

```text
selected state indices = [t_1, ..., t_Km]
selected states        = [r_t1, ..., r_tKm]
target                 = WHY_m text
```

Oracle group membership is **conditioning structure**, not a generated target.

There is no GROUP branch, grouping loss, group-ID token, or serialized `members` output.

### 7.3 HOW

Use:

```text
selected state indices = [1, ..., N]
selected states        = [r_1, ..., r_N]
target                 = HOW text
```

HOW does not consume generated WHAT/WHY text or oracle group IDs.

### 7.4 Semantic forward implementation

Every semantic example is a normal response-only InternVL conversation with direct continuous-state insertion.

Use the HF path owned by `03_STATE_SPEC.md`, conceptually:

```python
outputs = model(
    inputs_embeds=semantic_inputs_embeds,
    pixel_values=pixel_values,
    attention_mask=semantic_attention_mask,
    labels=semantic_labels,
    use_cache=False,
    ...
)
```

Semantic labels are:

```text
user text                  -> -100
image/context positions    -> -100
direct inserted state      -> -100
assistant target text      -> supervised token IDs
```

In `multibranch`, the semantic branches do not replay the raw few-shot support prefix. Personalization reaches them through the support-conditioned states `R`.

WHAT, WHY, and HOW all use the **same** InternVL/Qwen parameters and the same LM head. This is not a three-decoder architecture.

---

## 8. Multibranch semantic loss reduction

This section applies only to `semantic_mode="multibranch"`. The old structured-output scale-token weighting is removed completely.

There is no JSON punctuation to weight and no `L_STRUCT` in the primary multibranch path. The flat baseline has its own mode-specific semantic loss owned by `flat/04_SEMANTIC_SINGLE_SPEC.md` and is not redefined here.

### 8.1 Per-example loss

For each semantic example `e`, compute response-token **mean** negative log-likelihood:

$$
\ell_e
=
-\frac{1}{|\Omega_e|}
\sum_{\ell\in\Omega_e}
\log p_\theta(a_{e,\ell}\mid X_{e},a_{e,<\ell}),
$$

where `Omega_e` contains only supervised assistant target positions.

Do not use raw token-sum loss as the branch statistic because longer targets would receive larger weight solely due to length.

### 8.2 Within-branch averaging

For the `N` WHAT examples:

$$
\boxed{
\mathcal L_{\rm WHAT}
=
\frac{1}{N}
\sum_{t=1}^{N}\ell_t^{W}
}
$$

For the `M` WHY examples:

$$
\boxed{
\mathcal L_{\rm WHY}
=
\frac{1}{M}
\sum_{m=1}^{M}\ell_m^{Y}
}
$$

For the single HOW example:

$$
\boxed{
\mathcal L_{\rm HOW}=\ell^{H}
}
$$

### 8.3 Equal-scale semantic aggregation

The primary method gives WHAT, WHY, and HOW equal semantic-scale weight:

$$
\boxed{
\mathcal L_{\rm SEM}
=
\frac{1}{3}
\left(
\mathcal L_{\rm WHAT}
+
\mathcal L_{\rm WHY}
+
\mathcal L_{\rm HOW}
\right)
}
$$

This prevents WHAT from dominating simply because there are `N` fixation targets and prevents WHY from dominating simply because one trajectory contains many oracle groups.

Alternative branch weights are ablations/hyperparameter experiments and must not silently replace this primary reduction.

---

## 9. Primary multibranch joint objective

For `semantic_mode="multibranch"`, the complete primary objective is:

$$
\boxed{
\mathcal L_{\rm total}
=
\lambda_{\rm WHERE}\mathcal L_{\rm WHERE}
+
\lambda_{\rm SEM}\mathcal L_{\rm SEM}
}
$$

with primary defaults:

$$
\boxed{
\lambda_{\rm WHERE}=1,
\qquad
\lambda_{\rm SEM}=1.
}
$$

Equivalently:

$$
\mathcal L_{\rm total}
=
\mathcal L_{\rm WHERE}
+
\frac{1}{3}
\left(
\mathcal L_{\rm WHAT}
+
\mathcal L_{\rm WHY}
+
\mathcal L_{\rm HOW}
\right).
$$

The primary objective contains no:

- `L_STRUCT`;
- `L_GROUP`;
- `L_END_FIX`;
- coordinate-regression loss;
- duration-regression loss;
- contrastive semantic loss;
- RL/sequence-level reward loss.

---

## 10. Canonical multibranch execution and backward schedule

### 10.1 One training episode

For one `semantic_mode="multibranch"` episode, execute:

```text
1. Read one query + sampled ordered same-subject supports from the persisted SemGaze train split.
2. Render the complete few-shot WHERE causal conversation.
3. Build query-only WHERE labels; support assistant labels remain -100.
4. Teacher-force the query scanpath with output_hidden_states=True.
5. Obtain L_WHERE and final-query <END_FIX> states F.
6. Project F -> R with shared P_E without detachment.
7. Build N WHAT semantic examples from R.
8. Execute WHAT semantic forward batch(es) and compute L_WHAT.
9. Build M WHY semantic examples using oracle group selections.
10. Execute WHY semantic forward batch(es) and compute L_WHY.
11. Build one HOW semantic example using the full R sequence.
12. Execute HOW semantic forward and compute L_HOW.
13. Compute branch-balanced L_SEM.
14. Compute L_total.
15. Call backward once on L_total.
16. Apply gradient clipping if configured.
17. Optimizer step.
18. Scheduler step.
19. Zero gradients.
```

### 10.2 One shared computation graph

The primary `multibranch` implementation reuses the same teacher-forced WHERE states `F` for all semantic branches and keeps their graph alive until the joint backward.

Therefore:

```text
NO F.detach()
NO R.detach()
NO detached state cache
NO semantic-only backward that discards the WHERE graph
```

The required paths include:

$$
\mathcal L_{\rm WHAT}
\rightarrow R\rightarrow F\rightarrow\theta,
$$

$$
\mathcal L_{\rm WHY}
\rightarrow R\rightarrow F\rightarrow\theta,
$$

$$
\mathcal L_{\rm HOW}
\rightarrow R\rightarrow F\rightarrow\theta.
$$

### 10.3 Semantic batching

WHAT examples may be batched together; WHY examples may be batched together with selected-state padding/masks from `03_STATE_SPEC.md`; HOW examples may be batched across different episodes when compatible.

Batching is an implementation optimization only. It must preserve the loss definitions in Section 8.

If semantic examples are microbatched, the implementation must still produce the mathematically identical branch means before the joint objective is formed.

### 10.4 Memory-optimized recomputation

A future memory-optimized implementation may recompute the teacher-forced WHERE graph for semantic microbatches instead of retaining one large graph, but only if it is explicitly enabled and verified to produce the same objective/gradient semantics.

The primary specification does not authorize detaching `F` as a memory optimization.

---

## 11. Gradient-flow contract

### 11.1 WHERE path

`L_WHERE` updates:

- LoRA-adapted language-model parameters;
- the trainable `<END_FIX>` output/logit row through next-token prediction;
- the trainable `<END_FIX>` input row through normal causal-context usage;
- any other primary trainable parameters reached by the WHERE graph.

### 11.2 Semantic path (`multibranch`)

For `semantic_mode="multibranch"`, `L_WHAT`, `L_WHY`, and `L_HOW` update:

- the shared semantic LM/LoRA parameters;
- `P_E`;
- the WHERE computation through `F`;
- the trainable `<END_FIX>` **input embedding row** through hidden states read at `<END_FIX>` positions.

For untied Qwen3 embeddings, semantic loss does not directly supervise the `<END_FIX>` output/logit row as a target token; that row is directly trained by `L_WHERE`.

### 11.3 Support demonstrations

Support assistant scanpaths receive no direct LM target loss, but their tokens/images remain causal conditioning context.

Therefore gradients from the supervised query may flow through the model computation that processed support context. Masking support labels does **not** mean removing support influence.

---

## 12. Trainable parameter set

The trainable set below is shared by the primary multibranch method and, unless the corresponding semantic spec explicitly states otherwise, the `flat_single_output` baseline:

$$
\boxed{
\Theta_{\rm train}
=
\{
\theta_{\rm LoRA},
\theta_{P_E},
\theta_{\texttt{<END\_FIX>},\mathrm{in}},
\theta_{\texttt{<END\_FIX>},\mathrm{out}}
\}.
}
$$

If input/output vocabulary weights are tied, the last two terms are one shared trainable row.

Primary frozen parameters include:

- base Qwen3 weights outside LoRA adapters;
- native InternVL vision tower;
- native InternVL multimodal projector;
- all old vocabulary rows not selected by trainable-token tuning.

### 12.1 Required trainability assertions

Before the first optimizer step, print/log and assert:

```text
LoRA parameter count > 0
P_E parameter count > 0
END_FIX input row is trainable
END_FIX output row is trainable if embeddings are untied
vision tower frozen
native multimodal projector frozen
no full vocabulary matrix accidentally trainable in the primary configuration
```

After one non-empty backward pass, assert:

```text
finite nonzero gradient reaches P_E
finite gradient reaches END_FIX input row
finite gradient reaches END_FIX output row if untied and query contains END_FIX targets
at least one LoRA parameter receives finite gradient
```

Do not require every individual gradient element to be nonzero.

---

## 13. Optimizer and primary starting configuration

The optimizer family, scheduler, and starting configuration in this section are shared infrastructure reused by both semantic modes. To stay close to the released DeepGaze3.5-VL recipe while adding the SemGaze-specific parameters, use:

```text
optimizer        = AdamW
learning_rate    = 1e-4
scheduler        = cosine
warmup_ratio     = 0.10
LoRA rank        = 8
LoRA alpha       = 16
LoRA dropout     = 0.05
precision        = bf16 when supported
```

The same base learning rate is used for LoRA, `P_E`, and trainable `<END_FIX>` rows in the primary simple configuration.

Do not introduce a special larger `<END_FIX>` learning rate in the primary method.

Framework defaults such as Adam betas, epsilon, weight decay, and max-grad-norm must be written into the resolved run configuration/checkpoint metadata. They are engineering hyperparameters, not hidden methodological choices.

Per-device episode batch size and gradient accumulation may be reduced/increased for available memory, provided:

- the episode distribution from `01_FEWSHOT_SPEC.md` is unchanged;
- loss normalization from Sections 8-9 is unchanged;
- support membership/order is unchanged;
- direct semantic gradient flow to WHERE states is preserved.

---

## 14. Checkpoint and resume contract

A reproducible checkpoint for either semantic mode must preserve all shared trainable SemGaze state, not only the ordinary LoRA matrices. Mode-specific semantic metadata remains owned by `flat/04_SEMANTIC_SINGLE_SPEC.md`.

Save at least:

```text
tokenizer / processor containing <END_FIX>
END_FIX_TOKEN_ID
PEFT adapter weights
trainable-token weights for END_FIX input/output row(s)
shared projector P_E state_dict
optimizer state when resumable training is required
scheduler state when resumable training is required
resolved training configuration
SemGaze split manifest identity / hash
protocol version
```

When PEFT `trainable_token_indices` is used, save/load through the PEFT-supported adapter path so the selected vocabulary rows are part of the adapter state.

Resume order is normative:

```text
1. load checkpoint tokenizer/processor
2. load base HF InternVL model
3. resize embeddings to checkpoint tokenizer length
4. reconstruct the same PEFT/LoRA + trainable-token configuration
5. load adapter/trainable-token state
6. reconstruct and load P_E
7. restore optimizer/scheduler if resuming optimization
8. verify END_FIX token ID and trainability assertions
```

Fail resume if:

- `<END_FIX>` maps to a different token ID than the checkpoint metadata;
- the selected SemGaze split manifest differs unexpectedly;
- `P_E` is missing;
- required trainable-token state is missing;
- a row-selective checkpoint is silently loaded with the `<END_FIX>` row frozen.

---

## 15. Distributed / gradient-accumulation semantics

Distributed execution may shard batches and accumulate gradients, but it must preserve the mathematical objective.

For `multibranch` semantic branch losses, reductions must be based on the actual valid example/token counts, not on padded batch shapes. Flat-baseline loss reduction is not defined in this section.

In particular:

```text
WHAT branch mean = mean over valid WHAT example losses
WHY branch mean  = mean over valid WHY example losses
HOW branch mean  = mean over valid HOW example losses
```

If multiple episodes are accumulated before an optimizer step, average/sum them consistently with the framework's declared gradient-accumulation semantics. Do not allow trajectories with more fixations/groups to receive larger semantic weight merely because they create more branch examples.

Worker/rank seeding may change the sampled episode sequence but must implement the exact probability law in `01_FEWSHOT_SPEC.md`. The resolved global seed and distributed world size must be logged for reproducibility.

---

## 16. Fail-fast training assertions

Training must stop rather than guess or silently repair when any of the following occurs:

- runtime data was loaded from raw/original benchmark splits instead of the persisted SemGaze split artifact;
- a train query belongs to an unseen subject;
- support/query membership violates `01_FEWSHOT_SPEC.md`;
- support order changes between sampler output and WHERE rendering;
- support semantic annotations are inserted into the WHERE support turns;
- a support assistant token receives a supervised WHERE label;
- query `<END_FIX>` target count differs from query fixation count;
- query `<END_FIX>` readout count differs from query fixation count;
- `<END_FIX>` is non-atomic;
- `<END_FIX>` input/output row required by the loaded architecture is frozen;
- direct state insertion detaches `R`;
- a semantic branch replays raw supports;
- in `multibranch`, a semantic branch emits JSON/group membership instead of the raw target text contract;
- invalid semantic annotation reaches training;
- semantic state order differs from its displayed fixation-index order;
- semantic context overflow is repaired by dropping/pooling states;
- `P_E` is omitted from the optimizer;
- checkpoint saving omits required `<END_FIX>` trainable state or `P_E`;
- a structured-output `L_STRUCT` path is still active.

---

## 17. Teacher-forcing summary

### WHERE

- supports are observed demonstrations in the causal history;
- support assistant labels are masked;
- the GT query scanpath is teacher-forced;
- every query `<END_FIX>` is an ordinary supervised next-token target;
- final-layer query `<END_FIX>` hidden states are read as `F`.

### WHAT (`multibranch`)

- GT WHAT text is teacher-forced in one response-only semantic example per fixation;
- input conditioning uses query image/task + directly inserted `r_t`.

### WHY (`multibranch`)

- oracle group membership selects the input states;
- GT WHY text is teacher-forced in one response-only semantic example per oracle group;
- group membership itself is not generated.

### HOW (`multibranch`)

- GT HOW text is teacher-forced in one response-only semantic example for the full ordered trajectory.

For `multibranch`, there is no teacher-forced structured semantic JSON sequence. The separately defined flat baseline uses its own ordinary single-response target contract from `flat/04_SEMANTIC_SINGLE_SPEC.md`.

---

## 18. Primary training invariants

The following are non-negotiable for the primary `multibranch` implementation; shared initialization/state/optimizer/checkpoint items also apply to `flat_single_output` as specified by the corresponding semantic spec:

- runtime optimization uses only the persisted **curated SemGaze train split**;
- raw/original COCO-Search18 membership is not consulted at runtime;
- seen/unseen subject partition and episodic sampling come from `01_FEWSHOT_SPEC.md`;
- raw same-subject supports remain in the WHERE causal context;
- support assistant labels are masked, not deleted;
- final query scanpath is the only directly supervised WHERE response;
- `<END_FIX>` is trainable and receives ordinary query WHERE NTP supervision;
- no separate `L_END_FIX`;
- query `<END_FIX>` hidden states form `F`;
- one shared trainable `P_E` produces `R`;
- semantic branches use direct continuous-state insertion, not `<FIX_STATE>`;
- exactly `N` WHAT + `M` WHY + `1` HOW examples are defined per query;
- WHY uses oracle group membership as input-side state selection only;
- WHAT/WHY/HOW share the same InternVL/Qwen parameters and LM head;
- no structured semantic JSON generation;
- no GROUP-prediction branch or grouping loss;
- semantic loss is branch-balanced after within-branch averaging;
- `L_total = L_WHERE + L_SEM` under the primary unit weights;
- `F` and `R` are never detached before semantic supervision;
- primary execution uses one joint backward and one optimizer step per optimization unit;
- DeepGaze-style LoRA targets remain `q/k/v/o + gate/up/down`;
- native vision tower/projector stay frozen in the primary configuration;
- `<END_FIX>` row(s), LoRA adapters, and `P_E` are saved/restored in checkpoints.

---
