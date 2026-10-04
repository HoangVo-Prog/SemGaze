# SemGaze — Method Identity Specification

## 1. Scope

This file is the canonical owner for the **high-level identity and decomposition** of the current SemGaze method.

It describes, at a method level:

- the task setting;
- the personalization mechanism;
- the shared-model organization;
- the relationship between WHERE and the primary multibranch semantic path;
- the information flow used during training and evaluation;
- the current implementation scope and major methodological boundaries.

This file intentionally does **not** freeze paper claims, novelty claims, performance claims, or causal interpretations. Those should be formulated only after experiments are available.

Low-level contracts are owned elsewhere:

- curated COCO-Search18 dataset creation, split construction, few-shot subject/support/query construction, and frozen support manifests → `01_FEWSHOT_SPEC.md`;
- WHERE / duration-aware scanpath representation, prompt, serialization, training target, and free-running generation → `02_WHERE_SPEC.md`;
- `<END_FIX>` fixation-state extraction, shared projector, gradient preservation, and the low-level direct continuous-state insertion primitive → `03_STATE_SPEC.md`;
- primary `semantic_mode="multibranch"` WHAT / WHY / HOW targets, gold WHY groups, branch state consumption, prompts, and semantic-output contract → `multi/04_SEMANTIC_MULTI_SPEC.md`;
- `semantic_mode="flat_single_output"` prompt, full-state consumption, flat serialization/parser, and flat semantic loss → `flat/04_SEMANTIC_SINGLE_SPEC.md`;
- canonical training execution for the primary multibranch method, plus shared model/optimizer/checkpoint infrastructure reused by semantic modes → `05_TRAINING_SPEC.md`;
- evaluation paths, shared semantic-content metrics, paired semantic-mode comparison, controls, and ablations → `06_EVALUATION_SPEC.md`.

The current executable dataset scope is **COCO-Search18 only**. Dataset membership and preprocessing details are not duplicated here.

### Semantic-mode scope note

The **core SemGaze method identity in this file remains `semantic_mode="multibranch"`**. `flat/04_SEMANTIC_SINGLE_SPEC.md` defines `semantic_mode="flat_single_output"` as a plain-VLM comparison/control baseline. The baseline uses the same underlying semantic supervision and support-conditioned exploration states, but produces all WHAT/WHY/HOW content in one ordinary assistant response instead of the primary multibranch fan-out.

The flat baseline is not part of the core method identity. Its prompt, serialization, parser, state-consumption details, and semantic loss are intentionally not duplicated here and remain canonical only in `flat/04_SEMANTIC_SINGLE_SPEC.md`.


---

## 2. Problem setting

For an unseen subject $u^\star$, SemGaze receives an ordered few-shot support context containing previous visual-search demonstrations from that same subject:

$$
\widetilde{\mathcal C}_{u^\star}^{K}
=
\left(
(I_1,Q_1,S_1),\ldots,(I_K,Q_K,S_K)
\right),
\qquad
K\in\{1,5,10\},
$$

and a new query:

$$
(I_q,Q_q).
$$

Each support demonstration contains:

- the support image;
- the support visual-search task;
- the observed support scanpath.

Support semantic labels are not replayed as model input.

At a high level, SemGaze contains four prediction/supervision scales:

- **WHERE** — the duration-aware fixation trajectory;
- **WHAT** — one semantic description per fixation;
- **WHY** — one task-grounded rationale per gold/oracle fixation group;
- **HOW** — one description of the overall exploration/search strategy for the trajectory.

There is no test-time parameter update for a new subject.

---

## 3. Personalization identity

SemGaze does not introduce a persistent learned representation for each subject.

The primary method has no:

- subject-ID token;
- persistent user embedding;
- subject-specific encoder;
- per-user adapter;
- per-user fine-tuning or optimizer step at inference.

Subject identity is used by the data pipeline to construct same-subject support/query episodes and subject-disjoint evaluation.

The model-facing personalization path is:

$$
\boxed{
\widetilde{\mathcal C}_u^K
\rightarrow
\text{native multimodal causal context}
\rightarrow
\text{query computation}
}
$$

The ordered support demonstrations are rendered directly into the VLM causal history before the query. Personalization therefore enters through the support context rather than through a separate learned user representation.

---

## 4. Shared-model identity

The canonical backbone is the shared Hugging Face InternVL3.5 model specified by the component documents:

```text
OpenGVLab/InternVL3_5-8B-HF
InternVLForConditionalGeneration
```

The same underlying language-model parameters and LM head are used for:

- WHERE generation;
- fixation-state formation;
- WHAT generation;
- WHY generation;
- HOW generation.

The primary method is **not**:

- a separate gaze model plus a separate explanation model;
- a three-decoder WHAT/WHY/HOW architecture;
- a GROUP-prediction architecture;
- a unified structured-JSON semantic decoder.

For the primary `semantic_mode="multibranch"` path, WHAT, WHY, and HOW are independent semantic generation tasks implemented with shared model parameters. This statement does not exclude the separately defined `flat_single_output` comparison baseline in `flat/04_SEMANTIC_SINGLE_SPEC.md`.

---

## 5. WHERE representation

WHERE is modeled as an autoregressive duration-aware scanpath language.

The current primary representation contains one tuple per fixation:

$$
(x_t,y_t,d_t),
$$

followed by the learned atomic boundary token:

```text
<END_FIX>
```

The current method design uses the duration-enabled WHERE path only. There is no duration-free primary variant in this specification. The frozen COCO-Search18 declaration is `T[t] = fixation dwell duration in milliseconds`, with identity conversion to milliseconds; serialization follows `02_WHERE_SPEC.md`.

The query prompt receives the requested trajectory length:

$$
N_q=|S_q^{GT}|.
$$

Therefore the current WHERE protocol is oracle-length-conditioned and does not define autonomous stopping.

The exact coordinate conversion, duration representation, prompt, serializer, parser, and free-running generation behavior are owned by `02_WHERE_SPEC.md`.

---

## 6. Shared exploration states

During the teacher-forced WHERE computation, one fixation state is read from the final language-model layer at each query `<END_FIX>` position:

$$
f_t=h_{\operatorname{END\_FIX}(t)}.
$$

The full chronological sequence is:

$$
F=[f_1,\ldots,f_N].
$$

A single shared trainable projector produces:

$$
r_t=P_E(f_t),
\qquad
R=[r_1,\ldots,r_N].
$$

The projected states are continuous vectors. They are inserted directly into semantic LLM embedding sequences and are not serialized as text or represented by a semantic special token.

Semantic supervision remains differentiable through:

$$
R
\rightarrow
P_E
\rightarrow
F
\rightarrow
\text{WHERE computation}.
$$

The exact readout, projection, insertion, and gradient contracts are owned by `03_STATE_SPEC.md`.

---

## 7. Primary multibranch semantic identity

The primary `semantic_mode="multibranch"` SemGaze path uses three independent semantic generation branches sharing the same InternVL/Qwen parameters and LM head.

### 7.1 WHAT

For fixation $t$:

$$
(I_q,Q_q,r_t)
\rightarrow
w_t.
$$

There is one WHAT target and one WHAT response per fixation.

### 7.2 WHY

WHY uses the **gold/oracle fixation groups** supplied by the semantic annotation.

For:

$$
G_m^{GT}=\{t_1<\cdots<t_{K_m}\},
$$

select:

$$
R_m^Y=[r_{t_1},\ldots,r_{t_{K_m}}],
$$

then generate:

$$
(I_q,Q_q,R_m^Y)
\rightarrow
y_m.
$$

Gold group membership is input-side conditioning for WHY in both semantic training and semantic inference/evaluation. Group membership is **not predicted** by the current method.

Groups may be noncontiguous.

### 7.3 HOW

HOW uses the complete ordered trajectory state sequence:

$$
(I_q,Q_q,R)
\rightarrow
h.
$$

HOW does not consume generated WHAT/WHY text or group IDs.

### 7.4 No semantic generation chain

The primary `multibranch` method does not generate:

$$
WHAT\rightarrow WHY\rightarrow HOW.
$$

The branches share upstream exploration states but generate their targets independently.

---

## 8. High-level training flow

For one training episode, the primary `multibranch` method-level flow is:

```text
same-subject ordered support demonstrations
        +
query image / task / GT duration-aware scanpath
        ↓
teacher-forced WHERE computation
        ├── L_WHERE
        └── query <END_FIX> states F
                    ↓
              shared projector P_E
                    ↓
                    R
          ┌─────────┼─────────┐
          ↓         ↓         ↓
        WHAT       WHY       HOW
      forwards   forwards   forward
          ↓         ↓         ↓
       L_WHAT    L_WHY     L_HOW
          └─────────┼─────────┘
                    ↓
                  L_SEM
                    ↓
          L_total = L_WHERE + L_SEM
                    ↓
             joint backward
```

WHY state selection uses gold/oracle groups during training.

The number of concrete model forward calls is an execution detail, not part of method identity. WHAT, WHY, and HOW may require multiple shared-parameter semantic forwards because they operate on different selected state sequences.

The exact loss reduction, branch balancing, optimizer schedule, PEFT configuration, and backward behavior are owned by `05_TRAINING_SPEC.md`.

---

## 9. High-level evaluation identity

The current evaluation separates WHERE prediction from semantic generation.

### 9.1 WHERE evaluation

WHERE is generated free-running from:

- the frozen same-subject support context;
- the query image/task;
- the oracle query trajectory length $N_q$.

The query GT coordinates, durations, and semantic annotations are not supplied to the WHERE generator.

### 9.2 Semantic evaluation

The current primary `multibranch` semantic evaluation is GT-trajectory-conditioned:

1. teacher-force the GT query scanpath through WHERE;
2. extract/project the GT-conditioned exploration states;
3. build independent WHAT, WHY, and HOW examples;
4. use gold/oracle groups to select WHY states;
5. generate semantic responses.

Thus the current semantic evaluation does not require predicted-to-GT fixation alignment or predicted group discovery.

A free-running predicted-state semantic path is not part of the current primary method.

The exact evaluation protocol and metrics are owned by `06_EVALUATION_SPEC.md`.

---

## 10. Interpretation boundaries

This identity document is descriptive rather than claim-setting.

The following should not be inferred from the architecture alone:

- that WHAT or WHY causally generates WHERE;
- that WHAT, WHY, and HOW form a human cognitive hierarchy;
- that gold WHY groups are literal internal cognitive episodes;
- that semantic pseudo-labels are direct measurements of private human cognition;
- that the model predicts autonomous scanpath termination;
- that the method has demonstrated cross-dataset generalization.

The semantic annotations are task-grounded supervisory descriptions. The method-level dependency from WHERE states to semantic branches describes the implemented information flow, not a cognitive-causal theory.

No stronger scientific or empirical claim is frozen in this file.

---

## 11. Current methodological boundaries

### 11.1 Dataset scope

The current executable scope is COCO-Search18. Other datasets are outside the current protocol unless separately specified.

### 11.2 Context cost

Raw support images and support scanpaths increase multimodal context cost with $K$.

### 11.3 No support retrieval

The canonical few-shot protocol consumes the frozen support demonstrations selected by `01_FEWSHOT_SPEC.md`; it does not learn or execute support retrieval.

### 11.4 Full-image semantic conditioning

Semantic execution receives the query image in addition to exploration states. For `multibranch`, `multi/04_SEMANTIC_MULTI_SPEC.md` defines the selected state subsequences; the separately defined flat baseline consumes the shared state representation according to `flat/04_SEMANTIC_SINGLE_SPEC.md`. Evaluation may therefore include controls that test how much semantic generation uses the exploration states.

### 11.5 Gold-group WHY

WHY does not discover group membership. Gold/oracle groups are supplied as conditioning structure in semantic training and semantic inference/evaluation.

### 11.6 GT-trajectory-conditioned semantic evaluation

The primary semantic evaluation uses states extracted from the teacher-forced GT query trajectory. It is therefore distinct from free-running WHERE evaluation.

### 11.7 Pseudo-label quality

WHAT, WHY, and HOW targets are offline semantic annotations defined by the dataset-creation protocol. Their quality limits the semantic supervision available to the model.

### 11.8 Oracle trajectory length

The current WHERE protocol receives $N_q$, so autonomous stopping is outside the primary method.

---

## 12. Method in one view

- **Dataset scope:** COCO-Search18 current protocol.
- **Backbone:** shared `OpenGVLab/InternVL3_5-8B-HF` causal multimodal VLM.
- **Personalization:** ordered raw same-subject image + task + observed-scanpath demonstrations.
- **Persistent user representation:** none.
- **Test-time parameter update:** none.
- **WHERE:** autoregressive `(x, y, duration)` scanpath language.
- **Fixation boundary/readout:** trainable atomic `<END_FIX>`.
- **Trajectory length:** oracle $N_q$ supplied.
- **Shared state:** final-layer hidden state at each query `<END_FIX>`.
- **Semantic interface:** one shared projector $P_E$ followed by direct continuous-state insertion.
- **Primary semantic mode:** `semantic_mode="multibranch"`.
- **Comparison semantic mode:** `semantic_mode="flat_single_output"`, defined canonically in `flat/04_SEMANTIC_SINGLE_SPEC.md` and not part of the core method identity.
- **WHAT (multibranch):** one independent generation task per fixation.
- **WHY (multibranch):** one independent generation task per **gold/oracle group**; group membership is not predicted.
- **HOW (multibranch):** one independent generation task per complete trajectory.
- **Semantic structured JSON:** none.
- **GROUP branch / grouping loss:** none.
- **Top-level semantic objective (multibranch):** branch-balanced $L_{SEM}$ from WHAT/WHY/HOW.
- **Total objective (multibranch):** $L_{WHERE}+L_{SEM}$ under the primary unit weights defined in `05_TRAINING_SPEC.md`.
- **Semantic evaluation:** GT-trajectory-conditioned and gold-group-conditioned.
- **Free-running predicted-state semantic path:** not part of the current primary method.
- **Paper claims:** intentionally not frozen by this identity specification.
