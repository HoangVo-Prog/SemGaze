# SemGaze — Evaluation Specification

## 1. Scope

This file is the canonical owner for the current SemGaze evaluation protocol, including paired semantic comparison between `semantic_mode="multibranch"` and `semantic_mode="flat_single_output"` under shared data/state/target conditions.

It defines:

- epoch-end test evaluation after complete query coverage;
- unseen-subject final evaluation;
- the free-running WHERE evaluation path;
- the GT-trajectory-conditioned semantic evaluation path for both semantic modes;
- the gold/oracle-group WHY evaluation contract shared by both modes;
- paired `multibranch` vs `flat_single_output` semantic-content comparison and flat-format validity reporting;
- draw-level aggregation;
- valid/invalid output accounting;
- personalization controls and component ablations that do not change the primary method definition.

It does **not** redefine:

- dataset creation, split membership, unseen subjects, support manifests, support sampling, or support order → `01_FEWSHOT_SPEC.md`;
- WHERE serialization, duration representation, oracle trajectory-length conditioning, parser, or generation policy → `02_WHERE_SPEC.md`;
- fixation-state extraction and projected-state construction → `03_STATE_SPEC.md`;
- primary multibranch WHAT/WHY/HOW targets, gold groups, prompts, and execution → `multi/04_SEMANTIC_MULTI_SPEC.md`;
- flat single-output prompt/execution, parsing, and `flat_format_valid` → `flat/04_SEMANTIC_SINGLE_SPEC.md`;
- training loss definitions or optimizer behavior → `05_TRAINING_SPEC.md`.

The current executable dataset/variant scope is:

```text
dataset = COCO-Search18
runtime variant = all
unseen subjects = {7, 8, 9}
evaluation K = {1, 5, 10}
final support draws = 10 frozen exclusive draws per K
```

Evaluation must consume the persisted split files and frozen support manifests produced by `01_FEWSHOT_SPEC.md`. It must not recreate split membership or resample frozen test supports.

---

## 2. Primary evaluation paths

The current method has two primary evaluation paths with different purposes.

### 2.1 Path A — free-running WHERE evaluation

For unseen subject $u^\star$, fixed support block $\widetilde{\mathcal C}_{u^\star}^{K,d}$, and query $(I_q,Q_q)$, evaluate:

$$
P_\theta
\left(
S_q
\mid
\widetilde{\mathcal C}_{u^\star}^{K,d},
I_q,
Q_q,
N_q
\right),
$$

where:

$$
N_q=|S_q^{GT}|.
$$

The model receives:

- the frozen ordered same-subject support context;
- the query image;
- the query task;
- the scalar oracle trajectory length $N_q$.

The model does **not** receive query GT:

- fixation coordinates;
- fixation durations;
- WHAT annotations;
- WHY text;
- HOW text;
- gold WHY groups.

WHERE generation is free-running and duration-enabled under `02_WHERE_SPEC.md`.

This path evaluates scanpath prediction only.

### 2.2 Path B — GT-trajectory-conditioned semantic evaluation for both modes

The primary semantic evaluation teacher-forces the GT query duration-aware trajectory through WHERE:

$$
S_q^{GT}
\rightarrow
F_q^{GT}
\rightarrow
R_q^{GT}.
$$

The shared semantic state source is the same for both modes. The corresponding semantic spec owns mode-specific consumption and decoding.

For the primary `semantic_mode="multibranch"`, evaluate the three semantic branches independently:

#### WHAT

For every fixation $t$:

$$
(I_q,Q_q,r_t^{GT})
\rightarrow
\hat w_t.
$$

#### WHY

For every gold/oracle group:

$$
G_m^{GT}=\{t_1<\cdots<t_{K_m}\},
$$

select:

$$
R_{m}^{Y,GT}
=
[r_{t_1}^{GT},\ldots,r_{t_{K_m}}^{GT}],
$$

then generate:

$$
(I_q,Q_q,R_m^{Y,GT})
\rightarrow
\hat y_m.
$$

Gold/oracle group membership is supplied as input-side state selection during semantic inference/evaluation, exactly as in semantic training.

The model does **not** predict group membership.

#### HOW

Use the complete chronological state sequence:

$$
(I_q,Q_q,R_q^{GT})
\rightarrow
\hat h.
$$

For `semantic_mode="flat_single_output"`, the same query set, query image/task, teacher-forced states, and gold/oracle WHY groups are used; `flat/04_SEMANTIC_SINGLE_SPEC.md` owns the one-response parsing contract. The extracted WHAT 1..N, WHY 1..M, and HOW content is evaluated against the same targets defined below.

Thus Path B evaluates semantic generation under the same GT trajectory and annotated gold WHY grouping for both modes.

### 2.3 No current predicted-state semantic path

A semantic path of the form:

$$
\hat S
\rightarrow
\hat F
\rightarrow
\hat R
\rightarrow
(\hat W,\hat Y,\hat H)
$$

is **not** part of the current primary evaluation protocol.

The current documents do not define:

- predicted-to-GT fixation matching;
- transfer of gold groups onto a predicted trajectory;
- predicted grouping;
- interpolation or nearest-fixation alignment.

Evaluation code must not invent such a mapping.

---

## 3. Information available by path

| Information | Free-running WHERE | GT-conditioned WHAT | GT-conditioned WHY | GT-conditioned HOW |
|---|---:|---:|---:|---:|
| Frozen support context | yes | indirectly through GT-conditioned states | indirectly through GT-conditioned states | indirectly through GT-conditioned states |
| Query image/task | yes | yes | yes | yes |
| Oracle trajectory length $N_q$ | yes | yes through WHERE teacher forcing | yes through WHERE teacher forcing | yes through WHERE teacher forcing |
| GT query trajectory content | no | yes, to construct $R^{GT}$ | yes, to construct $R^{GT}$ | yes, to construct $R^{GT}$ |
| Gold WHY group membership | no | no | **yes** | no |
| Generated WHAT as input | no | no | no | no |
| Generated WHY as input | no | no | no | no |

The table above describes the primary `multibranch` branch inputs. The flat baseline receives the corresponding shared query image/task, full chronological `R^{GT}`, and gold/oracle WHY-group bookkeeping according to `flat/04_SEMANTIC_SINGLE_SPEC.md`. Neither mode replays the raw support prefix; personalization reaches semantic execution through the support-conditioned states extracted from the teacher-forced WHERE computation.

---

## 4. Epoch-end test evaluation

COCO-Search18 has no validation split. Complete the current query-coverage epoch,
evaluate the unseen-subject test queries using the frozen draws below, log `test`
metrics, then continue training. Epoch boundaries must not be defined by arbitrary
optimizer-step counts. No automatic checkpoint-selection scalar is introduced.
Test support membership and order are frozen and cannot be resampled on overflow.

## 5. Unseen-subject final evaluation

The unseen subjects are fixed to:

$$
\mathcal U_{test}=\{7,8,9\}.
$$

For final evaluation:

```text
support source -> data/COCO_Search18/split_95_5/all/train.json
query source   -> data/COCO_Search18/split_95_5/all/test.json
```

For each:

$$
K\in\{1,5,10\},
$$

there are exactly 10 frozen support draws:

$$
d\in\{0,\ldots,9\}.
$$

For fixed `(K, draw_id)`:

- subjects 7, 8, and 9 use the same ordered `(task, image_name)` support trial keys;
- those trial keys resolve to the corresponding subject-specific observed scanpaths;
- one frozen ordered support block is reused for every final-test query of that subject;
- support is never resampled per query;
- support order is never changed;
- all eligible test queries of subjects 7, 8, and 9 are evaluated.

Support exclusivity across draw IDs and independence across different $K$ families are owned by `01_FEWSHOT_SPEC.md`.

### 5.1 Current runtime variant

The current evaluation runtime uses only:

```text
variant = all
```

Therefore evaluation must not:

- silently switch to `tp_only`;
- silently switch to `ta_only`;
- filter the loaded `all` test split by `condition`;
- reconstruct an original COCO-Search18 target-present benchmark split at runtime.

A separate target-present-only external comparison would require an explicitly separate protocol/version and is not the current canonical evaluation path.

---

## 6. Draw-level aggregation

For a metric $M$, let:

$$
M_{K,d}
$$

denote the metric computed over the full frozen final query set for one support draw $d$ at shot count $K$.

The canonical reported value for that $K$ is the mean over the 10 frozen draws:

$$
\boxed{
\bar M_K
=
\frac{1}{10}
\sum_{d=0}^{9}M_{K,d}
}
$$

When useful, also report draw-to-draw dispersion, but the mean remains the required central aggregate.

Do not first pool predictions from all draws into one artificial query set and compute a single metric over that pooled set.

For subject-level diagnostics, per-subject values may additionally be reported, but they do not replace the canonical draw-level aggregate.

---

## 7. WHERE metrics

### 7.1 Canonical scanpath metrics

The current canonical WHERE report contains:

- **ScanMatch (SM) ↑**;
- **MultiMatch (MM) ↑**;
- **String Edit Distance (SED) ↓**.

The concrete implementations and parameter conventions used for these metrics must be fixed in the evaluation code/configuration and kept identical across compared methods.

### 7.2 Duration-aware prediction

The current WHERE model predicts duration for every fixation.

However, this specification does not invent a new duration-only scalar metric. If a duration-sensitive variant of SM/MM/SED or an additional duration metric is used, the exact definition and implementation must be explicitly frozen and reported before it is treated as canonical.

Until then, duration prediction remains part of the model output and parser contract, while SM/MM/SED remain the frozen top-level scanpath metric names.

### 7.3 Oracle-length disclosure

All primary WHERE results must be labeled as **oracle-length-conditioned** because:

$$
N_q=|S_q^{GT}|
$$

is supplied in the prompt.

A comparison that uses a different stopping/length protocol must be identified explicitly rather than presented as directly protocol-equivalent.

---

## 8. WHERE output validity and failure accounting

The canonical parser and accepted fixation grammar are owned by `02_WHERE_SPEC.md`.

For every WHERE query, evaluation must record at least:

```text
accepted_fixation_count
requested_fixation_count = N_q
canonical_format_valid
under_generated
```

where:

```text
under_generated = (accepted_fixation_count < N_q)
```

Parser behavior remains:

- over-generation → retain the first $N_q$ accepted fixation tuples;
- under-generation → mark the trajectory invalid/failed;
- never fabricate missing fixations;
- never copy GT fixations to repair the output.

### 8.1 Metric handling for invalid trajectories

The benchmark-specific numeric treatment of an under-generated/invalid trajectory must be implemented consistently with the chosen SM/MM/SED evaluation implementation.

Evaluation code must additionally report the invalid/under-generation rate so failure cases cannot disappear silently through metric preprocessing.

If the selected metric implementation cannot score an invalid trajectory directly, the handling rule must be written into the resolved evaluation configuration before producing the final table; the evaluator must not choose a rule ad hoc per sample or per method.

---

## 9. Semantic evaluation unit

Semantic evaluation operates on validated gold annotations from `multi/04_SEMANTIC_MULTI_SPEC.md`. The semantic **scoring units** are the same for `multibranch` and `flat_single_output`, regardless of whether they are produced by separate branch responses or parsed from one flat response.

For one query with:

- $N$ fixations;
- $M$ gold/oracle WHY groups;

the evaluator scores exactly:

```text
N WHAT content units
M WHY content units
1 HOW content unit
```

In `multibranch`, these come from the corresponding independent semantic generations. In `flat_single_output`, they are extracted by the parser owned by `flat/04_SEMANTIC_SINGLE_SPEC.md`.

Neither mode predicts GROUP membership, so there is no GROUP metric. Neither mode uses semantic JSON output. For the flat baseline, `flat_format_valid` is reported only as a separate format diagnostic and must not replace WHAT/WHY/HOW content metrics.

---

## 10. Semantic output validity

For `multibranch`, each semantic branch returns one raw natural-language response per semantic example. For `flat_single_output`, parsing and `flat_format_valid` are owned entirely by `flat/04_SEMANTIC_SINGLE_SPEC.md`; this file consumes the parsed semantic fields for scoring.

Evaluation must check at minimum:

### WHAT

- exactly one generated response per GT fixation example;
- response is non-empty after allowed whitespace normalization.

### WHY

- exactly one generated response per gold/oracle group example;
- response is non-empty after allowed whitespace normalization;
- the evaluator uses the same gold group membership that selected the branch input states;
- no group matching is required because group membership is not predicted.

### HOW

- exactly one generated response per query;
- response is non-empty after allowed whitespace normalization.

The evaluator must not reconstruct a structured semantic JSON object merely for scoring. For the flat baseline, format validity is a separate diagnostic; validly extracted semantic fields are still evaluated with the same content metrics and targets.

---

## 11. Semantic text metrics

The semantic branches produce open-ended natural-language text.

The exact canonical automatic text metric set for WHAT, WHY, and HOW is **not frozen by this specification version**.

Candidate metrics may include metrics such as:

- BERTScore;
- ROUGE-L;

but they must not be presented as canonical until the experiment protocol explicitly freezes them.

When a text metric is selected:

- use the same implementation/version/settings for `multibranch` and `flat_single_output`;
- score WHAT 1..N against the same corresponding fixation-level GT WHAT texts in both modes;
- score WHY 1..M directly against the same GT WHY texts for the **same supplied gold groups** in both modes;
- score HOW against the same trajectory-level GT HOW text in both modes;
- report WHAT, WHY, and HOW separately rather than collapsing them into one undeclared scalar;
- do not replace semantic-content metrics with `flat_format_valid`.

Because both modes use supplied oracle groups and neither predicts grouping, there is no bipartite/Hungarian group matching, co-membership F1, ARI, group-count error, or other grouping metric in this comparison.

---

## 12. Personalization controls

Personalization controls are optional diagnostics. They do not change the primary protocol.

### 12.1 Correct-subject support

Use the canonical frozen same-subject support block for the query subject.

### 12.2 No-support control

Evaluate the corresponding non-personalized context with no support demonstrations.

This is a control condition outside the canonical $K_{\mathrm{eval}}\in\{1,5,10\}$ personalized evaluation settings and must be labeled as such.

### 12.3 Shuffled-subject support control

Replace the query subject's support behavior with support behavior from another subject while keeping the diagnostic comparison otherwise controlled as closely as possible.

Because this intentionally violates the canonical same-subject support contract, it must be implemented only as an explicitly named diagnostic and never mixed into the primary result table.

### 12.4 Reporting

Do not encode a required inequality such as `correct > shuffled` as a method assumption.

Instead, report the selected evaluation metrics for each control condition and interpret the observed differences empirically after experiments are run.

---

## 13. Current component ablations

Ablations must preserve the dataset split, query membership, frozen support units, and evaluation protocol unless the ablation explicitly targets one of those variables.

### A1. Semantic supervision

Compare:

$$
\mathcal L_{WHERE}
$$

against the primary joint objective:

$$
\mathcal L_{WHERE}+\mathcal L_{SEM}.
$$

`L_SEM` is the branch-balanced WHAT/WHY/HOW objective defined by `05_TRAINING_SPEC.md`.

There is no `L_STRUCT` in the current method.

### A2. Semantic branch balancing

The primary semantic objective gives equal scale-level weight to WHAT, WHY, and HOW after within-branch averaging.

Any alternative weighting/reduction is an explicit ablation and must be named/configured rather than silently replacing the primary reduction.

### A3. Fixation-state readout

Compare the readouts defined by `03_STATE_SPEC.md`:

- primary: $f_t=h_{\operatorname{END\_FIX}(t)}$;
- ablation: $f_t=h_{\operatorname{lastToken}(s_t)}$.

### A4. Shared projector

Compare:

- primary: $r_t=P_E(f_t)$;
- ablation: $r_t=f_t$.

### A5. Exploration-state utilization

For semantic generation compare:

$$
(I_q,Q_q,R_{selected})
$$

against an image/task-only semantic control:

$$
(I_q,Q_q).
$$

The exact removal adapter must preserve the rest of the branch prompt/target protocol.

### A6. Raw support replay in semantic branches

Primary semantic branches do not replay raw support demonstrations.

A diagnostic may compare against a variant that additionally supplies raw support context to the semantic forward, but this must be labeled as an alternative input design.

### A7. Support semantic content

The primary support contract is:

$$
(I,Q,S)_{sup}.
$$

A richer-support diagnostic may expose support semantic annotations only if explicitly defined as an ablation. Such a diagnostic is not the primary few-shot interface.

### 13.1 No group-discovery ablation in the current primary specification

The current method uses gold/oracle WHY groups in both semantic training and semantic inference/evaluation.

Predicted grouping would introduce a new component and a new evaluation problem; it is therefore outside the current ablation set unless separately specified as a future method variant.

---

## 14. Paired method comparison

For any direct method comparison on the current curated `all` benchmark:

- use the same final query records;
- use support demonstrations derived from the same frozen `(K, draw_id)` support units;
- preserve the same oracle-length protocol when claiming direct WHERE comparability;
- preserve the same metric implementation/settings;
- do not resample support independently for different methods.

For the direct `multibranch` vs `flat_single_output` semantic comparison, use the same curated records, query set, frozen few-shot episode, teacher-forced WHERE states, shared projector `P_E`, query image/task, gold WHY groups, and WHAT/WHY/HOW annotations. The intended comparison difference is semantic execution as defined by the corresponding semantic spec.

A baseline may encode selected support information differently in other architectural comparisons if required, but underlying support trial identity and query membership must remain paired.

---

## 15. Reporting separation

Primary results should keep the following conceptually separate:

### WHERE

- SM;
- MM;
- SED;
- invalid/under-generation rate;
- $K$ and draw aggregation;
- explicit oracle-length conditioning disclosure.

### Semantic WHAT

- fixation-level text metric(s), once frozen;
- response validity rate.

### Semantic WHY

- gold-group-conditioned WHY text metric(s), once frozen;
- response validity rate.

### Semantic HOW

- trajectory-level text metric(s), once frozen;
- response validity rate.

### Flat baseline format diagnostic

- `flat_format_valid`, as defined and parsed only by `flat/04_SEMANTIC_SINGLE_SPEC.md`;
- report it separately from WHAT/WHY/HOW semantic-content metrics.

Do not report:

- GROUP discovery metrics;
- semantic JSON parse rate;
- one monolithic WHAT/GROUP/WHY/HOW structured-output score.

---

## 16. Required evaluation assertions

Evaluation must fail loudly if:

- the loaded benchmark variant is not `all`;
- unseen subjects differ from `{7,8,9}`;
- final support blocks differ from the frozen manifest;
- a final support record is not in `all/train.json`;
- a final query is not in `all/test.json`;
- a frozen support list is reordered or resampled;
- a WHERE query is evaluated without the declared oracle-length protocol;
- WHERE parsing fabricates missing fixations;
- semantic evaluation uses a predicted-state-to-GT matching heuristic;
- WHY is evaluated without the supplied gold/oracle group selection;
- either semantic mode predicts or scores GROUP membership;
- semantic evaluation code expects structured JSON output from either current mode;
- the flat baseline is evaluated without the parser/`flat_format_valid` contract owned by `flat/04_SEMANTIC_SINGLE_SPEC.md`;
- `flat_format_valid` is substituted for WHAT/WHY/HOW semantic-content metrics;
- different compared methods receive different frozen support units for the same paired comparison;
- metric configuration changes across methods within the same reported comparison.

---

## 17. Current unresolved evaluation choices

The following choices are intentionally **not** silently invented by this specification and must be frozen in the experiment configuration before final reporting:

1. exact implementation/settings for SM, MM, and SED;
2. whether and how duration receives an additional duration-sensitive metric beyond the canonical SM/MM/SED names;
3. exact numerical handling used by the selected scanpath metric implementation for an under-generated invalid trajectory;
4. exact automatic text metric(s) for WHAT, WHY, and HOW;
5. the scalar metric used for checkpoint selection if checkpoint selection requires one metric across several reported quantities.

These are evaluation-configuration decisions, not permission for runtime code to make sample-specific choices.

---

## 18. Primary evaluation invariants

- Current dataset/variant is COCO-Search18 `all`.
- Unseen subjects are exactly 7, 8, and 9.
- Canonical evaluation shot settings are $K_{\mathrm{eval}}\in\{1,5,10\}$; this does not define the train-time K distribution.
- Final evaluation uses 10 frozen exclusive support draws per evaluation $K$.
- One frozen support block is reused across the full corresponding query set.
- Subjects 7, 8, and 9 use the same ordered support trial identities for a fixed `(K, draw_id)`, resolved to subject-specific scanpaths.
- WHERE is free-running but oracle-length-conditioned.
- WHERE predicts duration as part of every fixation tuple.
- Primary semantic evaluation is GT-trajectory-conditioned.
- WHAT is evaluated one fixation target at a time.
- WHY is gold/oracle-group-conditioned in both training and inference/evaluation.
- GROUP membership is not predicted or scored by either semantic mode.
- HOW is evaluated once per complete trajectory.
- Semantic content units remain WHAT, WHY, and HOW under both modes. `multibranch` emits independent raw-text responses; `flat_single_output` emits one ordinary response whose parsing is owned by `flat/04_SEMANTIC_SINGLE_SPEC.md`. Neither mode uses structured JSON.
- No predicted-state semantic path is part of the current primary method.
- No `L_STRUCT` or grouping objective exists in current evaluation ablations.
- Draw-level means are computed over the 10 frozen support draws.
- Invalid WHERE outputs are never silently repaired.
