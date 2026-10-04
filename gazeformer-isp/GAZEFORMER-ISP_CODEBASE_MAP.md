# Gazeformer-ISP Codebase Map

## 0. Map Metadata

- Mapped directory: `gazeformer-isp/`; inspected 2026-09-25.
- Purpose: implementation navigation before changing architecture, branches, data contracts, objectives, or evaluation. Static source map, not a reproduction report or paper summary.
- Source code remains the source of truth. Inventory: 143 Python files, eight shell launchers, one environment YAML; no experiment YAML shipped. No training, extraction, checkpoint loading, or dataset-dependent evaluation was executed.
- **Path notation:** paths below are relative to this mapped directory. `G(D)` means `<D>/GazeformerISP/src`; `C(D)` means `<D>/ChenLSTMISP/src`; `D` is exactly `AiR`, `OSIE`, `OSIE_ASD`, or `COCO_Search18`. Unqualified `G/...` uses `G(OSIE)` as representative and applies to other Gazeformer copies unless differences are stated. Example: `G/models/models.py::Transformer.forward` resolves to `OSIE/GazeformerISP/src/models/models.py::Transformer.forward`. These are independent copies, not a shared package.
- `train.py::main.train` denotes a nested function inside `main`; `<module>` denotes executed top-level code, not an invented function.

## 1. Architecture at a Glance

There are **four dataset-specific Gazeformer applications and four separate ChenLSTM applications**, not one configurable model factory. Each has its own parser, datasets, model, losses, sampler, and evaluation utilities.

```text
Offline image/text extraction → .pth image tokens + embeddings.npy
annotation JSON → grouped dataset → collate → flattened observer examples
    → gazeformer(src, subjects, task)
        → projected visual tokens → transformer encoder
        → task attention + subject attention → spatial/channel integration
        → parallel fixation-query decoder
        → subject-weighted spatial/stop heads + log-normal duration parameters
    → supervised CE + duration NLL, then sampled ScanMatch policy-gradient training
    → Sampling → structured fixation vectors → pairwise / subject-aware metrics
```

Gazeformer training does **not** run an image backbone or text encoder. Personalized processing is inside the model; no separately launched personalization stage exists. ChenLSTM consumes pixels and runs a dilated ResNet plus recurrent spatial/semantic memory. Several shipped paths are inconsistent or not runnable as written; see §§12–14, particularly the AiR training import failure.

## 2. Repository Structure

Repeated subtrees are shown once to avoid implying that the copies share code.

```text
gazeformer-isp/
├── README.md                      # orientation/external assets, not runtime specification
├── user_scanpath.yml              # environment pins, not training configuration
├── AiR/                           # question-conditioned examples; subject_idx IDs
├── OSIE/                          # free-viewing; subject IDs
├── OSIE_ASD/                      # free-viewing; 39-subject parser default
└── COCO_Search18/                  # task/image groups; one-based annotation subject IDs
    (each dataset directory has README.md and both subtrees below)

<D>/GazeformerISP/
├── bash/train.sh                  # snapshot source, select GPUs, invoke src/train.py
└── src/
    ├── train.py                   # construction, two stages, validation, checkpoint IO
    ├── test.py                    # independent parser/checkpoint evaluation; COCO writes JSON
    ├── opts.py                    # training argparse + optional YAML/override merge
    ├── dataset/dataset.py         # supervised/RL/evaluation datasets and collators
    ├── preprocess/
    │   ├── feature_extractor.py   # ResNetCOCO image_data and SentenceTransformer text_data
    │   └── preprocess_fixations.py # top-level conversion; dataset mismatches exist
    ├── models/
    │   ├── gazeformer.py          # gazeformer, CrossAttentionPredictor, output attention_module
    │   ├── models.py              # transformer layers/wrappers, attention, integration
    │   ├── positional_encodings.py # fixed 2D sine positions
    │   ├── loss.py                # supervised likelihoods and sampled log probabilities
    │   └── sampling.py            # sampling, termination, fixation-vector conversion
    └── utils/
        ├── evaluation.py         # comprehensive_evaluation_by_subject, pairs_eval, p2g/g2p
        ├── checkpointing.py      # current/best checkpoint serialization
        ├── recording.py          # epoch/iteration/best-metric history
        ├── config.py             # YACS-derived CfgNode, _BASE_ inheritance
        ├── logger.py             # file/console metric logging
        └── evaltools/
            ├── scanmatch.py      # directly used discretization/alignment
            └── visual_attention_metrics.py # directly used SED/STDE implementations

<D>/ChenLSTMISP/
├── bash/train.sh                  # independent recurrent-model launcher
└── src/
    ├── train.py, test.py, opts.py # independent training/evaluation/configuration
    ├── dataset/dataset.py         # pixels, targets, optional attention-map/task inputs
    ├── models/
    │   ├── baseline_attention.py # baseline, ConvLSTM, memory attention, predict_head
    │   ├── resnet.py             # local pretrained ResNet implementation
    │   ├── loss.py               # local objective copy
    │   └── sampling.py           # local sampling copy
    └── utils/                    # local evaluation/checkpoint/record/config/logger/evaltools
```

Additional paths:

- `AiR/ChenLSTMISP/src/preprocess/preprocess_fixations.py` and `OSIE/ChenLSTMISP/src/preprocess/preprocess_fixations.py`: annotation preparation, not engines.
- `OSIE_ASD/{GazeformerISP,ChenLSTMISP}/src/visualization/{vis_tsne.py,leave_one_out_classification.py}`: checkpoint-derived observer-feature analysis; stale imports prevent treating these as working inference launchers.
- `COCO_Search18/GazeformerISP/src/visualization/vis_repeat_results_for_qual.py`: reads saved predictions/annotations and plots scanpaths, not model inference.
- `COCO_Search18/ChenLSTMISP/src/visualization/{vis_gt.py,vis_results_for_qual.py,vis_repeat_results_for_qual.py}`: offline plotting consumers. Some branches also have `src/README.md`; descriptions there do not override executable paths.

## 3. Execution Flows

### Training flow

Launch from `<D>/GazeformerISP/`: `bash bash/train.sh` invokes `python src/train.py`. Shell scripts depend on this working directory and Unix `rsync`/`cp`. No root dispatcher exists.

```text
<D>/GazeformerISP/bash/train.sh
  → G(D)/train.py::<module>: imports → opts.parse_opt → seeds → transform object
  → G(D)/train.py::main
      → select/create run directory; write hparams.json on fresh run
      → construct supervised, RL, evaluation datasets
      → DataLoader(..., collate_fn=dataset.collate_func, num_workers=4)
      → models.models.Transformer(...) → models.gazeformer.gazeformer(...)
      → Sampling(...) → AdamW(model.parameters(), ...)
      → RecordManager.init_record/load → CheckpointManager(...)
      → optional checkpoint.pth model + optimizer restore
      → LambdaLR(main.lr_lambda, last_epoch=iteration) → optional DataParallel
      → epochs range(start_epoch + 1, args.epoch)
          → main.train(iteration, epoch): supervised or RL
          → main.validation(iteration), only if epoch > no_eval_epoch
          → harmonic mean of validation ScanMatch values (otherwise -1)
          → CheckpointManager.step → RecordManager.save
          → optional shell copy of run at start_rl_epoch - 1
```

**AiR exception:** `AiR/GazeformerISP/src/train.py` imports four nonexistent evaluation functions at module scope. The downstream flow exists in source but is unreachable through ordinary imports as shipped (§12).

| Branch | Supervised / RL / evaluation symbols in `G(D)/dataset/dataset.py` | Grouping; validation split |
| --- | --- | --- |
| OSIE | `OSIE`, `OSIE_rl`, `OSIE_evaluation` | image `name`; `validation` |
| OSIE_ASD | `OSIE`, `OSIE_rl`, `OSIE_evaluation` | image `name`; `validation` |
| AiR | `AiR`, `AiR_rl`, `AiR_evaluation` | `question_id`; `validation` |
| COCO_Search18 | `COCOSearch`, `COCOSearch_rl`, `COCOSearch_evaluation` | `task/name`; `valid` |

### Supervised and policy-gradient flows

```text
main.train, epoch < start_rl_epoch:
  model.train → supervised tensors → CUDA → view(-1, *shape[2:])
  → model(src=images, subjects=subjects, task=task_embeddings)
  → CrossEntropyLoss(actions, target_scanpaths, action_masks)
    + lambda_1 * MLPLogNormalDistribution(mu, sigma2, durations, duration_masks)
  → backward → optional clip_grad_norm_ → optimizer.step → scheduler.step

main.train, epoch >= start_rl_epoch:
  model.eval (gradients still enabled) → RL data, flatten GT observer lists
  → one model forward → all_actions_prob, mu, sigma2
  → repeat accepted rl_sample_number times:
      Sampling.random_sample → Sampling.generate_scanpath
      → pairs_eval(GT, prediction, duration/non-duration ScanMatch)
      → reject whole draw if any reward is NaN
      → LogAction + LogDuration on sampled actions/detached, clipped durations
  → reward = harmonic mean of reward columns 5:7 (two ScanMatch scores)
  → baseline = mean reward across accepted draws, separately per example
  → sum((negative log action + negative log duration) * (reward - baseline))
  → backward → optional clipping → same optimizer/scheduler
```

This is a sampled-mean baseline, **not a greedy-decoded baseline**. Stage switching neither rebuilds the model nor reloads the best checkpoint.

### Validation / inference flow

```text
G(D)/train.py::main.validation or G(D)/test.py::main
  → evaluation dataset.collate_func → flatten tensors, keep grouped fix_vectors
  → model.eval → no_grad model forward (all sequence positions at once)
  → Sampling.random_sample repeated eval_repeat_num times
  → Sampling.generate_scanpath → fixed subject_num slices regroup predictions
  → comprehensive_evaluation_by_subject → metric means, stds, score details
  → training: TensorBoard/logger and checkpoint criterion
  → test: logger; COCO_Search18 additionally test_predicts.json
```

`test.py` parses at module scope using its **own parser**, reconstructs the model directly, and loads `<evaluation_dir>/checkpoints/...`; it does not read `hparams.json` or call `opts.parse_opt`. Gazeformer test loads `checkpoint.pth` for AiR/OSIE/OSIE_ASD and `checkpoint_best.pth` for COCO. `--mode` does not select the split: construction explicitly uses `type="test"`.

### Other executable modes

- `G(D)/preprocess/feature_extractor.py::<module>` parses `dataset_path`, `lm_model`, `cuda`. OSIE/ASD/COCO invoke `image_data(overwrite=True)` and `text_data`; AiR invokes only `text_data` (image call commented out).
- `G(D)/preprocess/preprocess_fixations.py::<module>` executes conversion immediately with hard-coded paths, no main/CLI. OSIE reads MATLAB fixations, fixes test images, seeds/shuffles the remainder, splits train/validation, and writes combined `processed/fixations.json`. ASD and COCO copies are identical OSIE converters, not confirmed dataset-specific preparation.
- `AiR/GazeformerISP/src/preprocess/preprocess_fixations.py::<module>` reads question/answer/fixation inputs, derives image dimensions and sorted subject-ID indices, writes three `AiR_fixations_<split>.json` files.
- `C(D)/train.py::main` has the same nested supervised/RL/validation lifecycle but constructs `baseline` and `Adam`; `C(D)/test.py::main` is separate. These are alternate architectures, not later Gazeformer stages.

## 4. Core Component Map

| Component | Path / Symbol | Responsibility | Main Inputs | Main Outputs | Used By |
| --- | --- | --- | --- | --- | --- |
| Experiment control | `G(D)/train.py::main`, `main.train`, `main.validation` | Construct/run both stages | args, batches, state | parameters, metrics, checkpoints | shell |
| Test entry | `G(D)/test.py::main` | Reconstruct/load/evaluate | test args, checkpoint | metrics; COCO JSON | Python entry |
| Grouped data | `G(D)/dataset/dataset.py` classes in §3, `.__getitem__`, `.collate_func` | Group observers, load features, create targets | annotations, caches | tensors + metadata | train/test loaders |
| Offline visual features | `G/preprocess/feature_extractor.py::ResNetCOCO.forward`, `image_data` | Run pretrained backbone body | normalized pixels | cached tokens | preprocessing only |
| Offline language features | `G(D)/preprocess/feature_extractor.py::text_data` | Encode prompt/question | strings | embeddings dictionary | dataset lookup |
| Top-level model | `G/models/gazeformer.py::gazeformer` | Embedding, queries, prediction heads | src, subjects, task | prediction dictionary | train/test |
| Transformer composition | `G/models/models.py::Transformer` | Encoder → personalization → decoder | tokens, zero targets, vectors | memory, sequence states | gazeformer |
| Task attention | `G/models/models.py::TransformerEncoderWrapper`, `AttentionModule` | Project modalities; score spatial tokens | image/task vectors | memory, visual_att | Transformer |
| Observer attention | `G/models/models.py::SubjectAttentionModule.forward` | Score memory using observer | memory, subject vector | spatial weights | Transformer |
| Personalized integration | `G/models/models.py::IntegrationModule.forward` | Combine weighted visual/observer features | memory, two attentions, subject | personalized memory | Transformer |
| Decoder | `G/models/models.py::TransformerDecoderWrapper`, `TransformerDecoder`, `TransformerDecoderLayer` | Query self-/cross-attention + FFN | zeros, query positions, memory | sequence states | Transformer |
| Spatial heads | `G/models/gazeformer.py::CrossAttentionPredictor.forward` | Predict spatial log attention | states, memory | log map per module | gazeformer head loop |
| Map prioritization | `G/models/gazeformer.py::attention_module.forward` | Subject-dependent map mixing | memory, log maps, subject | per-step map weights | training_process/inference |
| Duration heads | `G/models/gazeformer.py::gazeformer.__init__`: `generator_t_mu`, `generator_t_logvar` | Log-duration parameters | states | mu, exp(logvar) | loss/sampler |
| Objectives | `G/models/loss.py` (§7) | Masked target/sample likelihood | predictions, targets, masks | losses/log scores | main.train |
| Sampling | `G/models/sampling.py::Sampling` | Sample/terminate/convert | probabilities, duration parameters | vectors, masks | RL/evaluation |
| Metrics | `G(D)/utils/evaluation.py::pairs_eval`, `comprehensive_evaluation_by_subject`, `p2g` | Rewards and subject metrics | fixation vectors | arrays/dictionaries | RL/evaluation |
| Persistence | `G/utils/checkpointing.py::CheckpointManager`, `G/utils/recording.py::RecordManager` | Weights/optimizer and epoch history | model, optimizer, score | .pth, JSON | main |
| Alternate model | `C(D)/models/baseline_attention.py::baseline` | Pixel backbone + personalized recurrent prediction | pixels, subject, optional guidance | action/duration sequences | ChenLSTM entry points |

## 5. Model Architecture

### Construction and representations

No registry, `build_model`, or separate base-model subclass exists: `G(D)/train.py::main` and `G(D)/test.py::main` each construct `models.models.Transformer(...)`, pass it to `models.gazeformer.gazeformer(...)`, then move to CUDA. Both inherit `nn.Module`.

Important interfaces/attributes in `G/models/gazeformer.py`:

- `gazeformer.__init__(transformer, spatial_dim, args, subject_num, subject_feature_dim, action_map_num, dropout=0.4, max_len=7, patch_size=16, device="cuda:0")`: owns transformer, observer/query embeddings, positions, duration/stop/spatial heads, and map mixing.
- `gazeformer.forward(self, src: Tensor, subjects: Tensor, task: Tensor)`: dispatches to `training_process` or `inference` according to `self.training`. They contain essentially duplicated prediction code.
- `subject_embed = nn.Embedding(subject_num, subject_feature_dim)`: learned discrete observer table; no history-based observer encoder or unseen-user inference path.
- `querypos_embed = nn.Embedding(max_len, hidden_dim)`: learned positional queries. Decoder content starts at zero, not ground truth or sampled fixations.
- `models/positional_encodings.py::PositionEmbeddingSine2d`: fixed sine tensor stored as `nn.Parameter(requires_grad=False)`; `forward(x)` adds it to x. Image positions serve transformer and prediction attention.
- `queryfix_embed` is allocated at pixel resolution but unused: first-fixation initialization is commented out. `get_fixation_map` and `reparameterize` are not called by forward; the former references `x_grid/y_grid` not initialized by this class. Do not infer an active coordinate-Gaussian spatial head from these helpers.

### Actual forward graph

Let `N` = flattened observer examples, `S=im_h*im_w`, `T=max_length`, `E=hidden_dim`, `U=subject_feature_dim`, `K=action_map_num`. Shapes follow explicit projections/reshapes; actual cache contents were not inspected.

```text
src [N,S,img_hidden_dim]                         task [N,lm_hidden_dim]
  → TransformerEncoderWrapper.input_proj → [S,N,E]       │
  → TransformerEncoder + 2D patch positions             text_transform
  → img_transform → memory [S,N,E]                     → [N,E]
                       ├─ AttentionModule(memory batch-first, task) → visual_att [N,S]
subjects [N] → gazeformer.subject_embed → [N,U]
                       ├─ SubjectAttentionModule(memory, subject) → [S,N] → [N,S]
                       └─ IntegrationModule(memory, visual_att, subject_att, subject)
                              → personalized memory [S,N,E]
zeros [T,N,E] + querypos_embed [T,1,E]
  → TransformerDecoderWrapper → TransformerDecoder layers
      self-attention across queries; memory cross-attention; FFN
  → outs [T,N,E] → top-level dropout
      ├─ generator_t_mu → [T,N,1] → nominal mu [N,T]
      ├─ exp(generator_t_logvar) → nominal sigma2 [N,T]
      ├─ K CrossAttentionPredictor modules → log maps [N,T,K,S]
      └─ token_predictor → stop logits [N,T,K]
log maps + personalized memory + subject embedding
  → gazeformer.attention_module → weights over K [N,T,K]
  → weighted sum of [stop logit | spatial log map] → [N,T,S+1]
  → training: actions logits; eval: softmax → all_actions_prob
  → weighted spatial log map → action_map [N,T,im_h,im_w]
```

`G/models/models.py::TransformerEncoderWrapper.forward` projects task only for `AttentionModule`: it does not concatenate text tokens into encoder/decoder memory. Its `subjects` argument is unused there. `Transformer.forward` applies subject attention/integration after the encoder.

`IntegrationModule.forward(features, visual_att, subject_att, subjects)`:

1. Multiply batch-first memory by task and subject attention separately; concatenate along channels to `[N,S,2E]`.
2. Sum channels → `vision_spatial_feature`, add `subj_spatial_feature(subjects)` → `[N,S,1]`.
3. Sum spatial positions → `vision_semantic_feature`, add `subj_semantic_feature(subjects)` → `[N,1,E]`.
4. Outer product → `[N,S,E]`; concatenate with original memory and apply `linear_layer(2E→E)`.

`TransformerDecoderLayer.forward` selects pre/post normalization; entry-point construction uses default post-normalization. No causal/target padding mask is passed. `TransformerDecoder.return_intermediate` defaults false and is not enabled by entries. No intermediate decoder loss is wired.

`CrossAttentionPredictor.forward` runs its own query self-attention, then memory cross-attention and returns `log(attention_weights + 1e-16)`, not attention output vectors. The output `attention_module` aggregates memory using these **log maps**, averages over spatial positions, combines observer projections, and softmaxes over K. Final aggregation is a weighted sum of log maps/logits, not a mixture of normalized spatial probabilities. Returned `action_map` excludes stop and is not softmax-normalized.

Duration paths use bare `.squeeze()` after permutation; `[N,T]` survives only when neither dimension is one. No auxiliary head is trained. `TransformerDecoderWrapper.linear(3E→E)` is constructed but never called.

### Offline backbone and language encoder

`G/preprocess/feature_extractor.py::ResNetCOCO` constructs `maskrcnn_resnet50_fpn(weights=...COCO_V1).backbone.body`, runs stem/layers 1–4 explicitly, and flattens to `[batch,spatial,channels]`. Detector heads and FPN output processing do not execute. `image_data` resizes to `(768,1024)`, normalizes using ImageNet mean/std, saves `.squeeze().detach().cpu()` tokens. Cached channels must match `img_hidden_dim` (default 2048).

`G/models/models.py::ResNetCOCO` also exists and is imported by `gazeformer.py`, but is **not instantiated** by active model construction. Editing it does not change training inputs. `text_data` uses `SentenceTransformer(...stsb-roberta-base-v2).encode`; training receives saved vectors, not strings/token IDs.

### ChenLSTM sibling boundary

`C(D)/models/baseline_attention.py::baseline` is independent, not a Gazeformer subclass. Its constructor calls local `models/resnet.py::resnet50(pretrained=True)` and `baseline.dilate_resnet`, removes classification layers, adds `sal_conv`, `ConvLSTM`, subject embedding/projections, `subject_attention_module`, `spatial_att`, `semantic_att`, and `predict_head`.

`baseline.training_process/inference` run pixels through the backbone, combine initial guidance and subject attention with spatial/channel projections, recurrently call `ConvLSTM.forward` for `convLSTM_length` steps, then update memory from predicted action maps. `predict_head.forward` produces subject-weighted map/stop logits and duration parameters; inference exposes `all_actions_prob`. This recurrent feedback is absent from Gazeformer.

| Branch | Actual `baseline.forward` interface | Extra conditioning |
| --- | --- | --- |
| OSIE / OSIE_ASD | `(images, subject)` | initial zero attention map |
| AiR | `(images, subject, attention_maps)` | question-specific .npy attention bbox map |
| COCO_Search18 | `(images, subject, attention_maps, tasks)` | detector bbox map; batch `taskints` selects one of 18 `object_sal_layer` ModuleDict convolutions |

ChenLSTM datasets execute resizing/normalization. Recurrent layers hard-code `30×40`, `1200`, and channels `512`; changing map CLI sizes alone is insufficient. AiR/COCO train constructors do not pass `action_map_num`, unlike OSIE/ASD. Do not interchange family checkpoints/configuration.

## 6. Data Pipeline

### Annotation and cache contracts

```text
raw annotations → preprocess_fixations.py (or externally supplied JSON)
image/question → image_data/text_data (or externally supplied caches)
→ G(D)/dataset/dataset.py::<Dataset>.__init__: filter JSON, build group index
→ __getitem__: load .pth, duplicate per observer annotation, lookup task vector
→ supervised targets OR variable-length evaluation fix_vectors
→ collate_func: concatenate observer tensors, add singleton leading dimension
→ train/test: .cuda(), view(-1, *tensor.shape[2:]) → flattened observer examples
```

One item is an image/question/task-image group, **not one observer**. Group/observer order follows JSON insertion order; datasets neither sort observers nor synthesize missing IDs. The transform is stored but not executed by active Gazeformer datasets; cached features replace pixel transforms.

| Branch | File/annotation contract and task lookup | Spatial/duration conversion |
| --- | --- | --- |
| OSIE / OSIE_ASD | `<fix_dir>/fixations.json`, filter `split`; `name, subject, X, Y, T, length`; feature `<feat_dir>/<name with jpg→pth>`; embedding key `free-viewing` | Supervised: `(X-1)/(origin_width/im_w)` and analogous Y, integer cast. RL/eval: original-to-resized coordinate scaling, no -1. T milliseconds → seconds. |
| AiR | `AiR_fixations_<type>.json`; `question_id, image_id, subject_idx, height, width, X, Y, T_start, T_end, length, subject_answer, answer`; image-ID feature; question-ID embedding | Per-record dimensions. Supervised X/Y divided by grid scale **without -1**; eval resized coordinates. Duration `(T_end-T_start)/1000`. |
| COCO_Search18 | `fixations.json`; `split` train/valid/test; `task, name, subject, X, Y, T, length`; feature `<feat_dir>/<task>/<name>.pth`; task embedding | Subject becomes `subject-1`. Grouped supervised class scales `(X-1)` using **resize/grid**, not original/grid. RL/eval use X/Y directly. Annotations must already match resized coordinates. T/1000. |

Embeddings load from the parent of `stimuli_dir`: COCO uses `join(stimuli_dir, '..', ...)`, others split the path on `/`. OSIE/ASD/COCO `text_data` currently emit only free-viewing: COCO task embeddings and task-subfolder features are not produced by its shipped extractor. AiR extraction reads all three split JSONs and encodes questions.

### Targets and batch fields

Per observer, allocate T steps and `S+1` actions. Index 0 is termination; spatial index is `1 + y*im_w + x`. A fixation becomes one-hot, optionally Gaussian-blurred/renormalized when `blur_sigma` is truthy. Unused steps target termination, but `action_mask` includes only real fixations and the first stop (if length < T). `duration_mask` includes real fixations only. Supervised targets truncate long GT; RL/eval vectors use annotation length.

| Batch field | Producer: item key → collated key; meaning/shape | Consumer |
| --- | --- | --- |
| `images` | `image` → cached tokens `[1,N,S,img_hidden_dim]` | flatten to `[N,S,C]`; model src; sampler device/count |
| `subjects` | `subject` → indices `[1,N]` | flatten `[N]`; subject_embed |
| `task_embeddings` | `task_embedding` → `[1,N,lm_hidden_dim]` | model task → text_transform |
| `target_scanpaths` | `target_scanpath` → distributions `[1,N,T,S+1]` | CrossEntropyLoss |
| `durations` | `duration` → seconds, zero padding `[1,N,T]` | MLPLogNormalDistribution |
| `action_masks` | `action_mask` → valid fixation/stop `[1,N,T]` | action loss |
| `duration_masks` | `duration_mask` → fixations only `[1,N,T]` | duration loss |
| `fix_vectors` | RL/eval lists per group then observer; structured NumPy arrays with `start_x, start_y, duration`, all f8 | RL flattens lists for pairs_eval; evaluation preserves groups |
| `img_names` | `img_name` → one string per group | fixed subject_num prediction regrouping; COCO JSON |
| `tasks` | OSIE/ASD/COCO `task` lists | COCO JSON; model uses embeddings, not these strings |
| `firstfixs` | OSIE/ASD/COCO RL/eval `firstfix`: image center, Y/X order | collated but not passed to Gazeformer; absent in AiR |
| `qids, performances` | AiR `qid, performance` (answer equality and not literal `"faild"`) | metadata, not read by Gazeformer objectives |

RL/eval do not create supervised targets/masks; sampling creates masks for generated sequences. Numeric arrays use `torch.from_numpy`; every tensor gains a leading singleton, while nested lists remain lists.

**Adding a field:** update the selected branch's applicable three `__getitem__/collate_func` pairs, then explicit batch-key lists/unpacking in supervised train, RL train, validation, and test. One dataset-class edit does not propagate to the other two.

ChenLSTM grouping is analogous, but `images` are normalized RGB and supervised targets are consumed as `scanpaths`. AiR adds `attention_maps` from `<att_dir>/<qid>.npy`; COCO builds maps from category/threshold-filtered `coco_search18_detector.json` and adds `taskints`. These are actual model arguments, unlike Gazeformer AiR performance metadata.

## 7. Training and Objectives

### Stages and optimization

| Stage/mode | Entry and model state | Initialization / trainability | Update behavior |
| --- | --- | --- | --- |
| Supervised | `G(D)/train.py::main.train`, epoch < start_rl_epoch; model.train | Fresh modules or resume; fixed sine positions have requires_grad=False; no stage-specific freezing | CE + weighted duration NLL; optional clip; optimizer/scheduler per batch |
| Policy gradient | same function, epoch >= start_rl_epoch; **model.eval with autograd enabled** | Same parameters/optimizer; no reinitialization, best-weight load, or selective unfreezing | sampled-mean ScanMatch advantage; action + duration terms, unweighted relative to each other |
| Validation | `main.validation`; model.eval, forward under no_grad | current weights | no update; called each epoch strictly after no_eval_epoch |
| ChenLSTM alternative | `C(D)/train.py::main.train` | same two-stage policy; backbone inside trainable model; no explicit requires_grad freeze | local copies of losses/reward flow, but Adam instead of AdamW |

Gazeformer optimizer is `AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.98), eps=1e-9, weight_decay=args.weight_decay)`. Defaults: lr `1e-4`, weight decay `5e-5`. No named parameter groups. Unused parameters are registered but not thereby executed. Cached backbone/text features are outside the training graph, rather than backbone modules frozen in it. ChenLSTM uses `Adam(... betas=(0.9,0.999), eps=1e-8)`.

`G/train.py::main.lr_lambda(iteration)` uses loader lengths and iteration boundaries:

- Through `len(train_loader)*warmup_epoch`: linear warmup, factor `iteration / warmup_steps`.
- Through `len(train_loader)*start_rl_epoch`: linear decay from warmup to zero.
- Thereafter: `rl_lr_initial_decay * (1 - elapsed_rl_iterations / (len(train_rl_loader)*(epoch-start_rl_epoch)))`, where `epoch` in this formula is `args.epoch`, the configured total.

Stage switch uses **epoch**, schedule uses **iteration**. `RecordManager` initializes both counters to -1; fresh outer epochs start at 0. No early stopping; total epoch range has exclusive upper bound. Changing loader length on resume changes the reconstructed schedule.

### Losses

| Loss / Objective | Defined At | Inputs | Applied To | Weight / Config |
| --- | --- | --- | --- | --- |
| Target action CE | `G/models/loss.py::CrossEntropyLoss(input, gt, mask)` | logits, target distribution, action mask | stop + spatial actions; softmax then log with epsilon; total divided by mask.sum | coefficient 1 |
| Duration NLL | `G/models/loss.py::MLPLogNormalDistribution(log_normal_mu, log_normal_sigma2, gt, mask)` | mu, variance, GT seconds, duration mask | valid fixations; log-normal log PDF selected at mask==1 | `lambda_1`, default 1 |
| Sampled action log score | `G/models/loss.py::LogAction(input, mask)` | gathered sampled probabilities, generated action mask | per-example time sum of log probabilities / **whole-batch** mask.sum | negative score × advantage in RL |
| Sampled duration log score | `G/models/loss.py::LogDuration(input, log_normal_mu, log_normal_sigma2, mask)` | detached sampled seconds clipped [0,100], parameters, generated mask | per-example log-PDF sum / **whole-batch** mask.sum | negative score × same advantage; lambda_1 not used |
| RL reward/baseline | `G/train.py::main.train` + `G(D)/utils/evaluation.py::pairs_eval` | all accepted draws, paired GT | hmean of columns 5/6, minus draw mean per example | `rl_sample_number` default 5; sum reduction |

`DurationSmoothL1Loss`, `MLPRayleighDistribution`, `CrossEntropyProbLoss`, saliency `NSS/CC/KLD`, and Gaussian helpers exist/import in places but **are not called by active Gazeformer training**. No attention supervision, embedding regularizer, or intermediate auxiliary loss is wired.

### Checkpoints and selection

- `main` auto-sets `resume_dir=log_root` when that directory already contains `checkpoints/checkpoint.pth`. Fresh runs write `hparams.json`; resumed runs use current args, not automatic config restoration.
- `utils/checkpointing.py::CheckpointManager.step(metric)` always saves `checkpoint.pth` with `model` and `optimizer` state dictionaries. Best comparison is >= in max mode; `checkpoint_best.pth` contains model state only. DataParallel is unwrapped when applicable; the manager is normally constructed before wrapping.
- Selection score is harmonic mean of the two validation `ScanMatch` values, not training loss, MultiMatch, or retrieval. Before validation is enabled, score -1 still reaches the saver.
- Resume restores optimizer and model, then reconstructs LambdaLR with recorded iteration. Scheduler/RNG state is not separately checkpointed; epoch/iteration/best score live in `history_record.json` via `RecordManager`.
- `supervised_save` invokes `os.system('cp -r ...')` at the final supervised epoch, copying the run to a `_supervised_save` directory; not a distinct model-training entry.

## 8. Evaluation and Prediction

### Generation and post-processing

`G/models/sampling.py::Sampling.random_sample(all_actions_prob, log_normal_mu, log_normal_sigma2)`:

1. Clone/detach action probabilities and zero stop probability in the first `min_length` positions. `torch.distributions.Categorical(probs=...)` samples each step; there is no beam/greedy search or feedback into the model.
2. Gather selected probabilities from the **original** differentiable probability tensor (not the stop-masked/renormalized one). These feed RL `LogAction`.
3. Sample duration as `exp(randn * log_normal_sigma2 + log_normal_mu)`. This uses sigma2 directly as scale, whereas the loss treats it as variance (§12).
4. Return `selected_actions`, `selected_actions_probs`, `durations`, and `scanpath_length`. The latter is not what terminates serialized vectors.

`Sampling.generate_scanpath(images, prob_sample_actions, durations, sample_actions)` iterates until the first sampled zero or T steps. It includes the terminal step in action mask but not duration mask. For action `a>0`, it maps `a-1` into row-major grid and returns the cell center in configured width/height coordinates. It emits NumPy structured `(start_x,start_y,duration)` vectors in seconds. It does not force an initial center fixation or clamp evaluation durations; RL clipping occurs later, only for duration log-likelihood input.

### Metrics and aggregation

- `G(D)/utils/evaluation.py::pairs_eval` compares flat GT/prediction pairs and returns 11 columns: five MultiMatch components, ScanMatch without/with duration, SED, STDE, SED-best, STDE-best. RL consumes columns 5/6 only, but computes all and rejects a draw if **any** returned value is NaN.
- `comprehensive_evaluation_by_subject(gt_fix_vectors, predict_fix_vectors, args, is_eliminating_nan=True)` allocates group × subject × subject score arrays. Matched-observer diagonal entries produce means/stds; duration-aware ScanMatch pair scores drive `p2g` observer retrieval. Returns `cur_metrics, cur_metrics_std, scores_of_each_images` (nine metrics per pair).
- MultiMatch calls external `multimatch_gaze.docomparison`. Short paths are padded to at least three fixations using `(1,1,0.001)` for MultiMatch. In OSIE/ASD, local variables are reassigned to padded paths, so padding also affects the later metrics; AiR/COCO keep original vectors for later metrics.
- `utils/evaltools/scanmatch.py::ScanMatch.fixationToSequence` and `ScanMatch.match` discretize/align paths. Callers convert seconds back to milliseconds; grid bins are hard-coded `16×12`, temporal bin 50 ms for duration mode, threshold 3.5.
- `utils/evaltools/visual_attention_metrics.py::string_edit_distance` and `scaled_time_delay_embedding_similarity` consume coordinate arrays and a zero-image canvas for dimensions. This pipeline does not evaluate image-dependent saliency losses/metrics merely because those helpers exist.
- Metric dictionaries include `MultiMatch`, `ScanMatch`, `VAME`, and retrieval results. `p2g` ranks the matching subject among GT subjects; reported `pr1/pr3/pr5` are percentages and `rsum` their sum. `g2p` exists but its comprehensive-evaluation calls are commented out.
- `SED_best/STDE_best` are assigned from the same diagonal arrays in comprehensive evaluation, not a best-of-repeated-samples selection.

Variant differences matter: AiR/COCO compute all pairwise metrics; OSIE/ASD compute MultiMatch, SED/STDE and non-duration ScanMatch only on the diagonal, but duration-aware ScanMatch for all pairs. OSIE/ASD off-diagonal detail construction reuses the current `score` variable for the non-duration slot; do not interpret all detail slots as independently measured. AiR explicitly removes -1 unfilled entries in aggregation/retrieval. These utilities are not interchangeable without comparing implementations.

**Repeat caveat:** callers append repeated predictions into each group's subject list; evaluator arrays still allocate only `subject_num` rows. With complete groups and `eval_repeat_num>1`, indexing exceeds that allocation. There is no implemented repeat-axis aggregation. Group slicing also assumes exactly `subject_num` observer examples per group (§12).

### Outputs

- Gazeformer AiR/OSIE/ASD test logs metric means only; `predict_results=[]` does not imply a serialization path.
- `COCO_Search18/GazeformerISP/src/test.py::main` writes `<evaluation_dir>/test_predicts.json` with `img_names, task, subject, X, Y, T, length, score`. T is converted back to milliseconds. Subject comes from positional index modulo `subject_num`, not batch subject IDs. Each attached score contains comparisons with GT observers.
- ChenLSTM test checkpoint selection: AiR uses current `checkpoint.pth`; OSIE/ASD/COCO use best. `OSIE/ChenLSTMISP/src/test.py` also writes prediction JSON. Do not infer this behavior from the Gazeformer test counterpart.

## 9. Configuration Surface

Training path: `G(D)/opts.py::parse_opt` → module-global `args` in train → explicit constructor arguments and closures. `utils/config.py::CfgNode` inherits YACS CfgNode; `load_yaml_with_base` recursively resolves `_BASE_`. Precedence: parser defaults < YAML < `set_cfgs` < explicit CLI (second parse into namespace). Flat config keys become args attributes, with warnings for unknown names. `set_cfgs` merges through YACS and is intended for already-defined config keys. Test uses a separate argparse declaration and no YAML merge. `user_scanpath.yml` is not supplied through `--cfg`.

Unless marked otherwise, defaults below are from `G(OSIE)/opts.py::parse_opt` and common to the four Gazeformer branches.

| Argument / Config | Defined At | Consumed By | Effect |
| --- | --- | --- | --- |
| `img_dir, feat_dir, fix_dir` | G(D)/opts.py; separate test parser | dataset constructors | embedding-parent path, feature and annotation locations |
| `width=512, height=384; origin_width=800, origin_height=600` | opts/test | datasets, Sampling, validation metrics | pixel coordinate frames; COCO height=320, originals=1680×1050; Gazeformer test omits origin_size when constructing dataset |
| `im_h=24, im_w=32` | opts/test | dataset targets, Transformer via args, gazeformer spatial_dim, Sampling | S tokens/actions and integration layer dimensions; COCO im_h=20 |
| `img_hidden_dim=2048, lm_hidden_dim=768` | opts/test | Transformer constructor → input_proj/text_transform | required feature-cache widths |
| `hidden_dim=512` | opts/test | Transformer d_model, dim_feedforward; IntegrationModule; heads | encoder/decoder width and FFN width (caller sets both equal) |
| `num_encoder=6, num_decoder=6, nhead=8` | opts/test | Transformer and CrossAttentionPredictor constructors | layer counts and internal multihead attention; nhead is not action_map_num |
| `subject_num` | opts/test | gazeformer embedding table; evaluation slicing/arrays | defaults OSIE 15, ASD 39, AiR 20, COCO 10 |
| `subject_feature_dim=128` | opts/test | subject_embed, SubjectAttentionModule, IntegrationModule, output attention | observer representation/projection widths |
| `action_map_num=4` | opts/test | gazeformer attention_map_predictors and token_predictor | K independently parameterized spatial/stop map branches |
| `max_length=16, min_length=1` | opts/test | targets, query embeddings, Sampling | T positions and earliest allowable sampled termination; COCO T=7 |
| `encoder_dropout=.1, decoder_dropout=.2, cls_dropout=.4` | opts/test | Transformer/IntegrationModule, gazeformer/predictors | distinct encoder/integration, decoder, output dropout; disabled in RL eval mode |
| `patch_size=16` | opts/test | **not forwarded by entry constructors** | constructor default still sizes unused queryfix_embed; does not change cached feature extraction |
| `blur_sigma=None` | opts | supervised dataset | optional smoothed target distribution |
| `batch=1, test_batch=1` | opts | training/RL loaders, validation loader | counts groups, not observer examples; COCO train batch=3; test uses its own batch |
| `epoch=40, warmup_epoch=1, start_rl_epoch=25` | opts | outer loop, main.train, main.lr_lambda | total epochs, warmup and stage boundary |
| `lr=1e-4, weight_decay=5e-5, rl_lr_initial_decay=.1` | opts | AdamW, main.lr_lambda | base optimization and RL schedule scale |
| `lambda_1=1, rl_sample_number=5, clip=-1` | opts | main.train | supervised duration weighting, number of accepted RL draws, clipping only if >0 |
| `no_eval_epoch=5, eval_repeat_num=1` | opts; repeats also test | main epoch condition, sampling loops | evaluate for epoch >5; repeat integration currently unsafe above 1 |
| `log_root, resume_dir, supervised_save=True` | opts | main, record/checkpoint managers | run location, resume, stage snapshot |
| `evaluation_dir` | test module parser | test.main | checkpoint/log/COCO JSON root; architecture not restored from saved args |
| `seed=0, cuda=0, gpu_ids=[0,1]` | opts/test | module seed setup, Transformer device, DataParallel | RNG/backend configuration and devices; bare .cuda() calls limit device consistency |
| `mode` | opts/test | no behavioral dispatch in entry loops | train/test split is selected by entry code instead |
| `cfg, set_cfgs` | opts only | parse_opt → CfgNode | flat configuration override path |
| ChenLSTM `embedding_dim, map_width/map_height, dropout, att_dir/detector_dir, detector_threshold` | C(D)/opts.py and separate test parser | baseline, datasets | recurrent embedding/map sizes and task guidance; inspect explicit constructor forwarding |

**Shell overrides:** Gazeformer scripts use seeds OSIE=1, others=10; COCO explicitly uses epoch=40/start_rl_epoch=20, others inherit stage=25. All select visible GPUs 0,1. ChenLSTM scripts differ: OSIE=40/20, ASD=40/25, COCO=20/10 (total/stage); AiR specifies epoch=40, seed=1, visible GPU 0. All shell labels `MODEL_NAME='baseline'` are run-directory labels, not a model registry.

## 10. Dependency / Call Map

```text
G(D)/train.py::<module>
 ├─ calls      → opts.parse_opt → CfgNode.load_yaml_with_base / merge_from_list
 └─ calls      → main (under __main__)
main
 ├─ constructs → selected dataset classes → DataLoader(dataset.collate_func)
 ├─ constructs → models.models.Transformer → models.gazeformer.gazeformer
 ├─ constructs → Sampling, AdamW, LambdaLR, RecordManager, CheckpointManager
 ├─ calls      → main.train → model.forward → training_process OR inference
 │                           ├─ passes output to → losses (supervised)
 │                           └─ passes output to → Sampling → pairs_eval → RL losses
 └─ calls      → main.validation → model.inference → Sampling → comprehensive evaluation

models.gazeformer.gazeformer (inherits nn.Module)
 ├─ contains   → subject_embed, querypos_embed, PositionEmbeddingSine2d
 ├─ calls      → Transformer.forward
 │              ├─ encoder wrapper → input_proj → TransformerEncoder → img_transform
 │              │                                      task → text_transform → AttentionModule
 │              ├─ SubjectAttentionModule → IntegrationModule
 │              └─ decoder wrapper → TransformerDecoder → TransformerDecoderLayer(s)
 ├─ calls      → generator_t_mu / generator_t_logvar
 ├─ calls      → CrossAttentionPredictor(s) → output attention_module
 ├─ calls      → token_predictor; combines weighted stop/spatial logits
 └─ returns    → actions OR all_actions_prob, log_normal_mu, log_normal_sigma2, action_map

models/gazeformer.py
 └─ imports only → models.models.ResNetCOCO (not constructed in this model)
preprocess/feature_extractor.py::image_data
 └─ constructs/calls → its OWN ResNetCOCO → torchvision detection backbone.body

utils.evaluation
 ├─ calls      → multimatch_gaze.docomparison
 ├─ constructs/calls → ScanMatch.fixationToSequence / match
 ├─ calls      → string_edit_distance / scaled_time_delay_embedding_similarity
 └─ calls      → p2g (g2p calls commented out)

C(D)/train.py::main
 ├─ constructs → local dataset classes, baseline, Sampling, Adam, LambdaLR
 └─ calls      → baseline.forward → training_process/inference
                → local ResNet → ConvLSTM + spatial_att/semantic_att → predict_head

utils.config.CfgNode
 └─ inherits from → yacs.config.CfgNode (configuration, not architecture registry)
```

No source imports another dataset's Gazeformer package to share its model. `models/models.py`, `models/sampling.py`, and `models/loss.py` are byte-identical across Gazeformer copies at this commit. `models/gazeformer.py` differs only in COCO embedding initialization: normal std 0.01 vs 1 for the other three. Dataset/evaluation scripts differ materially.

## 11. Implementation Navigation Guide

Use the intended dataset/family copy first, not whichever similarly named file is found first.

| If you want to change... | Start Here | Then Inspect | Why |
| --- | --- | --- | --- |
| Model architecture | G/models/gazeformer.py::gazeformer.__init__, forward | G/models/models.py::Transformer; both train/test constructors | Composition is explicit and duplicated across entries/copies |
| Decoder behavior | G/models/models.py::TransformerDecoderLayer.forward_post, TransformerDecoder.forward | TransformerDecoderWrapper; gazeformer training_process/inference | Noncausal parallel queries and output shape are upstream contracts |
| Visual features | G(D)/preprocess/feature_extractor.py::image_data, ResNetCOCO | dataset __getitem__; TransformerEncoderWrapper.input_proj; im_h/im_w | Training consumes caches, not models.models.ResNetCOCO |
| Task/text features | feature_extractor.py::text_data; dataset embedding_dict lookup | TransformerEncoderWrapper.text_transform, AttentionModule | Vocabulary/key and embedding width must agree |
| Observer representation | gazeformer.__init__::subject_embed | dataset subject normalization; SubjectAttentionModule; evaluation indexing | Table IDs, feature widths and positional evaluation have different contracts |
| Personalized integration | models.py::IntegrationModule.forward | Transformer.forward; SubjectAttentionModule/AttentionModule | Both spatial attentions and subject projections feed the factorized memory |
| Fixation/stop prediction | gazeformer.py::CrossAttentionPredictor; token_predictor; attention_module | dataset target encoding; Sampling.generate_scanpath; CrossEntropyLoss | Stop index 0 and grid flattening span all consumers |
| Duration prediction | gazeformer generator_t_mu/generator_t_logvar | MLPLogNormalDistribution, LogDuration, Sampling.random_sample | Variance-versus-sampling-scale discrepancy must be considered |
| Training loss | loss.py::CrossEntropyLoss, MLPLogNormalDistribution | train.py::main.train supervised branch | Imported losses are not necessarily applied |
| Add auxiliary loss | gazeformer.training_process output dict; main.train | new target/mask producer + collate; inference branch if needed | Outputs do not train themselves; RL uses inference, not training_process |
| New output/head/branch | gazeformer.__init__, training_process AND inference | train/test construction, loss consumer, checkpoints, sampler if required | Both forward implementations and strict state_dict loading must stay aligned |
| New annotation field | selected dataset __init__/__getitem__ | all applicable collate_func methods and train/RL/val/test unpacking | No generic batch-schema forwarding |
| Batch structure | selected dataset.collate_func | .view(-1,*shape[2:]) in entries; subject_num slicing | Artificial leading axis and group metadata use different flattening |
| Training stages | train.py::main.train, main.lr_lambda, epoch loop | checkpoint manager, RecordManager, supervised_save | Stage is epoch-based; schedule/history are iteration-based |
| Freeze/unfreeze modules | train.py::main after construction/load | optimizer parameter collection; stage mode calls; fixed positions | No existing selective stage freeze; eval is not a gradient freeze |
| CLI/config option | opts.py::parse_opt | separate test parser; both constructors; optional YAML | Declared options can remain unused without forwarding |
| Checkpoint loading | train.py::main resume; test.py::main | CheckpointManager.step; RecordManager; hparams.json | Current/best files carry different state and no auto model config |
| Evaluation metrics | utils/evaluation.py::comprehensive_evaluation_by_subject | pairs_eval, p2g, evaltools; checkpoint criterion | Validation/reporting and RL reward are separate call paths |
| Inference/decoding | Sampling.random_sample, generate_scanpath | gazeformer.inference; test/validation regrouping and serialization | Sampling/termination happen outside parallel transformer forward |
| Repeat evaluation / missing observers | train.py::main.validation and test.py::main grouping | comprehensive_evaluation_by_subject allocation/diagonal indexing | Fixed subject counts currently conflict with repeat/missing groups |
| ChenLSTM recurrence/task branch | C(D)/models/baseline_attention.py::baseline, ConvLSTM, predict_head | C(D)/dataset/dataset.py; train model call | Pixels and attention_maps/taskints are not Gazeformer interfaces |

## 12. High-Risk Couplings and Invariants

Confirmed source constraints, not proposed fixes:

1. **Independent copies:** edits in OSIE do not affect AiR/ASD/COCO. Train and test independently reconstruct architecture; test never loads saved hyperparameters. Preserve intended checkpoint keys/shapes.
2. **Known AiR startup blocker:** `AiR/GazeformerISP/src/train.py` imports `human_evaluation, evaluation, human_evaluation_by_subject, evaluation_by_subject` from local evaluation.py, which defines none of them. An unused imported name can still stop execution before main.
3. **Cache/grid coupling:** source token count must equal `im_h*im_w` for positions, integration spatial Linear layers, and action-map reshape. Source/text channel widths must match their projections. Extractor resize is hard-coded, not controlled by training CLI.
4. **Head dimensions:** hidden_dim must satisfy MultiheadAttention head division and sine-position channel construction (the latter splits channels into sin/cos pairs). action_map_num controls independent predictor modules, not attention heads. Changing K also changes stop logits/checkpoint shapes.
5. **Observer indexing:** dataset forwards IDs to nn.Embedding; IDs must be in range. Evaluation uses list positions/diagonals and fixed group slices, not those IDs. Missing/reordered observers can corrupt grouping/matching; more than subject_num predictions (including repeats) exceeds allocated score rows.
6. **Collation:** observer examples are concatenated under an extra leading singleton; consumer view drops that axis. Metadata/GT remain nested lists. N is not loader batch size. Bare duration squeeze can remove N/T axes for singleton workloads or DataParallel shards.
7. **Stops/masks:** action 0 = stop; spatial actions offset by 1. First stop is included in action likelihood but excluded from duration likelihood. No zero-denominator guard exists for mask.sum. With min_length=0, scanpath_length's zero sentinel cannot distinguish termination at position 0, although generate_scanpath still stops there.
8. **Train/eval outputs:** train returns logits under actions; eval returns probabilities under all_actions_prob. RL intentionally runs eval **with gradients**, disabling dropout. Changing the dispatch/output keys affects both evaluation and optimization.
9. **Duration semantics:** generator returns exp(logvar); NLL treats it as variance, but sampler multiplies noise by it without sqrt. RL logs detached durations clipped to [0,100] after reward vectors have already been generated. Evaluation does not apply this clip.
10. **RL likelihood/baseline:** early stop suppression changes the sampling distribution but LogAction sees original probabilities. LogAction/LogDuration normalize by whole-batch mask counts; accepted-draw mean baseline includes each draw itself. rl_sample_number=1 gives zero advantage. NaN rejection has no retry cap.
11. **Coordinate differences:** OSIE/ASD subtract 1 only in supervised discretization; AiR does not. COCO active grouped loaders require already-resized XY. Gazeformer test parsers expose original sizes but do not pass them into dataset construction; OSIE/ASD test uses dataset defaults instead.
12. **Metric frame/padding differences:** OSIE/ASD pairs_eval hard-codes 320×240 despite Gazeformer 512×384 defaults. COCO pairs_eval uses 512×320 stimulus but MultiMatch screensize [320,240]. AiR uses [512,384]. Validation comprehensive evaluation uses args dimensions. OSIE/ASD padding and NaN fallback behavior differ from AiR/COCO; compare before consolidating utilities.
13. **Preprocessing mismatch:** ASD/COCO fixation converters are identical OSIE converters with /home/OSIE paths. COCO extractor is flat stimuli/free-viewing with OSIE_autism default root, while its loader needs task subfolders, task embeddings and 20×32 tokens. AiR extractor main skips image_data. These scripts do not establish an end-to-end preparation pipeline for every branch.
14. **Device/environment assumptions:** entries use bare .cuda(), some constructors use cuda:<args.cuda>, sampler uses get_device(), and DataParallel has default IDs [0,1]. This is not a verified CPU/multi-device-independent path. gpu_ids uses argparse type=list and supervised_save uses type=bool, not dedicated list/boolean flag parsers. POSIX path splitting and unquoted cp commands are also embedded.
15. **Persistence boundaries:** current checkpoint includes optimizer, best does not; history is separate and scheduler/RNG state absent. Resume treats every non-optimizer checkpoint entry as model state. Auto-resume may override the expected fresh-run behavior. Pre-validation -1 still produces checkpoints.
16. **Dormant/analysis code:** queryfix_embed and decoder-wrapper linear are not in active computation; alternative *_by_subject dataset classes are not selected by launchers. ASD visualization scripts refer to unavailable model modules/symbols; they are not evidence of another implemented architecture.
17. **Repeated draw serialization:** COCO test derives subject from position and converts structured vectors to a two-dimensional array before extracting XYZ/T columns; an empty predicted path does not satisfy that indexing contract. Default min_length=1 normally prevents an immediate stop.

## 13. External Dependencies That Matter

These are dependency relationships found in local source/pins, not claims about current upstream APIs.

| Dependency | Concrete use |
| --- | --- |
| PyTorch / torchvision | nn.MultiheadAttention, DataLoader/custom collators, AdamW/Adam, LambdaLR, DataParallel, Categorical, checkpoint IO; torchvision detection COCO backbone in offline extraction; transforms in pixel pipelines |
| sentence-transformers | feature_extractor.text_data constructs SentenceTransformer and caches encode results; not a training-time language model |
| NumPy / SciPy | annotation/cache arrays, structured fixation vectors; Gaussian target blur, MATLAB conversion, harmonic-mean rewards/selection |
| multimatch-gaze | utils/evaluation.py calls docomparison with structured fixation vectors/screensize; algorithm internals external |
| Local ScanMatch / visual_attention_metrics | actively called adapted metric implementations, not dead vendored files; scanmatch fixationToSequence/match and SED/STDE directly affect reward/evaluation |
| YACS / PyYAML | opts.parse_opt and utils.config.CfgNode implement optional YAML inheritance/merges; no experiment configs shipped |
| PIL / scikit-image | image IO and resizing in preprocessing/ChenLSTM; AiR/COCO attention map preparation |
| TensorBoard | train SummaryWriter consumes losses/metrics/LR, does not control optimization |
| scikit-learn / matplotlib / seaborn | observer classification/t-SNE and offline scanpath plotting; not active Gazeformer forward dependencies |

`user_scanpath.yml` pins torch 1.12.1+cu116, torchvision 0.13.1+cu116, multimatch-gaze 0.1.3, sentence-transformers 2.2.2, and yacs 0.1.8 among others. Installed versions/download availability were not checked; checkpoint deserialization and legacy dependency compatibility should not be assumed on a different environment.

## 14. Uncertainties / Things Not Established From Static Inspection

- **Data/cache contents:** existence, actual token shape/order, text-vector dimensions/dtypes, positive durations, valid coordinate ranges, observer completeness/order, and split identity require inspecting external assets. Defaults and projections specify expectations, not evidence that files satisfy them.
- **Dataset preparation provenance:** the actual ASD/COCO conversion/extraction used for published/downloaded assets is not recoverable from the mismatched scripts. README links do not establish their schema. COCO target-present paths appear in defaults; no dedicated target-absent switch is wired.
- **Checkpoint provenance/compatibility:** no weights were opened. Actual subject vocabularies, chosen model dimensions, training stage, run arguments, and compatibility with shipped test defaults remain unknown.
- **Runtime viability:** all Python files were syntax-parsed and local import targets inspected, not imported/executed with model dependencies. AiR/analysis missing imports are confirmed source defects; additional CUDA, dependency, filesystem, numerical, or data failures may exist.
- **Numerical intent:** the duration sampling variance discrepancy, log-map feature aggregation, global mask normalization, and metric frame differences are confirmed operations. Whether intentional or responsible for measured behavior cannot be established statically.
- **Metrics/dependency internals:** local callers and adapted metrics were inspected; installed multimatch-gaze, sentence-transformers, PyTorch/torchvision internal execution and pretrained weight revisions were not independently audited.
- **Observer-analysis scripts:** their unavailable imports and external labels/checkpoint paths prevent a verified t-SNE/classification execution path. Do not use them to infer a missing trained branch.

## 15. Fast Navigation Index

```text
PATH ALIASES          → §0; G(D)=<D>/GazeformerISP/src, C(D)=<D>/ChenLSTMISP/src
TRAINING ENTRY        → G(D)/train.py::main; <D>/GazeformerISP/bash/train.sh
SUPERVISED / RL       → G(D)/train.py::main.train
VALIDATION ENTRY      → G(D)/train.py::main.validation
TEST / INFERENCE      → G(D)/test.py::main → gazeformer.inference → Sampling
CONFIGURATION         → G(D)/opts.py::parse_opt; test.py::<module>; utils/config.py::CfgNode
DATASET CLASSES       → G(D)/dataset/dataset.py::{OSIE,AiR,COCOSearch}[,_rl,_evaluation]
COLLATE               → same classes::collate_func (see §6 field contracts)
MODEL CONSTRUCTION    → train.py::main AND test.py::main (no factory)
TOP-LEVEL MODEL       → G/models/gazeformer.py::gazeformer
VISUAL EXTRACTION     → G(D)/preprocess/feature_extractor.py::image_data / ResNetCOCO
TEXT EXTRACTION       → G(D)/preprocess/feature_extractor.py::text_data
VISUAL/TASK ENCODER   → G/models/models.py::TransformerEncoderWrapper
OBSERVER EMBEDDING    → G/models/gazeformer.py::gazeformer.subject_embed (attribute)
OBSERVER ATTENTION    → G/models/models.py::SubjectAttentionModule.forward
FEATURE INTEGRATION   → G/models/models.py::IntegrationModule.forward
DECODER               → G/models/models.py::TransformerDecoderLayer / TransformerDecoder
SPATIAL/STOP HEADS    → G/models/gazeformer.py::CrossAttentionPredictor / token_predictor attribute
PERSONALIZED MIXING   → G/models/gazeformer.py::attention_module.forward
DURATION HEADS        → gazeformer.generator_t_mu / generator_t_logvar attributes
LOSSES                → G/models/loss.py::CrossEntropyLoss / MLPLogNormalDistribution / LogAction / LogDuration
OPTIMIZER / SCHEDULE  → G(D)/train.py::main / main.lr_lambda
CHECKPOINT IO         → G/utils/checkpointing.py::CheckpointManager.step; train/test main loading
EPOCH HISTORY         → G/utils/recording.py::RecordManager
SAMPLING / STOPPING   → G/models/sampling.py::Sampling.random_sample / generate_scanpath
RL REWARD             → G(D)/utils/evaluation.py::pairs_eval + train.py::main.train
EVALUATION METRICS    → G(D)/utils/evaluation.py::comprehensive_evaluation_by_subject / p2g
PREDICTION JSON       → COCO_Search18/GazeformerISP/src/test.py::main
CHENLSTM ALTERNATIVE  → C(D)/models/baseline_attention.py::baseline / ConvLSTM / predict_head
CONFIRMED HAZARDS     → §12; unresolved external/runtime questions → §14
```