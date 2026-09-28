# ISP-SENet Codebase Map

## 0. Map Metadata

- Scope: `isp-senet/**`.
- Purpose: implementation navigation before changing personalization, support processing, architecture, objectives, datasets, or evaluation. Source code remains the source of truth; this is not a paper summary or a claim that published commands run unchanged.
- Paths below are relative to `isp-senet/`. For compact references, **S** = `SE-Net`, **O** = `ISP/OSIE/GazeformerISP/src`, **F** = `ISP/COCO_FV/GazeformerISP/src`, **C** = `ISP/COCO_Search18/GazeformerISP/src`; **P** means each of O/F/C, not a shared package. Expand aliases before searching. `::<module>` denotes executable module-level code; `main.train`/`main.validation` are nested functions.
- Inspected source/configs/launchers; did not execute training, deserialize checkpoints, or inspect dataset contents. No implementation changes accompany this map.
- Fast reuse: read sections 1 and 18 first, section 13 for edit routing, then only the relevant contracts in 6/7/9/14. All 19 numbered sections are intentional; alias-expanded file references and explicit top-level Python symbols were statically checked.

## 1. Architecture at a Glance

```text
S/train.py -> UserEmbeddingNet(image, complete observed scanpath, task vector)
           -> per-scanpath user_emb + subject logits -> classification + triplet training

S/train.py --eval-only -> evaluate_user_siamese -> per-subject accumulation/mean
                      -> saved [subject rows, embedding dimensions] tensor

P/models/gazeformer.py::gazeformer loads that tensor, indexes it by subjects
cached query-image features + task vector + selected embedding row
  -> Transformer -> subject attention + IntegrationModule -> parallel decoder
  -> subject-weighted action maps/stop logits + duration parameters -> Sampling
```

SE-Net has its own training entry, optimizer, and weights checkpoint. Embedding export is a separate execution mode of that entry. Predictor training has its own entry and supervised/RL phases. **The predictor does not construct or run SE-Net and does not load its weights checkpoint.** It consumes detached, offline-exported embedding tensors; no predictor gradient can reach SE-Net. Those tensors are ordinary attributes, not trainable embeddings or registered buffers.

## 2. Repository Structure

```text
isp-senet/
  README.md                              # orientation, not execution authority
  SE-Net/
    train.py                             # CLI, losses/loop, export/evaluation dispatch
    configs/
      osie_useremb.json                   # OSIE SE-Net settings
      coco_freeview_useremb.json          # COCO-Freeview settings
      coco_search18_TP_useremb.json       # target-present search settings
    common/
      config.py                          # JsonConfig nested attributes / serialization
      dataset.py                         # split processing, transforms, datasets
      data.py                            # Siamese_Triplet_Gaze and sample fields
      utils.py                           # ID mapping, support sampling, fixation tokenization
      position_encoding.py               # spatial and duration encodings
      losses.py, sinkhorn.py, metrics.py  # inherited helpers, not active training objectives
    src/
      builder.py                         # loaders, model, AdamW, checkpoint restore
      models.py                          # ImageFeatureEncoder, UserEmbeddingNet, attention
      eval_user.py                       # classification metrics and embedding export
      config.py                          # Detectron2/MaskFormer config extensions
      backbone/swin.py                    # registered D2SwinTransformer option
      pixel_decoder/msdeformattn.py       # active MSD feature pyramid decoder
      pixel_decoder/fpn.py                # alternative TransformerEncoderPixelDecoder
      pixel_decoder/ops/                  # required deformable-attention extension
      transformer_decoder/               # position/transformer helpers; optional segmentation
    environment.yml                      # separate SE-Net environment
  ISP/
    README.md, environment.yml           # ISP setup/environment
    OSIE/GazeformerISP/                   # O implementation
    COCO_FV/GazeformerISP/                # F implementation
    COCO_Search18/GazeformerISP/          # C implementation
      README.md                          # dataset-specific test commands
      bash/train.sh                      # training launcher (one per implementation)
      src/
        train.py, test.py                 # training and prediction/evaluation entries
        opts.py                          # training CLI + YAML overrides
        dataset/dataset.py               # train/RL/eval datasets, collators, ID filters
        models/gazeformer.py             # predictor, embedding load, action/duration heads
        models/models.py                 # transformer, subject/task attention, fusion
        models/positional_encodings.py   # fixed-grid position encodings
        models/loss.py                   # supervised and policy-gradient losses
        models/sampling.py               # actions, durations, scanpath conversion
        preprocess/feature_extractor.py  # offline ResNet and sentence embeddings
        preprocess/preprocess_fixations.py # annotation conversion, hard-coded paths
        utils/config.py                  # CfgNode YAML/base/overrides
        utils/checkpointing.py           # latest/best checkpoint writer
        utils/recording.py               # epoch/iteration/best-metric JSON
        utils/evaluation.py              # subject-aligned metrics and RL reward pairs
        utils/evaltools/                  # ScanMatch, SED, STDE implementations
        utils/logger.py                  # train/test text logging
```

The `src/` subtree repeats under all three dataset directories. O additionally has `utils/data_postprocess.py` (prediction serialization/ID recovery) and `preprocess/pseudo_fixations.py` (manual pseudo-label utility). Weights, annotations, arrays, results, and logs are omitted. Model copies are not imported from the sibling repository `gazeformer-isp/`.

## 3. Execution Flows

### 3.1 SE-Net training flow

```text
S/train.py::<module> -> parse_args -> common/config.py::JsonConfig(--hparams)
  -> overwrite Data.subject / Data.fewshot_subject from CLI
  -> src/builder.py::build(hparams, dataset_root, device, is_eval, split)
     -> load fixation JSON (+ bbox_annos.npy for COCO-Search18)
     -> adjust_subjects; condition/fixOnTarget filtering
     -> common/dataset.py::process_data -> Siamese_Triplet_Gaze
     -> train/valid DataLoader (default nested-dictionary collate)
     -> UserEmbeddingNet -> ImageFeatureEncoder -> pretrained backbone/pixel decoder
     -> AdamW; optional load checkpoint['model'] / restore global_step
  -> train_iter -> compute_loss -> compute_output for anchor, positive, negative
     -> transform_fixations -> UserEmbeddingNet.forward
     -> subject CE + triplet -> weighted sum -> backward -> optimizer.step
  -> epoch-start MultiStepLR.step
  -> periodic run_evaluation -> evaluate_user_siamese
  -> periodic torch.save({model, optimizer, step})
```

Entry is `S/train.py::<module>`, not a `main()` function. `build` constructs both loaders and the optimizer even in evaluation-only mode. CUDA is selected unconditionally. `split` is accepted/passed into `build` but is not used to create a split there.

### 3.2 ISP-SENet / personalized predictor training flow

Each `ISP/<dataset>/GazeformerISP/bash/train.sh` launches `src/train.py` from its GazeformerISP directory:

```text
P/train.py::<module> -> opts.py::parse_opt -> main
  -> OSIE / COCOSearch + corresponding _rl and _evaluation datasets
  -> dataset.collate_func -> DataLoaders
  -> models/models.py::Transformer -> models/gazeformer.py::gazeformer
     -> torch.load(args.user_emb_path) [already aggregated subject table]
  -> Sampling + AdamW + RecordManager + CheckpointManager + LambdaLR
  -> optional predictor resume
  -> main.train:
       epoch < start_rl_epoch: model.train -> actions/duration losses -> update
       otherwise: model.eval -> sampled predictions -> ScanMatch reward -> update
  -> main.validation -> sampled scanpaths -> comprehensive_evaluation_by_subject
  -> checkpoint_manager.step -> record_manager.save
```

There is **no in-batch support -> SE-Net path** here. `subjects` selects exported rows; `images` contains cached query features. Supervised query target scanpaths enter only the loss, not the decoder. C has a legacy fine-tuning branch; see sections 8/14 before attempting it.

### 3.3 Validation / test / inference flow

- SE-Net: `S/train.py::run_evaluation` normally uses the validation loader (annotation split `test`); `--eval-only` switches to the training loader for export, unless `--mode evaluate-net` selects validation again. All routes call `S/src/eval_user.py::evaluate_user_siamese` and may write embeddings, not just metrics.
- Predictor validation: `P/train.py::main.validation` uses annotation split **O=`validation`, F=`test`, C=`valid`**, `model.eval()` and `torch.no_grad()` around forward; samples outputs `eval_repeat_num` times and evaluates them. In F, checkpoint selection therefore uses the test split.
- Predictor test: `P/test.py::main` has its **own module-level parser**, constructs `_evaluation(type="test")`, constructs the model (loading the external embedding table), loads `<evaluation_dir>/checkpoints/checkpoint_best.pth`, samples/evaluates, and writes predictions.
- Low-level inference: `P/models/gazeformer.py::gazeformer.forward(src, subjects, task)` dispatches to `training_process` or `inference` according to `self.training`. No separate service or image-only inference entry exists.

### 3.4 Additional modes

- `S/train.py --eval-only --fewshot_subject ...`: unseen-subject embedding export; does not fine-tune SE-Net.
- Predictor RL belongs to the same training program. It uses eval-mode prediction outputs **with autograd enabled**.
- `P/preprocess/feature_extractor.py::{image_data,text_data}` performs offline feature generation; training does not call it. Annotation conversion and O pseudo-label utilities are manual, path-sensitive scripts.
- No provided launcher orchestrates repeated random support trials or the complete SE-Net/export/predictor pipeline.

## 4. Core Component Map

| Component | Path / Symbol | Responsibility | Main Inputs | Main Outputs | Used By |
| --- | --- | --- | --- | --- | --- |
| SE-Net builder | `S/src/builder.py::build` | Loaders, network, optimizer, checkpoint | JsonConfig, dataset root | model/optimizer/loaders/step | SE-Net entry |
| Subject network | `S/src/models.py::UserEmbeddingNet` | Encode one scanpath in image context | RGB, fixation tokens/mask, duration, task | embedding, subject logits, attention | training/export |
| SE-Net image encoder | `S/src/models.py::ImageFeatureEncoder` | Detectron2 backbone + pixel decoder | normalized RGB | high-resolution/multiscale maps | UserEmbeddingNet |
| Pixel decoder | `S/src/pixel_decoder/msdeformattn.py::MSDeformAttnPixelDecoder.forward_features` | Deformable feature pyramid | backbone dictionary | 256-channel maps | ImageFeatureEncoder |
| Context/scanpath encoder | `S/src/models.py::UserEmbeddingNet.forward`, `working_memory_encoder` | Gather fixated image features, joint attention | context/fixation tokens, masks | contextualized fixation tokens | subject readout |
| Readout/classifier | `S/src/models.py::UserEmbeddingNet.user_fix_cross_attn`, `subject_predictor` | CLS attention; subject MLP | fixation memory | user vector, subject logits | triplet / CE / export |
| Support aggregation | `S/src/eval_user.py::evaluate_user_siamese` | Detached per-class accumulation/mean | anchor embeddings and IDs | saved subject table | predictor constructor |
| Predictor | `P/models/gazeformer.py::gazeformer` | Row lookup, transformer, output heads | cached features, IDs, task | actions, duration parameters | train/test |
| Offline visual backbone | `P/preprocess/feature_extractor.py::ResNetCOCO` | Mask R-CNN ResNet50 body features | resized RGB | cached spatial tokens | image_data, not forward |
| Query/task encoder | `P/models/models.py::TransformerEncoderWrapper`, `AttentionModule` | Encode visual tokens, task spatial weights | features, task vector | memory, visual attention | Transformer |
| Observer attention/fusion | `P/models/models.py::SubjectAttentionModule`, `IntegrationModule` | Subject spatial attention and spatial/channel fusion | memory, attention, subject vector | personalized memory | Transformer.forward |
| Parallel decoder | `P/models/models.py::TransformerDecoderWrapper`, `TransformerDecoder`, `TransformerDecoderLayer` | Decode all timestep queries | zero targets, memory, positions | per-step states | gazeformer heads |
| Fixation maps | `P/models/gazeformer.py::CrossAttentionPredictor` | State self-attention, memory cross-attention | states, memory | log spatial attention | gazeformer |
| Adaptive prioritization | `P/models/gazeformer.py::attention_module` | Subject-conditioned map weighting | memory, log maps, subject | softmax over map heads | output aggregation |
| Stop/duration heads | `P/models/gazeformer.py::gazeformer.token_predictor`, `generator_t_mu`, `generator_t_logvar` | Stop per map; log-normal parameters | decoder states | stop logits, mu, exp(logvar) | losses/Sampling |
| Generation | `P/models/sampling.py::Sampling` | Sample/decode actions and durations | probabilities, mu, sigma2 | fixation arrays, masks | RL/validation/test |

## 5. Model Architecture

### 5.1 SE-Net architecture

`S/src/models.py::UserEmbeddingNet.forward(img, tgt_seq, tgt_padding_mask, tgt_seq_high, duration, task_emb)`:

1. `ImageFeatureEncoder` loads a Detectron2-configured backbone and MSD/FPN weights. `input_proj` maps 256 channels to `hidden_dim = Model.embedding_dim` (384 in shipped JSONs).
2. Flatten the low-resolution map into dorsal/context tokens; gather high-resolution features at `tgt_seq_high` into ventral/fixation tokens. Token 0 gathers a prepended zero vector. Training/export always request high-resolution tokens.
3. Spatial positions enter visual token contents. Ventral attention positions receive modality and learned fixation-index embeddings, zeroed for padding. **Duration encoding is added to `ventral_pos`, immediately followed by `ventral_pos.fill_(0)`; its duration contribution does not reach attention as written.** The signature alone does not establish duration conditioning.
4. For `ntask != 1`, `task_transform` projects a 768-D task vector, prepends it to dorsal tokens, and `task_dorsal_encoder` self-attends. `build` uses 18 tasks for search, otherwise 1 (FV forces 1). The dataset supplies the first dictionary embedding, not a task-specific lookup.
5. Concatenate context and fixation tokens; `working_memory_encoder` self-attends them jointly. The last `Data.max_traj_length` tokens become fixation memory.
6. Learned `cls_embed` attends that memory through `user_fix_cross_attn`. Its squeezed value is `user_emb`; MLP `subject_predictor` uses the **same value** for subject logits. There is no separate metric projection or explicit L2 normalization.

`query_embed`, `query_pos`, `transformer_ffn_layers`, and `cls_pos` are constructed but unused in this forward. Segmentation/saliency options on `ImageFeatureEncoder` are not enabled here. Swin is locally registered, but the shipped JSONs reference external `data/resnet50.yaml`; its actual configuration/weights must be supplied.

### 5.2 ISP-SENet architecture

- `gazeformer` contains the local ISP-style `Transformer`, not an imported Gazeformer package. `subject_embed = torch.load(user_emb_path)` is indexed by IDs and moved to the current device in `forward`; O alone unsqueezes a 1-D table. No subject `nn.Embedding` is constructed.
- `TransformerEncoderWrapper`: `img_hidden_dim -> hidden_dim` projection, positional transformer encoder, `img_transform`; separate `lm_hidden_dim -> hidden_dim` task projection drives `AttentionModule` spatial softmax.
- `Transformer.forward`: `SubjectAttentionModule` computes subject-conditioned spatial softmax. `IntegrationModule` combines task-/subject-weighted visual features, adds subject-derived spatial and semantic terms, multiplies them into a spatial/channel map, concatenates with memory, and linearly fuses.
- `TransformerDecoderWrapper` passes zero targets and learned timestep positions to the decoder. Training and inference provide no causal target mask or teacher-forced fixation history: all `[L,N,H]` states are computed together.
- `action_map_num` separate `CrossAttentionPredictor`s produce log spatial-attention maps. `token_predictor` supplies one stop logit per map. Lowercase `attention_module` uses spatial means of memory weighted by these **log maps**, plus subject projections, to softmax across heads. Weighted logits place termination at action 0 and spatial cells at 1..G.
- Duration heads emit mu and `exp(generator_t_logvar(outs))`. Training returns `actions` logits; inference returns `all_actions_prob` softmax. Both return `log_normal_mu`, `log_normal_sigma2`, `action_map`. Decoder states and attention weights remain local variables, not public outputs.

### 5.3 Actual forward path

```text
S/common/data.py::Siamese_Triplet_Gaze + S/common/utils.py::transform_fixations
 -> S/src/models.py::UserEmbeddingNet.forward -> user_emb
 -> S/src/eval_user.py::evaluate_user_siamese -> torch.save(subject table)
 -> P/models/gazeformer.py::gazeformer.__init__ -> subject_embed[subjects]
query features + task -> P/models/models.py::TransformerEncoderWrapper.forward
 -> Transformer.forward -> SubjectAttentionModule -> IntegrationModule
 -> TransformerDecoderWrapper -> TransformerDecoder -> outs
 -> P/models/gazeformer.py::CrossAttentionPredictor + stop/duration heads
 -> attention_module -> weighted actions/probabilities
 -> P/models/sampling.py::Sampling.random_sample -> generate_scanpath
```

## 6. Data Pipeline

Notation: `B` = SE-Net batch, `Tse` = its padded scanpath length, `Dse` = embedding width; `N` = flattened predictor examples, `L` = predictor maximum length, `G = im_h * im_w`, `H` = predictor hidden width, `M` = map heads. These two sequence lengths and two hidden widths are separate settings.

### 6.1 SE-Net data path

`S/src/builder.py::build` reads `Data.fix_path` relative to dataset root; search additionally reads bbox annotations. Excluded subjects are removed/reindexed. `TAP=TP` filters `condition=present` and truthy `fixOnTarget`; `FV` filters `condition=freeview`.

`S/common/dataset.py::process_data` rescales coordinates from dataset-specific sizes to `Data.im_h/im_w` when the height differs; constructs Resize/ToTensor/ImageNet normalization; uses `train` and `test` annotation splits. Only training trajectories undergo `select_fewshot_subject` when requested. `preprocess_fixations` produces prefix/terminal records, then `filter_scanpath` retains only `is_last` records: SE-Net gets whole observed scanpaths, not next-fixation training samples. COCO initial fixation is forced to image center; truncation and optional discretization/return-fixation removal occur here.

`S/common/data.py::Siamese_Triplet_Gaze.__getitem__` returns `anchor`, `positive`, `negative`. Positive: another record of the same subject; negative: any other subject, **not restricted to the same image** despite its comment. Indices are randomly selected on each access. If `num_fewshot == 1`, both reuse the anchor. No hard-negative mining or empty-candidate fallback exists.

| Per-instance field | Producer / transformation | Shape / type | Consumer |
| --- | --- | --- | --- |
| `true_state` | `Siamese_Triplet_Gaze.process_data`; image path is directly under `Data.image_path`, task subdirectory for search | `[3,im_h,im_w]` normalized RGB | `train.compute_output`, export -> image encoder |
| `normalized_fixations` | pad by repeating final fixation; divide x/y by `[im_w+1,im_h+1]` | `[Tse,2]` | `transform_fixations` |
| `is_padding` | 1 after original scanpath length | `[Tse]` | token padding then attention mask |
| `duration` | raw annotation `T`, truncated or zero-padded; no seconds conversion here | `[Tse]` | SE-Net forward; duration contribution is erased (section 5) |
| `task_emb` | load root `coco_search18_embeddings.npy` for search, otherwise `osie_embeddings.npy`; always first dictionary value | expected 768-D | task projection only when ntask != 1 |
| `subject_id` | annotation ID after applicable remapping | integer | CE target, export row index |
| `img_name`, `task_name`, `original_fixs`, `scanpath_length` | sample metadata | strings/tensors/scalar | diagnostics/export metadata |
| `task_id`, `true_action`, `bbox` | category/action metadata; absent bbox uses `[1,2,3,4]` | scalars / 4-vector | not used by active CE/triplet graph; bbox branch is incomplete |

`S/common/utils.py::transform_fixations(..., sample_scanpath=False, return_highres=True)` maps normalized x/y into coarse `[im_w//32, im_h//32]` and fine `[im_w//4, im_h//4]` flattened grids, offsets real indices by 1, and sets padding to `pad_idx` (0). Callers derive `[B,Tse]` boolean mask from coarse tokens and use fine tokens for feature gathering. Default DataLoader collation adds the batch axis to each nested instance.

### 6.2 ISP-SENet data path

Active datasets: `O/dataset/dataset.py::{OSIE,OSIE_rl,OSIE_evaluation}` and `F/C/dataset/dataset.py::{COCOSearch,COCOSearch_rl,COCOSearch_evaluation}`. `COCOSearch_by_subject` is not used by these entry points (and references unset `self.args`).

1. Read JSON and filter exact split; `ex_subject` takes precedence over `fewshot_subject`. Group by image name for O, by task/name for F/C (C normalizes task spaces to underscores). Examples remain in annotation order, not sorted by the collator.
2. Load `.pth` query features from `feat_dir`; one feature matrix is repeated for each subject's trajectory. Despite the name `image`, this is not RGB. Training passes RGB transforms into datasets, but the active `__getitem__` reads features instead of applying them.
3. Training bins `(X-1,Y-1)` by action-grid downscales (original size in O, configured resize size in F/C); limits to `max_length`; constructs Gaussian-blurred, normalized spatial targets plus termination at index 0. Durations are milliseconds / 1000. `action_mask` covers observed fixations plus one stop if there is room; `duration_mask` covers fixations only.
4. Evaluation/RL in O rescale coordinates to `width/height`; F/C use annotation coordinates directly (rescaling is commented out). All convert duration to seconds and keep variable-length structured `fix_vectors` with fields `start_x,start_y,duration`. `firstfix` is constructed but does not enter the predictor call.
5. Training `collate_func` concatenates subjects across images, then adds a leading singleton axis; loops flatten the first two axes. Evaluation collators stack image groups, assuming equal subject counts, and loops flatten image/subject axes.

| Batch field | Produced at `P/dataset/dataset.py` | Shape after loop flattening | Used by |
| --- | --- | --- | --- |
| `images` | `__getitem__.image` -> collate | `[N,G,img_hidden_dim]` (default 2048 channels) | predictor `src` |
| `subjects` | remapped `subject` -> collate | `[N]` integer | table lookup, not passed to SE-Net |
| `task_embeddings` | embedding dictionary: free-viewing for O/F, normalized category for C | `[N,lm_hidden_dim]` (default 768) | predictor `task` |
| `target_scanpaths` | spatial/stop distributions -> collate | `[N,L,G+1]` | supervised CrossEntropyLoss |
| `durations` | annotation T / 1000 -> collate | `[N,L]` | supervised duration NLL |
| `action_masks`, `duration_masks` | target construction -> collate | `[N,L]` | separate action/duration reductions |
| `fix_vectors` | `_rl` / `_evaluation` samples | nested lists of structured arrays | pairs_eval / subject evaluation |
| `img_names`, `tasks`, `firstfixs` | sample metadata -> collate | grouped metadata | grouping/serialization; firstfixs unused by forward |

Preprocessing is not synchronized automatically. O's `preprocess_fixations.py` converts MATLAB data to zero-based subjects and fixed image-list splits; F changes `val` to `validation` and normalizes selected task names; C subtracts one from subject IDs, sorts task/name/subject/split, and writes TP/TA subsets. Their hard-coded paths/output locations need review before reuse. `feature_extractor.py::image_data` uses a pretrained torchvision Mask R-CNN ResNet50 body and 768x1024 input resizing, yielding a 24x32 spatial grid at stride 32. `text_data` uses SentenceTransformer; no language model is trained in predictor loops.

## 7. Few-Shot Personalization Contract

| Item | Produced At | Shape / Type if Explicit | Consumed At | Semantics |
| --- | --- | --- | --- | --- |
| Seen ID mapping | `S/common/utils.py::adjust_subjects`; `P/dataset/dataset.py::adjust_subjects` | sorted retained IDs -> 0..n-1 | training/export/table lookup | mappings must agree across separately supplied annotations |
| Unseen ID mapping | each `select_fewshot_subject` | CLI list order -> 0..kSubjects-1 | export/query table lookup | row 0 means first listed unseen subject, not original ID 0 |
| Selected support names | `S/common/utils.py::select_fewshot_subject` | local list of up to `num_fewshot` image names | SE-Net training-split dataset | no support-index field is passed to model |
| Support image/scanpath | `Siamese_Triplet_Gaze.process_data` | RGB, `[Tse,2]`, mask | tokenization and SE-Net | one whole trajectory per encoder example |
| Support durations/task | same | `[Tse]`, 768-D | SE-Net forward | task/erased-duration caveats above |
| Per-example vector | `UserEmbeddingNet.forward['user_emb']` | `[B,Dse]` | triplet, classifier, export | shared CLS readout, not explicitly L2-normalized |
| Aggregated vector table | `evaluate_user_siamese` | `[Data.num_subjects,Model.embedding_dim]` CPU tensor | `gazeformer.__init__` | index-add/count mean, except one-shot shortcut |
| Selected subject vectors | `gazeformer.training_process` / `inference` | `[N,subject_feature_dim]` required | Transformer + map prioritizer | direct table indexing; no online support encoding |
| Personalization weights | `SubjectAttentionModule`; lowercase `attention_module` | spatial `[G,N]`; map `[N,L,M]` | IntegrationModule; action aggregation | learned query-side weights, not support-set pooling |

**Sampling and aggregation details:**

- SE-Net chooses support once during dataset construction: collect unique image names across requested subjects, Python `random.shuffle`, take first K, keep **all** matching trajectories for each available subject. It does not ensure K trajectories per subject, task balance, or one trajectory per image/task. Same selected names are shared across subjects; missing subject/image pairs are skipped.
- K is `Data.num_fewshot`, not a SE-Net CLI flag. `Data.random_support` is passed as `k` but does not seed or alter selection; sample persistence code is commented out. SE-Net seeds Python/NumPy/Torch to 0 at import, but unordered set-to-list conversion makes cross-process image ordering another dependency.
- `evaluate_user_siamese` processes only anchors, calls `model.eval()` (without a no-grad context), detaches vectors to CPU, uses `index_add_` and actual processed counts. Normal path divides nonzero rows by counts. Unobserved rows stay zero. There is no empty-support warning or learned fallback.
- If `fewshot_subject` is active and K=1, export saves the accumulated table and returns **before averaging or metrics**. Multiple matching trajectories therefore sum rather than average. The export training loader still has `drop_last=True` with half the configured batch size: small support sets can be partly or entirely dropped.
- Support comes from SE-Net annotation split `train`; predictor queries use `validation`/`test`. No cross-stage image/task-disjointness check or shared manifest exists; actual non-overlap requires inspecting the supplied annotations. Same-task support/query is not forbidden.
- Predictor validation/test only filters/remaps requested subjects; its `select_fewshot_subject` immediately returns all filtered records when split is not `train`. Test `num_fewshot` does **not** regenerate or select the embedding file; `user_emb_path` does that independently.
- Predictor's optional few-shot training samples its own K image names and writes `result/<experiment>/sampling/<K>_shot/sample_<random_support>.json`; `_rl` reads that file. This is query-training selection, not SE-Net execution. Predictor import-time seeding does not call Python `random.seed`, and `random_support` is a filename tag rather than a selection seed.
- Aggregated vectors are cached on disk and loaded once per model construction; no per-query/iteration SE-Net recomputation or gradient flow exists. `eval_repeat_num` repeats scanpath sampling, **not support selection**. No automatic K/trial-averaging protocol is implemented.

## 8. Training and Objectives

### 8.1 Training stages

| Stage | Entry Point | Trainable | Frozen / Absent | Initialization | Objective | Produces |
| --- | --- | --- | --- | --- | --- | --- |
| SE-Net training | `S/train.py::<module>` -> `train_iter` | readout/context/task modules; default pixel decoder | backbone parameters frozen by shipped config | pretrained image components; optional SE-Net model checkpoint | subject CE + triplet | `ckp_<step>.pt` |
| Embedding export | same, `--eval-only` -> `run_evaluation` | no updates | whole network in eval mode | same builder/checkpoint | classification metrics + accumulation | subject tensor files |
| Predictor supervised | `P/train.py::main.train`, epoch < start_rl_epoch | registered predictor parameters with gradients | SE-Net/backbone extractor absent; subject table unregistered | newly initialized predictor, or resume | spatial+stop CE, duration NLL | latest/best predictor checkpoints |
| Predictor RL | same, epoch >= start_rl_epoch | same registered parameters, despite model.eval | same offline dependencies | continues supervised/resumed weights | sampled ScanMatch policy gradient | latest/best checkpoints |

SE-Net freezing: `UserEmbeddingNet.__init__` first freezes the entire image encoder if `train_backbone=False`; `train_pixel_decoder=True` re-enables its pixel-decoder parameters. All shipped JSONs use this combination. However, `train_iter` calls `model.train()`, recursively changing modes again; there is no overriding `train()` to keep the backbone in eval mode. Parameter freezing and running-stat/dropout behavior must be treated separately.

SE-Net optimizer: `S/src/builder.py::build` constructs one AdamW group over `model.parameters()` with `Train.adam_lr/adam_betas`. `S/train.py` steps MultiStepLR at the start of each epoch (`lr_steps`, gamma 0.1), evaluates/saves by global iteration, stops at `max_iters`; no early stopping/best-metric checkpoint selection. Saves only when checkpoint interval hits **and global_step < 30000**. Restoring global_step does not restore optimizer or scheduler state.

Predictor optimizer: one AdamW group, `lr`, betas `(0.9,0.98)`, epsilon `1e-9`, `weight_decay`; optional gradient clipping. Nested `main.lr_lambda` warms up by supervised-loader iteration count, linearly decays toward RL boundary, then uses `rl_lr_initial_decay` and RL-loader counts. LambdaLR steps after each update. No module-specific learning rates. Epoch range is `start_epoch+1 .. epoch-1`; fresh RecordManager starts at -1. Validation runs when `epoch > no_eval_epoch` (O additionally requires `<60`); no early stopping. Best checkpoint maximizes harmonic mean of the two ScanMatch summary scores; skipped epochs pass -1 to checkpoint manager.

**Legacy fine-tuning is not a working extra stage as written:** `C/train.py::main` has active `fewshot_finetune_path` logic expecting checkpoint key `subject_embed.weight`, then calling `model.subject_embed.parameters()` after freezing everything. Current `subject_embed` is a tensor, not a module, and current checkpoints omit that key. F has analogous code commented out; O train does not implement it. Do not interpret these blocks as supported joint training or subject-vector optimization.

### 8.2 Losses

| Loss / Objective | Defined At | Inputs / Target | Applied To | Weight / Config |
| --- | --- | --- | --- | --- |
| Subject CE | `S/train.py::<module>.loss_funcs`, `compute_loss` | `pred_subject_id` vs each triplet member's `subject_id` | sum of anchor/positive/negative CE | `Train.subject_id_pred_weight` |
| Triplet margin | same | anchor/positive/negative `user_emb` | same CLS vectors used by classifier/export | margin 5, p=2; `Train.triplet_loss` |
| Spatial + termination CE | `P/models/loss.py::CrossEntropyLoss` | softmax of `actions`, soft `target_scanpaths`, `action_masks` | sum over targets divided by mask sum | 1 |
| Duration NLL | `P/models/loss.py::MLPLogNormalDistribution` | mu, sigma2, seconds targets, `duration_masks` | masked negative log-normal log density | `lambda_1` in supervised phase |
| RL action/duration terms | `P/models/loss.py::{LogAction,LogDuration}`; `P/train.py::main.train` | sampled action probabilities/durations, generated masks, reward advantage | negative log probabilities times reward minus trial mean | action + duration, no lambda_1 multiplier |

RL calls `P/utils/evaluation.py::pairs_eval`; reward is harmonic mean of columns 5:7 (ScanMatch without/with duration), baseline is mean across `rl_sample_number` accepted trials for each example. NaN rewards cause retries without an attempt limit; sampled durations are clipped to [0,100] for LogDuration. Each LogAction/LogDuration uses the global mask sum in its per-example reduction. No greedy baseline is computed.

No active contrastive/pairwise/mining objective was found. `S/common/losses.py::PairwiseContrastiveLoss` exists but is not called by training; a `pairwise_loss` entry is allocated in `loss_funcs` but never added by `compute_loss`. `bbox_pred` is an incomplete option: the network does not return that field, and its loss block mistakenly appends subject CE rather than bbox L1. Imported predictor NSS/CC/KLD/Rayleigh/SmoothL1 alternatives are not used in the active loops. No separate coordinate-regression or auxiliary personalized loss is active.

## 9. Checkpoint and Stage Dependency

```text
External image-component weights
  loaded by -> S/src/models.py::ImageFeatureEncoder.__init__
  config    -> Data.backbone_config -> Detectron2 cfg.MODEL.WEIGHTS
  loading   -> torch.load; backbone keys starting 'res' gain 'stages.'
               MSD keys adapter*/layer* gain lateral_convs./output_convs.
  strict    -> default strict load_state_dict
  companion -> *_MSDeformAttnPixelDecoder.pkl (or *_FPN.pkl)

SE-Net ckp_<global_step>.pt
  produced by -> S/train.py::<module> periodic save
  contains    -> model state_dict, optimizer state_dict, step=global_step+1
  loaded by   -> S/src/builder.py::build
  path        -> join(Train.log_dir, Model.checkpoint)
  loading     -> model.load_state_dict(ckp['model'], strict=False)
  resume      -> restores step, NOT optimizer; does not restore scheduler
  downstream  -> SE-Net evaluation/export, NOT predictor weight loading

Exported subject embedding tensor
  produced by -> S/src/eval_user.py::evaluate_user_siamese (detach().cpu())
  names       -> train_user_embedding_no_vsencoder.pt [ordinary export]
                fewshot_user_embedding_<Data.num_fewshot>.pt [few-shot]
  loaded by   -> P/models/gazeformer.py::gazeformer.__init__
  API         -> torch.load(args.user_emb_path), no load_state_dict
  contents    -> bare tensor; no subject-ID map, support manifest, or model metadata
  training    -> plain attribute, not Parameter/buffer; omitted from optimizer/state_dict

Predictor checkpoints
  produced by -> P/utils/checkpointing.py::CheckpointManager.step
  latest      -> checkpoint.pth = {model: state_dict, optimizer: state_dict}
  best        -> checkpoint_best.pth = {model: state_dict}
  bookkeeping -> P/utils/recording.py::RecordManager -> history_record.json
  loaded by   -> P/train.py::main resume; P/test.py::main best-model load
  strict      -> default strict load_state_dict; no general prefix conversion
```

SE-Net defaults already name checkpoints: OSIE `ckp_11999.pt`; F/C `ckp_27999.pt`. Empty `Model.checkpoint` starts at step 0, but still requires image-component weights. Missing configured files fail directly: no download/random fallback. `strict=False` does not make incompatible tensor sizes safe.

Predictor always requires a subject tensor, including before loading a predictor checkpoint. Default `user_emb_path=''` in training is not a usable fallback. Ordinary training does not require/load a pretrained Gazeformer checkpoint; existing `log_root/checkpoints/checkpoint.pth` triggers automatic resume, otherwise `resume_dir` may request it. Latest checkpoint restores model/optimizer; epoch/iteration/best metric come separately from RecordManager. Best checkpoint alone does not include optimizer, scheduler, embedding table, or IDs. Writers unwrap DataParallel when applicable.

**Producer/launcher mismatch:** all three `bash/train.sh` files point to `train_user_embedding.pt`, but current SE-Net export writes `train_user_embedding_no_vsencoder.pt`. No rename/copy bridge is implemented. Set the path to the actual generated artifact or explicitly manage the filename. The export suffix does not mean the working-memory encoder is disabled; its forward loop executes.

## 10. Evaluation and Prediction

**SE-Net:** `S/src/eval_user.py::evaluate_user_siamese` computes per-class top-1/3/5 accuracy, averages across `Data.num_subjects` including zero-observation classes as zero, writes `metrics_user_<step>.json`, and exports embeddings. Classifier always runs even for unseen embedding generation; there is no unseen classifier-head replacement. Unless the K=1 early return is taken, it also writes an embedding-dictionary pickle, but filling that dictionary is disabled by local `save_user_emb_dict=False`. Do not treat that pickle as the populated embedding table.

**Predictor decoding:** `P/models/sampling.py::Sampling.random_sample`:

1. Clone action probabilities with `.data`, prohibit action 0 for the first `min_length` steps, sample Categorical actions, and gather their probabilities from the original tensor for RL gradients.
2. Sample durations as `exp(randn * log_normal_sigma2 + log_normal_mu)`. This uses **sigma2 directly as noise scale**, whereas the NLL treats it as variance; document this discrepancy rather than silently assuming a standard log-normal sampler.
3. `generate_scanpath` stops at the first action 0 or at L. Nonzero actions subtract one, convert flattened cell indices to cell-center x/y at `width/height`, and attach durations in seconds. It returns NumPy structured arrays and masks (stop belongs to action mask, not duration mask). No autoregressive feedback, beam search, or IOR update exists in this path.

**Metrics:** `P/utils/evaluation.py::comprehensive_evaluation_by_subject` calls external `multimatch_gaze.docomparison`, local `utils/evaltools/scanmatch.py::ScanMatch`, and `utils/evaltools/visual_attention_metrics.py::{string_edit_distance,scaled_time_delay_embedding_similarity}`. Summaries are MultiMatch vector/direction/length/position/duration, ScanMatch without/with duration, and VAME SED/STDE (including labels for best). Short scanpaths are padded to at least three fixations with `(1,1,0.001)` for comparisons; duration is multiplied by 1000 for ScanMatch. Diagonal, position-aligned subject comparisons produce means/std over image/subject slots; MultiMatch optionally removes NaN rows. Off-diagonal duration-aware ScanMatch is also computed, but not the main diagonal summary. The SED/STDE best summaries reuse the same diagonal arrays, not a separate best-of-support search.

**Grouping constraints:** validation/test slice generated examples into chunks of `args.subject_num` per image, assuming each group has exactly that many trajectories in matching subject order. Metrics use list positions rather than passed subject IDs. Repeated predictions are appended to each image list, while metric arrays allocate only `subject_num` rows: `eval_repeat_num>1` can index outside those arrays. This is not a verified working multi-trial aggregation protocol; defaults use 1.

**Serialization:** O `test.py` calls `utils/data_postprocess.py::get_prediction_list`, rounds/casts coordinates/duration, and restores original subject IDs using excluded/few-shot lists; saves `log/prediction.json`. F/C serialize directly in `test.py` with local subject indices and milliseconds; F uses key `name`, C uses `img_names`. O test explicitly stops when `i_batch > 100`, so it evaluates at most 101 batches; F/C do not have that cap. All depend on a separately supplied subject table; none resamples support during evaluation.

## 11. Configuration Surface

SE-Net is JSON-based: `S/train.py::parse_args` -> `S/common/config.py::JsonConfig` -> `S/src/builder.py::build`/datasets/model/loop. Predictor training is argparse-based: `P/opts.py::parse_opt` first parses, loads optional YAML through `P/utils/config.py::CfgNode`, applies `set_cfgs`, then reparses CLI; explicit CLI wins. Test duplicates its parser rather than loading training `hparams.json`. Relative paths are resolved against process working directory, not automatically against the JSON or script location.

| Argument / Config | Defined At | Consumed By | Effect |
| --- | --- | --- | --- |
| `--hparams`, `--dataset-root` | `S/train.py::parse_args` | JsonConfig/build | select SE-Net configuration and annotation root |
| `--ex_subject`, `--fewshot_subject` | S parser; P opts/test parsers | respective subject filters | excluded seen IDs / ordered unseen IDs; both default `[-1]` |
| `--eval-only`, `--mode` | S parser | run_evaluation | export/training dispatch; exact special mode `evaluate-net` |
| `Data.name`, `TAP`, `fix_path`, `image_path` | S configs | builder/process_data/process_data method | annotation filters, paths, rescaling, task count |
| `Data.num_fewshot` (10 shipped) | S configs | select_fewshot_subject, dataset, export | K image names, triplet reuse at 1, early-return export at 1 |
| `Data.random_support` | S configs | support helper argument | currently no active selection effect |
| `Data.num_subjects` | S configs | classifier and export allocation | number of logits/table rows; not automatically changed for unseen list |
| `Model.embedding_dim` (384) | S configs | builder -> UserEmbeddingNet | user vector width, attention width, input projection |
| `Model.n_heads/n_enc_layers/n_dec_layers/hidden_dim/num_output_layers` | S configs | UserEmbeddingNet constructor | attention heads/depth, FFN width, classifier MLP depth; some allocated layers unused |
| `Data.max_traj_length`, `im_h/im_w`, `pad_idx` | S configs | preprocessing, position embeddings, forward | fixed scanpath length/grid and padding convention |
| `Data.backbone_config/pixel_decoder` | S configs | ImageFeatureEncoder | external Detectron2 YAML and MSD/FPN choice |
| `Train.train_backbone/train_pixel_decoder/dropout` | S configs | UserEmbeddingNet | requires_grad flags and dropout, not persistent eval-mode freezing |
| `Train.losses/subject_id_pred_weight/triplet_loss` | S configs | compute_loss/train_iter | active objective names and weights |
| `Train.adam_lr/adam_betas/lr_steps` | S configs | build, module loop | optimizer and epoch scheduler |
| `Train.batch_size/n_workers/max_iters/checkpoint_every/evaluate_every` | S configs | loaders/loop | batch, stopping, save/eval frequency |
| `Model.checkpoint`, `Train.log_dir` | S configs | build and export/save | SE-Net restore/output; nonempty shipped checkpoint defaults |
| `--user_emb_path`, `--subject_feature_dim` | P opts/test | gazeformer, subject attention/fusion/head | mandatory table file and expected row width (384 default) |
| `--subject_num` | P opts/test | grouping/metrics | expected subjects per query group; not used to create a learned subject table |
| `--num_fewshot`, `--random_support` | P opts/test | train support filter, sample-file path; test log name | separate predictor few-shot data selection; does not choose table filename automatically |
| `--feat_dir/emb_dir/fix_dir/img_dir` | P opts/test | datasets | cached image/text/annotation paths; RGB path not used by active feature-loading forward |
| `--hidden_dim/img_hidden_dim/lm_hidden_dim` | P opts/test | Transformer/head construction | default 512/2048/768; must match caches |
| `--num_encoder/num_decoder/nhead` | P opts/test | Transformer and CrossAttentionPredictor | depth/heads; default 6/6/8 |
| `--im_h/im_w`, `--width/height`, `--origin_width/origin_height` | P opts/test | model positions/fusion; datasets/Sampling | feature grid vs predicted coordinates vs annotation coordinates |
| `--max_length/min_length`, `--action_map_num` | P opts/test | targets/query embeddings/Sampling; heads | parallel horizon, earliest stop, candidate maps (4 default) |
| `--blur_sigma`, `--lambda_1` | P opts | target construction; supervised loop | spatial smoothing and duration loss weight |
| `--epoch/start_rl_epoch/rl_sample_number` | P opts | main.train | epoch budget, supervised/RL boundary, accepted reward trials |
| `--lr/weight_decay/clip/warmup_epoch/rl_lr_initial_decay` | P opts | optimizer/scheduler/train | one-group learning schedule and clipping |
| `--resume_dir/log_root` | P opts | main/record/checkpoint managers | resume/output; existing latest checkpoint can auto-resume |
| `--fewshot_finetune_path` | C opts/train; F opts (inactive block) | legacy loading block | incompatible subject-Embedding fine-tune path, not SE-Net transfer |
| `--no_eval_epoch/eval_repeat_num` | P opts; repeat also test | validation/sampling | evaluation threshold/repeats; repeat>1 caveat above |
| `--evaluation_dir` | P test parser | test.main | predictor best checkpoint and test logs |
| `--cfg/--set_cfgs` | P opts | CfgNode/parse_opt | training YAML defaults/overrides; not the test parser |

Shipped SE-Net JSONs: OSIE has 10 classifier rows, Tse=20, max_iters=15000; F has 7 rows, Tse=20, max_iters=30000; C has 7 rows, Tse=10, max_iters=30000. All use 384-D embeddings, CE+triplet, frozen backbone parameters and trainable pixel decoder. Many inherited JSON keys (`Model.name="HAT"`, `freeze_trained_params`, `resume_training`, `use_sinkhorn`, etc.) do not select active branches in builder/train; follow consumers rather than names.

Launcher overrides matter: O excludes 10..14, uses 10 rows, batch 2, 40 epochs, no-eval threshold 30 (RL default 25). F excludes 0..2, uses 7 rows, batch 2, RL at 20, threshold 18. C excludes 7..9, uses 7 rows, batch 4, length 10, RL at 20, threshold 20. Default predictor L is O=16/F=20/C=7, so C launcher differs. All predictor grid defaults are 24x32, even though F/C output coordinates default to 320x512; grid and output image dimensions are not interchangeable.

## 12. Dependency / Call Map

```text
S/train.py
  imports/calls -> src.builder.build
  calls         -> compute_loss -> compute_output -> transform_fixations -> model
  calls         -> run_evaluation -> src.eval_user.evaluate_user_siamese
  saves         -> model/optimizer checkpoint
S/src/builder.py::build
  calls         -> common.dataset.process_data -> Siamese_Triplet_Gaze (inherits Dataset)
  constructs    -> UserEmbeddingNet (inherits nn.Module), AdamW, DataLoaders
  loads         -> SE-Net checkpoint model state
S/src/models.py::UserEmbeddingNet
  constructs    -> ImageFeatureEncoder, local attention layers, MLP
  freezes       -> image encoder parameters conditionally; re-enables pixel decoder conditionally
  calls         -> pixel_decoder.forward_features; joint attention; CLS readout; classifier
  passes output -> compute_loss OR evaluate_user_siamese aggregation
S/src/eval_user.py::evaluate_user_siamese
  detaches/saves -> user_emb -> external subject table (no parameters shared across processes)
P/train.py / P/test.py
  constructs    -> local datasets, Transformer, gazeformer, Sampling
  loads         -> predictor checkpoint when requested
P/models/gazeformer.py::gazeformer
  loads         -> external table, NOT UserEmbeddingNet checkpoint
  indexes       -> subject_embed[subjects]
  calls         -> Transformer; CrossAttentionPredictor; lowercase attention_module
P/models/models.py::Transformer
  contains      -> TransformerEncoderWrapper, SubjectAttentionModule, IntegrationModule,
                  TransformerDecoderWrapper
  passes output -> personalized memory + decoder states -> predictor heads
P/train.py::main.train
  passes output -> models.loss functions [supervised]
                  Sampling -> utils.evaluation.pairs_eval -> LogAction/LogDuration [RL]
P/train.py::main.validation / P/test.py::main
  calls         -> Sampling -> comprehensive_evaluation_by_subject -> metric implementations
```

Import is not execution: `ResNetCOCO` is imported in `gazeformer.py` but not instantiated there; visual extraction is offline. SE-Net segmentation classes and predictor alternative losses are present/imported without participating in the active paths. Three predictor copies share designs, not live parameters or Python imports.

## 13. Implementation Navigation Guide

These are edit starting points, not changes made by this documentation task. Apply predictor changes deliberately to O/F/C rather than assuming a shared implementation.

| If you want to change... | Start Here | Then Inspect | Why |
| --- | --- | --- | --- |
| SE-Net architecture | `S/src/models.py::UserEmbeddingNet` | `S/src/builder.py::build` | constructor and active forward are separate from configuration wiring |
| Support scanpath encoder | `UserEmbeddingNet.forward` in same file | `S/common/utils.py::transform_fixations`, `Siamese_Triplet_Gaze.process_data` | gathered indices, padding, temporal positions must agree |
| Subject embedding dimension | `S/configs/*.json::Model.embedding_dim` | SE-Net projections/export; `P/opts.py::parse_opt` (`--subject_feature_dim`); predictor fusion/head | artifact width is an offline cross-stage contract |
| Support-set aggregation | `S/src/eval_user.py::evaluate_user_siamese` | support selection and export DataLoader | averaging is outside the model; K=1/drop-last differ |
| K-shot behavior | `S/common/utils.py::select_fewshot_subject` | `Data.num_fewshot`, dataset triplets, export; separately P helper | K currently counts names, not balanced subject/task examples |
| Subject classification objective | `S/train.py::compute_loss` | `UserEmbeddingNet.subject_predictor`, num_subjects, loss weights | same readout supplies classifier and metric loss |
| Metric-learning objective | `S/train.py::compute_loss` | `Siamese_Triplet_Gaze.__getitem__`, loss_funcs | triplet selection is data-side; no hard-negative miner |
| SE-Net checkpoint loading | `S/src/builder.py::build` | `ImageFeatureEncoder.__init__`, model checkpoint shapes | backbone initialization precedes full model restore |
| Freezing/unfreezing SE-Net | `UserEmbeddingNet.__init__` | `S/train.py::train_iter`, optimizer | requires_grad flags do not preserve eval mode |
| Base scanpath architecture | `P/models/models.py::Transformer` | `P/models/gazeformer.py::gazeformer`, both constructors in train/test | local copies, no external model factory |
| Decoder behavior | `TransformerDecoderLayer.forward`, `TransformerDecoder.forward` in same file | gazeformer zero tgt/querypos setup | non-autoregressive, unmasked targets; all steps influence each other |
| Visual features | `P/preprocess/feature_extractor.py::image_data` | dataset feature loads, input_proj, position grid, IntegrationModule | changing caches must change grid/channels coherently |
| SE-Net image encoder | `S/src/models.py::ImageFeatureEncoder` | external backbone YAML, pixel_decoder, input_proj | independent from predictor ResNet cache path |
| Task representation | `Siamese_Triplet_Gaze.process_data`; `P/dataset/dataset.py` | SE-Net task_transform; predictor text_transform/AttentionModule; text_data | SE-Net first-entry lookup differs from predictor category/free-viewing lookup |
| Subject/user representation | `P/models/gazeformer.py::gazeformer.__init__` | export ID mapping, subject lookups, dimensions | replace ordinary tensor loading, not an active nn.Embedding |
| Personalized feature integration | `P/models/models.py::IntegrationModule.forward` | Transformer.forward, SubjectAttentionModule | both spatial and channel terms inject subject information |
| Observer-centric attention | `P/models/models.py::SubjectAttentionModule.forward` | Transformer.forward -> IntegrationModule | subject attention acts before decoder, alongside task attention |
| Adaptive fixation prioritization | `P/models/gazeformer.py::attention_module.forward` | training_process/inference, action_map_num | subject controls map mixing after decoder/head prediction |
| Fixation prediction | `CrossAttentionPredictor.forward` in same file | token_predictor, weighted logits, target construction, Sampling | grid head and stop slot form one categorical action space |
| Duration prediction | `gazeformer.generator_t_mu/generator_t_logvar` | MLPLogNormalDistribution, LogDuration, random_sample | variance semantics must match sampling/losses |
| Scanpath training loss | `P/train.py::main.train` | `P/models/loss.py`, dataset target/mask producers | supervised and RL objectives consume different output keys |
| Add auxiliary loss | appropriate `train_iter` / `main.train` | model outputs, batch targets, weights/parser | define forward output and train/eval behavior together |
| Add model output | `gazeformer.training_process` AND `inference` | validation/test/RL consumers, serialization | mode-specific dictionaries otherwise diverge |
| Add head/branch | `gazeformer.__init__`, both forward paths | checkpoint strict loading, optimizer construction | new registered parameters alter checkpoint contracts |
| Add explanation branch | same, near `outs` / personalized memory / head attention | dataset annotation fields, loss and evaluation consumers | those intermediates exist but are not returned; no explanation branch exists yet |
| Consume full predicted fixation sequence | `gazeformer` action/duration outputs -> `Sampling.generate_scanpath` | train/test/RL consumers and desired gradients | decoded paths stop dynamically and are NumPy/detached; differentiable full-sequence input would need a tensor interface |
| Consume per-timestep decoder state | `gazeformer.training_process` / `inference` local `outs` | `TransformerDecoderWrapper.forward` | `[L,N,H]` is available before/after dropout, not in output dictionary |
| Add dataset annotation field | `Siamese_Triplet_Gaze.process_data` or P active `__getitem__` | corresponding collator, train/test extraction, forward | default nested collation vs manually listed predictor fields differ |
| Change batch structure | P dataset `collate_func` | train/validation/test `.view` flattening and subject regrouping | extra singleton axis and constant subject-count assumptions are coupled |
| Change support/query structure | S support helper/export; P subject filters | row-map serialization, query grouping, checkpoint inputs | there is no shared episodic sample contract today |
| Change training stages | `S/train.py::<module>`; `P/train.py::main.train` | scheduler, RecordManager, checkpoint manager | separate processes plus an epoch-based supervised/RL boundary |
| Joint-train SE-Net and predictor | `gazeformer` table boundary + `UserEmbeddingNet` | datasets/collators, optimizers, checkpoint formats, both environments | requires replacing offline detach/save/load with an explicit differentiable model/data path |
| Freeze arbitrary modules | relevant model constructor + optimizer setup | repeated `.train()` / `.eval()` in loops | training mode is not gradient freezing; RL still backpropagates |
| Add CLI/config option | S parse_args/JsonConfig OR P opts.parse_opt | P test parser, constructor calls, launch scripts | train/test are not one unified configuration surface |
| Predictor checkpoint loading | P main resume/test blocks | CheckpointManager, RecordManager, external user_emb_path | state_dict does not carry subject table or IDs |
| Evaluation metrics | `P/utils/evaluation.py::comprehensive_evaluation_by_subject` | evaltools; `pairs_eval`; grouping and selection score | changing summary need not change RL reward unless wired explicitly |
| Inference/decoding | `P/models/sampling.py::Sampling` | inference softmax, min/max lengths, test serialization | stopping/coordinates are downstream of parallel logits |
| Repeat support experiments | S selection/export and artifact naming | P user_emb_path, external experiment orchestration | prediction-repeat knob does not resample supports |

## 14. High-Risk Couplings and Invariants

1. **Offline interface, not an integrated model:** `S/src/eval_user.py` exports a bare detached tensor; `P/models/gazeformer.py` loads it outside state_dict. Predictor checkpoints are not self-contained and cannot restore/verify the table or its subject mapping.
2. **ID/order contract:** seen IDs are compacted in sorted order; unseen IDs follow list order. Classifier/table row allocation is `Data.num_subjects`, not the unseen-list length. Predictor metrics regroup by `subject_num` and list position; missing/duplicate/reordered trajectories can break alignment even if tensor indexing succeeds.
3. **Three separate copies:** models/models.py O/F are identical, C differs only in a blank line at inspection. F/C gazeformer.py are identical; O differs in one-dimensional table handling, query-position initialization (std 1 versus 0.01), and training duration `.squeeze()` versus `.squeeze(2)`. O supervised batch size 1 can therefore lose a required dimension.
4. **Sequence/grid coupling:** SE-Net pads exactly Tse and slices the last Tse working-memory tokens; changing padding length changes readout. Predictor zero targets/query embeddings/output reshapes share L; positional encodings, cached token count, fusion spatial linear layers and action maps share G. Dse must match `subject_feature_dim`, but need not equal predictor H.
5. **Different token conventions:** SE-Net uses 0 padding and real fixation indices starting at 1 in feature-gather grids; predictor 0 is termination and 1..G are spatial cells. Predictor target rows beyond observed fixations are stop targets but only the first stop is action-masked in.
6. **Coordinates/units differ:** S uses resized RGB/normalized coordinates and raw durations; O predictor bins original coordinates and rescales evaluation coordinates; F/C bin configured-size coordinates and evaluate raw x/y. Predictor losses/sampled arrays use seconds; serialized/ScanMatch durations use milliseconds. Query firstfix metadata does not force an initial center fixation.
7. **Export can silently undercount:** training loader retains `drop_last=True` in eval-only export; K selects unique names, not fixed records per subject; zero-count rows stay zero; K=1 skips averaging. Export evaluation still asks for positive/negative dataset samples unless K=1, so missing same/other-subject candidates can fail even though only anchors are consumed.
8. **Duration/task path surprises in SE-Net:** duration PE is erased before attention; task vector always comes from the first dictionary entry. Unused FFN/query modules are registered but not forward dependencies. These are concrete source behaviors, not assumed architecture choices.
9. **Train/eval is an output contract:** supervised requires `actions`; RL/validation/test require `all_actions_prob`. RL calls eval without no-grad. Calling model.eval does not freeze parameters, and SE-Net model.train does not preserve backbone eval mode.
10. **Evaluation partitions are not uniform:** S validates on `test`; O predictor validates on `validation`; F on `test`; C on `valid`. All predictor test entries use `test`. Do not assume independent held-out validation in F.
11. **Repeat and test limits:** O test caps at 101 batches. More than one prediction repeat appends extra subject rows into fixed-size metric arrays. Scanpath duration sampling uses sigma2 rather than its square root. These paths require review before extending evaluation.
12. **Legacy branches are not reliable APIs:** C fine-tune expects an nn.Embedding/checkpoint key that no longer exists; bbox training in S expects missing outputs and computes the wrong loss; `COCOSearch_by_subject` references unset args; `select_fewshot_subject_task` in C is not called by the launch paths.
13. **Paths/configs are operational dependencies:** S image YAML/weights are external; launchers' seen-embedding filename differs from exporter; saved artifacts carry no metadata. O feature-extractor module entry calls `text_data(dataset_path=args.p, ...)` although parser defines `dataset_path`, not `p`; preprocessing commands are not validated turnkey setup.
14. **Resume is partial/dataset-sensitive:** S restores step but not optimizer/scheduler; predictor uses a separate history JSON and reconstructs LambdaLR from iteration/loader lengths. Changing dataset size or stage boundary changes the resumed schedule. No general automatic checkpoint migration exists.
15. **No safe absent-data fallback:** configured checkpoints/features/embedding dictionaries are loaded directly; empty triplet pools use random.choice; all-pad SE-Net memory has no explicit special handling. Inspect data and runtime behavior instead of inferring graceful fallback.

## 15. Differences From the Base Predictor

Within the requested `isp-senet/**` scope there is **no separate pristine base Gazeformer/Gazeformer-ISP implementation to diff against**. The predictor is the local ISP-style implementation, not an import of the sibling `gazeformer-isp/` tree. Upstream equivalence, original losses, and all historical modifications are therefore not established by this map. Comments showing an old embedding lookup are evidence of inactive code, not a verified base executable.

| Area | Base Path / Symbol | ISP-SENet Path / Symbol | Actual Difference / Evidence Limit |
| --- | --- | --- | --- |
| Observer representation | commented `nn.Embedding` and lookup lines in local gazeformer.py; no active base | `P/models/gazeformer.py::gazeformer.__init__` | active code loads a tensor and indexes it; no trainable subject embedding |
| Training boundary | no separate base entry inside scope | `S/train.py`, `P/train.py` | separate SE-Net training/export and predictor training confirmed; not an inferred paper-stage mapping |
| Feature integration/decoder | no pristine base for comparison | `P/models/models.py::{Transformer,IntegrationModule}` | local modules execute as mapped; unchanged-from-upstream claim not established |
| Objectives | no pristine base for comparison | `P/models/loss.py`, `P/train.py::main.train` | CE/log-normal and ScanMatch RL verified; historical additions/removals not established |
| Query/support interface | no pristine base for comparison | S support dataset/export; P datasets | P still consumes features/IDs/task vectors, not an online support batch |

Verified **local variant** differences (not upstream comparisons): OSIE versus COCO task/path/grouping rules; validation split names and coordinate handling; O versus F/C query-embedding initialization/duration squeeze; C's active legacy fine-tune block; O test cap and subject-ID recovery. These affect edits even where the transformer and loss/sampling implementations otherwise match. `models/loss.py`, `models/sampling.py`, and `utils/checkpointing.py` are byte-identical across O/F/C at this commit.

## 16. External Dependencies That Matter

- **PyTorch / torchvision:** module/state_dict/optimizer/DataLoader contracts throughout; torchvision transforms for SE-Net RGB; pretrained Mask R-CNN ResNet50 body in offline predictor feature extraction. Cache generation and current predictors are separate computations.
- **Detectron2 / fvcore:** `S/src/models.py::ImageFeatureEncoder`, `S/src/config.py`, backbone registry, ShapeSpec, layers/initialization. External YAML and weights control the concrete image backbone.
- **Local deformable-attention extension:** `S/src/pixel_decoder/ops/modules/ms_deform_attn.py::MSDeformAttn` calls the custom operation used by MSD. Build/install route is `ops/setup.py` / `ops/make.sh`; a compatible compiled extension is a runtime prerequisite, not a trivial utility.
- **SentenceTransformers:** only offline `P/preprocess/feature_extractor.py::text_data`; serializes embedding dictionaries. Current predictor forward uses loaded vectors, not tokenizer/text-model APIs.
- **NumPy / SciPy / Pillow:** serialized arrays/structured fixation vectors, Gaussian target smoothing, harmonic-mean rewards/selection, MATLAB preprocessing and RGB image loading.
- **multimatch_gaze:** direct metric API in `P/utils/evaluation.py`; local ScanMatch and visual-attention metrics supply the other reward/evaluation measures. Version/runtime numerical behavior was not tested.
- **PyYAML/yacs-style CfgNode implementation and TensorBoard:** P config loading/overrides and train logging respectively. See `P/utils/config.py` and the two environment manifests rather than treating them as a single interchangeable environment.

## 17. Uncertainties / Things Not Established From Static Inspection

- Actual tensor/checkpoint contents, compatible shapes, row provenance, and whether downloaded files match current architectures or filenames were not inspected. The source producer/consumer contract is established; artifact compatibility is not.
- External backbone YAML/weights, datasets, pretrained feature caches, and task dictionaries are needed to establish the exact loaded backbone, token-grid provenance, coordinate units, full subject coverage/order, and support/query image disjointness.
- Intended corrections for erased duration encoding, first-task-only SE-Net lookup, the one-shot export shortcut, incomplete fine-tuning/bbox branches, sampler variance mismatch, and repeat evaluation are unknown. This map records behavior, not suggested paper-aligned replacements.
- Whether historical experiments used the current validation splits, O test cap, exporter filename, a renamed tensor, or locally patched scripts cannot be determined from these sources.
- No repeated-support experiment orchestrator or shared support/query manifest was found. External experiments may exist, but the provided sources do not establish their seeds, averaging protocol, or reproducibility.
- Runtime effects of frozen-backbone training mode, all-padding attention, empty/small loaders, GPU/device/DataParallel behavior, custom-extension compatibility, and installed library versions need execution. No smoke run or metric reproduction was performed.
- No upstream/base behavioral comparison was performed outside the requested scope; identical-looking terminology/comments are not evidence of equivalence.

## 18. Fast Navigation Index

Aliases are defined in section 0; each P reference means inspect the appropriate O/F/C copy.

```text
SE-NET TRAIN ENTRY       -> S/train.py::<module> / train_iter
ISP-SENET TRAIN ENTRY    -> P/train.py::main / main.train
SE-NET EXPORT / EVAL     -> S/train.py::run_evaluation -> S/src/eval_user.py::evaluate_user_siamese
PREDICTOR VALIDATION     -> P/train.py::main.validation
PREDICTOR TEST           -> P/test.py::main

SE-NET BUILDER           -> S/src/builder.py::build
SE-NET DATASET          -> S/common/data.py::Siamese_Triplet_Gaze
SUPPORT SAMPLER          -> S/common/utils.py::select_fewshot_subject
TRIPLET SAMPLER          -> S/common/data.py::Siamese_Triplet_Gaze.__getitem__
FIXATION TOKENIZATION    -> S/common/utils.py::transform_fixations
QUERY DATASET            -> O/dataset/dataset.py::OSIE; F/C/dataset/dataset.py::COCOSearch
QUERY RL / EVAL          -> same files, *_rl / *_evaluation
QUERY COLLATE            -> active dataset class::collate_func

SE-NET                  -> S/src/models.py::UserEmbeddingNet
SUPPORT IMAGE ENCODER    -> S/src/models.py::ImageFeatureEncoder
SUPPORT CONTEXT ENCODER  -> UserEmbeddingNet.forward / working_memory_encoder
SUBJECT READOUT          -> UserEmbeddingNet.user_fix_cross_attn / subject_predictor
SUPPORT AGGREGATION      -> S/src/eval_user.py::evaluate_user_siamese

TOP-LEVEL PREDICTOR      -> P/models/gazeformer.py::gazeformer
SUBJECT TABLE LOAD       -> gazeformer.__init__ / subject_embed
QUERY VISUAL EXTRACTION  -> P/preprocess/feature_extractor.py::image_data / ResNetCOCO
VISUAL / TASK ENCODER    -> P/models/models.py::TransformerEncoderWrapper / AttentionModule
OBSERVER ATTENTION       -> P/models/models.py::SubjectAttentionModule
PERSONALIZED INTEGRATION -> P/models/models.py::IntegrationModule
DECODER                  -> P/models/models.py::TransformerDecoder / TransformerDecoderLayer
PER-TIMESTEP STATES      -> gazeformer.training_process / inference local outs
FIXATION HEAD            -> P/models/gazeformer.py::CrossAttentionPredictor / token_predictor
MAP PRIORITIZATION       -> P/models/gazeformer.py::attention_module
DURATION HEAD            -> gazeformer.generator_t_mu / generator_t_logvar

SE-NET LOSSES            -> S/train.py::compute_loss / loss_funcs
SCANPATH LOSSES          -> P/models/loss.py::CrossEntropyLoss / MLPLogNormalDistribution
RL LOSSES / REWARD       -> P/train.py::main.train; P/models/loss.py::LogAction / LogDuration;
                           P/utils/evaluation.py::pairs_eval
SE-NET OPTIMIZER         -> S/src/builder.py::build
PREDICTOR OPTIMIZER      -> P/train.py::main
SCHEDULERS               -> S/train.py::<module>; P/train.py::main.lr_lambda
SE-NET CHECKPOINT IO     -> S/train.py::<module> save; S/src/builder.py::build load
PREDICTOR CHECKPOINT IO  -> P/utils/checkpointing.py::CheckpointManager; P/train.py / test.py::main
DECODING                 -> P/models/sampling.py::Sampling.random_sample / generate_scanpath
METRICS                  -> P/utils/evaluation.py::comprehensive_evaluation_by_subject
CONFIG                   -> S/train.py::parse_args + S/configs/*.json;
                           P/opts.py::parse_opt + P/test.py::<module> parser
```