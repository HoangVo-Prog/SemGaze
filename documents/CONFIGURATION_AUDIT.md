> Historical implementation/performance report. Split, sampling, schedule and evaluation examples below predate the 95/5 query-coverage migration. Use [COCO_955_MIGRATION.md](COCO_955_MIGRATION.md) and the current YAML/CLI for new runs. Historical measurements are not evidence for current sampling throughput.

# Configuration plumbing audit — 2026-10-05

The reference `configs/flat_single.yaml` is an experiment, not a protocol validator.
The pre-existing user choice `initialization_adapter: null` / `adapter_load_mode: fresh`
is retained and now executes fresh PEFT initialization. No baseline checkout, fixture
golden, scientific objective, normalized data format or projector architecture was changed.

## Resolution and consumers

`load_config(path, overrides)` parses YAML, fills **missing** fields from
`configs/defaults.yaml`, applies explicitly supplied CLI overrides, and validates the
result. Explicit null is retained: for example null adapter means fresh initialization,
null clipping disables clipping, and unresolved optimizer numeric values fail rather
than becoming hidden defaults. Required training horizon and generation choices are
resolved before model/data execution. `training.max_steps` takes precedence over
`epochs * steps_per_epoch`; partial final epochs are supported.

Training passes the resulting mapping directly to `build_flat_model_bundle(config=...)`,
`TrainingEpisodeSampler(data_config=...)`, data adapters and scheduling. The bundle owns
the same resolved experiment values used by WHERE, semantics, optimizers, validation,
prediction and checkpoints. The generated/overridden output directory is recorded before
execution. Every run saves `resolved_config.json`, even if the optional copy inside each
checkpoint is disabled. Standalone evaluation inherits checkpoint config, then accepts
runtime/evaluation YAML overrides and CLI budget/output overrides. Model/data identity
remains attached to the checkpoint.

Resume restores weights, moments, RNG and history. Current optimizer hyperparameters,
precision, batching, logging, evaluation and schedule reach their consumers; loading old
optimizer state does not silently replace the new hyperparameters. Changing saved model/
token structure or optimizer implementation is rejected because their state cannot be
restored into a different structure. The seed initializes new runs; resume restores RNG
state. Model initialization source and fresh LoRA settings describe how saved weights
were created and are not reapplied to overwrite resumed weights.

## Previously enforced or ignored settings now wired

- Removed required base-model ID and released-adapter path/mode; added fresh LoRA and
  arbitrary compatible local continued adapters. Removed released rank/alpha/dropout/
  target equality checks. Fresh targets are restricted to actual matching LM modules.
- Removed fixed data/image paths, variant, unseen-subject list, K list, uniform K law,
  context length and episode support/image-capacity requirements. Manifest membership
  must still agree with configured subjects/variant. Evaluation requires actual persisted
  support blocks for the requested K; it never invents frozen supports.
- Removed fixed AdamW/cosine selection, epoch-only evaluation, all-K evaluation,
  one prediction batch, mandatory validation prediction scope, mandatory full-epoch
  horizons and all-checkpoint-flags-on enforcement. Training batch size and accumulation
  were already used; tests now explicitly prove their product reaches the step reducer.
- Fixed the post-CLI model reload, hard-coded runtime data construction, cached default
  context limit, constant evaluation K/draw loops and implicit null optimizer defaults.
- Added YAML horizon/output/save/logging controls previously confined to CLI literals.
  Default prediction behavior is retained; additional configured batches reuse accepted
  training episodes without consuming extra sampler draws.
- Wired boundary-token spelling through token installation, prompt, codec, collator,
  generation parser and checkpoint loader. Atomicity remains mandatory.
- Wired checkpoint component flags and optional gradient diagnostic gates. All finite
  loss/gradient and tensor integrity checks remain mandatory. Disabling required saved
  model components creates a partial export, with a clear load-time failure.

## Validation intentionally retained

Missing files and images; malformed JSON/records; checksum and persisted membership
mismatches; duplicate IDs and train/validation leakage; unresolved coordinate frames;
finite numeric fixation fields and aligned X/Y/duration lengths; semantic partitions and
nonempty text; same-subject/distinct/excluded-query supports and frozen order; exclusion
of configured unseen subjects from updates; tokenizer IDs/templates and atomic boundary
compatibility; Git LFS pointers; exact adapter tensor keys/shapes/values; supported native
InternVL architecture; native image expansion, response labels and token spans; context
overflow; tensor dimensions, state counts, masks and connected gradients; finite losses
and gradients; strict parsers; checkpoint tensor/ID/tied-weight/manifest integrity.

## Remaining implementation limits

These limits are distinct from experiment recommendations. Every restricted reference
field is listed below with its concrete reason. Additional limits:

- Base model IDs/paths are customizable within native HF InternVL, whose image-token and
  inputs-embeds behavior this package implements. Alternative VLM architectures need a
  model/processor integration. Continued adapters currently take local PEFT directories
  with safetensors weights; they may be downloaded separately. They need not include a
  tokenizer; if tokenizer artifacts are present, compatibility is verified.
- `adapter_load_mode` implements `fresh` and `continue_trainable`. Fresh requires a null
  adapter source; continuing requires a source. Continued adapters use their own tensor
  architecture; `model.lora.*` is explicitly **fresh-only**. Effective continued metadata
  is recorded in `preflight.json`, without overwriting the requested fresh settings.
- `merge_adapter_for_training: true` is unsupported because merging removes the retained
  trainable LoRA tensors required by this training/checkpoint lifecycle.
- Precision supports FP32/BF16. FP16 needs a GradScaler path that is not implemented.
  Optimizers support AdamW/Adam/SGD; Adam betas/epsilon apply to Adam-family optimizers.
  Schedulers support cosine/linear/constant/constant_with_warmup; constant deliberately
  has no warmup. No mixed backend or multibranch work was introduced.
- Physical execution remains sequential per episode; configured batch size times
  gradient accumulation controls the exact episode-mean optimizer step.
- Validation loss aggregation remains episode mean, with full seen query coverage;
  prediction scope supports all_seen or none. `train_batches` can be zero or any count
  available before evaluation. No examples are sampled solely to fill prediction output.
- KV caching is configurable for WHERE, but HF checkpointed gradients disable it, so
  `use_cache: true` together with gradient checkpointing raises an incompatibility error.

## Complete reference-field inventory

`Implemented contract` below means alternative behavior has no implementation and fails
explicitly; it is not silently overwritten. Defaults for direct standalone utility calls
remain supported, but runtime entrypoints pass resolved settings explicitly.

| Field | Consumer / status |
|---|---|
| `experiment.name` | config.resolve_output_dir (generated run directory prefix). |
| `experiment.semantic_mode` | Implemented contract: Only one flat WHAT/WHY/HOW response using full chronological states and gold groups is implemented; other semantic paths/objectives change the requested architecture. |
| `experiment.protocol_version` | checkpoint metadata and restore compatibility. |
| `experiment.seed` | model initialization RNG and TrainingEpisodeSampler RNG. |
| `data.dataset` | Implemented contract: Only the persisted COCO-Search18 record/manifest adapter is implemented. |
| `data.variant` | read_persisted_splits compares configured variant with manifest/records. |
| `data.split_root` | train/evaluate -> read_persisted_splits. |
| `data.images_root` | train/evaluate -> CocoSearch18Adapter. |
| `data.annotation_frame` | CocoSearch18Adapter.image_info. |
| `data.unseen_subjects` | manifest checks, sampler, training eligibility, validation and final evaluation. |
| `data.duration.source_field` | raw record length check and CocoSearch18Adapter -> duration_ms. |
| `data.duration.semantics` | Implemented contract: Input adaptation implements dwell milliseconds with identity conversion; timestamp/unit conversion is not implemented. |
| `data.duration.source_unit` | Implemented contract: Input adaptation implements dwell milliseconds with identity conversion; timestamp/unit conversion is not implemented. |
| `data.duration.conversion_to_ms` | Implemented contract: Input adaptation implements dwell milliseconds with identity conversion; timestamp/unit conversion is not implemented. |
| `data.fewshot.k_values` | TrainingEpisodeSampler candidate/query pools and weighted draws. |
| `data.fewshot.train_k_probabilities` | TrainingEpisodeSampler.rng.choices weights; zero-probability K need no pool. |
| `data.fewshot.require_same_subject` | Implemented contract: Same-subject, distinct-image, query-excluding, ordered supports are structural episode invariants; changing them changes the requested scientific data contract. |
| `data.fewshot.require_distinct_support_stimuli` | Implemented contract: Same-subject, distinct-image, query-excluding, ordered supports are structural episode invariants; changing them changes the requested scientific data contract. |
| `data.fewshot.exclude_query_stimulus` | Implemented contract: Same-subject, distinct-image, query-excluding, ordered supports are structural episode invariants; changing them changes the requested scientific data contract. |
| `data.fewshot.preserve_input_order` | Implemented contract: Same-subject, distinct-image, query-excluding, ordered supports are structural episode invariants; changing them changes the requested scientific data contract. |
| `model.base_model` | AutoProcessor and AutoModelForImageTextToText loaders, including checkpoint reconstruction. |
| `model.backend` | Implemented contract: Only the differentiable HF + PEFT model path is implemented. |
| `model.initialization_adapter` | inspect_adapter and PeftModel.from_pretrained; null for fresh. |
| `model.adapter_load_mode` | validated fresh/continue combination -> selected initialization path. |
| `model.merge_adapter_for_training` | Unsupported true: merged models lack retained trainable LoRA state. |
| `model.lora.rank` | build_flat_model_bundle -> real LoraConfig for fresh initialization; continued checkpoint metadata is authoritative. |
| `model.lora.alpha` | build_flat_model_bundle -> real LoraConfig for fresh initialization; continued checkpoint metadata is authoritative. |
| `model.lora.dropout` | build_flat_model_bundle -> real LoraConfig for fresh initialization; continued checkpoint metadata is authoritative. |
| `model.lora.target_modules` | build_flat_model_bundle -> real LoraConfig for fresh initialization; continued checkpoint metadata is authoritative. |
| `model.freeze.vision_tower` | Implemented contract: The trainable-set checker and checkpoint lifecycle implement LM LoRA + token rows + P_E, not dense base/vision/vocabulary training and persistence. |
| `model.freeze.native_multimodal_projector` | Implemented contract: The trainable-set checker and checkpoint lifecycle implement LM LoRA + token rows + P_E, not dense base/vision/vocabulary training and persistence. |
| `model.freeze.base_lm_outside_lora` | Implemented contract: The trainable-set checker and checkpoint lifecycle implement LM LoRA + token rows + P_E, not dense base/vision/vocabulary training and persistence. |
| `model.freeze.old_vocab_rows` | Implemented contract: The trainable-set checker and checkpoint lifecycle implement LM LoRA + token rows + P_E, not dense base/vision/vocabulary training and persistence. |
| `where.mode` | Implemented contract: Only the XYD serialization and parsing path is implemented. |
| `where.end_fix_token` | model token installation, WHERE prompt/codec/collation/parser and checkpoint loader. |
| `where.require_atomic_end_fix` | Implemented contract: Query-only supervision and atomic boundary hidden-state extraction require these masks/states; alternate supervision/readout is not implemented. |
| `where.coordinates.grid_size` | Implemented contract: The XYD codec, prompt and parser implement the existing two-digit coordinates / three-digit millisecond data format; alternative codecs are not part of this refactor. |
| `where.coordinates.min_value` | Implemented contract: The XYD codec, prompt and parser implement the existing two-digit coordinates / three-digit millisecond data format; alternative codecs are not part of this refactor. |
| `where.coordinates.max_value` | Implemented contract: The XYD codec, prompt and parser implement the existing two-digit coordinates / three-digit millisecond data format; alternative codecs are not part of this refactor. |
| `where.coordinates.text_width` | Implemented contract: The XYD codec, prompt and parser implement the existing two-digit coordinates / three-digit millisecond data format; alternative codecs are not part of this refactor. |
| `where.coordinates.conversion` | Implemented contract: The XYD codec, prompt and parser implement the existing two-digit coordinates / three-digit millisecond data format; alternative codecs are not part of this refactor. |
| `where.duration.unit` | Implemented contract: The XYD codec, prompt and parser implement the existing two-digit coordinates / three-digit millisecond data format; alternative codecs are not part of this refactor. |
| `where.duration.bin_width_ms` | Implemented contract: The XYD codec, prompt and parser implement the existing two-digit coordinates / three-digit millisecond data format; alternative codecs are not part of this refactor. |
| `where.duration.min_value` | Implemented contract: The XYD codec, prompt and parser implement the existing two-digit coordinates / three-digit millisecond data format; alternative codecs are not part of this refactor. |
| `where.duration.max_value` | Implemented contract: The XYD codec, prompt and parser implement the existing two-digit coordinates / three-digit millisecond data format; alternative codecs are not part of this refactor. |
| `where.duration.text_width` | Implemented contract: The XYD codec, prompt and parser implement the existing two-digit coordinates / three-digit millisecond data format; alternative codecs are not part of this refactor. |
| `where.duration.non_integer_rounding` | Implemented contract: The XYD codec, prompt and parser implement the existing two-digit coordinates / three-digit millisecond data format; alternative codecs are not part of this refactor. |
| `where.context.max_k` | config capacity validation and collate_where episode guard. |
| `where.context.max_images_per_episode` | config capacity validation and collate_where episode guard. |
| `where.context.max_total_sequence_length` | bundle.context_limit -> WHERE/semantic collation and generation budgets. |
| `where.context.visual_policy` | Implemented contract: Native collation and image-token expansion implement exactly one 448x448 tile per image; patch tiling is not implemented. |
| `where.context.overflow_training` | Implemented contract: The training loop implements whole-episode rejection/resampling; partial-support truncation changes the episode contract. |
| `where.supervision.support_assistant_labels` | Implemented contract: Query-only supervision and atomic boundary hidden-state extraction require these masks/states; alternate supervision/readout is not implemented. |
| `where.supervision.query_assistant_labels` | Implemented contract: Query-only supervision and atomic boundary hidden-state extraction require these masks/states; alternate supervision/readout is not implemented. |
| `where.supervision.supervise_query_end_fix` | Implemented contract: Query-only supervision and atomic boundary hidden-state extraction require these masks/states; alternate supervision/readout is not implemented. |
| `where.supervision.output_hidden_states` | Implemented contract: Query-only supervision and atomic boundary hidden-state extraction require these masks/states; alternate supervision/readout is not implemented. |
| `where.supervision.use_cache` | forward_where passes requested use_cache to HF. |
| `state.readout_layer` | Implemented contract: The implemented connected graph extracts final query boundary states and inserts d_model vectors via one Linear+LayerNorm projector; detached/alternate graphs or dimensions are not implemented. |
| `state.readout_position` | Implemented contract: The implemented connected graph extracts final query boundary states and inserts d_model vectors via one Linear+LayerNorm projector; detached/alternate graphs or dimensions are not implemented. |
| `state.projector.type` | Implemented contract: The implemented connected graph extracts final query boundary states and inserts d_model vectors via one Linear+LayerNorm projector; detached/alternate graphs or dimensions are not implemented. |
| `state.projector.input_dim` | Implemented contract: The implemented connected graph extracts final query boundary states and inserts d_model vectors via one Linear+LayerNorm projector; detached/alternate graphs or dimensions are not implemented. |
| `state.projector.output_dim` | Implemented contract: The implemented connected graph extracts final query boundary states and inserts d_model vectors via one Linear+LayerNorm projector; detached/alternate graphs or dimensions are not implemented. |
| `state.detach_where_states` | Implemented contract: The implemented connected graph extracts final query boundary states and inserts d_model vectors via one Linear+LayerNorm projector; detached/alternate graphs or dimensions are not implemented. |
| `state.direct_state_positions.attention_mask` | Implemented contract: The implemented connected graph extracts final query boundary states and inserts d_model vectors via one Linear+LayerNorm projector; detached/alternate graphs or dimensions are not implemented. |
| `state.direct_state_positions.label` | Implemented contract: The implemented connected graph extracts final query boundary states and inserts d_model vectors via one Linear+LayerNorm projector; detached/alternate graphs or dimensions are not implemented. |
| `semantic.mode` | Implemented contract: Only one flat WHAT/WHY/HOW response using full chronological states and gold groups is implemented; other semantic paths/objectives change the requested architecture. |
| `semantic.replay_support_context` | Implemented contract: Only one flat WHAT/WHY/HOW response using full chronological states and gold groups is implemented; other semantic paths/objectives change the requested architecture. |
| `semantic.consume_full_chronological_states` | Implemented contract: Only one flat WHAT/WHY/HOW response using full chronological states and gold groups is implemented; other semantic paths/objectives change the requested architecture. |
| `semantic.use_gold_why_groups` | Implemented contract: Only one flat WHAT/WHY/HOW response using full chronological states and gold groups is implemented; other semantic paths/objectives change the requested architecture. |
| `semantic.predict_groups` | Implemented contract: Only one flat WHAT/WHY/HOW response using full chronological states and gold groups is implemented; other semantic paths/objectives change the requested architecture. |
| `semantic.semantic_special_tokens` | Implemented contract: Only one flat WHAT/WHY/HOW response using full chronological states and gold groups is implemented; other semantic paths/objectives change the requested architecture. |
| `semantic.output_format` | Implemented contract: Only one flat WHAT/WHY/HOW response using full chronological states and gold groups is implemented; other semantic paths/objectives change the requested architecture. |
| `semantic.parser` | Implemented contract: Only one flat WHAT/WHY/HOW response using full chronological states and gold groups is implemented; other semantic paths/objectives change the requested architecture. |
| `semantic.loss` | Implemented contract: Only one flat WHAT/WHY/HOW response using full chronological states and gold groups is implemented; other semantic paths/objectives change the requested architecture. |
| `training.lambda_where` | run_flat_training_step and validation weighted joint objective. |
| `training.lambda_sem` | run_flat_training_step and validation weighted joint objective. |
| `training.optimizer` | make_optimizer dispatches AdamW/Adam/SGD. |
| `training.learning_rate` | make_optimizer, scheduler base rate, resume hyperparameters. |
| `training.scheduler` | make_scheduler dispatches supported HF scheduler. |
| `training.warmup_ratio` | make_scheduler computes horizon * ratio. |
| `training.precision` | model/checkpoint loader dtype (FP32/BF16). |
| `training.steps_per_epoch` | train entrypoint and run_training_loop epoch boundaries. |
| `training.epochs` | train entrypoint derives max_steps when max_steps is null. |
| `training.max_steps` | train entrypoint -> scheduler and loop horizon; CLI overrides supported. |
| `training.max_episode_retries` | run_training_loop complete-episode overflow retry limit. |
| `training.per_device_train_batch_size` | episode-mean optimizer step and prediction batch size. |
| `training.gradient_accumulation_steps` | run_training_loop episode count and loss scaling per optimizer step. |
| `training.gradient_checkpointing` | model/checkpoint enable nonreentrant checkpointing; resume updates mode. |
| `training.max_grad_norm` | flat_step/loop clipping; null disables clipping. |
| `training.weight_decay` | make_optimizer and resume hyperparameters. |
| `training.adam_beta1` | Adam-family optimizer beta1 and resume hyperparameters. |
| `training.adam_beta2` | Adam-family optimizer beta2 and resume hyperparameters. |
| `training.adam_epsilon` | Adam-family optimizer epsilon and resume hyperparameters. |
| `training.backward.one_joint_backward` | Implemented contract: The implemented connected graph extracts final query boundary states and inserts d_model vectors via one Linear+LayerNorm projector; detached/alternate graphs or dimensions are not implemented. |
| `training.backward.detach_f` | Implemented contract: The implemented connected graph extracts final query boundary states and inserts d_model vectors via one Linear+LayerNorm projector; detached/alternate graphs or dimensions are not implemented. |
| `training.backward.detach_r` | Implemented contract: The implemented connected graph extracts final query boundary states and inserts d_model vectors via one Linear+LayerNorm projector; detached/alternate graphs or dimensions are not implemented. |
| `evaluation.strategy` | run_training_loop epoch/steps/no evaluation scheduling. |
| `evaluation.every_steps` | run_training_loop when strategy is steps. |
| `evaluation.k_values` | validate_epoch, predict_epoch and evaluate_flat loops. |
| `evaluation.loss_aggregation` | Implemented contract: Validation aggregates episode means; a token-weighted/subject-weighted reducer is not implemented. |
| `evaluation.predictions.train_batches` | loop retains selected accepted episodes; predict_epoch validates/exports requested count. |
| `evaluation.predictions.validation_scope` | predict_epoch all_seen/none query iterator. |
| `evaluation.predictions.semantic_max_new_tokens` | epoch and standalone semantic model.generate max_new_tokens. |
| `runtime.device` | model/checkpoint placement and CUDA preflight. |
| `runtime.output_dir` | resolved model/train/evaluate output paths and saved config. |
| `logging.every_steps` | run_training_loop train-log cadence (also logs final step). |
| `logging.filename` | run_training_loop JSONL destination under output directory. |
| `checkpoint.save_every` | run_training_loop step save cadence; null disables interval saves. |
| `checkpoint.save_at_epoch_end` | run_training_loop epoch-boundary saves. |
| `checkpoint.save_at_end` | run_training_loop final-step save. |
| `checkpoint.save_processor_and_tokenizer` | save_checkpoint includes/omits the named artifact/component; full reconstruction/resume verifies required state exists. |
| `checkpoint.save_end_fix_token_id` | save_checkpoint includes/omits the named artifact/component; full reconstruction/resume verifies required state exists. |
| `checkpoint.save_peft_adapter` | save_checkpoint includes/omits the named artifact/component; full reconstruction/resume verifies required state exists. |
| `checkpoint.save_trainable_end_fix_rows` | save_checkpoint includes/omits the named artifact/component; full reconstruction/resume verifies required state exists. |
| `checkpoint.save_projector` | save_checkpoint includes/omits the named artifact/component; full reconstruction/resume verifies required state exists. |
| `checkpoint.save_resolved_config` | save_checkpoint includes/omits the named artifact/component; full reconstruction/resume verifies required state exists. |
| `checkpoint.save_split_manifest_identity` | save_checkpoint includes/omits the named artifact/component; full reconstruction/resume verifies required state exists. |
| `checkpoint.save_optimizer` | save_checkpoint includes/omits the named artifact/component; full reconstruction/resume verifies required state exists. |
| `checkpoint.save_scheduler` | save_checkpoint includes/omits the named artifact/component; full reconstruction/resume verifies required state exists. |
| `smoke.fixture` | smoke runner and train entrypoint fixture loader. |
| `smoke.expected_where` | contract smoke golden loader. |
| `smoke.expected_flat_target` | contract smoke golden loader. |
| `smoke.require_finite_losses` | Implemented contract: Non-finite losses are runtime integrity failures; optional diagnostic switches cannot authorize NaN updates. |
| `smoke.require_lora_gradient` | flat_step optional presence/nonzero gradient diagnostic and smoke reporting; nonfinite gradients always fail. |
| `smoke.require_projector_nonzero_gradient` | flat_step optional presence/nonzero gradient diagnostic and smoke reporting; nonfinite gradients always fail. |
| `smoke.require_end_fix_input_gradient` | flat_step optional presence/nonzero gradient diagnostic and smoke reporting; nonfinite gradients always fail. |
| `smoke.require_end_fix_output_gradient_if_untied` | flat_step optional presence/nonzero gradient diagnostic and smoke reporting; nonfinite gradients always fail. |

## Verification

`tests/test_config_plumbing.py` observes runtime consumers, not just parsed dictionaries:
real tiny HF/PEFT fresh and continued adapters, rank/alpha/dropout/target modules,
alternate model IDs, FP32/BF16 joint backward, custom atomic token, 4096 context,
checkpointing, WHERE caching, AdamW/Adam/SGD, three nondefault schedulers, optimizer
hyperparameters after resume, nonstandard K=2/3 and weighted probabilities, alternate
unseen subjects, K=5-only validation, multiple/disabled prediction batches, semantic
budgets, batch size 2 x accumulation 3, step evaluation, null clipping, logging cadence,
selective checkpoint components, relocated split/image roots and duration source field.
Entrypoint tests prove epochs/CLI overrides reach model construction, sampler, optimizer,
loop, evaluation and saved JSON. Checksum and manifest mismatch failures remain tested.

The contract smoke preserves the original goldens. Full configured 8B smoke was attempted
and stopped at CUDA preflight on this CPU-only host; it did not download/train the model.
Real tiny-model checks do not claim an 8B run. Exact final commands/results are recorded
in `IMPLEMENTATION_PLAN_FLAT.md`.
