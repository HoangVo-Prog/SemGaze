> Historical implementation/performance report. Split, sampling, schedule and evaluation examples below predate the 95/5 query-coverage migration. Use [COCO_955_MIGRATION.md](COCO_955_MIGRATION.md) and the current YAML/CLI for new runs. Historical measurements are not evidence for current sampling throughput.

# Section A: source audit (before refactoring, 2026-10-05)

## Call graph and evidence

`train_flat.main` builds/resumes one shared InternVL/PEFT model, runs fixture
smoke, constructs optimizer/scheduler, then calls `training.loop.run_training_loop`.
`TrainingEpisodeSampler.sample` samples subject -> K -> query -> distinct support
stimuli -> records -> shuffled support order with its own checkpointed Python RNG.
The loop calls `run_flat_training_step` B * accumulation times, each with loss
scale 1/(B * accumulation), then clips, steps optimizer and scheduler. Thus B was
sequential accumulation, not a GPU batch. Episode membership and sampler RNG must
remain unchanged at each optimizer boundary.

| Stage / original symbol | Shapes and retained work | Blocking / restriction |
|---|---|---|
| `where.collator.collate_native/where` | IDs, mask, labels [1,L]; pixels [K+1,3,448,448]; no gradients | PIL decode, native processor, decode/encode span verification, offset tokenization on CPU; scalar spans, `encoded['input_ids'][0]` |
| `where.forward.forward_where` | full model [1,L,D], logits [1,L,V], all LM hidden outputs | `output_hidden_states=True`; selects `states[0]`; GPU `mask.all()` |
| HF `InternVLForConditionalGeneration.forward` | `model` -> `lm_head` over every position | full vocabulary even for ignored support/image/user tokens |
| `model.trainable_tokens.RowHead.forward` | frozen base head then `cat` replacing one column | second [1,L,V] allocation; trainable replacement/input rows preserved |
| HF `ForCausalLMLoss` | logits.float(), shifted labels, mean CE | FP32 [1,L,V]; only query assistant target + EOS supervised |
| `state.extractor.extract_query_states` | query-only final hidden [B,Nmax,D] plus mask | already handles variable N! `where` wrapper discards batch; GPU nonzero indexing |
| `state.projector.build_projector` | Linear + LayerNorm [N,D] -> [N,D] | differentiable; no inherent batch restriction |
| `semantic.flat.forward.prepare_flat_inputs` | single query pixels [1,3,448,448], prompt+target IDs [1,S] | query decoded/preprocessed a second time; END_FIX `.any()` GPU check |
| `state.insertion.insert_states` | concatenate R into embeddings [1,S+N,D] | explicit B=1 restriction; GPU scalar label checks per boundary |
| `semantic.flat.forward.forward_flat` | same InternVL -> logits [1,S+N,V] -> mean CE | query vision recomputed; semantic graph retains R -> P_E -> WHERE |
| `training.flat_step.run_flat_training_step` | weighted joint scalar -> backward | 3 finite-loss syncs; all-parameter finite scan, then repeated LoRA/projector/row scans every episode |
| `training.loop` | detached losses -> Python floats; clip -> optimizer -> scheduler | per-episode host syncs, no explicit synchronized baseline timing |

The old objective is mean over supervised shifted tokens **within each episode**,
then mean over episodes across each optimizer minibatch. Native HF batched mean
would instead weight long responses more heavily. Neither branch needs logits
after training loss; evaluation diagnostic slices still do (outside scope).

## Allocations and checkpointing

Installed Transformers 5.8.0 source was inspected: `InternVLModel.forward`
returns `last_hidden_state` from the Qwen backbone without a vocabulary head.
Qwen3 uses `capture_outputs` to collect each decoder output when requested;
its final normalized output is available without this collection. The vision
feature-layer policy can independently require vision hidden outputs and must
not be altered. `build_flat_model_bundle` enables non-reentrant checkpointing
(`use_reentrant=False`) on supported Qwen3DecoderLayer / InternVLVisionLayer
modules. Keeping layer outputs alive defeats some checkpoint memory savings;
semantic backprop must still recompute WHERE blocks. This needs regression tests.

One bf16 logits tensor costs B*L*V*2 bytes, RowHead cat another equal allocation,
and native loss upcast B*L*V*4 bytes (plus CE/autograd workspace). The released
vocab.json + added_tokens.json contain 151,679 tokens; adding END_FIX yields
V=151,680. At B=1,L=4096: 1.157 GiB + 1.157 GiB + 2.314 GiB; L=8192 doubles these,
and B=2 doubles again. These are allocation estimates, not measured peak VRAM;
tensor lifetimes overlap only partly. Runtime profiler will record actual V.
All-layer output storage additionally scales as B*L*D*(layers+1)*dtype_size.
LoRA optimizer states scale with trainable parameters only; no evidence supports
changing optimizer format.

## Frozen computation and attention

`build.assert_trainable_set` verifies frozen vision tower and native projector;
only language LoRA, END_FIX row(s), and P_E are trainable. WHERE encodes K+1 images,
semantic encodes the query again. Supports are recomputed every sampled episode.
Within-step query feature reuse is safe only with identical preprocessing AND a
deterministic frozen producer (frozen parameters alone do not disable dropout).
No language-state caching or detachment is allowed. Production text/vision
attention backends are not explicitly selected by the loader: record actual
module config values rather than assuming flash attention.

## Discrepancies / environment

The local user-modified `flat_single.yaml` uses fresh LoRA (`initialization_adapter:
null`), whereas the requested invariant requires released DeepGaze initialization.
Keep that user edit; provide a separate continued-adapter benchmark/run config.
Example checkpointing is on, but missing-field defaults were off. The extractor
already supports B>1. `max_grad_norm: null` relies on the expensive finite scans
for failure behavior, so aggregate checking must also cover this case.

Both available Python runtimes report PyTorch 2.11.0+cpu and CUDA unavailable;
`nvidia-smi` is absent. `.venv/Scripts/python.exe` has the real HF/PEFT test stack.
A100 40GB acceptance cannot be measured here; the 133-byte adapter payload and
tokenizer.json are Git LFS pointers, confirmed by the continued-adapter preflight.
CPU tiny-model timings are only
engineering evidence, never an 8B throughput or VRAM claim.

## Patch gate: order and risks

Audit completed before behavioral edits. First add an opt-in synchronized
profiler and record the original baseline. Then: shared selected-position full-V
CE with episode reduction; direct final backbone output; sparse gradient scans;
batched metadata/collation/extraction/insertion and one backbone call per branch;
conventional accumulation; cost grouping only inside an optimizer minibatch;
guarded frozen query feature reuse. Keep checkpointing, precision, loss weights,
serialization, all masks and trainable parameter names. Record backend/runtime
metadata; defer backend/checkpoint/allocator tuning until target GPU evidence.
Risks: off-by-one shifted labels, pooled token loss, image flattening across rows,
padding readout/insertion, PEFT bypass, stochastic frozen vision, checkpoint
recomputation, and changed dropout RNG assignment under vectorization. Tests
must isolate deterministic numerical parity from stochastic training execution.

## Additional native implementation constraint

HF InternVL `get_placeholder_mask` also uses GPU boolean indexing and formats
`special_image_mask.sum()` in its mismatch message. That native check remains
in WHERE; the custom per-parameter diagnostic scans were the targeted sync
removal. Reused semantic features use host-validated explicit image masks.
This is a source observation, not a measured GPU bottleneck on this host.
