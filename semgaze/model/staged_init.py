"""Weight-only stage transitions: checkpoint WHERE-only -> a fresh training run.

The checkpoint is loaded and validated using its ORIGINAL architecture first.
Only after that do we switch trainable rows/projector and install the target
training config. Strict optimizer/scheduler/sampler resume is a separate path.
"""

from copy import deepcopy
import json
from pathlib import Path

from semgaze.model.build import assert_trainable_set
from semgaze.model.checkpoint import architecture_signature
from semgaze.state.projector import build_projector


def validate_staged_config(source, target):
    """Reject incompatible base weights/tokenizer before loading the 8B model.

    END_FIX activation and explicit/shared semantic conditioning are intentionally
    *not* hard architecture locks for WEIGHT-ONLY initialization.
    """
    for group, key in (
        ('experiment', 'protocol_version'),
        ('experiment', 'semantic_mode'),
        ('model', 'base_model'),
        ('model', 'backend'),
        ('model', 'lora'),
        ('model', 'freeze'),
        ('where', 'mode'),
        ('where', 'end_fix_token'),
        ('where', 'coordinates'),
        ('where', 'duration'),
    ):
        if source[group][key] != target[group][key]:
            raise ValueError(
                f'weight-only stage transition requires matching {group}.{key}; '
                'a checkpoint cannot initialize a different backbone/LoRA/vocabulary/WHERE codec'
            )
    if target['semantic']['conditioning'] == 'shared_states' and not target['checkpoint']['save_projector']:
        raise ValueError('shared_states needs checkpoint.save_projector=true for loadable new checkpoints')
    # The current loader requires these artifacts even when END_FIX is disabled.
    for key in ('save_trainable_end_fix_rows', 'save_end_fix_token_id',
                'save_peft_adapter', 'save_processor_and_tokenizer', 'save_resolved_config'):
        if not target['checkpoint'][key]:
            raise ValueError(f'weight-only stage transition requires checkpoint.{key}=true')


def adapt_checkpoint_for_new_run(bundle, target, *, source_checkpoint):
    """Preserve restored backbone/LoRA/END_FIX weights; adapt trainable graph.

    Must run ONLY after load_checkpoint_bundle has restored source weights AND
    before the new optimizer/scheduler are constructed. The caller seeds RNG
    for reproducible initialization of a newly required shared-state projector.
    """
    source = bundle.config
    validate_staged_config(source, target)
    old_arch = architecture_signature(source)
    new_arch = architecture_signature(target)

    old_shared = source['semantic']['conditioning'] == 'shared_states'
    new_shared = target['semantic']['conditioning'] == 'shared_states'
    # A WHERE-only checkpoint has never trained a semantic projector, even if
    # its architecture happened to contain one. Do not treat it as pretrained.
    projector_reinitialized = new_shared and (not old_shared or source['training']['lambda_sem'] == 0)
    projector_removed = old_shared and not new_shared
    if projector_reinitialized:
        hidden_size = bundle.model.get_base_model().config.text_config.hidden_size
        bundle.projector = build_projector(hidden_size, enabled=True).to(
            device=bundle.input_row.device, dtype=bundle.input_row.dtype)
    elif projector_removed:
        bundle.projector = build_projector(0, enabled=False)

    # Both tied and untied END_FIX rows already exist in the source checkpoint.
    # Its value is preserved; only whether it participates in optimization changes.
    use_end = target['where']['use_end_fix_token']
    bundle.input_row.requires_grad_(use_end)
    bundle.output_row.requires_grad_(use_end)

    # LoRA tensor geometry/provenance belong to the checkpoint (not a new adapter
    # initialization in the target YAML). Other training/evaluation choices belong
    # to the new run, including independent epochs, loss weights and scheduler.
    target['model'] = deepcopy(source['model'])
    bundle.config = target
    bundle.context_limit = target['where']['context']['max_total_sequence_length']
    bundle.trainer_history = []
    bundle.model.zero_grad(set_to_none=True)
    bundle.projector.zero_grad(set_to_none=True)

    diagnostics = assert_trainable_set(bundle.model, bundle.projector,
        bundle.input_row, bundle.output_row, require_end_fix=use_end)
    metadata = json.loads((Path(source_checkpoint) / 'metadata.json').read_text(encoding='utf-8'))
    bundle.diagnostics = dict(bundle.diagnostics)
    bundle.diagnostics.update(diagnostics)
    bundle.diagnostics.update(
        end_fix_token_count=int(use_end),
        inactive_frozen_end_fix_rows=not use_end,
        optimizer_projector_active=target['training']['lambda_sem'] > 0,
        optimizer_trainable_parameter_count=sum(p.numel() for p in bundle.trainable_parameters()),
        trainable_parameters=sum(p.numel() for p in bundle.trainable_parameters()),
        staged_initialization={
            'source_checkpoint': str(Path(source_checkpoint).resolve()),
            'source_step': metadata['step'],
            'source_architecture': old_arch,
            'target_architecture': new_arch,
            'source_objective': {
                'lambda_where': source['training']['lambda_where'],
                'lambda_sem': source['training']['lambda_sem'],
            },
            'target_objective': {
                'lambda_where': target['training']['lambda_where'],
                'lambda_sem': target['training']['lambda_sem'],
            },
            'projector_reinitialized': projector_reinitialized,
            'projector_removed': projector_removed,
            'optimizer_scheduler_sampler_reset': True,
        },
    )
    return bundle
