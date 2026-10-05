"""Complete flat trainable state plus provenance and resumable optimizer/RNG."""
import json
from pathlib import Path
import random
import torch
from safetensors.torch import save_file, load_file
from .config import write_run_config
from .build import assert_trainable_set


def save_checkpoint(bundle, path, *, split_manifest_identity, step, sampler=None):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'checkpoint destination is not empty: {path}')
    path.mkdir(parents=True, exist_ok=True)
    if not split_manifest_identity:
        raise ValueError('split/manifest identity is required')
    options = bundle.config['checkpoint']
    if options['save_processor_and_tokenizer']:
        bundle.processor.save_pretrained(path / 'processor')
    if options['save_peft_adapter']:
        bundle.model.save_pretrained(path / 'adapter', safe_serialization=True, save_embedding_layers=False)
    tensors = {}
    if options['save_trainable_end_fix_rows']:
        tensors['end_fix_input'] = bundle.input_row.detach().cpu().clone()
        if not bundle.embeddings_tied:
            tensors['end_fix_output'] = bundle.output_row.detach().cpu().clone()
    if options['save_projector']:
        tensors.update({f'projector.{k}': v.detach().cpu().clone() for k, v in bundle.projector.state_dict().items()})
    if tensors:
        save_file(tensors, str(path / 'semgaze.safetensors'))
    if options['save_resolved_config']:
        write_run_config(bundle.config, path)
    metadata = {'protocol_version': bundle.config['experiment']['protocol_version'],
        'semantic_mode': 'flat_single_output', 'end_fix_id': bundle.end_fix_id if options['save_end_fix_token_id'] else None,
        'embeddings_tied': bundle.embeddings_tied, 'split_manifest_identity': split_manifest_identity if options['save_split_manifest_identity'] else None,
        'step': step, 'diagnostics': bundle.diagnostics,
        'trainable_names': [n for n, p in bundle.model.named_parameters() if p.requires_grad]}
    (path / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (path / 'trainer_state.json').write_text(json.dumps(
        {'step': step, 'log_history': bundle.trainer_history}, indent=2, allow_nan=False), encoding='utf-8')
    state = {'optimizer': bundle.optimizer.state_dict() if bundle.optimizer is not None and options['save_optimizer'] else None,
             'scheduler': bundle.scheduler.state_dict() if bundle.scheduler is not None and options['save_scheduler'] else None,
             'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             'sampler': sampler.state_dict() if sampler is not None else None}
    torch.save(state, path / 'training.pt')


def restore_checkpoint_state(bundle, path, *, split_manifest_identity, sampler=None, resume_optimizer=False, restore_hyperparameters=True):
    """Load into a bundle constructed with checkpoint processor and PEFT adapter."""
    path = Path(path)
    metadata = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
    if metadata['semantic_mode'] != 'flat_single_output' or metadata['protocol_version'] != bundle.config['experiment']['protocol_version']:
        raise ValueError('checkpoint protocol/mode mismatch')
    if metadata['split_manifest_identity'] != split_manifest_identity:
        raise ValueError('checkpoint split/manifest identity mismatch')
    if metadata['end_fix_id'] != bundle.end_fix_id or metadata['embeddings_tied'] != bundle.embeddings_tied:
        raise ValueError('checkpoint END_FIX ID or tied-weight policy mismatch')
    if metadata['trainable_names'] != [n for n, p in bundle.model.named_parameters() if p.requires_grad]:
        raise ValueError('checkpoint trainable parameter set mismatch')
    tensors = load_file(str(path / 'semgaze.safetensors'))
    expected = {'end_fix_input', *(f'projector.{k}' for k in bundle.projector.state_dict())}
    if not bundle.embeddings_tied:
        expected.add('end_fix_output')
    if set(tensors) != expected:
        raise ValueError('checkpoint is missing required row/projector state or contains unexpected tensors')
    with torch.no_grad():
        bundle.input_row.copy_(tensors['end_fix_input'])
        if not bundle.embeddings_tied:
            bundle.output_row.copy_(tensors['end_fix_output'])
    bundle.projector.load_state_dict({k.removeprefix('projector.'): v for k, v in tensors.items() if k.startswith('projector.')})
    assert_trainable_set(bundle.model, bundle.projector, bundle.input_row, bundle.output_row)
    trainer_state = path / 'trainer_state.json'
    if trainer_state.exists():
        history = json.loads(trainer_state.read_text(encoding='utf-8'))
        if history['step'] != metadata['step']:
            raise ValueError('trainer history step differs from checkpoint metadata')
        bundle.trainer_history = history['log_history']
    else:
        bundle.trainer_history = []  # backwards-compatible inference checkpoints
    if resume_optimizer:
        state = torch.load(path / 'training.pt', map_location='cpu', weights_only=True)
        if bundle.optimizer is None or bundle.scheduler is None or state['optimizer'] is None or state['scheduler'] is None:
            raise ValueError('resumable training requires saved and constructed optimizer/scheduler')
        configured_groups = [{k: v for k, v in group.items() if k != 'params'} for group in bundle.optimizer.param_groups]
        bundle.optimizer.load_state_dict(state['optimizer'])
        if restore_hyperparameters:
            bundle.scheduler.load_state_dict(state['scheduler'])
        else:
            for group, configured in zip(bundle.optimizer.param_groups, configured_groups):
                group.update(configured)
                group['lr'] = bundle.config['training']['learning_rate']
                group['initial_lr'] = group['lr']
        random.setstate(state['python_rng'])
        torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng']:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        if sampler is not None:
            if state['sampler'] is None:
                raise ValueError('missing episode sampler state')
            sampler.load_state_dict(state['sampler'])
    return metadata['step']


def load_checkpoint_bundle(path, *, split_manifest_identity, output_dir=None, runtime=None, precision=None):
    """Rebuild base -> checkpoint vocabulary -> continued adapter -> rows/P_E."""
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel
    from .build import FlatModelBundle
    from .config import resolve_config
    from .trainable_tokens import install_trainable_rows
    from semgaze.state.projector import build_projector
    path = Path(path)
    for required in ('resolved_config.json', 'metadata.json', 'processor', 'adapter', 'semgaze.safetensors'):
        if not (path / required).exists():
            raise ValueError(f'checkpoint is incomplete for loading: missing {required}')
    config = resolve_config(json.loads((path / 'resolved_config.json').read_text(encoding='utf-8')))
    if runtime is not None:
        config['runtime'].update(runtime)
    if precision is not None:
        config['training']['precision'] = precision
    config = resolve_config(config)
    metadata = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
    if metadata['split_manifest_identity'] != split_manifest_identity:
        raise ValueError('checkpoint split/manifest identity mismatch')
    processor = AutoProcessor.from_pretrained(path / 'processor')
    token_ids = processor.tokenizer.encode(config['where']['end_fix_token'], add_special_tokens=False)
    if token_ids != [metadata['end_fix_id']]:
        raise ValueError('checkpoint END_FIX is non-atomic or ID changed')
    dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32}[config['training']['precision']]
    base = AutoModelForImageTextToText.from_pretrained(config['model']['base_model'], dtype=dtype,
        revision=metadata['diagnostics'].get('base_commit'))
    if base.__class__.__name__ != 'InternVLForConditionalGeneration':
        raise ValueError('expected HF InternVLForConditionalGeneration')
    base.resize_token_embeddings(len(processor.tokenizer))
    model = PeftModel.from_pretrained(base, path / 'adapter', is_trainable=True).to(config['runtime']['device'])
    input_row, output_row, tied = install_trainable_rows(model, token_ids[0])
    projector = build_projector(base.config.text_config.hidden_size).to(input_row)
    if config['training']['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    bundle = FlatModelBundle(model, processor, projector, token_ids[0], input_row, output_row, tied,
        config, metadata['diagnostics'], Path(output_dir or path))
    restore_checkpoint_state(bundle, path, split_manifest_identity=split_manifest_identity)
    from semgaze.training.profiling import runtime_metadata
    bundle.diagnostics.update(runtime_metadata(bundle))
    if output_dir is not None:
        config['runtime']['output_dir'] = str(Path(output_dir).resolve())
        write_run_config(config, output_dir)
    return bundle
