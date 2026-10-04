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
    bundle.processor.save_pretrained(path / 'processor')
    bundle.model.save_pretrained(path / 'adapter', safe_serialization=True, save_embedding_layers=False)
    tensors = {'end_fix_input': bundle.input_row.detach().cpu().clone(),
               **{f'projector.{k}': v.detach().cpu().clone() for k, v in bundle.projector.state_dict().items()}}
    if not bundle.embeddings_tied:
        tensors['end_fix_output'] = bundle.output_row.detach().cpu().clone()
    save_file(tensors, str(path / 'semgaze.safetensors'))
    write_run_config(bundle.config, path)
    metadata = {'protocol_version': bundle.config['experiment']['protocol_version'],
        'semantic_mode': 'flat_single_output', 'end_fix_id': bundle.end_fix_id,
        'embeddings_tied': bundle.embeddings_tied, 'split_manifest_identity': split_manifest_identity,
        'step': step, 'diagnostics': bundle.diagnostics,
        'trainable_names': [n for n, p in bundle.model.named_parameters() if p.requires_grad]}
    (path / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    state = {'optimizer': bundle.optimizer.state_dict() if bundle.optimizer is not None else None,
             'scheduler': bundle.scheduler.state_dict() if bundle.scheduler is not None else None,
             'python_rng': random.getstate(), 'torch_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             'sampler': sampler.state_dict() if sampler is not None else None}
    torch.save(state, path / 'training.pt')


def restore_checkpoint_state(bundle, path, *, split_manifest_identity, sampler=None, resume_optimizer=False):
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
    if resume_optimizer:
        state = torch.load(path / 'training.pt', map_location='cpu', weights_only=True)
        if bundle.optimizer is None or bundle.scheduler is None or state['optimizer'] is None or state['scheduler'] is None:
            raise ValueError('resumable training requires saved and constructed optimizer/scheduler')
        bundle.optimizer.load_state_dict(state['optimizer'])
        bundle.scheduler.load_state_dict(state['scheduler'])
        random.setstate(state['python_rng'])
        torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng']:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        if sampler is not None:
            if state['sampler'] is None:
                raise ValueError('missing episode sampler state')
            sampler.load_state_dict(state['sampler'])
    return metadata['step']


def load_checkpoint_bundle(path, *, split_manifest_identity, output_dir=None):
    """Rebuild base -> checkpoint vocabulary -> continued adapter -> rows/P_E."""
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel
    from .build import FlatModelBundle
    from .config import validate_config
    from .trainable_tokens import install_trainable_rows
    from semgaze.state.projector import build_projector
    path = Path(path)
    config = json.loads((path / 'resolved_config.json').read_text(encoding='utf-8'))
    validate_config(config)
    metadata = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
    if metadata['split_manifest_identity'] != split_manifest_identity:
        raise ValueError('checkpoint split/manifest identity mismatch')
    processor = AutoProcessor.from_pretrained(path / 'processor')
    token_ids = processor.tokenizer.encode('<END_FIX>', add_special_tokens=False)
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
    if output_dir is not None:
        write_run_config(config, output_dir)
    return bundle
