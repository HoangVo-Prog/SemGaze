from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import random
import uuid
import torch
from .config import ROOT, load_config, resolve_config, write_run_config
from .trainable_tokens import install_trainable_rows
from semgaze.data.cocosearch18 import sha256_file
from semgaze.state.projector import build_projector


class ModelPreflightError(RuntimeError):
    pass


@dataclass
class FlatModelBundle:
    model: object
    processor: object
    projector: object
    end_fix_id: int
    input_row: object
    output_row: object
    embeddings_tied: bool
    config: dict
    diagnostics: dict
    output_dir: Path
    context_limit: int = 8192
    optimizer: object = None
    scheduler: object = None
    trainer_history: list = field(default_factory=list)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad] + list(self.projector.parameters())


def inspect_adapter(path):
    path = Path(path)
    metadata = json.loads((path / 'adapter_config.json').read_text())
    targets = {s.split('.')[-1] for s in metadata['target_modules']}
    expected = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
    if (metadata['base_model_name_or_path'] != 'OpenGVLab/InternVL3_5-8B-HF'
            or (metadata['r'], metadata['lora_alpha'], metadata['lora_dropout']) != (8, 16, 0.05)
            or targets != expected or metadata['peft_type'] != 'LORA'):
        raise ModelPreflightError('released adapter architecture does not match frozen initialization')
    hashes, blockers = {}, []
    for name in ('adapter_config.json', 'adapter_model.safetensors', 'tokenizer.json'):
        file = path / name
        if not file.exists():
            blockers.append(f'missing {file}')
            continue
        hashes[name] = sha256_file(file)
        with file.open('rb') as stream:
            if stream.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
                blockers.append(f'{file} is a Git LFS pointer, not the released payload')
    return {'adapter_metadata': metadata, 'adapter_file_sha256': hashes, 'blockers': blockers}


def verify_tokenizer_compatibility(processor, adapter_path):
    from transformers import AutoTokenizer
    released = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer = processor.tokenizer
    if tokenizer.get_vocab() != released.get_vocab():
        raise ModelPreflightError('base/adapter vocabulary or added-token IDs differ')
    for key in ('eos_token_id', 'pad_token_id'):
        if getattr(tokenizer, key) != getattr(released, key):
            raise ModelPreflightError(f'base/adapter {key} differs')
    template = (Path(adapter_path) / 'chat_template.jinja').read_text(encoding='utf-8')
    # Compare actual role/image rendering, allowing harmless template-source whitespace.
    from jinja2 import Environment
    cases = [[{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'query'}]}],
             [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'support'}]},
              {'role': 'assistant', 'content': '[(01, 02, 003)]'},
              {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'query'}]}]]
    for messages in cases:
        for generation in (False, True):
            expected = Environment(trim_blocks=True, lstrip_blocks=True).from_string(template).render(
                messages=messages, add_generation_prompt=generation)
            actual = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation)
            if actual != expected:
                raise ModelPreflightError('base/adapter native chat role/image rendering differs')


def verify_loaded_adapter(model, adapter_path):
    """Fail on missing/unexpected tensors, not only a nonzero trainable count."""
    from safetensors import safe_open
    actual = {name.replace('.default.', '.'): parameter for name, parameter in model.named_parameters()
              if 'lora_' in name}
    with safe_open(str(Path(adapter_path) / 'adapter_model.safetensors'), framework='pt', device='cpu') as source:
        if set(source.keys()) != set(actual):
            raise ModelPreflightError('loaded LoRA tensor keys differ from released checkpoint')
        for key, parameter in actual.items():
            expected = source.get_tensor(key).to(dtype=parameter.dtype)
            if parameter.shape != expected.shape or not torch.equal(parameter.detach().cpu(), expected):
                raise ModelPreflightError(f'released adapter tensor not loaded exactly: {key}')


def assert_trainable_set(model, projector, input_row, output_row):
    rows = {id(input_row), id(output_row)}
    lora = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if id(parameter) in rows:
                continue
            if 'lora_' not in name or '.language_model.' not in name:
                raise ModelPreflightError(f'unexpected trainable parameter: {name}')
            lora.append(parameter)
    if not lora or not input_row.requires_grad or not output_row.requires_grad:
        raise ModelPreflightError('required released LoRA/END_FIX parameters are frozen or absent')
    if not all(p.requires_grad for p in projector.parameters()):
        raise ModelPreflightError('P_E is not fully trainable')
    base = model.get_base_model()
    vision_frozen = not any(p.requires_grad for p in base.model.vision_tower.parameters())
    native_frozen = not any(p.requires_grad for p in base.model.multi_modal_projector.parameters())
    if not vision_frozen or not native_frozen:
        raise ModelPreflightError('native vision/projector must be frozen')
    return {'lora_trainable_parameter_count': sum(p.numel() for p in lora),
            'projector_trainable_parameter_count': sum(p.numel() for p in projector.parameters()),
            'vision_tower_frozen': vision_frozen, 'native_multimodal_projector_frozen': native_frozen}


def build_flat_model_bundle(config_path, *, output_dir=None):
    config = resolve_config(load_config(config_path))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output_dir = Path(output_dir or ROOT / 'runs' / f'flat-{stamp}-{uuid.uuid4().hex[:6]}')
    write_run_config(config, output_dir)
    adapter = ROOT / config['model']['initialization_adapter']
    diagnostics = inspect_adapter(adapter)
    diagnostics.update(adapter_source=adapter.as_posix(), torch_version=torch.__version__,
                       cuda_available=torch.cuda.is_available())
    for package in ('transformers', 'peft', 'accelerate', 'safetensors'):
        try:
            diagnostics[package + '_version'] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            diagnostics['blockers'].append(f'missing dependency: {package}')
    if config['runtime']['device'] == 'cuda' and not torch.cuda.is_available():
        diagnostics['blockers'].append('CUDA unavailable; released 8B gradient smoke needs a prepared model device')
    (output_dir / 'preflight.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
    if diagnostics['blockers']:
        raise ModelPreflightError('; '.join(diagnostics['blockers']) + f'; diagnostics: {output_dir}')
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel
    seed = config['experiment']['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    processor = AutoProcessor.from_pretrained(config['model']['base_model'])
    verify_tokenizer_compatibility(processor, adapter)
    precision = config['training']['precision']
    dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32}[precision]
    if dtype == torch.bfloat16 and config['runtime']['device'] == 'cuda' and not torch.cuda.is_bf16_supported():
        raise ModelPreflightError('bf16 unsupported; explicitly choose supported precision')
    base = AutoModelForImageTextToText.from_pretrained(config['model']['base_model'], dtype=dtype)
    if base.__class__.__name__ != 'InternVLForConditionalGeneration':
        raise ModelPreflightError('expected HF InternVLForConditionalGeneration')
    tokenizer = processor.tokenizer
    tokenizer.add_tokens(['<END_FIX>'], special_tokens=True)
    token_ids = tokenizer.encode('<END_FIX>', add_special_tokens=False)
    if len(token_ids) != 1:
        raise ModelPreflightError('END_FIX is not atomic')
    base.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    verify_loaded_adapter(model, adapter)
    model.to(device=config['runtime']['device'])
    input_row, output_row, tied = install_trainable_rows(model, token_ids[0])
    projector = build_projector(base.config.text_config.hidden_size).to(device=input_row.device, dtype=input_row.dtype)
    if config['training']['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    diagnostics.update(assert_trainable_set(model, projector, input_row, output_row))
    diagnostics.update(adapter_trainable=True, end_fix_token_count=1, end_fix_token_id=token_ids[0],
                       embeddings_tied=tied, base_commit=getattr(base.config, '_commit_hash', None))
    (output_dir / 'preflight.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
    return FlatModelBundle(model, processor, projector, token_ids[0], input_row, output_row, tied,
                           config, diagnostics, output_dir)
