from dataclasses import dataclass, field
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import random
import torch
from .config import ROOT, load_config, resolve_config, write_run_config, resolve_output_dir
from .trainable_tokens import install_trainable_rows
from semgaze.data.cocosearch18 import sha256_file
from semgaze.state.projector import build_projector


class ModelPreflightError(RuntimeError):
    pass


def validate_attention_backend(config, diagnostics=None):
    """Validate the explicitly selected text attention backend before loading weights."""
    backend = config['model'].get('attention_backend', 'sdpa')
    if backend not in ('sdpa', 'flash_attention_2'):
        raise ModelPreflightError(f'unsupported model.attention_backend={backend!r}')
    if diagnostics is not None:
        diagnostics['attention_backend'] = backend
        diagnostics['text_attention'] = backend
        diagnostics['fa2_active'] = backend == 'flash_attention_2'
    if backend == 'flash_attention_2':
        if not torch.cuda.is_available():
            raise ModelPreflightError(
                'model.attention_backend=flash_attention_2 requires CUDA; '
                'run on the target CUDA environment or choose sdpa for CPU tests')
        if importlib.util.find_spec('flash_attn') is None:
            raise ModelPreflightError(
                'model.attention_backend=flash_attention_2 requires the flash-attn package; '
                'install a build compatible with this PyTorch/CUDA/Python environment')
        try:
            importlib.import_module('flash_attn')
        except Exception as error:
            raise ModelPreflightError(
                'flash-attn is present but cannot be imported; install a build compatible '
                f'with this PyTorch/CUDA/Python environment ({error})') from error


def set_text_attention_backend(base, backend):
    """Select FA2 only on InternVL's Qwen language model, preserving vision attention."""
    language = getattr(getattr(base, 'model', None), 'language_model', None)
    if language is None:
        raise ModelPreflightError('InternVL language_model module is missing; cannot configure text attention')
    try:
        language.set_attn_implementation(backend)
    except Exception as error:
        raise ModelPreflightError(
            f'failed to configure text attention backend {backend!r}: {error}') from error
    actual = getattr(language.config, '_attn_implementation', None)
    if actual != backend:
        raise ModelPreflightError(f'text attention backend requested {backend!r}, observed {actual!r}')
    return actual


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
    context_limit: int = None
    optimizer: object = None
    scheduler: object = None
    trainer_history: list = field(default_factory=list)

    def __post_init__(self):
        if self.context_limit is None:
            self.context_limit = self.config['where']['context']['max_total_sequence_length']

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad] + list(self.projector.parameters())


def inspect_adapter(path):
    path = Path(path)
    metadata = json.loads((path / 'adapter_config.json').read_text())
    if metadata.get('peft_type') != 'LORA':
        raise ModelPreflightError('this trainer implements LoRA adapters only')
    hashes, blockers = {}, []
    files = ['adapter_config.json', 'adapter_model.safetensors']
    files += [name for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja') if (path / name).exists()]
    for name in files:
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
    if not (Path(adapter_path) / 'tokenizer_config.json').exists():
        return  # Standard PEFT checkpoints need not contain a tokenizer.
    released = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer = processor.tokenizer
    if tokenizer.get_vocab() != released.get_vocab():
        raise ModelPreflightError('base/adapter vocabulary or added-token IDs differ')
    for key in ('eos_token_id', 'pad_token_id'):
        if getattr(tokenizer, key) != getattr(released, key):
            raise ModelPreflightError(f'base/adapter {key} differs')
    if not (Path(adapter_path) / 'chat_template.jinja').exists():
        return
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


def build_flat_model_bundle(config_path=None, *, config=None, output_dir=None):
    config = resolve_config(config) if config is not None else load_config(config_path)
    output_dir = resolve_output_dir(config, output_dir)
    write_run_config(config, output_dir)
    source = config['model']['initialization_adapter']
    adapter = ROOT / source if source is not None else None
    diagnostics = inspect_adapter(adapter) if adapter else {'blockers': []}
    diagnostics.update(adapter_source=adapter.as_posix() if adapter else None, torch_version=torch.__version__,
                       cuda_available=torch.cuda.is_available())
    validate_attention_backend(config, diagnostics)
    for package in ('transformers', 'peft', 'accelerate', 'safetensors'):
        try:
            diagnostics[package + '_version'] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            diagnostics['blockers'].append(f'missing dependency: {package}')
    if torch.device(config['runtime']['device']).type == 'cuda' and not torch.cuda.is_available():
        diagnostics['blockers'].append('CUDA unavailable; configured runtime.device requires a prepared CUDA device')
    (output_dir / 'preflight.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
    if diagnostics['blockers']:
        raise ModelPreflightError('; '.join(diagnostics['blockers']) + f'; diagnostics: {output_dir}')
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel, LoraConfig, get_peft_model
    seed = config['experiment']['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    processor = AutoProcessor.from_pretrained(config['model']['base_model'])
    if adapter is not None:
        verify_tokenizer_compatibility(processor, adapter)
    precision = config['training']['precision']
    dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32}[precision]
    if dtype == torch.bfloat16 and torch.device(config['runtime']['device']).type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise ModelPreflightError('bf16 unsupported; explicitly choose supported precision')
    base = AutoModelForImageTextToText.from_pretrained(config['model']['base_model'], dtype=dtype)
    if base.__class__.__name__ != 'InternVLForConditionalGeneration':
        raise ModelPreflightError('expected HF InternVLForConditionalGeneration')
    set_text_attention_backend(base, config['model'].get('attention_backend', 'sdpa'))
    diagnostics['vision_attention'] = getattr(getattr(base.config, 'vision_config', None), '_attn_implementation', None)
    tokenizer = processor.tokenizer
    token = config['where']['end_fix_token']
    tokenizer.add_tokens([token], special_tokens=True)
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ModelPreflightError('END_FIX is not atomic')
    base.resize_token_embeddings(len(tokenizer))
    if adapter is not None:
        model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
        verify_loaded_adapter(model, adapter)
        # Continued weights own their tensor architecture; record effective metadata,
        # without replacing the user's fresh-initialization LoRA settings.
        diagnostics['effective_lora'] = diagnostics['adapter_metadata']
    else:
        lora = config['model']['lora']
        targets = [name for name, module in base.named_modules()
                   if '.language_model.' in name and any(name == target or name.endswith('.' + target)
                       for target in lora['target_modules'])]
        missing = [target for target in lora['target_modules'] if not any(name == target or name.endswith('.' + target) for name in targets)]
        if missing:
            raise ModelPreflightError(f'configured LoRA targets match no language-model modules: {missing}')
        model = get_peft_model(base, LoraConfig(r=lora['rank'], lora_alpha=lora['alpha'],
            lora_dropout=lora['dropout'], target_modules=targets, task_type='CAUSAL_LM'))
        diagnostics['effective_lora'] = lora

    model.to(device=config['runtime']['device'])
    input_row, output_row, tied = install_trainable_rows(model, token_ids[0])
    projector = build_projector(base.config.text_config.hidden_size).to(device=input_row.device, dtype=input_row.dtype)
    if config['training']['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    diagnostics.update(assert_trainable_set(model, projector, input_row, output_row))
    diagnostics.update(adapter_trainable=True, end_fix_token_count=1, end_fix_token_id=token_ids[0],
                       embeddings_tied=tied, base_commit=getattr(base.config, '_commit_hash', None))
    (output_dir / 'preflight.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
    bundle = FlatModelBundle(model, processor, projector, token_ids[0], input_row, output_row, tied,
                             config, diagnostics, output_dir)
    from semgaze.training.profiling import runtime_metadata
    diagnostics.update(runtime_metadata(bundle))
    (output_dir / 'preflight.json').write_text(json.dumps(diagnostics, indent=2), encoding='utf-8')
    return bundle
