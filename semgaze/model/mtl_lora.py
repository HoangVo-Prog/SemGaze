"""Full two-task MTL-LoRA for existing PEFT LoRA Linear layers.

A and B1 remain ordinary PEFT weights. B2, rank-space task transforms and
mixture logits are saved separately in the SemGaze checkpoint. A task scope must
remain active THROUGH backward when gradient checkpointing is enabled.
"""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from threading import local
from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn

_TASK = ContextVar('intergaze_mtl_lora_task', default=None)
_TASKS = {'where': 0, 'semantic': 1}
_KEYS = ('mtl_extra_b', 'mtl_lambdas', 'mtl_mixture_logits')


@contextmanager
def task_scope(task):
    if task not in _TASKS:
        raise ValueError(f'unsupported MTL-LoRA task: {task!r}')
    token = _TASK.set(task)
    try:
        yield
    finally:
        _TASK.reset(token)


class _RecomputeTaskContext:
    """Reusable checkpoint-recomputation scope for repeated autograd.grad/backward."""
    def __init__(self, task):
        self.task = task
        self._thread = local()

    def __enter__(self):
        tokens = getattr(self._thread, 'tokens', None)
        if tokens is None:
            tokens = []
            self._thread.tokens = tokens
        tokens.append(_TASK.set(self.task))
        return self

    def __exit__(self, *exc):
        _TASK.reset(self._thread.tokens.pop())


def checkpoint_task_context_fn():
    """Snapshot forward task and restore it for each checkpoint recomputation."""
    task = _TASK.get()
    if task is None:
        raise RuntimeError('MTL-LoRA checkpoint created without a selected task')
    return nullcontext(), _RecomputeTaskContext(task)


def enabled(bundle):
    return bundle.config['model'].get('mtl_lora', {}).get('enabled', False)


def _mtl_forward(self, x, *args, **kwargs):
    """Replace only the adapted Linear math; no wrapper around InternVL blocks."""
    if self.merged or self.disable_adapters:
        raise RuntimeError('merging/disabling the PEFT adapter is unsupported by active MTL-LoRA')
    active = self.active_adapters
    if list(active) != ['default']:
        raise RuntimeError(f'MTL-LoRA expects only active PEFT adapter default: {active}')
    task = _TASK.get()
    if task is None:
        raise RuntimeError('MTL-LoRA task not selected; this also catches bad checkpoint recomputation')
    idx = _TASKS[task]
    result = self.base_layer(x, *args, **kwargs)
    A, B1 = self.lora_A['default'], self.lora_B['default']
    x_low = self.lora_dropout['default'](x.to(A.weight.dtype))
    projected = A(x_low)
    projected = F.linear(projected, self.mtl_lambdas[idx])
    # Explicit fp32 softmax for stable BF16 mixture gradients.
    weights = torch.softmax(self.mtl_mixture_logits[idx].float() / self.mtl_temperature, dim=0)
    mix = weights[0].to(B1.weight.dtype) * B1.weight + weights[1].to(B1.weight.dtype) * self.mtl_extra_b
    adapted = F.linear(projected.to(mix.dtype), mix)
    return result + adapted.to(result.dtype) * self.scaling['default']


def install_full_mtl_lora(model, config):
    """Install on PEFT's existing LoRA Linear instances without renaming A/B.

    B2 is initially cloned from B1, preserving a trained WHERE-only checkpoint
    *exactly* for both tasks. Distinct mixture logits break expert symmetry.
    """
    if not config.get('enabled', False):
        return 0
    temperature = float(config['temperature'])
    if not 0 < temperature:
        raise ValueError('MTL-LoRA temperature must be positive')
    count = 0
    for name, module in model.named_modules():
        if not (name.startswith('language_model.') or '.language_model.' in name) or not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
            continue
        if not hasattr(module, 'base_layer') or not isinstance(module.base_layer, nn.Linear):
            raise TypeError(f'MTL-LoRA requires linear PEFT targets: {name}')
        if hasattr(module, 'mtl_lambdas'):
            raise ValueError(f'MTL-LoRA is already installed: {name}')
        if list(module.lora_A.keys()) != ['default'] or list(module.lora_B.keys()) != ['default']:
            raise ValueError(f'MTL-LoRA requires one default LoRA adapter: {name}')
        A, B = module.lora_A['default'], module.lora_B['default']
        rank = A.out_features
        if B.in_features != rank:
            raise ValueError(f'LoRA dimensions do not match: {name}')
        module.register_parameter('mtl_extra_b', nn.Parameter(B.weight.detach().clone()))
        eye = torch.eye(rank, device=A.weight.device, dtype=A.weight.dtype)
        module.register_parameter('mtl_lambdas', nn.Parameter(eye.unsqueeze(0).repeat(2, 1, 1)))
        # Nonidentical routing initially, but identical effective B1/B2 weights.
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float32,
                              device=B.weight.device)
        module.register_parameter('mtl_mixture_logits', nn.Parameter(logits))
        module.mtl_temperature = temperature
        module.forward = MethodType(_mtl_forward, module)
        count += 1
    if count == 0:
        raise ValueError('MTL-LoRA found no language_model PEFT LoRA Linear modules')
    return count


def mtl_parameter_tensors(model):
    """Stable safetensors mapping independent of PEFT adapter serializer."""
    return {'mtl.' + name: param for name, param in model.named_parameters()
            if name.rsplit('.', 1)[-1] in _KEYS}


def load_mtl_parameters(model, tensors):
    expected = mtl_parameter_tensors(model)
    if set(expected) != set(tensors):
        raise ValueError('MTL-LoRA checkpoint tensors do not match model architecture')
    with torch.no_grad():
        for key, parameter in expected.items():
            tensor = tensors[key]
            if parameter.shape != tensor.shape:
                raise ValueError(f'MTL-LoRA tensor shape mismatch: {key}')
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
