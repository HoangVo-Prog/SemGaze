"""CPU-only tests for the task-routed full MTL-LoRA adapter core."""
import copy

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from safetensors.torch import save_file, load_file

from semgaze.model.mtl_lora import (
    task_scope, install_full_mtl_lora, mtl_parameter_tensors, load_mtl_parameters,
)


class FakePeftLinear(nn.Module):
    def __init__(self, dim=12, rank=4, dropout=0.0):
        super().__init__()
        self.base_layer = nn.Linear(dim, dim, bias=False)
        self.base_layer.weight.requires_grad_(False)
        self.lora_A = nn.ModuleDict({'default': nn.Linear(dim, rank, bias=False)})
        self.lora_B = nn.ModuleDict({'default': nn.Linear(rank, dim, bias=False)})
        self.lora_dropout = nn.ModuleDict({'default': nn.Dropout(dropout)})
        self.scaling = {'default': 2.0}
        self.merged = False
        self.disable_adapters = False
        self.active_adapters = ['default']

    def forward(self, x):
        return self.base_layer(x) + self.lora_B['default'](
            self.lora_A['default'](self.lora_dropout['default'](x))) * self.scaling['default']


class ToyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([FakePeftLinear()])

    def forward(self, x, checkpointing=False):
        layer = self.language_model.layers[0]
        if checkpointing:
            return checkpoint(lambda z: layer(z), x, use_reentrant=False)
        return layer(x)


def installed(model=None):
    model = ToyNetwork() if model is None else model
    assert install_full_mtl_lora(model, {'enabled': True, 'temperature': 1.0}) == 1
    return model


def test_initialization_matches_pretrained_lora():
    torch.manual_seed(10)
    model = ToyNetwork()
    x = torch.randn(2, 3, 12)
    before = model(x).detach().clone()
    installed(model)
    with task_scope('where'):
        assert torch.allclose(model(x), before, atol=1e-6, rtol=1e-6)
    with task_scope('semantic'):
        assert torch.allclose(model(x), before, atol=1e-6, rtol=1e-6)


def test_task_mixing_and_isolation():
    torch.manual_seed(11)
    model = installed()
    layer = model.language_model.layers[0]
    with torch.no_grad():
        layer.mtl_extra_b.add_(1.0)
        layer.mtl_lambdas[1].mul_(1.5)
    x = torch.randn(2, 3, 12, requires_grad=True)
    with task_scope('where'):
        w = model(x)
        w.square().mean().backward()
    assert layer.mtl_lambdas.grad[0].abs().sum() > 0
    assert torch.count_nonzero(layer.mtl_lambdas.grad[1]) == 0
    assert layer.mtl_mixture_logits.grad[0].abs().sum() > 0
    assert torch.count_nonzero(layer.mtl_mixture_logits.grad[1]) == 0
    assert layer.lora_A['default'].weight.grad is not None
    model.zero_grad(set_to_none=True)
    with task_scope('semantic'):
        s = model(x)
        s.square().mean().backward()
    assert not torch.allclose(w, s)
    assert layer.mtl_lambdas.grad[1].abs().sum() > 0
    assert torch.count_nonzero(layer.mtl_lambdas.grad[0]) == 0


def test_requires_explicit_routing():
    model = installed()
    with pytest.raises(RuntimeError, match='task not selected'):
        model(torch.randn(1, 2, 12))


def test_roundtrip_and_corrupt_state(tmp_path):
    model = installed()
    tensors = {k: v.detach().clone().contiguous() for k, v in mtl_parameter_tensors(model).items()}
    assert len(tensors) == 3
    target = tmp_path / 'mtl.safetensors'
    save_file(tensors, str(target))
    clone = installed(copy.deepcopy(ToyNetwork()))
    load_mtl_parameters(clone, load_file(str(target)))
    for key, param in mtl_parameter_tensors(clone).items():
        assert torch.equal(param, tensors[key])
    with pytest.raises(ValueError, match='do not match'):
        load_mtl_parameters(clone, {})


def _grads(use_checkpoint):
    torch.manual_seed(23)
    model = installed()
    layer = model.language_model.layers[0]
    with torch.no_grad():
        layer.mtl_extra_b.add_(0.13)
    x = torch.randn(2, 3, 12, requires_grad=True)
    # Forward AND backward occur under the same task, matching sequential
    # task-specific backward in the InterGaze training patch.
    for name in ('where', 'semantic'):
        with task_scope(name):
            loss = model(x, checkpointing=use_checkpoint).square().mean()
            loss.backward()
    return {key: p.grad.detach().clone() for key, p in model.named_parameters()
            if p.requires_grad}


def test_checkpointing_parity_for_both_tasks():
    control = _grads(False)
    candidate = _grads(True)
    for name, ref in control.items():
        assert torch.allclose(candidate[name], ref, atol=1e-6, rtol=1e-5), name


def test_recomputation_without_task_context_fails_closed():
    model = installed()
    x = torch.randn(2, 3, 12, requires_grad=True)
    with task_scope('where'):
        loss = model(x, checkpointing=True).square().mean()
    # The chosen design prevents this failure by keeping task_scope active
    # around autograd.backward; a lost task must raise, not silently route SEM.
    with pytest.raises(RuntimeError, match='task not selected'):
        loss.backward()
