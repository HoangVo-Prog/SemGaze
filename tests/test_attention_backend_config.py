import pytest
import torch

from semgaze.model.build import ModelPreflightError, validate_attention_backend
from semgaze.model.config import resolve_config


def test_attention_backend_and_prefetch_config_validate():
    config = resolve_config({'model': {'attention_backend': 'sdpa'},
                             'training': {'prefetch': {'enabled': True, 'depth': 2, 'pin_memory': True}}})
    diagnostics = {}
    validate_attention_backend(config, diagnostics)
    assert diagnostics == {'attention_backend': 'sdpa', 'text_attention': 'sdpa', 'fa2_active': False}


def test_attention_backend_rejects_unknown_value():
    with pytest.raises(ValueError, match='attention_backend'):
        resolve_config({'model': {'attention_backend': 'eager'}})


def test_fa2_fails_early_without_runtime_support():
    config = resolve_config({'model': {'attention_backend': 'flash_attention_2'}})
    if torch.cuda.is_available():
        pytest.skip('local CUDA runtime may provide FA2')
    with pytest.raises(ModelPreflightError, match='requires CUDA'):
        validate_attention_backend(config)
