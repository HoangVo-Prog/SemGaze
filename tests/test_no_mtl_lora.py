"""Regression tests for the standard-LoRA-only architecture; no GPU needed."""
import pytest
from semgaze.model.config import ROOT, load_config, resolve_config
from semgaze.model.checkpoint import architecture_signature, reject_removed_mtl_checkpoint


def test_default_config_has_only_standard_lora():
    cfg = load_config(ROOT / 'configs/flat_single.yaml')
    assert 'mtl_lora' not in cfg['model']
    assert cfg['training']['backward']['one_joint_backward'] is True
    assert 'mtl_lora' not in architecture_signature(cfg)


def test_legacy_disabled_mtl_is_stripped_before_validation():
    cfg = resolve_config({'model': {'mtl_lora': {'enabled': False, 'temperature': 1.0}}})
    assert 'mtl_lora' not in cfg['model']


def test_active_mtl_config_is_never_silently_interpreted_as_standard_lora():
    with pytest.raises(ValueError, match='MTL-LoRA is removed'):
        resolve_config({'model': {'mtl_lora': {'enabled': True, 'temperature': 1.0}}})


def test_standard_lora_uses_joint_backward():
    with pytest.raises(ValueError, match='one_joint_backward=true'):
        resolve_config({'training': {'backward': {'one_joint_backward': False}}})


def test_mtl_checkpoint_rejected_and_standard_signature_unchanged():
    with pytest.raises(ValueError, match='MTL-LoRA checkpoint'):
        reject_removed_mtl_checkpoint({'architecture': {'mtl_lora': {'type': 'full'}}})
    cfg = resolve_config({})
    reject_removed_mtl_checkpoint({'architecture': architecture_signature(cfg)})
