"""Read-only safety checks for proposed training visual feature reuse.

NOT a feature cache. Static checks are necessary but not sufficient.
No caching is enabled by this module.
"""
from torch import nn


def inspect_frozen_training_producers(vision_tower, native_projector):
    """Fail closed on trainable/stochastic/mutable sources detectable statically.

    Dynamic evidence still required: repeated-forward equality, RNG state,
    frozen buffer stability, batch-independence, and all-trainable gradient parity.
    """
    findings = []
    for name, producer in (('vision_tower', vision_tower),
                           ('native_projector', native_projector)):
        if any(parameter.requires_grad for parameter in producer.parameters()):
            findings.append(f'{name} contains trainable parameters')
        for module in producer.modules():
            if module.training:
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    findings.append(f'{name} includes train-mode batch normalization')
                for attr in ('p', 'drop_prob', 'attention_dropout'):
                    probability = getattr(module, attr, 0)
                    if isinstance(probability, (float, int)) and probability > 0:
                        findings.append(f'{name} includes stochastic {attr}={probability}')
    return {'static_gate_passed': not findings, 'blockers': findings,
            'dynamic_parity_required': True, 'training_cache_enabled': False}
