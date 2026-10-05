"""Opt-in profiling. Synchronization is confined to explicit profile boundaries."""
from contextlib import contextmanager
from time import perf_counter
import torch


class TrainingProfiler:
    def __init__(self, device):
        self.device = torch.device(device)
        self.times = {}

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @contextmanager
    def stage(self, name):
        self.sync()
        start = perf_counter()
        yield
        self.sync()
        self.times[name] = self.times.get(name, 0.0) + perf_counter() - start

    def memory(self):
        if self.device.type != 'cuda':
            return dict.fromkeys(('allocated', 'reserved', 'peak_allocated', 'peak_reserved'))
        return {name: fn(self.device) for name, fn in (
            ('allocated', torch.cuda.memory_allocated), ('reserved', torch.cuda.memory_reserved),
            ('peak_allocated', torch.cuda.max_memory_allocated),
            ('peak_reserved', torch.cuda.max_memory_reserved))}


def runtime_metadata(bundle):
    base = bundle.model.get_base_model()
    return {
        'text_attention': base.model.language_model.config._attn_implementation,
        'vision_attention': base.model.vision_tower.config._attn_implementation,
        'vocabulary_size': base.get_output_embeddings().weight.shape[0],
        'dtype': str(base.get_input_embeddings().weight.dtype),
        'checkpointed_modules': [name for name, module in base.named_modules()
                                 if getattr(module, 'gradient_checkpointing', False)],
        'trainable_parameters': sum(p.numel() for p in bundle.trainable_parameters()),
        'optimizer_state_bytes': sum(v.numel() * v.element_size()
            for state in bundle.optimizer.state.values() for v in state.values()
            if isinstance(v, torch.Tensor)) if bundle.optimizer else 0,
    }
