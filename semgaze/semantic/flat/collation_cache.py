"""Bounded CPU-only cache for deterministic, native semantic text collation.

Native processor/tokenizer performs the full audited serialization on miss.
On hit only the pixel row comes from the current WHERE sample, so model image
inputs never become stale. Never cache embeddings, R, F or autograd tensors.
"""
from collections import OrderedDict
from dataclasses import replace
from semgaze.model.visual_cache import processor_signature


class SemanticNativeLRU:
    def __init__(self, max_entries=128):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError('positive semantic collation cache capacity required')
        self.max_entries = max_entries
        self.items = OrderedDict()
        self.hits = self.misses = self.evictions = 0

    @staticmethod
    def key(processor, query, prompt, target):
        return (processor_signature(processor), query.record_id,
                query.image_path, prompt.text, tuple(prompt.boundaries), target)

    def get(self, key, *, pixels):
        if pixels.device.type != 'cpu' or tuple(pixels.shape) != (1, 3, 448, 448):
            raise ValueError('cache hit requires native canonical CPU query pixels')
        item = self.items.get(key)
        if item is None:
            self.misses += 1
            return None
        self.hits += 1
        self.items.move_to_end(key)
        # Cached tensors are private and cloned to prevent any accidental caller
        # mutation from tainting later episodes.
        return replace(item, inputs={**{k: v.clone() for k, v in item.inputs.items()},
                                     'pixel_values': pixels})

    def put(self, key, native):
        if any(v.device.type != 'cpu' for v in native.inputs.values()):
            raise ValueError('only CPU-native tensors are cacheable')
        item = replace(native, inputs={k: v.clone() for k, v in native.inputs.items()
                                       if k != 'pixel_values'})
        self.items[key] = item
        self.items.move_to_end(key)
        if len(self.items) > self.max_entries:
            self.items.popitem(last=False)
            self.evictions += 1

    def statistics(self):
        bytes_cached = sum(v.numel() * v.element_size()
                           for item in self.items.values() for v in item.inputs.values())
        return dict(entries=len(self.items), hits=self.hits, misses=self.misses,
                    evictions=self.evictions, cpu_tensor_bytes=bytes_cached,
                    tokenizer_calls_avoided=self.hits,
                    hit_rate=self.hits / max(1, self.hits + self.misses))

    def close(self):
        self.items.clear()
