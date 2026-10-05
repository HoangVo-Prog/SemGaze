"""Bounded, invocation-scoped reuse of native deterministic image work.

The processor still owns text expansion and the backbone owns vision projection.
No language states, trainable producers, or cross-checkpoint entries are cached.
"""
from collections import OrderedDict
from copy import copy
import torch


class InferenceVisualCache(OrderedDict):
    def __init__(self, *, preprocessing=True, features=True, max_entries=256):
        super().__init__()
        self.preprocessing, self.features = preprocessing, features
        self.max_entries = max_entries
        self.pixels, self.visual = OrderedDict(), OrderedDict()
        self.hits = dict(preprocessing=0, visual=0)
        self.misses = dict(preprocessing=0, visual=0)

    def __setitem__(self, key, value):
        if not self.preprocessing:
            return
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.max_entries:
            self.popitem(last=False)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def _put(self, store, key, value):
        store[key] = value
        store.move_to_end(key)
        while len(store) > self.max_entries:
            store.popitem(last=False)

    def processor_for(self, processor, paths):
        if not self.preprocessing:
            return processor
        owner, original = self, processor.image_processor

        class CachedImageProcessor:
            def __getattr__(self, name):
                return getattr(original, name)

            def __call__(self, images, **kwargs):
                if len(images) != len(paths):
                    raise ValueError('cached preprocessing image mapping mismatch')
                rows = []
                for path, image in zip(paths, images):
                    key = (path, repr(sorted(kwargs.items())))
                    if key in owner.pixels:
                        owner.hits['preprocessing'] += 1
                        owner.pixels.move_to_end(key)
                        row = owner.pixels[key]
                    else:
                        owner.misses['preprocessing'] += 1
                        row = original(images=[image], **kwargs)
                        if set(row) != {'pixel_values', 'num_patches'} or list(row['num_patches']) != [1]:
                            raise ValueError('cache requires native one-tile preprocessing')
                        owner._put(owner.pixels, key, row)
                    rows.append(row)
                return {'pixel_values': torch.cat([r['pixel_values'] for r in rows]),
                        'num_patches': [1] * len(rows)}

        result = copy(processor)
        result.image_processor = CachedImageProcessor()
        return result

    def fuse(self, bundle, inputs, paths):
        if not self.features:
            return inputs, None
        from semgaze.semantic.flat.forward import frozen_vision_is_reusable
        if bundle.model.training or torch.is_grad_enabled() or not frozen_vision_is_reusable(bundle.model):
            raise ValueError('visual cache requires inference and frozen deterministic vision producers')
        pixels = inputs['pixel_values']
        if len(paths) != len(pixels):
            raise ValueError('cached visual image mapping mismatch')
        # Input preprocessing is fixed for this cache invocation; keep pixel identity
        # checks on CPU in semantic preparation, and never key on LM state.
        keys = [(p, str(pixels.dtype), str(pixels.device)) for p in paths]
        missing = list(dict.fromkeys(k for k in keys if k not in self.visual))
        fresh = {key: self.visual[key] for key in keys if key in self.visual}
        if missing:
            indices = [keys.index(k) for k in missing]
            native = bundle.model.get_base_model().model
            values = native.get_image_features(pixels[indices], return_dict=True).pooler_output
            # Own each row's storage: a cached view must not retain an evicted
            # image's entire vision batch allocation.
            fresh.update((key, value.clone()) for key, value in zip(missing, values.unbind()))
        rows = []
        for key in keys:
            if key in self.visual:
                self.hits['visual'] += 1
                row = self.visual[key]
                self.visual.move_to_end(key)
            else:
                self.misses['visual'] += 1
                row = fresh[key]
                self._put(self.visual, key, row)
            rows.append(row)
        features = torch.stack(rows)
        ids = inputs['input_ids']
        mask = ids == bundle.processor.image_token_id
        if int(mask.sum()) != features.shape[0] * features.shape[1]:
            raise ValueError('cached visual placeholder mapping mismatch')
        embeddings = bundle.model.get_input_embeddings()(ids)
        embeddings = embeddings.masked_scatter(mask.unsqueeze(-1), features.to(embeddings))
        return {k: v for k, v in inputs.items() if k not in ('input_ids', 'pixel_values')} | {
            'inputs_embeds': embeddings}, features

    def statistics(self):
        return {name: dict(entries=len(store), hits=self.hits[name], misses=self.misses[name],
                          hit_rate=self.hits[name] / max(1, self.hits[name] + self.misses[name]))
                for name, store in (('preprocessing', self.pixels), ('visual', self.visual))}

    def close(self):
        self.clear()
        self.pixels.clear()
        self.visual.clear()
