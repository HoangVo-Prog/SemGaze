"""Bounded, invocation-scoped reuse of native deterministic image work.

The processor still owns text expansion and the backbone owns vision projection.
No language states, trainable producers, or cross-checkpoint entries are cached.
"""
from collections import OrderedDict
from copy import copy
import json
import torch


def parameter_signature(module):
    values = []
    for name, p in module.named_parameters():
        try:
            version = p._version
        except RuntimeError:
            version = 'inference'
        values.append((name, id(p), version, str(p.device), str(p.dtype)))
    return tuple(values)


def processor_signature(processor):
    image_processor = processor.image_processor
    to_dict = getattr(image_processor, 'to_dict', None)
    image_config = to_dict() if to_dict is not None else repr(image_processor)
    return (id(processor), id(processor.image_processor), id(processor.tokenizer),
            json.dumps(image_config, sort_keys=True, default=str),
            repr(getattr(processor, 'chat_template', None)), processor.image_seq_length,
            processor.image_token_id,
            False, 448, 448, 1)


class InferenceVisualCache(OrderedDict):
    def __init__(self, *, preprocessing=True, features=True, max_entries=256):
        super().__init__()
        self.preprocessing, self.features = preprocessing, features
        self.max_entries = max_entries
        self.pixels, self.visual = OrderedDict(), OrderedDict()
        self.hits = dict(preprocessing=0, visual=0)
        self.misses = dict(preprocessing=0, visual=0)
        self.closed = False
        self._processor_signature = None
        self._binding = None
        self._bound_objects = None
        self.visual_pixels = {}
        self._preprocessing_kwargs = None

    def _check_processor(self, processor):
        if self.closed:
            raise ValueError('visual cache is closed')
        signature = processor_signature(processor)
        if self._processor_signature is not None and signature != self._processor_signature:
            raise ValueError('incompatible visual cache processor/preprocessing')
        self._processor_signature = signature

    def validate(self, bundle):
        from semgaze.semantic.flat.forward import frozen_vision_is_reusable
        self._check_processor(bundle.processor)
        native = bundle.model.get_base_model().model
        embedding = bundle.model.get_input_embeddings().weight
        signature = (id(bundle.model), id(native.vision_tower), id(native.multi_modal_projector),
                     str(embedding.device), str(embedding.dtype),
                     parameter_signature(native.vision_tower), parameter_signature(native.multi_modal_projector))
        if self._binding is not None and signature != self._binding:
            raise ValueError('incompatible visual cache model/device/dtype')
        if self.features and (bundle.model.training or torch.is_grad_enabled() or
                              not frozen_vision_is_reusable(bundle.model)):
            raise ValueError('visual cache requires inference and frozen deterministic vision producers')
        self._binding = signature
        self._bound_objects = (bundle.model, native.vision_tower, native.multi_modal_projector, bundle.processor)

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
        self._check_processor(processor)
        if not self.preprocessing and not self.features:
            return processor
        owner, original = self, processor.image_processor

        class CachedImageProcessor:
            def __getattr__(self, name):
                return getattr(original, name)

            def __call__(self, images, **kwargs):
                owner._check_processor(processor)
                if len(images) != len(paths):
                    raise ValueError('cached preprocessing image mapping mismatch')
                signature = repr(sorted(kwargs.items()))
                if owner._preprocessing_kwargs is not None and signature != owner._preprocessing_kwargs:
                    raise ValueError('incompatible visual cache preprocessing arguments')
                if kwargs.get('crop_to_patches') is not False or kwargs.get('size') != {'height': 448, 'width': 448}:
                    raise ValueError('cache requires canonical 448x448 one-tile preprocessing')
                owner._preprocessing_kwargs = signature
                rows = []
                for path, image in zip(paths, images):
                    # Visual entries pin their canonical CPU preprocessing even
                    # when the independent preprocessing LRU has evicted it.
                    row = owner.visual_pixels.get(path)
                    if row is None:
                        row = owner.pixels.get(path)
                    if row is not None:
                        owner.hits['preprocessing'] += 1
                        if path in owner.pixels:
                            owner.pixels.move_to_end(path)
                    else:
                        owner.misses['preprocessing'] += 1
                        row = original(images=[image], **kwargs)
                        if set(row) != {'pixel_values', 'num_patches'} or list(row['num_patches']) != [1]:
                            raise ValueError('cache requires native one-tile preprocessing')
                        owner._put(owner.pixels, path, row)
                    rows.append(row)
                return {'pixel_values': torch.cat([r['pixel_values'] for r in rows]),
                        'num_patches': [1] * len(rows)}

        result = copy(processor)
        result.image_processor = CachedImageProcessor()
        return result

    def _transfer_misses(self, pixels, bundle):
        # Same target as the baseline to_model_device(pixel_values), derived
        # from the live model before touching any device-resident pixel tensor.
        embedding = bundle.model.get_input_embeddings().weight
        return pixels.to(device=embedding.device, dtype=embedding.dtype, non_blocking=True)

    def get_or_compute_visual_features(self, bundle, *, paths, host_pixel_values):
        self.validate(bundle)
        if not self.features:
            raise ValueError('final visual feature caching is disabled')
        rows = list(host_pixel_values)
        if not paths or len(paths) != len(rows) or any(row.device.type != 'cpu' for row in rows):
            raise ValueError('cached visual lookup requires aligned host pixel rows')
        if any(tuple(row.shape) != (3, 448, 448) for row in rows):
            raise ValueError('cache requires canonical 448x448 one-tile pixels')
        first = {}
        for index, path in enumerate(paths):
            if path in first and not torch.equal(rows[first[path]], rows[index]):
                raise ValueError('same image identity has different preprocessing')
            first.setdefault(path, index)
            if path in self.visual:
                canonical = self.visual_pixels[path]['pixel_values'][0]
                if canonical.dtype != rows[index].dtype or not torch.equal(canonical, rows[index]):
                    raise ValueError('cached image preprocessing differs from canonical host pixels')
        missing = [path for path in first if path not in self.visual]
        fresh = {path: self.visual[path] for path in first if path in self.visual}
        was_hit = set(fresh)
        if missing:
            pixels = self._transfer_misses(torch.stack([rows[first[path]] for path in missing]), bundle)
            native = bundle.model.get_base_model().model
            values = native.get_image_features(pixels, return_dict=True).pooler_output
            if values.shape[:2] != (len(missing), bundle.processor.image_seq_length):
                raise ValueError('native image feature groups differ from requested image ordering')
            fresh.update((path, value.clone()) for path, value in zip(missing, values.unbind()))
        # Assemble from request-local references so LRU eviction within a batch
        # cannot reorder or discard a hit needed later in that same batch.
        result = torch.stack([fresh[path] for path in paths])
        for path in paths:
            self.hits['visual'] += int(path in was_hit)
            self.misses['visual'] += int(path not in was_hit)
            self.visual[path] = fresh[path]
            self.visual.move_to_end(path)
            self.visual_pixels[path] = {'pixel_values': rows[first[path]].unsqueeze(0).clone(), 'num_patches': [1]}
            while len(self.visual) > self.max_entries:
                evicted, _ = self.visual.popitem(last=False)
                self.visual_pixels.pop(evicted)
        return result

    def fuse(self, bundle, inputs, paths, *, host_pixel_values=None):
        self.validate(bundle)
        if not self.features:
            return inputs, None
        pixels = inputs.get('pixel_values') if host_pixel_values is None else host_pixel_values
        features = self.get_or_compute_visual_features(bundle, paths=paths, host_pixel_values=pixels)
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
        self.visual_pixels.clear()
        self.closed = True
        self._bound_objects = None
