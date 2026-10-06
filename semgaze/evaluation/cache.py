"""Exact projected GT WHERE states, owned by one loss -> prediction cycle."""
import json
import math
import torch
from semgaze.model.visual_cache import processor_signature, parameter_signature


PROTOCOL_VERSION = 'flat-gt-end-fix-r-v1'
_UNSET = object()


def _record_key(record):
    identity = tuple(getattr(record, name) for name in (
        'record_id', 'stimulus_id', 'subject', 'image_path', 'task', 'condition',
        'image_width', 'image_height'))
    trajectories = tuple(tuple(getattr(record, name)) for name in ('x_px', 'y_px', 'duration_ms'))
    if not trajectories[0] or len({len(t) for t in trajectories}) != 1 or not all(
            type(v) in (int, float) and math.isfinite(v) for t in trajectories for v in t):
        raise ValueError('invalid WHERE trajectory cache key')
    return identity + trajectories


def episode_key(episode):
    key = (_record_key(episode.query), tuple(_record_key(s) for s in episode.supports),
           len(episode.supports), episode.draw_id)
    hash(key)  # Validate immutability; the tuple itself, not its hash, is the key.
    return key


class ProjectedWhereCache:
    def __init__(self, *, bundle, split_manifest_identity, cycle_id):
        self.bundle = bundle
        self.split_manifest_identity = split_manifest_identity
        self.cycle_id = cycle_id
        self._ownership = (split_manifest_identity, cycle_id)
        self._provenance = self._signature(bundle)
        self._manifest = None
        self._entries = {}
        self.closed = False

    @staticmethod
    def _signature(bundle):
        return (id(bundle.model), id(bundle.projector), processor_signature(bundle.processor),
                bundle.end_fix_id, bundle.config['where']['end_fix_token'], PROTOCOL_VERSION,
                json.dumps(bundle.config['where'], sort_keys=True),
                parameter_signature(bundle.model), parameter_signature(bundle.projector))

    def validate(self, bundle, *, split_manifest_identity=_UNSET, cycle_id=_UNSET, manifest=None):
        if self.closed:
            raise ValueError('projected WHERE cache is closed')
        if (self._signature(bundle) != self._provenance or
                (self.split_manifest_identity, self.cycle_id) != self._ownership or
                (split_manifest_identity is not _UNSET and split_manifest_identity != self.split_manifest_identity) or
                (cycle_id is not _UNSET and cycle_id != self.cycle_id)):
            raise ValueError('incompatible projected WHERE cache provenance/model/split/cycle')
        if manifest is not None:
            signature = json.dumps(manifest, sort_keys=True, allow_nan=False)
            if self._manifest is not None and signature != self._manifest:
                raise ValueError('incompatible projected WHERE cache split manifest')
            self._manifest = signature

    def key_for(self, episode):
        return episode_key(episode)

    def _check_access(self):
        self.validate(self.bundle)
        if torch.is_grad_enabled() or self.bundle.model.training or self.bundle.projector.training:
            raise ValueError('projected WHERE cache requires inference/eval')

    def put(self, episode, projected_r):
        self._check_access()
        if projected_r.shape != (len(episode.query.x_px), self.bundle.projector[0].in_features):
            raise ValueError('projected WHERE cache requires full chronological [N,d_model] R')
        # Own exact CPU storage, including when the producer already runs on CPU.
        # No detach, dtype conversion, or serialization of the representation.
        with torch.inference_mode():
            self._entries[self.key_for(episode)] = projected_r.to(device='cpu', copy=True)

    def get(self, episode):
        self._check_access()
        try:
            key = self.key_for(episode)
        except (AttributeError, TypeError, ValueError):
            return None
        return self._entries.get(key)

    def contains(self, episode):
        return self.get(episode) is not None

    def close(self):
        self._entries.clear()
        self.closed = True
