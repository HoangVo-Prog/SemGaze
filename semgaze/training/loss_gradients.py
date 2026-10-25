"""Opt-in, read-only per-objective gradients over a complete optimizer window.

One ``record`` call per microbatch BEFORE the normal joint ``backward``.  The
probe accumulates lambda- and microbatch-weighted gradients on CPU so its norms
and cosines describe the exact (pre-clipping) optimizer update, not just the
last microbatch.  Never writes to Parameter.grad or optimizer state.
"""
import math

import torch


class LossGradientProbe:
    """Capture WHERE/semantic gradient contributions without changing training."""

    def __init__(self, bundle):
        # The optimizer operates on exactly this set (projector excluded for
        # WHERE-only), so don't silently measure frozen/non-optimized weights.
        self.params = list(bundle.trainable_parameters())
        if len({id(p) for p in self.params}) != len(self.params):
            raise ValueError('duplicate trainable Parameter in gradient probe')
        model_names = {id(p): n for n, p in bundle.model.named_parameters()}
        projector_ids = {id(p) for p in bundle.projector.parameters()}
        input_row_id, output_row_id = id(bundle.input_row), id(bundle.output_row)
        self.groups = {'all_trainable': list(range(len(self.params))),
                       'shared_lora': [], 'mtl_task_specific': [],
                       'end_fix_rows': [], 'projector': []}
        for idx, param in enumerate(self.params):
            pid = id(param)
            if pid in projector_ids:
                group = 'projector'
            elif pid in (input_row_id, output_row_id):
                group = 'end_fix_rows'
            elif model_names.get(pid, '').endswith(('mtl_lambdas', 'mtl_mixture_logits')):
                group = 'mtl_task_specific'
            elif 'lora_' in model_names.get(pid, '') or model_names.get(pid, '').endswith('mtl_extra_b'):
                group = 'shared_lora'
            else:
                raise ValueError(f'unclassified trainable parameter: {model_names.get(pid, pid)}')
            self.groups[group].append(idx)
        self.gradients = {'where': [None] * len(self.params),
                          'semantic': [None] * len(self.params)}
        self.microbatches = 0

    def record(self, *, where_loss=None, semantic_loss=None, lambda_where=1.0,
               lambda_sem=1.0, micro_weight=1.0, advance=True):
        """Capture the same weighted component means used by joint backward.

        ``where_loss`` and ``semantic_loss`` are differentiable BATCH MEANS,
        not detached logging tensors. ``micro_weight`` is n_micro/n_window.
        Requires non-reentrant checkpointing for autograd.grad compatibility.
        """
        for name, loss, coefficient in (
            ('where', where_loss, lambda_where),
            ('semantic', semantic_loss, lambda_sem),
        ):
            if loss is None or coefficient == 0:
                continue
            grads = torch.autograd.grad(
                loss * (float(coefficient) * float(micro_weight)),
                self.params, retain_graph=True, allow_unused=True,
            )
            buffer = self.gradients[name]
            for i, grad in enumerate(grads):
                if grad is None:
                    continue
                # Keep limited-duration diagnostic buffers off the GPU.
                value = grad.detach().to(device='cpu', dtype=torch.float32)
                if not bool(torch.isfinite(value).all()):
                    raise RuntimeError(f'non-finite {name} objective gradient')
                if buffer[i] is None:
                    buffer[i] = value.clone()  # materialize expanded/zero-stride gradients
                else:
                    buffer[i].add_(value)
        if advance:
            self.microbatches += 1

    def _group_stats(self, indices):
        a, b = self.gradients['where'], self.gradients['semantic']
        aa = bb = ab = 0.0
        support_a = support_b = 0
        for i in indices:
            wa, sb = a[i], b[i]
            if wa is not None:
                aa += float(torch.dot(wa.reshape(-1), wa.reshape(-1)))
                support_a += 1
            if sb is not None:
                bb += float(torch.dot(sb.reshape(-1), sb.reshape(-1)))
                support_b += 1
            if wa is not None and sb is not None:
                ab += float(torch.dot(wa.reshape(-1), sb.reshape(-1)))
        nw, ns = math.sqrt(max(0.0, aa)), math.sqrt(max(0.0, bb))
        cosine = max(-1.0, min(1.0, ab / (nw * ns))) if nw > 0 and ns > 0 else None
        return {'where_norm': nw, 'semantic_norm': ns,
                'semantic_to_where_norm_ratio': ns / nw if nw > 0 else None,
                'cosine': cosine, 'dot': ab,
                'conflict': cosine < 0 if cosine is not None else None,
                'where_parameter_count': support_a,
                'semantic_parameter_count': support_b}

    def summary(self):
        if not self.microbatches:
            raise ValueError('no microbatches recorded')
        return {'basis': 'lambda_weighted_gradient_accumulated_before_clipping',
                'microbatches': self.microbatches,
                'groups': {key: self._group_stats(ids)
                           for key, ids in self.groups.items()}}
