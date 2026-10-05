"""Audit reference only: original full logits/all-hidden training computation.

Never used as a production fallback. Keep original RowHead cat and HF CE so
parity tests compare independent loss implementations, including the boundary.
"""
from contextlib import contextmanager, nullcontext
from unittest.mock import patch
import torch
from torch.nn import functional as F
from semgaze.where.collator import collate_where, to_model_device
from semgaze.state.extractor import extract_query_states
from semgaze.semantic.flat.forward import prepare_flat_inputs


@contextmanager
def original_row_head(bundle):
    head = bundle.model.get_output_embeddings()
    def forward(hidden):
        logits = head.base(hidden)
        replacement = F.linear(hidden, head.end_fix_row.to(hidden.dtype).unsqueeze(0))
        if getattr(head.base, 'bias', None) is not None:
            replacement = replacement + head.base.bias[head.token_id]
        return torch.cat((logits[..., :head.token_id], replacement,
                          logits[..., head.token_id+1:]), dim=-1)
    with patch.object(head, 'forward', forward):
        yield


def reference_forward(bundle, episode, profiler=None):
    stage = profiler.stage if profiler else lambda name: nullcontext()
    with original_row_head(bundle):
        with stage('where'):
            batch = collate_where(bundle.processor, episode, bundle.end_fix_id,
                                  bundle.context_limit, config=bundle.config)
            inputs = to_model_device(batch.inputs, bundle.model)
            output = bundle.model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            states, mask, positions = extract_query_states(output.hidden_states[-1], inputs['input_ids'],
                inputs['labels'], bundle.end_fix_id, [len(episode.query.x_px)])
            assert mask.all()
            where_loss = output.loss
            fixation_states = states[0]
            del output  # original forward_where releases its output object here
        with stage('semantic'):
            projected = bundle.projector(fixation_states)
            semantic, positions, native = prepare_flat_inputs(bundle, episode.query, projected)
            flat = bundle.model(**semantic, output_hidden_states=False, use_cache=False, return_dict=True)
            flat_loss = flat.loss
            del flat  # original forward_flat returns only loss and positions
            total = bundle.config['training']['lambda_where'] * where_loss + bundle.config['training']['lambda_sem'] * flat_loss
        metadata = dict(record_id=episode.query.record_id, K=len(episode.supports),
            where_length=inputs['input_ids'].shape[1], semantic_length=semantic['inputs_embeds'].shape[1],
            image_count=len(batch.image_paths), fixation_count=len(fixation_states),
            supervised_where_tokens=int((batch.inputs['labels'][:, 1:] != -100).sum()),
            supervised_semantic_tokens=int((semantic['labels'][:, 1:] != -100).sum()))
        return where_loss, flat_loss, total, fixation_states, metadata


def reference_step(bundle, episode, *, zero_grad=True, loss_scale=1, profiler=None):
    from semgaze.training.flat_step import check_gradients
    bundle.model.train()
    bundle.projector.train()
    if zero_grad:
        bundle.model.zero_grad(set_to_none=True)
        bundle.projector.zero_grad(set_to_none=True)
    where, flat, total, _, metadata = reference_forward(bundle, episode, profiler)
    if not all(bool(torch.isfinite(x)) for x in (where, flat, total)):
        raise RuntimeError('non-finite joint loss')
    with profiler.stage('backward') if profiler else nullcontext():
        (total * loss_scale).backward()
    check_gradients(bundle)
    return {'loss_total': total.detach(), 'episodes': [metadata]}
