import torch
from semgaze.data.schema import normalized_episode_from_dict, require_seen_training_episode
from semgaze.where.forward import forward_where_batch
from semgaze.semantic.flat.forward import forward_flat_batch
from contextlib import nullcontext


def make_optimizer(bundle):
    t = bundle.config['training']
    kwargs = dict(lr=t['learning_rate'], weight_decay=t['weight_decay'])
    if t['optimizer'] in ('AdamW', 'Adam'):
        kwargs.update(betas=(t['adam_beta1'], t['adam_beta2']), eps=t['adam_epsilon'])
    return getattr(torch.optim, t['optimizer'])(bundle.trainable_parameters(), **kwargs)


def make_scheduler(bundle, max_steps, start_step=0):
    from transformers import get_scheduler
    t = bundle.config['training']
    scheduler = get_scheduler(t['scheduler'], bundle.optimizer,
        num_warmup_steps=int(max_steps * t['warmup_ratio']), num_training_steps=max_steps)
    if start_step:
        scheduler.last_epoch = start_step
        for group, base_lr, factor in zip(bundle.optimizer.param_groups, scheduler.base_lrs, scheduler.lr_lambdas):
            group['lr'] = base_lr * factor(start_step)
        scheduler._last_lr = [g['lr'] for g in bundle.optimizer.param_groups]
    return scheduler


def gradient_diagnostics(bundle):
    def finite(p):
        return p.grad is not None and bool(torch.isfinite(p.grad).all())
    lora = [p for name, p in bundle.model.named_parameters() if 'lora_' in name and p.requires_grad]
    projector = list(bundle.projector.parameters())
    return {'lora_grad_finite': any(finite(p) for p in lora) and all(p.grad is None or finite(p) for p in lora),
            'projector_grad_finite': all(finite(p) for p in projector),
            'projector_grad_nonzero': any(finite(p) and bool(p.grad.ne(0).any()) for p in projector),
            'end_fix_input_grad_finite': finite(bundle.input_row),
            'end_fix_output_grad_finite': finite(bundle.output_row)}


def check_gradients(bundle):
    diagnostics = gradient_diagnostics(bundle)
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
           for p in bundle.trainable_parameters()):
        raise RuntimeError('non-finite parameter gradient')
    smoke = bundle.config['smoke']
    checks = {
        'require_lora_gradient': diagnostics['lora_grad_finite'],
        'require_projector_nonzero_gradient': diagnostics['projector_grad_finite'] and diagnostics['projector_grad_nonzero'],
        'require_end_fix_input_gradient': diagnostics['end_fix_input_grad_finite'],
        'require_end_fix_output_gradient_if_untied': bundle.embeddings_tied or diagnostics['end_fix_output_grad_finite'],
    }
    if any(smoke[key] and not valid for key, valid in checks.items()):
        raise RuntimeError(f'gradient contract failed: {diagnostics}')
    return diagnostics


def clip_and_check_gradients(bundle):
    # One aggregate finite check per optimizer step, even when clipping is disabled.
    limit = bundle.config['training']['max_grad_norm']
    return torch.nn.utils.clip_grad_norm_(bundle.trainable_parameters(),
        float('inf') if limit is None else limit, error_if_nonfinite=True)


def run_flat_training_step(model_bundle, episode, optimizer_step=False, *,
                           zero_grad=True, loss_scale=1.0, diagnostics=True,
                           where_batch=None, profiler=None, semantic_cache=None,
                           audit_disable_where_attention_mask=False,
                           audit_capture=None):
    """One vectorized WHERE and semantic forward, one backward per physical batch.

    A scalar episode keeps the public smoke/evaluation-facing result convention.
    Lists/tuples are physical batches, never sequential model execution.
    """
    bundle = model_bundle
    single = not isinstance(episode, (list, tuple))
    episodes = [episode] if single else list(episode)
    episodes = [normalized_episode_from_dict(e) if isinstance(e, dict) else e for e in episodes]
    if not episodes:
        raise ValueError('empty physical episode batch')
    for e in episodes:
        require_seen_training_episode(e, bundle.config['data']['unseen_subjects'])
    bundle.model.train()
    bundle.projector.train()
    if zero_grad:
        bundle.model.zero_grad(set_to_none=True)
        bundle.projector.zero_grad(set_to_none=True)
    stage = profiler.stage if profiler is not None else lambda name: nullcontext()
    with stage('where'):
        where = forward_where_batch(bundle, episodes, batch=where_batch, profiler=profiler,
                                    audit_disable_where_attention_mask=audit_disable_where_attention_mask)
    if audit_capture is not None:
        for state in where.states:
            state.retain_grad()
        audit_capture['rng_state_after_where_cpu'] = torch.get_rng_state().clone()
        device = bundle.input_row.device
        if device.type == 'cuda':
            audit_capture['rng_state_after_where_cuda'] = torch.cuda.get_rng_state(device).clone()
    with stage('semantic'):
        # One shared projector on all valid states; no padding enters P_E.
        counts = [len(e.query.x_px) for e in episodes]
        projected = bundle.projector(torch.cat(where.states, dim=0))
        states = list(projected.split(counts))
        if not all(f.requires_grad and r.requires_grad for f, r in zip(where.states, states)):
            raise RuntimeError('WHERE F / projected R were detached')
        flat, positions, semantic_metadata = forward_flat_batch(bundle, [e.query for e in episodes], states,
                                                               where=where, profiler=profiler,
                                                                semantic_cache=semantic_cache)
        t = bundle.config['training']
        episode_total = t['lambda_where'] * where.episode_losses + t['lambda_sem'] * flat.episode_losses
        loss_total = episode_total.mean()
    if diagnostics and not bool(torch.isfinite(torch.stack((where.loss_where, flat.loss, loss_total))).all()):
        raise RuntimeError('non-finite joint loss')
    with stage('backward'):
        (loss_total * loss_scale).backward()
    if audit_capture is not None:
        audit_capture.update({
            'where_episode_losses': where.episode_losses.detach().clone(),
            'where_states': [state.detach().clone() for state in where.states],
            'loss_where': where.loss_where.detach().clone(),
            'loss_flat': flat.loss.detach().clone(),
            'loss_total': loss_total.detach().clone(),
        })
        audit_capture['state_gradients'] = [
            state.grad.detach().clone() if state.grad is not None else None
            for state in where.states
        ]
    checks = check_gradients(bundle) if diagnostics else {}
    if optimizer_step:
        if bundle.optimizer is None:
            bundle.optimizer = make_optimizer(bundle)
        with stage('optimizer'):
            clip_and_check_gradients(bundle)
            bundle.optimizer.step()
            if bundle.scheduler is not None:
                bundle.scheduler.step()
    metadata = [{k: v for k, v in m.items() if k != 'semantic_target'} | sm
                for m, sm in zip(where.batch.metadata, semantic_metadata)]
    result = {'loss_where': where.loss_where.detach(), 'loss_flat': flat.loss.detach(),
              'loss_total': loss_total.detach(), 'physical_batch_size': len(episodes),
              'episodes': metadata, 'embeddings_untied': not bundle.embeddings_tied, **checks}
    for key, values in (
        ('query_fixation_count', counts), ('query_end_fix_state_count', [len(f) for f in where.states]),
        ('supervised_query_end_fix_count', [len(p) for p in where.positions]),
        ('supervised_support_end_fix_count', [int((s.inputs['labels'][:, :s.response_start] == bundle.end_fix_id).sum())
                                             for s in where.batch.samples]),
        ('projector_output_shape', [tuple(r.shape) for r in states]),
        ('inserted_state_count', [len(p) for p in positions])):
        result[key] = values[0] if single else values
    return result
