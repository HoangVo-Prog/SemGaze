import torch
from semgaze.data.schema import normalized_episode_from_dict, require_seen_training_episode
from semgaze.where.forward import forward_where
from semgaze.semantic.flat.forward import forward_flat


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


def run_flat_training_step(model_bundle, episode, optimizer_step=False, *,
                           zero_grad=True, loss_scale=1.0):
    bundle = model_bundle
    if isinstance(episode, dict):
        episode = normalized_episode_from_dict(episode)
    require_seen_training_episode(episode, bundle.config['data']['unseen_subjects'])
    bundle.model.train()
    bundle.projector.train()
    if zero_grad:
        bundle.model.zero_grad(set_to_none=True)
        bundle.projector.zero_grad(set_to_none=True)
    where = forward_where(bundle, episode)
    states = bundle.projector(where.states)
    if not where.states.requires_grad or not states.requires_grad:
        raise RuntimeError('WHERE F / projected R were detached')
    loss_flat, positions = forward_flat(bundle, episode.query, states)
    t = bundle.config['training']
    loss_total = t['lambda_where'] * where.loss_where + t['lambda_sem'] * loss_flat
    if not all(bool(torch.isfinite(loss)) for loss in (where.loss_where, loss_flat, loss_total)):
        raise RuntimeError('non-finite joint loss')
    (loss_total * loss_scale).backward()
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in bundle.trainable_parameters()):
        raise RuntimeError('non-finite parameter gradient')
    diagnostics = gradient_diagnostics(bundle)
    smoke = bundle.config['smoke']
    checks = {
        'require_lora_gradient': diagnostics['lora_grad_finite'],
        'require_projector_nonzero_gradient': diagnostics['projector_grad_finite'] and diagnostics['projector_grad_nonzero'],
        'require_end_fix_input_gradient': diagnostics['end_fix_input_grad_finite'],
        'require_end_fix_output_gradient_if_untied': bundle.embeddings_tied or diagnostics['end_fix_output_grad_finite'],
    }
    if any(smoke[key] and not valid for key, valid in checks.items()):
        raise RuntimeError(f'gradient contract failed: {diagnostics}')
    if optimizer_step:
        if bundle.optimizer is None:
            bundle.optimizer = make_optimizer(bundle)
        if t['max_grad_norm'] is not None:
            torch.nn.utils.clip_grad_norm_(bundle.trainable_parameters(), t['max_grad_norm'], error_if_nonfinite=True)
        bundle.optimizer.step()
        if bundle.scheduler is not None:
            bundle.scheduler.step()
    return {'loss_where': where.loss_where.detach(), 'loss_flat': loss_flat.detach(),
            'loss_total': loss_total.detach(), 'query_fixation_count': len(episode.query.x_px),
            'query_end_fix_state_count': len(where.states), 'supervised_query_end_fix_count': len(where.positions),
            'supervised_support_end_fix_count': int((where.batch.inputs['labels'][:, :where.batch.response_start] == bundle.end_fix_id).sum()),
            'projector_output_shape': tuple(states.shape), 'inserted_state_count': len(positions),
            'embeddings_untied': not bundle.embeddings_tied, **diagnostics}
