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
    # Preserve numerical-safety diagnostics without fixture/smoke policies.
    diagnostics = gradient_diagnostics(bundle)
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
           for p in bundle.trainable_parameters()):
        raise RuntimeError('non-finite parameter gradient')
    return diagnostics


def clip_and_check_gradients(bundle):
    # One aggregate finite check per optimizer step, even when clipping is disabled.
    limit = bundle.config['training']['max_grad_norm']
    return torch.nn.utils.clip_grad_norm_(bundle.trainable_parameters(),
        float('inf') if limit is None else limit, error_if_nonfinite=True)


def run_flat_training_step(model_bundle, episode, optimizer_step=False, *,
                           zero_grad=True, loss_scale=1.0, diagnostics=True,
                           where_batch=None, profiler=None, semantic_cache=None):
    """Run active objectives; explicit-text Semantic needs no WHERE forward.

    Both losses positive: joint WHERE + Semantic objectives.
    lambda_sem=0: WHERE-only (no projector/semantic forward/loss).
    lambda_where=0 with explicit text: Semantic-only (GT XYD, no WHERE forward).
    Shared-state Semantic still requires WHERE forward even if its loss is zero.
    """
    bundle = model_bundle
    single = not isinstance(episode, (list, tuple))
    episodes = [episode] if single else list(episode)
    episodes = [normalized_episode_from_dict(e) if isinstance(e, dict) else e for e in episodes]
    if not episodes:
        raise ValueError('empty physical episode batch')
    for e in episodes:
        from semgaze.data.joint import unseen_subject_ids
        require_seen_training_episode(e, unseen_subject_ids(bundle.config['data']))
    bundle.model.train()
    bundle.projector.train()
    if zero_grad:
        bundle.model.zero_grad(set_to_none=True)
        bundle.projector.zero_grad(set_to_none=True)
    stage = profiler.stage if profiler is not None else lambda name: nullcontext()
    training_config = bundle.config['training']
    where_enabled = training_config['lambda_where'] > 0
    semantic_enabled = training_config['lambda_sem'] > 0
    if not (where_enabled or semantic_enabled):
        raise ValueError('training must supervise WHERE and/or semantic')
    counts = [len(e.query.x_px) for e in episodes]
    explicit = bundle.config['semantic']['conditioning'] == 'explicit_scanpath_text'
    need_where_forward = where_enabled or (semantic_enabled and not explicit)
    where = None
    if need_where_forward:
        with stage('where'):
            where = forward_where_batch(bundle, episodes, batch=where_batch, profiler=profiler,
                                        compute_loss=where_enabled,
                                        return_fixation_states=semantic_enabled and not explicit)
    where_episode_losses = where.episode_losses if where is not None else None
    flat, positions, semantic_metadata = None, [], [{} for _ in episodes]
    states = [None] * len(episodes)
    component_values = {'loss_what': None, 'loss_why': None, 'loss_how': None}
    if semantic_enabled:
        with stage('semantic'):
            if not explicit:
                projected = bundle.projector(torch.cat(where.states, dim=0))
                states = list(projected.split(counts))
                if not all(f.requires_grad and r.requires_grad for f, r in zip(where.states, states)):
                    raise RuntimeError('WHERE F / projected R were detached')
            flat, positions, semantic_metadata = forward_flat_batch(bundle, [e.query for e in episodes], states,
                where=None if explicit else where, profiler=profiler,
                semantic_cache=None if explicit else semantic_cache, response_offsets=True)
            from semgaze.semantic.flat.diagnostics import summarize_semantic_component_nll
            component_values = summarize_semantic_component_nll(flat.token_nll, episodes, semantic_metadata)
    if where_enabled and semantic_enabled:
        # Preserve Full-model arithmetic, loss order, and per-episode reduction.
        episode_total = training_config['lambda_where'] * where_episode_losses + training_config['lambda_sem'] * flat.episode_losses
    elif where_enabled:
        episode_total = training_config['lambda_where'] * where_episode_losses
    else:
        episode_total = training_config['lambda_sem'] * flat.episode_losses
    loss_total = episode_total.mean()
    if diagnostics:
        objective_losses = [loss_total]
        if where_enabled:
            objective_losses.append(where.loss_where)
        if semantic_enabled:
            objective_losses.append(flat.loss)
        if not bool(torch.isfinite(torch.stack(objective_losses)).all()):
            raise RuntimeError('non-finite active supervision loss')
    with stage('backward'):
        (loss_total * loss_scale).backward()
    checks = check_gradients(bundle) if diagnostics else {}
    if optimizer_step:
        if bundle.optimizer is None:
            bundle.optimizer = make_optimizer(bundle)
        with stage('optimizer'):
            clip_and_check_gradients(bundle)
            bundle.optimizer.step()
            if bundle.scheduler is not None:
                bundle.scheduler.step()
    where_metadata = (where.batch.metadata if where is not None else
                      where_batch.metadata if where_batch is not None else
                      [dict(record_id=e.query.record_id, K=len(e.supports),
                            fixation_count=len(e.query.x_px)) for e in episodes])
    metadata = [{k: v for k, v in m.items() if k != 'semantic_target'}
                | {k: v for k, v in sm.items() if k != 'response_offsets'}
                for m, sm in zip(where_metadata, semantic_metadata)]
    result = {'loss_where': where_episode_losses.detach().mean() if where_enabled else None,

              'loss_flat': flat.loss.detach() if semantic_enabled else None,
              'loss_what': component_values['loss_what'],
              'loss_why': component_values['loss_why'],
              'loss_how': component_values['loss_how'],
              'loss_total': loss_total.detach(), 'physical_batch_size': len(episodes),
              'episodes': metadata, 'embeddings_untied': not bundle.embeddings_tied, **checks}
    for key, values in (
        ('query_fixation_count', counts),
        ('query_end_fix_state_count', [len(f) if semantic_enabled and bundle.config['where']['use_end_fix_token']
                                      and not explicit else 0
                                      for f in (where.states if where is not None else [None] * len(episodes))]),
        ('supervised_query_end_fix_count', [int((s.inputs['labels'] == bundle.end_fix_id).sum()) if where_enabled else 0
                                            for s in where.batch.samples]),
        ('supervised_support_end_fix_count', [int((s.inputs['labels'][:, :s.response_start] == bundle.end_fix_id).sum()) if where_enabled else 0
                                              for s in where.batch.samples]),
        ('projector_output_shape', [tuple(r.shape) if r is not None else None for r in states]),
        ('inserted_state_count', [len(p) for p in positions] if semantic_enabled else [0] * len(episodes))):
        result[key] = values[0] if single else values
    return result
