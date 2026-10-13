#!/usr/bin/env python3
"""Apply SemGaze SS+LWCE source patch to an exact-style no-attn checkout.

Usage: python apply_semgaze_patch.py --repo /path/to/SemGaze
Fails closed if an expected source anchor is absent or repeated. Does not touch Git.
"""
import argparse
from pathlib import Path
import shutil


def replace(text, old, new, label):
    matches = text.count(old)
    if matches != 1:
        raise RuntimeError(f'{label}: expected exactly one anchor, found {matches}')
    return text.replace(old, new, 1)


def update_selected_loss(src):
    src = replace(src,
        '    token_counts: torch.Tensor\n',
        '    token_counts: torch.Tensor\n    unweighted_episode_losses: torch.Tensor | None = None\n', 'selected_loss dataclass')
    src = replace(src,
        'def compute_selected_causal_nll(final_hidden, labels, head):',
        'def compute_selected_causal_nll(final_hidden, labels, head, *, target_weights=None):', 'selected_loss signature')
    src = replace(src,
        '    sums = token_nll.new_zeros(labels.shape[0]).index_add(0, episode_device, token_nll)\n'
        '    counts = counts.to(device)\n'
        '    return SelectedCausalNLL(token_nll, sums / counts, counts)',
        '    sums = token_nll.new_zeros(labels.shape[0]).index_add(0, episode_device, token_nll)\n'
        '    counts = counts.to(device)\n'
        '    unweighted = sums / counts\n'
        '    if target_weights is None:\n'
        '        return SelectedCausalNLL(token_nll, unweighted, counts, unweighted)\n'
        '    if target_weights.shape != labels.shape:\n'
        '        raise ValueError("target weights/labels shapes do not align")\n'
        '    weights = target_weights[:, 1:][episode, position].to(device=device, dtype=token_nll.dtype)\n'
        '    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):\n'
        '        raise ValueError("fixation weights must be finite and nonnegative")\n'
        '    weighted_sums = sums.new_zeros(sums.shape).index_add(0, episode_device, token_nll * weights)\n'
        '    denominators = sums.new_zeros(sums.shape).index_add(0, episode_device, weights)\n'
        '    if bool((denominators <= 0).any()):\n'
        '        raise ValueError("each episode must contain positive supervised weight")\n'
        '    return SelectedCausalNLL(token_nll, weighted_sums / denominators, counts, unweighted)',
        'selected_loss reduction')
    return src


def update_where_forward(src):
    src = replace(src,
        'from .collator import collate_where_batch, to_model_device\n',
        'from .collator import collate_where_batch, to_model_device\n'
        'from .weighting import build_fixation_target_weights\n', 'where forwarding imports')
    src = replace(src,
        '    query_image_features: object = None\n',
        '    query_image_features: object = None\n    unweighted_episode_losses: object = None\n', 'where output dataclass')
    src = replace(src,
        'audit_disable_where_attention_mask=False):',
        'audit_disable_where_attention_mask=False, fixation_weighting=None):',
        'where forward signature')
    src = replace(src,
        '        loss = compute_selected_causal_nll(outputs.last_hidden_state, batch.inputs[\'labels\'], bundle.model.get_output_embeddings())',
        '        target_weights = build_fixation_target_weights(\n'
        '            batch, episodes, config=fixation_weighting,\n'
        '            end_fix_token=bundle.config[\'where\'][\'end_fix_token\'])\n'
        '        loss = compute_selected_causal_nll(outputs.last_hidden_state, batch.inputs[\'labels\'],\n'
        '            bundle.model.get_output_embeddings(), target_weights=target_weights)', 'where forward NLL')
    src = replace(src,
        '                       loss.episode_losses, [features[m[\'query_image_index\']] for m in batch.metadata])',
        '                        loss.episode_losses, [features[m[\'query_image_index\']] for m in batch.metadata],\n'
        '                        loss.unweighted_episode_losses)', 'where forward return')
    return src


def update_flat_step(src):
    src = replace(src,
        'audit_capture=None):',
        'audit_capture=None, global_step=0):', 'flat step signature')
    src = replace(src,
        '    stage = profiler.stage if profiler is not None else lambda name: nullcontext()\n    with stage(\'where\'):',
        '    stage = profiler.stage if profiler is not None else lambda name: nullcontext()\n'
        '    training_config = bundle.config[\'training\']\n'
        '    lwce_config = training_config[\'late_fixation_ce\']\n'
        '    active_weighting = lwce_config if lwce_config[\'enabled\'] else None\n'
        '    with stage(\'where\'):', 'flat step config')
    src = replace(src,
        'audit_disable_where_attention_mask=audit_disable_where_attention_mask)',
        'audit_disable_where_attention_mask=audit_disable_where_attention_mask,\n'
        '                                     fixation_weighting=active_weighting)', 'flat step where forwarding')
    src = replace(src,
        '    with stage(\'semantic\'):\n',
        '    where_episode_losses = where.episode_losses\n'
        '    ss_stats = {\'ss_eligible\': 0, \'ss_attempted\': 0, \'ss_accepted\': 0,\n'
        '                \'ss_probability\': 0.0, \'ss_changed\': 0,\n'
        '                \'ss_positions\': {}, \'ss_failure_reasons\': {}}\n'
        '    ss_loss_mixed_mean = None\n'
        '    if training_config[\'scheduled_sampling\'][\'enabled\']:\n'
        '        from semgaze.where.ss import scheduled_sampling_loss\n'
        '        with stage(\'scheduled_sampling\'):\n'
        '            sampled = scheduled_sampling_loss(bundle, episodes, global_step=global_step,\n'
        '                weighting=active_weighting, image_cache=where.batch.image_cache)\n'
        '        ss_stats = sampled.diagnostics\n'
        '        ss_loss_mixed_mean = ss_stats.get(\'ss_loss_mixed_mean\')\n'
        '        beta = float(training_config[\'scheduled_sampling\'][\'loss_mix\'])\n'
        '        where_episode_losses = torch.where(sampled.applied,\n'
        '            (1.0 - beta) * where.episode_losses + beta * sampled.episode_losses,\n'
        '            where.episode_losses)\n'
        '    with stage(\'semantic\'):\n', 'flat step ss hook')
    src = replace(src, 'semantic_cache=semantic_cache)',
        'semantic_cache=semantic_cache, response_offsets=True)',
        'semantic response offsets')
    src = replace(src,
        "response_offsets=True)\n        t = bundle.config['training']\n",
        "response_offsets=True)\n        from semgaze.semantic.flat.diagnostics import summarize_semantic_component_nll\n"
        "        component_values = summarize_semantic_component_nll(flat.token_nll, episodes, semantic_metadata)\n"
        "        t = bundle.config['training']\n", 'semantic diagnostics')
    src = replace(src,
        "    metadata = [{k: v for k, v in m.items() if k != 'semantic_target'} | sm\n",
        "    metadata = [{k: v for k, v in m.items() if k != 'semantic_target'}\n"
        "                | {k: v for k, v in sm.items() if k != 'response_offsets'}\n", 'metadata filter')
    src = replace(src,
        '        episode_total = t[\'lambda_where\'] * where.episode_losses + t[\'lambda_sem\'] * flat.episode_losses',
        '        episode_total = t[\'lambda_where\'] * where_episode_losses + t[\'lambda_sem\'] * flat.episode_losses', 'flat step total objective')
    src = replace(src,
        '    result = {\'loss_where\': where.loss_where.detach(), \'loss_flat\': flat.loss.detach(),',
        '    result = {\'loss_where\': where_episode_losses.detach().mean(),\n'
        '              \'loss_where_gt\': where.loss_where.detach(),\n'
        '              \'loss_where_gt_unweighted\': where.unweighted_episode_losses.detach().mean(),\n'
        '              \'loss_where_mixed\': ss_loss_mixed_mean,\n'
        '              **ss_stats, \'lwce_enabled\': bool(lwce_config[\'enabled\']),\n'
        '              \'loss_flat\': flat.loss.detach(),\n'
        '              \'loss_what\': component_values[\'loss_what\'],\n'
        '              \'loss_why\': component_values[\'loss_why\'],\n'
        '              \'loss_how\': component_values[\'loss_how\'],\n'
        '              \'lwce_alpha\': float(lwce_config[\'alpha\']) if lwce_config[\'enabled\'] else 0.0,', 'flat step result')
    return src


def update_loop(src):
    src = replace(src,
        "            component_losses = {'loss_where': [], 'loss_flat': []}",
        "            component_losses = {'loss_where': [], 'loss_where_gt': [],\n"
        "                                'loss_where_gt_unweighted': [], 'loss_flat': []}\n"
        "            ss_counters = {'ss_eligible': 0, 'ss_attempted': 0, 'ss_accepted': 0, 'ss_changed': 0}\n"
        "            ss_failures = {}\n"
        "            ss_positions = {}\n"
        "            ss_prob = 0.0\n"
        "            ss_mixed_losses = []", 'loop aggregate init')
    src = replace(src,
        '                     profiler=profiler, semantic_cache=semantic_cache)',
        '                     profiler=profiler, semantic_cache=semantic_cache, global_step=step)', 'loop global step')
    src = replace(src,
        '                losses.append(result[\'loss_total\'] * weights[micro])',
        '                losses.append(result[\'loss_total\'] * weights[micro])\n'
        '                for key in ss_counters:\n'
        '                    ss_counters[key] += int(result.get(key, 0))\n'
        '                ss_prob = max(ss_prob, float(result.get(\'ss_probability\', 0.0)))\n'
        '                if result.get(\'loss_where_mixed\') is not None:\n'
        '                    ss_mixed_losses.append(float(result[\'loss_where_mixed\']))\n'
        '                for position, count in result.get(\'ss_positions\', {}).items():\n'
        '                    ss_positions[position] = ss_positions.get(position, 0) + count\n'
        '                for reason, count in result.get(\'ss_failure_reasons\', {}).items():\n'
        '                    ss_failures[reason] = ss_failures.get(reason, 0) + count', 'loop aggregation')
    src = replace(src,
        '                for key, values in diagnostic_losses.items():\n',
        "                entry.update(ss_counters)\n"
        "                entry['ss_probability'] = ss_prob\n"
        "                entry['ss_failure_reasons'] = ss_failures\n"
        "                entry['ss_positions'] = ss_positions\n"
        "                entry['loss_where_mixed'] = (sum(ss_mixed_losses) / len(ss_mixed_losses)\n"
        "                                             if ss_mixed_losses else None)\n"
        "                entry['lwce_enabled'] = bool(t['late_fixation_ce']['enabled'])\n"
        "                entry['ss_enabled'] = bool(t['scheduled_sampling']['enabled'])\n"
        "                entry['lwce_alpha'] = (float(t['late_fixation_ce']['alpha'])\n"
        "                                       if t['late_fixation_ce']['enabled'] else 0.0)\n"
        "                entry['lwce_mode'] = t['late_fixation_ce']['mode']\n"
        "                entry['lwce_threshold'] = t['late_fixation_ce']['threshold']\n"
        "                entry['ss_loss_mix'] = t['scheduled_sampling']['loss_mix']\n"
        '                for key, values in diagnostic_losses.items():\n', 'loop logging')
    return src


def update_config(src):
    src = replace(src,
        "    model, t, evaluation = config['model'], config['training'], config['evaluation']\n",
        "    model, t, evaluation = config['model'], config['training'], config['evaluation']\n"
        "    loading = config['checkpoint_loading']\n"
        "    if any(loading[key] is not None and (not isinstance(loading[key], str) or not loading[key].strip())\n"
        "           for key in ('resume_from', 'init_from')):\n"
        "        raise ValueError('checkpoint_loading paths must be null or nonempty strings')\n"
        "    if loading['resume_from'] is not None and loading['init_from'] is not None:\n"
        "        raise ValueError('choose only checkpoint_loading.resume_from or init_from')\n"
        "    ss = t['scheduled_sampling']\n"
        "    lwce = t['late_fixation_ce']\n"
        "    if any(type(x.get('enabled')) is not bool for x in (ss, lwce)):\n"
        "        raise ValueError('SS/LWCE enabled must be bool')\n"
        "    for key in ('probability', 'warmup_ratio', 'max_probability', 'loss_mix'):\n"
        "        value = ss[key]\n"
        "        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:\n"
        "            raise ValueError(f'scheduled_sampling.{key} must be in [0, 1]')\n"
        "    if ss['sampling_strategy'] != 'greedy_single_fixation' or ss['target_tasks'] != ['where']:\n"
        "        raise ValueError('only WHERE greedy_single_fixation SS is implemented')\n"
        "    if type(ss['seed']) is not int or ss['seed'] < 0:\n"
        "        raise ValueError('scheduled_sampling.seed must be a nonnegative integer')\n"
        "    positive_int(ss['max_new_tokens'], 'scheduled_sampling.max_new_tokens')\n"
        "    if lwce['mode'] not in ('late_only', 'linear'):\n"
        "        raise ValueError('late_fixation_ce.mode must be late_only or linear')\n"
        "    if type(lwce['alpha']) not in (int, float) or not math.isfinite(lwce['alpha']) or lwce['alpha'] < 0:\n"
        "        raise ValueError('late_fixation_ce.alpha must be finite and nonnegative')\n"
        "    positive_int(lwce['threshold'], 'late_fixation_ce.threshold', minimum=0)\n", 'config validate features')
    return src


def update_defaults(src):
    src = replace(src, '\ntraining:\n',
        '\ncheckpoint_loading:\n'
        '  # Strict resume restores optimizer, scheduler, and sampler.\n'
        '  resume_from: null\n'
        '  # Start a fresh adaptation run from weights, with a new optimizer/sampler.\n'
        '  init_from: null\n\n'
        'training:\n'
        '  scheduled_sampling:\n'
        '    enabled: false\n'
        '    probability: 0.30\n'
        '    max_probability: 0.30\n'
        '    warmup_ratio: 0.10\n'
        '    sampling_strategy: greedy_single_fixation\n'
        '    target_tasks: [where]\n'
        '    loss_mix: 0.50\n'
        '    max_new_tokens: 48\n'
        '    seed: 2027\n'
        '  late_fixation_ce:\n'
        '    enabled: false\n'
        '    mode: late_only\n'
        '    alpha: 0.50\n'
        '    threshold: 5\n', 'defaults training anchor')
    return src


def update_train_flat(src):
    src = replace(src,
        "    parser.add_argument('--resume', type=Path)\n",
        "    parser.add_argument('--resume', type=Path, help='Strict optimizer/scheduler/sampler resume')\n"
        "    parser.add_argument('--init-from-checkpoint', type=Path,\n"
        "                        help='Weight-only initialization for a new adaptation run')\n", 'train args')
    src = replace(src,
        "    from semgaze.model.config import resolve_output_dir\n",
        "    loading = config['checkpoint_loading']\n"
        "    resume_path = args.resume or (Path(loading['resume_from']) if loading['resume_from'] else None)\n"
        "    init_path = args.init_from_checkpoint or (Path(loading['init_from']) if loading['init_from'] else None)\n"
        "    if resume_path is not None and init_path is not None:\n"
        "        parser.error('use either --resume or --init-from-checkpoint, not both')\n"
        "    from semgaze.model.config import resolve_output_dir\n", 'train checkpoint config')
    src = replace(src,
        '    if args.resume:\n        bundle = load_checkpoint_bundle(args.resume, split_manifest_identity=identity, output_dir=output_dir, runtime=config[\'runtime\'],',
        '    source_checkpoint = resume_path or init_path\n'
        '    if source_checkpoint is not None:\n'
        '        bundle = load_checkpoint_bundle(source_checkpoint, split_manifest_identity=identity, output_dir=output_dir, runtime=config[\'runtime\'],',
        'train init path')
    src = replace(src,
        "        # A resume retains optimizer schedule and episode-window geometry.\n",
        "        # Checkpoint initialization always retains source architecture/tokenizer.\n", 'train checkpoint comment')
    src = replace(src,
        "        if bundle.config['model'] != config['model'] or bundle.config['where']['end_fix_token'] != config['where']['end_fix_token']:\n"
        "            raise ValueError('resume model/token settings differ from saved weight structure; initialize a new run instead')\n",
        "        if resume_path is not None:\n"
        "            if bundle.config['model'] != config['model'] or bundle.config['where']['end_fix_token'] != config['where']['end_fix_token']:\n"
        "                raise ValueError('strict resume model/token settings differ from checkpoint')\n"
        "        else:\n"
        "            old_config = bundle.config\n"
        "            if old_config['model']['base_model'] != config['model']['base_model']:\n"
        "                raise ValueError('weight initialization cannot change base model')\n"
        "            for key in ('mode', 'end_fix_token', 'coordinates', 'duration'):\n"
        "                if old_config['where'][key] != config['where'][key]:\n"
        "                    raise ValueError(f'weight initialization cannot change WHERE codec: {key}')\n"
        "            for key in ('state', 'semantic'):\n"
        "                if old_config[key] != config[key]:\n"
        "                    raise ValueError(f'weight initialization cannot change {key} architecture')\n"
        "            # Preserve the loaded checkpoint's actual LoRA initialization provenance.\n"
        "            config['model'] = old_config['model']\n", 'train model resume and init separation')
    src = replace(src,
        "        if bundle.config['training']['optimizer'] != config['training']['optimizer']:\n",
        "        if resume_path is not None:\n"
        "            for key in ('scheduled_sampling', 'late_fixation_ce'):\n"
        "                if bundle.config['training'][key] != config['training'][key]:\n"
        "                    raise ValueError(f'strict resume cannot change training.{key}; use init_from')\n"
        "        if resume_path is not None and bundle.config['training']['optimizer'] != config['training']['optimizer']:\n", 'train optimizer resume guard')
    src = replace(src,
        "        for key in ('per_device_train_batch_size', 'gradient_accumulation_steps', 'scheduler',\n"
        "                    'warmup_ratio', 'learning_rate', 'total_optimizer_updates', 'num_train_epochs'):\n"
        "            if bundle.config['training'][key] != config['training'][key]:\n"
        "                raise ValueError(f'resume training.{key} differs; start a new run to change it')\n",
        "        if resume_path is not None:\n"
        "            for key in ('per_device_train_batch_size', 'gradient_accumulation_steps', 'scheduler',\n"
        "                        'warmup_ratio', 'learning_rate', 'total_optimizer_updates', 'num_train_epochs'):\n"
        "                if bundle.config['training'][key] != config['training'][key]:\n"
        "                    raise ValueError(f'resume training.{key} differs; start a new run to change it')\n", 'train strict horizon')
    src = replace(src,
        '    if args.resume:\n        if config[\'training\'][\'gradient_checkpointing\']:',
        '    if source_checkpoint is not None:\n        if config[\'training\'][\'gradient_checkpointing\']:', 'train checkpoint gc')
    src = replace(src,
        '    if args.resume:\n        start = restore_checkpoint_state(bundle, args.resume, split_manifest_identity=identity,\n',
        '    if resume_path is not None:\n        start = restore_checkpoint_state(bundle, resume_path, split_manifest_identity=identity,\n', 'train strict resume restore')
    src = replace(src,
        "        bundle.config = config\n",
        "        if init_path is not None:\n"
        "            bundle.trainer_history = []  # adaptation starts with fresh history\n"
        "            import random\n"
        "            random.seed(config['experiment']['seed'])\n"
        "            torch.manual_seed(config['experiment']['seed'])\n"
        "            if torch.cuda.is_available():\n"
        "                torch.cuda.manual_seed_all(config['experiment']['seed'])\n"
        "        bundle.config = config\n", 'new adaptation log history')
    src = replace(src,
        "    if args.semantic_max_new_tokens is not None:\n",
        "    if args.resume is not None and args.init_from_checkpoint is not None:\n"
        "        parser.error('choose --resume or --init-from-checkpoint')\n"
        "    if args.resume is not None:\n"
        "        overrides['checkpoint_loading'] = {'resume_from': str(args.resume), 'init_from': None}\n"
        "    elif args.init_from_checkpoint is not None:\n"
        "        overrides['checkpoint_loading'] = {'init_from': str(args.init_from_checkpoint), 'resume_from': None}\n"
        "    if args.semantic_max_new_tokens is not None:\n", 'checkpoint CLI overrides')
    return src


TRANSFORMS = {
    'semgaze/model/selected_loss.py': update_selected_loss,
    'semgaze/where/forward.py': update_where_forward,
    'semgaze/training/flat_step.py': update_flat_step,
    'semgaze/training/loop.py': update_loop,
    'semgaze/model/config.py': update_config,
    'configs/defaults.yaml': update_defaults,
    'train_flat.py': update_train_flat,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = args.repo.resolve()
    bundled = Path(__file__).resolve().parent / 'new'
    updates = {}
    for relative, transform in TRANSFORMS.items():
        path = root / relative
        old = path.read_text(encoding='utf-8')
        updates[path] = transform(old)
        if old == updates[path]:
            raise RuntimeError(f'no changes to {relative}')
    # Create full experiment YAMLs from the exact current flat_single.yaml,
    # preserving its data paths, batch sizes, evaluation and optimizer settings.
    flat_source = (root / 'configs/flat_single.yaml').read_text(encoding='utf-8')
    for name, run_name, use_ss, use_lwce in (
        ('flat_continue_sft.yaml', 'semgaze_continue_sft_control', False, False),
        ('flat_ss_only.yaml', 'semgaze_ss_only', True, False),
        ('flat_lwce_only.yaml', 'semgaze_lwce_only', False, True),
        ('flat_ss_lwce.yaml', 'semgaze_ss_lwce', True, True),
    ):
        # Use safe, complete YAML augmentation, preserving existing keys and comments.
        cfg = replace(flat_source, '  name: semgaze_flat_single',
                      f'  name: {run_name}', 'flat example name')
        cfg = replace(cfg, '\ntraining:\n',
                      '\ncheckpoint_loading:\n'
                      '  resume_from: null  # for strict same-run resume only\n'
                      '  init_from: null    # SET to SFT checkpoint directory for adaptation\n\n'
                      'training:\n'
                      '  scheduled_sampling:\n'
                      f'    enabled: {str(use_ss).lower()}\n'
                      '    probability: 0.30\n'
                      '    max_probability: 0.30\n'
                      '    warmup_ratio: 0.10\n'
                      '    sampling_strategy: greedy_single_fixation\n'
                      '    target_tasks: [where]\n'
                      '    loss_mix: 0.50\n'
                      '    max_new_tokens: 48\n'
                      '    seed: 2027\n'
                      '  late_fixation_ce:\n'
                      f'    enabled: {str(use_lwce).lower()}\n'
                      '    mode: late_only\n'
                      '    alpha: 0.50\n'
                      '    threshold: 5\n', 'flat experiment config')
        dest = root / 'configs' / name
        if dest.exists():
            raise FileExistsError(f'experiment config already exists: {dest}')
        updates[dest] = cfg
    for src in bundled.rglob('*'):
        if src.is_file():
            dest = root / src.relative_to(bundled)
            if dest.exists():
                raise FileExistsError(f'new file already exists: {dest}')
            updates[dest] = src.read_text(encoding='utf-8')
    if args.dry_run:
        print('Dry-run anchors OK:', len(updates), 'files')
        return
    for dest, content in updates.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding='utf-8')
    print('SemGaze SS + LWCE patch applied:', len(updates), 'files; run pytest before training')


if __name__ == '__main__':
    main()
