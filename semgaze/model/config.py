"""YAML -> missing-field defaults -> explicit overrides -> validated run config."""
import copy
import json
import math
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = ROOT / 'configs/defaults.yaml'


def _merge(defaults, values):
    result = copy.deepcopy(defaults)
    for key, value in values.items():
        if key == 'batch_size_by_k' and isinstance(value, dict):
            result[key] = {str(k): v for k, v in result.get(key, {}).items()} | {str(k): v for k, v in value.items()}
            continue
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def load_config(path, overrides=None):
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    payload = json.loads(text) if path.suffix.lower() == '.json' else yaml.safe_load(text)
    return resolve_config(payload, overrides)


def resolve_config(config, overrides=None):
    if not isinstance(config, dict):
        raise ValueError('config must be a YAML mapping')
    defaults = yaml.safe_load(DEFAULTS_PATH.read_text(encoding='utf-8'))
    result = _merge(defaults, config)
    if overrides:
        result = _merge(result, overrides)
    # Keep older run/config files backward-compatible even if they predate the
    # explicit semantic draw switch.
    result['evaluation'].setdefault('semantic_use_draws', True)
    validate_config(result)
    return result


def default_section(section):
    # Compatibility for direct utility callers; runtime passes resolved sections.
    return yaml.safe_load(DEFAULTS_PATH.read_text(encoding='utf-8'))[section]


def positive_int(value, name, *, allow_none=False, minimum=1):
    if value is None and allow_none:
        return
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}' + (' or null' if allow_none else ''))


def validate_config(config):
    # Capability checks: changing these requires another codec, objective, architecture,
    # or dataset contract. Do not add experiment recommendations here.
    capabilities = {
        'data.dataset': 'COCO-Search18',
        'model.backend': 'huggingface',
        'where.require_atomic_end_fix': True,
        'where.coordinates.grid_size': 100,
        'where.coordinates.min_value': 0,
        'where.coordinates.max_value': 99,
        'where.coordinates.text_width': 2,
        'where.coordinates.conversion': 'deepgaze_two_stage_python_round',
        'where.duration.unit': 'ms',
        'where.duration.bin_width_ms': 1,
        'where.duration.min_value': 0,
        'where.duration.max_value': 999,
        'where.duration.text_width': 3,
        'where.duration.non_integer_rounding': 'nearest_python_round',
        'where.context.visual_policy': 'one_448x448_tile_per_image',
        'where.context.overflow_training': 'resample_supports_same_query_and_k',
        'where.supervision.support_assistant_labels': 'ignore',
        'where.supervision.query_assistant_labels': 'supervise',
        'where.supervision.supervise_query_end_fix': True,
        'state.readout_layer': 'final_language_model_layer',
        'state.readout_position': 'query_end_fix_token_position',
        'state.projector.type': 'linear_layernorm',
        'state.projector.input_dim': 'model_hidden_size',
        'state.projector.output_dim': 'model_hidden_size',
        'state.direct_state_positions.attention_mask': 1,
        'state.direct_state_positions.label': -100,
        'semantic.output_format': 'flat_lines',
        'semantic.parser': 'strict_rule_based',
        'experiment.semantic_mode': 'flat_single_output',
        'semantic.mode': 'flat_single_output',
        'where.mode': 'xyd',
        'semantic.predict_groups': False,
        'semantic.replay_support_context': False,
        'semantic.consume_full_chronological_states': True,
        'semantic.use_gold_why_groups': True,
        'semantic.semantic_special_tokens': False,
        'semantic.loss': 'response_token_mean_nll',
        'state.detach_where_states': False,
        'training.backward.one_joint_backward': True,
        'training.backward.detach_f': False,
        'training.backward.detach_r': False,
        'model.freeze.vision_tower': True,
        'model.freeze.native_multimodal_projector': True,
        'model.freeze.base_lm_outside_lora': True,
        'model.freeze.old_vocab_rows': True,
        'data.fewshot.require_same_subject': True,
        'data.fewshot.require_distinct_support_stimuli': True,
        'data.fewshot.exclude_query_stimulus': True,
        'data.fewshot.preserve_input_order': True,
        'data.duration.semantics': 'dwell_duration',
        'data.duration.source_unit': 'ms',
        'data.duration.conversion_to_ms': 'identity',
        'evaluation.loss_aggregation': 'episode_mean',
        'smoke.require_finite_losses': True,
    }
    for dotted, implemented in capabilities.items():
        value = config
        for key in dotted.split('.'):
            value = value[key]
        if value != implemented:
            raise ValueError(f'{dotted}={value!r} is unsupported by the implemented flat operation ({implemented!r})')
    for section in ('checkpoint', 'smoke'):
        for key, value in config[section].items():
            if (key.startswith('save_') and key != 'save_every') or key.startswith('require_'):
                if type(value) is not bool:
                    raise ValueError(f'{section}.{key} must be boolean')
    if type(config['where']['supervision']['use_cache']) is not bool:
        raise ValueError('where.supervision.use_cache must be boolean')
    if type(config['where']['supervision']['output_hidden_states']) is not bool:
        raise ValueError('where.supervision.output_hidden_states must be boolean (legacy compatibility field)')
    positive_int(config['training']['gradient_diagnostics_every'], 'training.gradient_diagnostics_every', minimum=0)
    positive_int(config['training']['profile_every_steps'], 'training.profile_every_steps', minimum=0)
    positive_int(config['training']['preprocessing_cache_max_entries'], 'training.preprocessing_cache_max_entries')
    positive_int(config['training']['semantic_collation_cache_max_entries'], 'training.semantic_collation_cache_max_entries')
    for key in ('length_aware_batching', 'reuse_query_vision', 'cache_preprocessed_images', 'cache_semantic_collation'):
        if type(config['training'][key]) is not bool:
            raise ValueError(f'training.{key} must be boolean')
    if config['where']['supervision']['use_cache'] and config['training']['gradient_checkpointing']:
        raise ValueError('HF gradient checkpointing disables KV caching; choose use_cache=false or disable gradient_checkpointing')
    model, t, evaluation = config['model'], config['training'], config['evaluation']
    if 'validation' in config or 'validation_scope' in evaluation['predictions']:
        raise ValueError('obsolete validation configuration; use test/test_scope')
    if any(key in t for key in ('steps_per_epoch', 'epochs')):
        raise ValueError('use num_train_epochs; optimizer updates are derived from Q_train')

    if config['data']['unseen_subjects'] != [7, 8, 9]:
        raise ValueError('unseen subjects must remain [7,8,9]')

    if model['adapter_load_mode'] not in ('fresh', 'continue_trainable'):
        raise ValueError('adapter_load_mode supports fresh or continue_trainable')
    if model['adapter_load_mode'] == 'continue_trainable' and not model['initialization_adapter']:
        raise ValueError('continue_trainable requires initialization_adapter')
    if model['adapter_load_mode'] == 'fresh' and model['initialization_adapter'] is not None:
        raise ValueError('fresh requires initialization_adapter: null; choose continue_trainable to load weights')
    if model['merge_adapter_for_training']:
        raise ValueError('merged adapters have no trainable LoRA parameters; this trainer requires retained PEFT')
    if not isinstance(model['base_model'], str) or not model['base_model'].strip():
        raise ValueError('model.base_model must be a model ID or local path')
    positive_int(model['lora']['rank'], 'model.lora.rank')
    if model['lora']['alpha'] <= 0 or not 0 <= model['lora']['dropout'] < 1 or not model['lora']['target_modules']:
        raise ValueError('LoRA requires positive alpha, dropout in [0,1), and target modules')
    if t['precision'] not in ('fp32', 'bf16'):
        raise ValueError('precision supports fp32 or bf16; fp16 needs a GradScaler implementation')
    if t['optimizer'] not in ('AdamW', 'Adam', 'SGD'):
        raise ValueError('optimizer supports AdamW, Adam or SGD')
    if t['scheduler'] not in ('cosine', 'linear', 'constant', 'constant_with_warmup'):
        raise ValueError('unsupported learning-rate scheduler')
    for key in ('per_device_train_batch_size', 'gradient_accumulation_steps', 'max_episode_retries'):
        positive_int(t[key], 'training.' + key)
    for key in ('max_steps', 'num_train_epochs'):
        positive_int(t[key], 'training.' + key, allow_none=(key == 'max_steps'))
    for key in ('learning_rate', 'weight_decay', 'adam_beta1', 'adam_beta2', 'adam_epsilon', 'warmup_ratio', 'lambda_where', 'lambda_sem'):
        value = t[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(f'training.{key} must be an explicit finite nonnegative number')
    if t['learning_rate'] == 0 or t['adam_epsilon'] == 0 or not 0 <= t['warmup_ratio'] <= 1 or any(not 0 <= t[b] < 1 for b in ('adam_beta1','adam_beta2')):
        raise ValueError('invalid learning rate, epsilon, warmup ratio or Adam betas')
    if t['max_grad_norm'] is not None and (not math.isfinite(t['max_grad_norm']) or t['max_grad_norm'] <= 0):
        raise ValueError('max_grad_norm must be positive or null (disabled)')
    if type(t['gradient_checkpointing']) is not bool:
        raise ValueError('gradient_checkpointing must be an explicit boolean')
    if evaluation['strategy'] not in ('epoch', 'steps', 'no'):
        raise ValueError('evaluation.strategy supports epoch, steps or no (smoke/benchmark only)')
    positive_int(evaluation['draw'], 'evaluation.draw')
    if type(evaluation.get('semantic_use_draws', True)) is not bool:
        raise ValueError('evaluation.semantic_use_draws must be boolean')
    if evaluation['strategy'] == 'steps':
        positive_int(evaluation.get('eval_steps'), 'evaluation.eval_steps')
    for name, values in [('data.fewshot.k_values',config['data']['fewshot']['k_values']), ('evaluation.k_values',evaluation['k_values'])]:
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f'{name} must be a nonempty list of unique positive integers')
        for k in values:
            positive_int(k, name)
    probs = config['data']['fewshot']['train_k_probabilities']
    if len(probs) != len(config['data']['fewshot']['k_values']) or any(not math.isfinite(p) or p < 0 for p in probs) or not math.isclose(sum(probs), 1, abs_tol=1e-8):
        raise ValueError('train_k_probabilities must align with k_values and sum to one')

    split_root = str(config['data']['split_root']).replace('\\', '/')

    ctx = config['where']['context']
    if ctx['max_k'] != 10 or ctx['max_images_per_episode'] != 11:
        raise ValueError('COCO requires max_k=10 and max_images_per_episode=11')
    for key in ('max_k', 'max_images_per_episode', 'max_total_sequence_length'):
        positive_int(ctx[key], 'where.context.' + key)
    if max(config['data']['fewshot']['k_values'] + evaluation['k_values']) > min(ctx['max_k'], ctx['max_images_per_episode'] - 1):
        raise ValueError('configured K exceeds episode support/image capacity')
    if not isinstance(config['where']['end_fix_token'], str) or not config['where']['end_fix_token'].strip():
        raise ValueError('end_fix_token must be nonempty')
    predictions = evaluation['predictions']
    test = config['test']
    metrics = evaluation.get('metrics', {})
    if not isinstance(metrics, dict):
        raise ValueError('evaluation.metrics must be a mapping')
    if type(metrics.get('enabled')) is not bool:
        raise ValueError('evaluation.metrics.enabled must be boolean')
    if metrics.get('split') != 'test':
        raise ValueError('canonical evaluation metrics are test-split metrics')
    where_metrics = metrics.get('where', {})
    probability_metrics = metrics.get('probability', {})
    semantic_metrics = metrics.get('semantic', {})
    if not isinstance(where_metrics, dict) or not isinstance(probability_metrics, dict) or not isinstance(semantic_metrics, dict):
        raise ValueError('evaluation.metrics sections must be mappings')
    scanpath_enabled = any(bool(where_metrics.get(name)) for name in ('scanmatch', 'multimatch', 'sed'))
    if scanpath_enabled and where_metrics.get('coordinate_adapter') not in (
            'deepgaze_vl_predict_scanpath_round', 'deepgaze_historical_round'):
        raise ValueError('scanpath metrics require a verified DeepGaze-VL coordinate_adapter label')
    if probability_metrics.get('ig'):
        centerbias = probability_metrics.get('centerbias', {})
        if centerbias.get('source') != 'canonical_data' or centerbias.get('allow_synthetic_fallback') is not False or not centerbias.get('root'):
            raise ValueError('canonical IG requires a canonical center-bias root and forbids synthetic fallback')
    if probability_metrics.get('outcome') not in ('A', 'B'):
        raise ValueError('metrics.probability.outcome must explicitly select Outcome A or B')
    if probability_metrics.get('log_z', 0.0) != 0.0:
        raise ValueError('metrics.probability.log_z is a fixed no-op and must be 0.0')
    semantic_enabled = bool(semantic_metrics.get('bertscore', {}).get('enabled')) or bool(semantic_metrics.get('cider_r', {}).get('enabled'))
    if metrics['enabled'] and not (scanpath_enabled or probability_metrics.get('ll') or probability_metrics.get('ig') or semantic_enabled):
        raise ValueError('metrics.enabled requires at least one enabled metric')
    for branch in ('loss', 'prediction'):
        settings = test[branch]
        if settings['execution'] not in ('serial', 'same_k_batched'):
            raise ValueError('test execution must be serial or same_k_batched')
        from semgaze.evaluation.batching import physical_size
        for k in evaluation['k_values']:
            physical_size(settings, k)
        positive_int(settings['bucket_window'], 'test.bucket_window')
        if type(settings['bucket_by_length']) is not bool:
            raise ValueError('bucket_by_length must be boolean')
    cache = test['cache']
    positive_int(cache['max_entries'], 'test.cache.max_entries')
    for flag in ('frozen_visual_features', 'support_preprocessing', 'prefix_kv'):
        if type(cache[flag]) is not bool:
            raise ValueError('test cache flags must be boolean')
    if cache['prefix_kv']:
        raise ValueError('prefix KV caching requires separate server profiling and parity')
    positive_int(predictions['train_batches'], 'evaluation.predictions.train_batches', minimum=0)
    positive_int(predictions['semantic_max_new_tokens'], 'semantic_max_new_tokens', allow_none=True)
    if predictions['test_scope'] not in ('all_unseen', 'none'):
        raise ValueError('test_scope supports all_unseen or none')
    positive_int(config['checkpoint']['save_every'], 'checkpoint.save_every', allow_none=True)
    positive_int(config['logging']['every_steps'], 'logging.every_steps')
    subjects = config['data']['unseen_subjects']
    if not isinstance(subjects,list) or any(type(u) is not int for u in subjects) or len(set(subjects)) != len(subjects):
        raise ValueError('unseen_subjects must be unique integer subject IDs')


def write_run_config(config, output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / 'resolved_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')


def resolve_output_dir(config, output_dir=None):
    from datetime import datetime, timezone
    import uuid
    selected = output_dir or config['runtime']['output_dir']
    if selected is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        selected = ROOT / 'runs' / f"{config['experiment']['name']}-{stamp}-{uuid.uuid4().hex[:6]}"
    path = Path(selected).resolve()
    config['runtime']['output_dir'] = str(path)
    return path
