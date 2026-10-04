"""Resolved engineering choices are explicit run artifacts, not scientific defaults."""
import copy
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    validate_config(config)
    return config


def validate_config(config):
    required = {
        ('experiment', 'semantic_mode'): 'flat_single_output',
        ('semantic', 'mode'): 'flat_single_output',
        ('data', 'variant'): 'all',
        ('model', 'base_model'): 'OpenGVLab/InternVL3_5-8B-HF',
        ('model', 'adapter_load_mode'): 'continue_trainable',
        ('model', 'merge_adapter_for_training'): False,
        ('where', 'mode'): 'xyd',
        ('where', 'end_fix_token'): '<END_FIX>',
        ('semantic', 'predict_groups'): False,
        ('semantic', 'replay_support_context'): False,
        ('semantic', 'consume_full_chronological_states'): True,
        ('semantic', 'use_gold_why_groups'): True,
        ('semantic', 'semantic_special_tokens'): False,
        ('semantic', 'loss'): 'response_token_mean_nll',
        ('state', 'detach_where_states'): False,
    }
    for (section, key), expected in required.items():
        if config[section][key] != expected:
            raise ValueError(f'{section}.{key} must be {expected!r}')
    if (ROOT / config['data']['split_root']).resolve() != ROOT / 'data/COCO_Search18/split/all':
        raise ValueError('runtime split must be data/COCO_Search18/split/all')
    if (ROOT / config['model']['initialization_adapter']).resolve() != ROOT / 'DeepGaze-VL/model/visual_search_adapter':
        raise ValueError('initialization must use released visual_search_adapter')
    if config['data']['unseen_subjects'] != [7, 8, 9] or config['data']['fewshot']['k_values'] != [1, 5, 10]:
        raise ValueError('subject/K protocol differs from frozen contract')
    if config['data']['duration'] != dict(source_field='T', semantics='dwell_duration', source_unit='ms', conversion_to_ms='identity'):
        raise ValueError('duration declaration must be T -> dwell milliseconds, identity')
    if config['where']['context']['max_total_sequence_length'] != 8192:
        raise ValueError('primary context limit must be 8192')
    if not all(config['model']['freeze'].values()):
        raise ValueError('primary base/vision/projector/old vocabulary must be frozen')
    backward = config['training']['backward']
    if not backward['one_joint_backward'] or backward['detach_f'] or backward['detach_r']:
        raise ValueError('one connected joint backward required')
    fixed = {
        'data.dataset': 'COCO-Search18', 'model.backend': 'huggingface',
        'model.lora.rank': 8, 'model.lora.alpha': 16, 'model.lora.dropout': 0.05,
        'where.require_atomic_end_fix': True, 'where.coordinates.grid_size': 100,
        'where.coordinates.min_value': 0, 'where.coordinates.max_value': 99,
        'where.coordinates.text_width': 2, 'where.coordinates.conversion': 'deepgaze_two_stage_python_round',
        'where.duration.unit': 'ms', 'where.duration.bin_width_ms': 1,
        'where.duration.min_value': 0, 'where.duration.max_value': 999, 'where.duration.text_width': 3,
        'where.duration.non_integer_rounding': 'nearest_python_round',
        'where.context.max_k': 10, 'where.context.max_images_per_episode': 11,
        'where.context.visual_policy': 'one_448x448_tile_per_image',
        'where.context.overflow_training': 'reject_and_resample_complete_episode',
        'where.supervision.support_assistant_labels': 'ignore', 'where.supervision.query_assistant_labels': 'supervise',
        'where.supervision.supervise_query_end_fix': True, 'where.supervision.output_hidden_states': True,
        'where.supervision.use_cache': False, 'state.readout_layer': 'final_language_model_layer',
        'state.readout_position': 'query_end_fix_token_position', 'state.projector.type': 'linear_layernorm',
        'state.projector.input_dim': 'model_hidden_size', 'state.projector.output_dim': 'model_hidden_size',
        'state.direct_state_positions.attention_mask': 1, 'state.direct_state_positions.label': -100,
        'semantic.output_format': 'flat_lines', 'semantic.parser': 'strict_rule_based',
        'training.optimizer': 'AdamW', 'training.scheduler': 'cosine',
    }
    for dotted, expected in fixed.items():
        value = config
        for key in dotted.split('.'):
            value = value[key]
        if value != expected:
            raise ValueError(f'{dotted} must be {expected!r}')
    if set(config['model']['lora']['target_modules']) != {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}:
        raise ValueError('LoRA targets differ from the released adapter')
    for key in ('require_same_subject', 'require_distinct_support_stimuli', 'exclude_query_stimulus', 'preserve_input_order'):
        if config['data']['fewshot'][key] is not True:
            raise ValueError(f'data.fewshot.{key} must be true')
    probabilities = config['data']['fewshot']['train_k_probabilities']
    if len(probabilities) != 3 or any(abs(p - 1/3) > 1e-9 for p in probabilities):
        raise ValueError('K sampling must be uniform')
    if not all(config['checkpoint'].values()) or not all(v for k, v in config['smoke'].items() if k.startswith('require_')):
        raise ValueError('required checkpoint components/gradient gates cannot be disabled')
    for key in ('per_device_train_batch_size', 'gradient_accumulation_steps'):
        if type(config['training'][key]) is not int or config['training'][key] < 1:
            raise ValueError(f'training.{key} must be a positive integer')
    steps = config['training'].get('steps_per_epoch')
    if steps is not None and (type(steps) is not int or steps < 1):
        raise ValueError('training.steps_per_epoch must be null or a positive integer')
    evaluation = config.get('evaluation')
    # Older inference checkpoints may omit the new epoch schedule. New training
    # resolves it explicitly before creating a model or reading runtime data.
    if evaluation is not None and {k: v for k, v in evaluation.items() if k != 'predictions'} != {
            'strategy': 'epoch', 'k_values': [1, 5, 10], 'loss_aggregation': 'episode_mean'}:
        raise ValueError('evaluation requires epoch scheduling, all K values, and episode_mean losses')
    predictions = (evaluation or {}).get('predictions')
    if predictions is not None:
        if type(predictions.get('train_batches')) is not int or predictions['train_batches'] != 1 or predictions.get('validation_scope') != 'all_seen':
            raise ValueError('epoch predictions require one train batch and all seen validation queries')
        budget = predictions.get('semantic_max_new_tokens')
        if budget is not None and (type(budget) is not int or budget < 1):
            raise ValueError('semantic_max_new_tokens must be null or a positive integer')


def resolve_config(config):
    result = copy.deepcopy(config)
    # Declared engineering starting choices; written to resolved_config.json.
    defaults = {'gradient_checkpointing': False, 'max_grad_norm': 1.0, 'weight_decay': 0.01,
                'adam_beta1': 0.9, 'adam_beta2': 0.999, 'adam_epsilon': 1e-8}
    resolved = {}
    for key, value in defaults.items():
        if result['training'][key] is None:
            result['training'][key] = value
            resolved[key] = value
    result['engineering_resolutions'] = resolved
    result.setdefault('runtime', {})
    result['runtime'].setdefault('device', 'cuda')
    # annotation_frame stays unresolved until verified; fixtures carry explicit dimensions.
    return result


def write_run_config(config, output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / 'resolved_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
