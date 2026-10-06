"""Observe non-default YAML values at actual model/data/trainer consumers."""
import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from semgaze.model.config import load_config, resolve_config, ROOT
from semgaze.model.build import build_flat_model_bundle
from semgaze.model.checkpoint import save_checkpoint, restore_checkpoint_state, load_checkpoint_bundle
from semgaze.training.flat_step import make_optimizer, make_scheduler, run_flat_training_step
from semgaze.data.fewshot import TrainingEpisodeSampler
from semgaze.evaluation.test import evaluate_test_epoch, EVAL_KEYS
from semgaze.evaluation.predictions import predict_epoch
from semgaze.training.loop import run_training_loop
from test_model_path import tiny_bundle, episode
from test_epoch_validation import evaluation_data, FixedSampler


def write_yaml(tmp_path, values):
    path = tmp_path / 'experiment.yaml'
    path.write_text(yaml.safe_dump(values), encoding='utf-8')
    return path


def test_missing_only_defaults_cli_precedence_and_no_mutation(tmp_path):
    values = {'model': {'initialization_adapter': None, 'adapter_load_mode': 'fresh'},
              'training': {'learning_rate': 0.003, 'max_grad_norm': None},
              'data': {'annotation_frame': None}, 'runtime': {'device': 'cpu'}}
    config = load_config(write_yaml(tmp_path, values), {'training': {'learning_rate': 0.007}})
    assert config['training']['learning_rate'] == 0.007
    assert config['training']['max_grad_norm'] is None
    assert config['data']['annotation_frame'] is None
    assert config['model']['initialization_adapter'] is None
    assert config['training']['gradient_accumulation_steps'] == 1
    before = copy.deepcopy(config)
    resolve_config(config)
    assert config == before
    values['training']['adam_beta1'] = None
    with pytest.raises(ValueError, match='adam_beta1'):
        load_config(write_yaml(tmp_path, values))


@pytest.mark.parametrize('precision', ['fp32', 'bf16'])
def test_yaml_fresh_model_lora_token_context_reaches_real_peft(tmp_path, episode, monkeypatch, precision):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor, base = tiny_bundle(tmp_path, episode, bare=True)
    observed = []
    monkeypatch.setattr(AutoProcessor, 'from_pretrained', lambda name: observed.append(('processor', name)) or processor)
    def model_loader(name, **kwargs):
        observed.append(('model', name, kwargs['dtype']))
        return base.to(kwargs['dtype'])
    monkeypatch.setattr(AutoModelForImageTextToText, 'from_pretrained', model_loader)
    path = write_yaml(tmp_path, {'runtime': {'device': 'cpu', 'output_dir': str(tmp_path / 'run')},
        'model': {'base_model': 'local/alternate-internvl', 'initialization_adapter': None,
                  'adapter_load_mode': 'fresh', 'lora': {'rank': 4, 'alpha': 12, 'dropout': 0.0,
                                                       'target_modules': ['q_proj', 'v_proj']}},
        'where': {'end_fix_token': '<FIX_DONE>', 'context': {'max_total_sequence_length': 4096}},
        'training': {'precision': precision, 'gradient_checkpointing': True}})
    bundle = build_flat_model_bundle(path)
    assert observed == [('processor', 'local/alternate-internvl'),
                        ('model', 'local/alternate-internvl', {'fp32': torch.float32, 'bf16': torch.bfloat16}[precision])]
    peft = bundle.model.peft_config['default']
    assert (peft.r, peft.lora_alpha, peft.lora_dropout) == (4, 12, 0.0)
    assert all('.language_model.' in target for target in peft.target_modules)
    assert {t.split('.')[-1] for t in peft.target_modules} == {'q_proj', 'v_proj'}
    assert bundle.context_limit == 4096
    assert bundle.processor.tokenizer.encode('<FIX_DONE>', add_special_tokens=False) == [bundle.end_fix_id]
    result = run_flat_training_step(bundle, episode)
    assert result['projector_grad_nonzero'] and result['query_end_fix_state_count'] == 4
    assert json.loads((tmp_path / 'run/resolved_config.json').read_text()) == bundle.config


def test_continued_noncanonical_adapter_loads_exact_weights(tmp_path, episode, monkeypatch):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from semgaze.model.build import verify_loaded_adapter
    from peft import LoraConfig, get_peft_model
    _, source_base = tiny_bundle(tmp_path, episode, bare=True)
    targets = [name for name, _ in source_base.named_modules() if '.language_model.' in name and name.endswith('.q_proj')]
    source = get_peft_model(source_base, LoraConfig(r=4, lora_alpha=11, lora_dropout=0.2, target_modules=targets, task_type='CAUSAL_LM'))
    adapter = tmp_path / 'my-adapter'
    source.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
    processor, base = tiny_bundle(tmp_path, episode, bare=True)
    monkeypatch.setattr(AutoProcessor, 'from_pretrained', lambda *a, **kw: processor)
    monkeypatch.setattr(AutoModelForImageTextToText, 'from_pretrained', lambda *a, **kw: base)
    path = write_yaml(tmp_path, {'runtime': {'device': 'cpu'}, 'training': {'precision': 'fp32'},
        'model': {'base_model': 'local/base', 'initialization_adapter': str(adapter),
                  'adapter_load_mode': 'continue_trainable', 'lora': {'rank': 3, 'alpha': 7}}})
    bundle = build_flat_model_bundle(path, output_dir=tmp_path / 'run')
    verify_loaded_adapter(bundle.model, adapter)
    assert bundle.model.peft_config['default'].r == 4  # checkpoint architecture, not fresh knobs
    assert bundle.config['model']['lora']['rank'] == 3  # never silently overwrite the requested config
    assert bundle.diagnostics['effective_lora']['r'] == 4
    assert bundle.diagnostics['adapter_source'] == adapter.as_posix()


@pytest.mark.parametrize('optimizer,scheduler', [('AdamW','linear'), ('Adam','constant'), ('SGD','constant_with_warmup')])
def test_optimizer_and_scheduler_consume_yaml(tmp_path, episode, optimizer, scheduler):
    bundle = tiny_bundle(tmp_path, episode)
    bundle.config = load_config(write_yaml(tmp_path, {'training': {'optimizer': optimizer,
        'scheduler': scheduler, 'learning_rate': 0.004, 'weight_decay': 0.13,
        'adam_beta1': 0.71, 'adam_beta2': 0.81, 'adam_epsilon': 0.00001, 'warmup_ratio': 0.25}}))
    bundle.optimizer = make_optimizer(bundle)
    assert type(bundle.optimizer).__name__ == optimizer
    group = bundle.optimizer.param_groups[0]
    assert group['lr'] == 0.004 and group['weight_decay'] == 0.13
    if optimizer != 'SGD':
        assert group['betas'] == (0.71,0.81) and group['eps'] == 0.00001
    schedule = make_scheduler(bundle, 8)
    assert schedule.lr_lambdas[0](0) == (1 if scheduler == 'constant' else 0)
    assert schedule.lr_lambdas[0](8) == (0 if scheduler == 'linear' else 1)


@pytest.mark.parametrize('data', [
    {'unseen_subjects': [2]},
    {'fewshot': {'k_values': [1,5,10], 'train_k_probabilities': [0.3,0.3,0.4]}},
    {'split_root': 'data/COCO_Search18/split/all'},
])
def test_obsolete_protocol_overrides_rejected(tmp_path, data):
    with pytest.raises(ValueError):
        load_config(write_yaml(tmp_path, {'data': data}))


def test_evaluation_subset_multiple_prediction_batches_and_no_test_predictions(tmp_path, episode, monkeypatch):
    import semgaze.evaluation.test as validation
    import semgaze.evaluation.predictions as predictions
    bundle = tiny_bundle(tmp_path, episode)
    bundle.config = load_config(write_yaml(tmp_path, {'evaluation': {
        'predictions': {'train_batches':2, 'test_scope':'none', 'semantic_max_new_tokens':7}}}))
    train, queries, manifest = evaluation_data(episode)
    seen = []
    def losses(bundle, ep):
        seen.append(len(ep.supports))
        return dict.fromkeys(EVAL_KEYS, 2.0)
    monkeypatch.setattr(validation, 'batch_losses', lambda bundle, episodes, **kw: [losses(bundle,e) for e in episodes])
    assert evaluate_test_epoch(bundle, train, queries, manifest)['test_episodes'] == 30
    assert seen == [1]*10 + [5]*10 + [10]*10
    bundle.config['test']['prediction']['execution'] = 'serial'
    monkeypatch.setattr(predictions, 'evaluate_where_episode', lambda *a: {'text':'where'})
    budgets = []
    monkeypatch.setattr(predictions, 'evaluate_flat_episode', lambda *a, generation_budget: budgets.append(generation_budget) or {'text':'flat'})
    result = predict_epoch(bundle, [episode,episode], train, queries, manifest, epoch=1, step=2, split_manifest_identity='fixture')
    assert result['train_prediction_batches'] == 2 and result['train_prediction_episodes'] == 2
    assert result['test_prediction_episodes'] == 0 and budgets == [7,7]


def test_batch_accumulation_step_test_logging_and_checkpoint_controls(tmp_path, episode, monkeypatch):
    import semgaze.training.loop as loop
    bundle = tiny_bundle(tmp_path, episode)
    bundle.config = load_config(write_yaml(tmp_path, {'training': {'num_train_epochs':1,
        'per_device_train_batch_size':2,'gradient_accumulation_steps':3,'max_grad_norm':None},
        'evaluation': {'strategy':'epoch', 'predictions':{'train_batches':0,'test_scope':'none'}},
        'logging': {'every_steps':2,'filename':'custom/log.jsonl'},
        'checkpoint': {'save_every':None,'save_at_epoch_end':False,'save_at_end':False}}))
    bundle.optimizer = make_optimizer(bundle)
    bundle.scheduler = make_scheduler(bundle, 3)
    calls, evaluations = [], []
    monkeypatch.setattr(loop,'run_flat_training_step',lambda *a,**kw:calls.append(kw) or {'loss_total':1.0})
    monkeypatch.setattr(loop,'evaluate_test_epoch',lambda *a:evaluations.append(1) or dict.fromkeys(EVAL_KEYS,1.0) | {'test_queries':1,'test_episodes':1})
    monkeypatch.setattr(loop,'save_checkpoint',lambda *a,**kw:pytest.fail('saving was disabled'))
    run_training_loop(bundle,FixedSampler(episode, count=18),{},[],{},split_manifest_identity='fixture',max_steps=3,save_every=None)
    assert len(calls) == 9 and all(c['loss_scale'] == 1/3 for c in calls)
    assert all(len(c['where_batch'].episodes) == 2 for c in calls)
    assert [i for i, c in enumerate(calls) if c['diagnostics']] == [2]
    assert sum(c['zero_grad'] for c in calls) == 3 and len(evaluations) == 1
    entries = [json.loads(line) for line in (tmp_path/'custom/log.jsonl').read_text().splitlines()]
    assert [e['step'] for e in entries if e['event']=='train_step'] == [2,3]


def test_selective_checkpoint_and_resume_does_not_override_optimizer_choices(tmp_path, episode):
    bundle = tiny_bundle(tmp_path,episode)
    bundle.optimizer = make_optimizer(bundle)
    bundle.scheduler = make_scheduler(bundle,10)
    run_flat_training_step(bundle,episode,optimizer_step=True)
    path=tmp_path/'complete'
    save_checkpoint(bundle,path,split_manifest_identity='fixture',step=1)
    bundle.config['training'].update(learning_rate=0.003,weight_decay=0.22,adam_beta1=0.61)
    bundle.optimizer=make_optimizer(bundle)
    bundle.scheduler=make_scheduler(bundle,20)
    restore_checkpoint_state(bundle,path,split_manifest_identity='fixture',resume_optimizer=True,restore_hyperparameters=False)
    group=bundle.optimizer.param_groups[0]
    assert group['lr']==0.003 and group['weight_decay']==0.22 and group['betas'][0]==0.61
    bundle.config['checkpoint'].update(save_processor_and_tokenizer=False,save_peft_adapter=False,
        save_trainable_end_fix_rows=False,save_projector=False,save_optimizer=False,save_scheduler=False)
    partial=tmp_path/'partial'
    save_checkpoint(bundle,partial,split_manifest_identity='fixture',step=1)
    assert not (partial/'processor').exists() and not (partial/'adapter').exists()
    assert not (partial/'semgaze.safetensors').exists()
    state=torch.load(partial/'training.pt',weights_only=True)
    assert state['optimizer'] is None and state['scheduler'] is None
    with pytest.raises(ValueError,match='incomplete'):
        load_checkpoint_bundle(partial,split_manifest_identity='fixture')

@pytest.mark.parametrize('cli_max_steps,expected', [(None,6),(5,5)])
def test_train_entrypoint_passes_one_resolved_config_to_consumers(tmp_path, episode, monkeypatch, cli_max_steps, expected):
    import sys
    import train_flat
    output=tmp_path/'output'
    fixture=tmp_path/'custom-fixture.json'
    fixture.write_bytes((ROOT/'tests/fixtures/flat_episode.json').read_bytes())
    path=write_yaml(tmp_path, {'runtime':{'device':'cpu','output_dir':str(output)},
        'data':{'split_root':str(tmp_path/'split_95_5/all'),'images_root':str(tmp_path/'images')},
        'training':{'num_train_epochs':3,'per_device_train_batch_size':6,'optimizer':'SGD','learning_rate':0.02,
                    'scheduler':'constant','precision':'fp32'},
        'evaluation':{'strategy':'no'}, 'checkpoint':{'save_every':7},
        'smoke':{'fixture':str(fixture)}})
    records=[replace(episode.query, record_id=f'r{i}',stimulus_id=f'i{i}') for i in range(12)]
    calls={}
    def splits(root, *, data_config):
        calls['split']=(root,data_config)
        return {'train':[{'subject':1,'record':r} for r in records], 'test':[]},{},'identity'
    def adapter(root, **kwargs):
        calls['images']=(root,kwargs)
        return lambda raw:raw['record']
    bundle=tiny_bundle(tmp_path,episode)
    def build(*,config,output_dir):
        calls['build']=copy.deepcopy(config)
        bundle.config=config
        bundle.output_dir=output_dir
        return bundle
    monkeypatch.setattr(train_flat,'read_persisted_splits',splits)
    monkeypatch.setattr(train_flat,'CocoSearch18Adapter',adapter)
    monkeypatch.setattr(train_flat,'build_flat_model_bundle',build)
    monkeypatch.setattr(train_flat,'run_flat_training_step',lambda *a:{'loss_total':torch.tensor(1.)})
    monkeypatch.setattr(train_flat,'run_training_loop',lambda b,s,*a,**kw:calls.update(loop=kw, sampled_k=len(s.sample().supports),lr=b.optimizer.param_groups[0]['lr']))
    argv=['train_flat.py','--config',str(path)]
    if cli_max_steps:
        argv += ['--max-steps',str(cli_max_steps)]
    monkeypatch.setattr(sys,'argv',argv)
    train_flat.main()
    assert calls['build']['training']['optimizer_updates_per_epoch']==2
    assert calls['build']['training']['total_optimizer_updates']==expected
    assert calls['loop']['max_steps']==expected and calls['loop']['save_every']==7
    assert 1 <= calls['sampled_k'] <= 10 and calls['lr']==0.02
    assert calls['split'][0]==tmp_path/'split_95_5/all' and calls['images'][0]==tmp_path/'images'
    assert json.loads((output/'resolved_config.json').read_text())==bundle.config


def test_relocated_dataset_duration_field_and_integrity(tmp_path):
    import shutil
    from PIL import Image
    from semgaze.data.cocosearch18 import read_persisted_splits, CocoSearch18Adapter
    source = ROOT/'data/COCO_Search18/split_95_5'
    root = tmp_path/'split_95_5/all'
    root.mkdir(parents=True)
    for name in ('master_split_manifest.json', 'split_index.json'):
        shutil.copyfile(source/name, root.parent/name)
    for name in ('train.json', 'test.json', 'split_manifest.json'):
        shutil.copyfile(source/'all'/name, root/name)
    raw, manifest, identity = read_persisted_splits(root)
    record = dict(raw['train'][0])
    record['dwell_ms'] = record.pop('T')
    images = tmp_path/'images'
    images.mkdir()
    Image.new('RGB', (80,40)).save(images/record['name'])
    adapter = CocoSearch18Adapter(images, annotation_frame={'width':160,'height':80}, duration_field='dwell_ms')
    normalized = adapter(record)
    assert normalized.duration_ms == tuple(record['dwell_ms']) and normalized.image_width == 160
    (root/'test.json').write_text('[]')
    with pytest.raises(ValueError, match='checksum'):
        read_persisted_splits(root)


def test_evaluate_entrypoint_inherits_checkpoint_and_honors_overrides(tmp_path, episode, monkeypatch):
    import sys
    import evaluate_flat
    train, queries, manifest = evaluation_data(episode)
    checkpoint=tmp_path/'checkpoint'
    checkpoint.mkdir()
    config=resolve_config({'runtime':{'device':'cpu'},'evaluation':{'predictions':{'semantic_max_new_tokens':4}}})
    (checkpoint/'resolved_config.json').write_text(json.dumps(config))
    output=tmp_path/'evaluation'
    override=write_yaml(tmp_path,{'runtime':{'output_dir':str(output),'device':'cpu'},
                                  'evaluation':{'predictions':{'semantic_max_new_tokens':7}}})
    raw={'train':[{'record_id':r.record_id,'subject':r.subject,'record':r} for r in train.values()],
         'test':[{'record_id':r.record_id,'subject':r.subject,'record':r} for r in queries]}
    monkeypatch.setattr(evaluate_flat,'read_persisted_splits',lambda *a,**kw:(raw,manifest,'identity'))
    monkeypatch.setattr(evaluate_flat,'CocoSearch18Adapter',lambda *a,**kw:lambda r:r['record'])
    calls=[]
    bundle=tiny_bundle(tmp_path,episode)
    monkeypatch.setattr(evaluate_flat,'load_checkpoint_bundle',lambda *a,**kw:calls.append(kw) or bundle)
    budgets=[]
    def predicted(bundle, episodes, *, budget, cache, path):
        assert path == 'both'
        for ep in episodes:
            budgets.append(budget)
            yield ep, ({'under_generated':False}, {'flat_format_valid':True})
    monkeypatch.setattr(evaluate_flat,'prediction_batches',predicted)
    monkeypatch.setattr(sys,'argv',['evaluate_flat.py','--checkpoint',str(checkpoint),'--config',str(override),
        '--split','test','--path','both','--semantic-max-new-tokens','9'])
    evaluate_flat.main()
    rows=[json.loads(line) for line in (output/'predictions.jsonl').read_text().splitlines()]
    assert len(rows)==30 and rows[0]['k']==1 and len(rows[0]['support_ids'])==1
    assert budgets==[9]*30 and calls[0]['runtime']['device']=='cpu'
    resolved=json.loads((output/'resolved_config.json').read_text())
    assert resolved['evaluation']['k_values']==[1,5,10]
    assert resolved['evaluation']['predictions']['semantic_max_new_tokens']==9


def test_cache_setting_reaches_where_forward(tmp_path, episode, monkeypatch):
    from semgaze.where.forward import forward_where
    bundle=tiny_bundle(tmp_path,episode)
    bundle.config=resolve_config(bundle.config,{'where':{'supervision':{'use_cache':True}},
                                               'training':{'gradient_checkpointing':False}})
    seen=[]
    hook=bundle.model.get_base_model().model.register_forward_pre_hook(lambda m,a,kw:seen.append(kw['use_cache']),with_kwargs=True)
    try:
        forward_where(bundle,episode)
    finally:
        hook.remove()
    assert seen==[True]
    with pytest.raises(ValueError,match='checkpointing disables KV'):
        resolve_config(bundle.config,{'training':{'gradient_checkpointing':True}})
