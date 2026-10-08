"""Dependency-free targeted smoke of the new joint reader logic."""
import importlib.util, sys, types, tempfile, json
from pathlib import Path
root=Path(__file__).parent
for name in ['semgaze','semgaze.data']:
    mod=types.ModuleType(name); mod.__path__=[]; sys.modules[name]=mod
modules={'semgaze.data.json_splits': {'read_current_json_splits': lambda *x: None,
                                      '_repo_path': lambda x: Path(x),
                                      '_read_json': lambda x: json.loads(Path(x).read_text()),
                                      '_sha256': lambda p: 'testsha'},
 'semgaze.data.cocosearch18': {'CocoSearch18Adapter': object},
 'semgaze.data.schema': {'NormalizedRecord': object},
 'semgaze.data.semantic': {'semantic_from_prediction': lambda *x: None}}
for name, ns in modules.items():
    m=types.ModuleType(name); m.__dict__.update(ns); sys.modules[name]=m
spec=importlib.util.spec_from_file_location('semgaze.data.joint', root/'semgaze/data/joint.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
assert module.qualified_subject('AiR','7')=='AiR::7'
assert module.qualified_stimulus('AiR','x')=='AiR::x'
assert module.unseen_subject_ids({'dataset':'all','unseen_subjects':[7,8,9],'air_unseen_subjects':['JY']})=={7,8,9,'AiR::JY'}
with tempfile.TemporaryDirectory() as d:
 p=Path(d); (p/'split_manifest.json').write_text('{}');(p/'air.json').write_text('{}')
 coco_m={'unseen_subject_ids':[7,8,9],'train_stimulus_ids':['c'],'test_stimulus_ids':['ct'], 'support_draws':{'1':[],'5':[],'10':[]}}
 air_m={'unseen_subject_ids':['JY'],'train_stimulus_ids':['a'],'test_stimulus_ids':['at'],'support_draws':{}}
 module.read_current_json_splits=lambda *a:({'train':[{'dataset':'COCO-Search18','record_id':'c1'}],'test':[{'dataset':'COCO-Search18','record_id':'ct'}]},coco_m,'coco_hash')
 module._verify_air=lambda *a:({'train':[{'dataset':'AiR','record_id':'a1'}],'test':[{'dataset':'AiR','record_id':'at'}]},air_m,p/'air.json')
 combined={'dataset':'all','resplit_after_merge':False,'source_manifests':{'AiR':str(p/'air.json'),'COCO-Search18':'ignored'},'record_counts':{'train':2,'test':2},'per_dataset_record_counts':{'AiR':{'train':1,'test':1},'COCO-Search18':{'train':1,'test':1}},'support_sampling':{'per_dataset_support_draws':{'AiR':{}}}}
 raw,man,identity=module.read_joint_splits(p,combined,{'dataset':'all'})
 assert len(raw['train'])==2 and len(raw['test'])==2
 assert {v['dataset'] for v in raw['train']}=={'AiR','COCO-Search18'}
 assert 'AiR::JY' in man['unseen_subject_ids']
 assert 'AiR::at' in man['test_stimulus_ids']
print('PASS: new joint reader identity, union coverage, unseen subjects, stimulus namespaces')
