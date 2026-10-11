"""CPU-only unit tests: execute inside the SemGaze repo with: python -m unittest discover -s tests -p 'test_audit_where_k10_root_cause.py'"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import types
import unittest
from unittest.mock import patch
import torch

SCRIPT = Path(__file__).resolve().parents[1]/'tools/audit_where_k10_root_cause.py'
spec = importlib.util.spec_from_file_location('where_ablation', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class Record:
    def __init__(self, index, n=3):
        self.record_id=f'r{index}'
        self.subject=8
        self.dataset='COCO-Search18'
        self.task=f'task{index}'
        self.x_px=tuple(range(n))
        self.image_path=f'image{index}.jpg'
        self.stimulus_id=self.image_path


class Episode:
    def __init__(self, supports, query, draw_id=0):
        self.supports=tuple(supports)
        self.query=query
        self.draw_id=draw_id
        assert len(self.supports)>0
        assert len(set(x.stimulus_id for x in self.supports))==len(self.supports)


def schema_patch():
    m=types.ModuleType('semgaze.data.schema');m.FlatEpisode=Episode
    return patch.dict(sys.modules, {'semgaze':types.ModuleType('semgaze'),
         'semgaze.data':types.ModuleType('semgaze.data'),'semgaze.data.schema':m})


def mk_episode():
    return Episode([Record(i) for i in range(10)], Record(100))


def conversation(e):
    msgs=[]
    for s in e.supports:
        msgs.extend([{'role':'user','content':[{'type':'image'}, {'type':'text','text':f'{audit.INSTRUCTION} {s.task}'}]},
                     {'role':'assistant','content':f'GT-{s.record_id}'}])
    msgs.append({'role':'user','content':[{'type':'image'}, {'type':'text','text':'QUERY EXACT'}]})
    return msgs


class TinyTokenizer:
    eos_token_id=4
    pad_token_id=0
    def encode(self, text, add_special_tokens=False):
        return [ord(x)+100 for x in text]
    def decode(self, ids, skip_special_tokens=False,clean_up_tokenization_spaces=False):
        return ''.join(chr(i-100) for i in ids)
    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):return chr(ids-100)
        return [chr(x-100) for x in ids]


class TinyProcessor:
    def __init__(self):
        self.tokenizer=TinyTokenizer()
        self.image_token_id=9999
        self.image_seq_length=1
        self.last_text=None
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):
        return '<prompt>' + '|'.join(m['content'][1]['text'] if isinstance(m['content'],list) else m['content'] for m in messages) + '<assistant>'
    def __call__(self, *, text, images,return_tensors,crop_to_patches,size,add_special_tokens):
        self.last_text=text[0]
        ids=self.tokenizer.encode(text[0]); ids += [9999 for _ in images]
        # In production, expanded image tokens occur before the response; the tiny
        # processor puts them first, then makes the output end with the prefill.
        ids=[9999 for _ in images]+self.tokenizer.encode(text[0])
        return {'input_ids':torch.tensor([ids]),'attention_mask':torch.ones(1,len(ids),dtype=torch.long),
                'pixel_values':torch.zeros(len(images),3,448,448)}


class TransformTests(unittest.TestCase):
    def test_nested_identity_and_order(self):
        with schema_patch():
            e=mk_episode(); groups=audit.nested_supports(e)
            self.assertEqual(audit.support_ids(groups['A1_first1']),['r0'])
            self.assertEqual(audit.support_ids(groups['A2_last1']),['r9'])
            self.assertEqual(audit.support_ids(groups['A3_first5']),[f'r{i}' for i in range(5)])
            self.assertEqual(audit.support_ids(groups['A4_last5']),[f'r{i}' for i in range(5,10)])
            self.assertEqual(audit.support_ids(groups['A5_original10']),audit.support_ids(e))

    def test_order_permutations_deterministic_and_paired(self):
        with schema_patch():
            e=mk_episode()
            for name in ('C_reverse','C_perm1','C_perm2'):
                x=audit.reordered_supports(e,name)
                self.assertEqual(audit.support_ids(x),audit.support_ids(audit.reordered_supports(e,name)))
                self.assertCountEqual(audit.support_ids(x),audit.support_ids(e))
                self.assertNotEqual(audit.support_ids(x),audit.support_ids(e))
                self.assertEqual(x.query,e.query)
            self.assertEqual(audit.support_ids(audit.reordered_supports(e,'C_reverse')),
                             list(reversed(audit.support_ids(e))))

    def test_prompt_changes_only_support_user_text(self):
        e=mk_episode(); msgs=conversation(e)
        for variant in ('B1_original','B2_concise','B3_minimal'):
            new=audit.transform_messages(msgs,e,variant)
            self.assertEqual(len(new),len(msgs))
            self.assertEqual(new[-1],msgs[-1])
            for idx in range(10):
                self.assertEqual(new[2*idx+1],msgs[2*idx+1])
                self.assertEqual(new[2*idx]['content'][0],{'type':'image'})
                self.assertEqual(new[2*idx]['role'],'user')
                if variant!='B1_original':
                    self.assertNotEqual(new[2*idx]['content'][1]['text'],msgs[2*idx]['content'][1]['text'])
            self.assertEqual(msgs[0]['content'][1]['text'],f'{audit.INSTRUCTION} task0')

    def test_prefill_is_supplied_before_model_generation(self):
        processor=TinyProcessor()
        e=mk_episode()
        msgs=conversation(e)
        m=types.ModuleType('semgaze.where.collator')
        m.load_images=lambda paths:[None for _ in paths]
        with patch.dict(sys.modules,{'semgaze.where':types.ModuleType('semgaze.where'),
                                     'semgaze.where.collator':m}):
            encoded,rendered,text=audit.native_prefill(processor,msgs,[x.image_path for x in e.supports]+[e.query.image_path],'[(')
        self.assertTrue(processor.last_text.endswith('[('))
        self.assertTrue(rendered.endswith('[('))
        self.assertEqual(text,processor.last_text)
        self.assertEqual(int((encoded['input_ids']==processor.image_token_id).sum()),11)

    def test_prefill_not_posthoc_repair_and_raw_continuation(self):
        # Explicitly ensure the generation interface sees prefilled IDs.
        processor=TinyProcessor()
        e=mk_episode()
        mcollate=types.ModuleType('semgaze.where.collator')
        mcollate.load_images=lambda paths:[None for _ in paths]
        mcollate.to_model_device=lambda inputs,model:dict(inputs)
        mgen=types.ModuleType('semgaze.where.generation')
        mgen.where_generation_budget=lambda bundle,ep:64
        mgen.decode_response=lambda tok,ids:tok.decode(ids)
        mserial=types.ModuleType('semgaze.where.serialization')
        mserial.parse_xyd_output=lambda text,n,tok:{'fixations':[(32,50,220)] if text.startswith('[(') else [],
                             'accepted_fixation_count':int(text.startswith('[(')),
                             'requested_fixation_count':n,'under_generated':True,
                             'canonical_format_valid':text.startswith('[(')}
        class Model:
            def generate(self,**kwargs):
                self.last_prompt=kwargs['input_ids'][0].tolist()
                return torch.tensor([self.last_prompt+processor.tokenizer.encode('32, 50, 220)')])
        model=Model()
        bundle=SimpleNamespace(processor=processor,model=model,config={'where':{'end_fix_token':'<END_FIX>'}},context_limit=8192)
        msgs=conversation(e)
        with patch.dict(sys.modules,{'semgaze.where':types.ModuleType('semgaze.where'),
                 'semgaze.where.collator':mcollate,'semgaze.where.generation':mgen,
                 'semgaze.where.serialization':mserial}):
            ins,rendered,full=audit.native_prefill(processor,msgs,[x.image_path for x in e.supports]+[e.query.image_path],'[(')
            prepared={'inputs':ins,'rendered':rendered,'native_text':full,'messages':msgs,'prefill':'[('}
            from tempfile import TemporaryDirectory
            with TemporaryDirectory() as tmp:
                obs=audit.create_observation(bundle,e,'D2_prefill',prepared,Path(tmp))
        self.assertEqual(obs['raw_generated_text'],'32, 50, 220)')
        self.assertTrue(obs['response'].startswith('[('))
        self.assertFalse(obs['continuation_starts_with_open'])
        self.assertTrue(obs['prefill_applied_before_generation'])
        self.assertEqual(processor.tokenizer.decode(model.last_prompt[-2:]),'[(')
        self.assertEqual(obs['raw_generated_ids'],processor.tokenizer.encode('32, 50, 220)'))


if __name__=='__main__':
    unittest.main()
