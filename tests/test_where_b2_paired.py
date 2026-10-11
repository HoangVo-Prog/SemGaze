"""CPU-only selection and paired comparison contract tests."""
import importlib.util
from pathlib import Path
from unittest.mock import patch
import unittest

P = Path(__file__).resolve().parents[1] / 'tools' / 'where_b2_paired.py'
s = importlib.util.spec_from_file_location('paired', P)
paired = importlib.util.module_from_spec(s)
s.loader.exec_module(paired)


def fake_row(sub, j, category):
    return {'query_id': f'coco_semgaze::{sub}::car::{j:012d}.jpg', 'subject':sub,
            'support_ids':[f'{sub}:{j}:support:{i}' for i in range(10)],
            'where_generation':{'requested_fixation_count':(3 if j%2==0 else 4)}}


def data():
    rows=[]
    for sub in (7,8,9):
        for j in range(35):rows.append(fake_row(sub,j,'valid'))
        for j in range(35,47):rows.append(fake_row(sub,j,'prefix_only'))
    return rows


def category(row, **kw):
    return 'prefix_only' if int(row['query_id'].split('::')[-1].split('.')[0]) >= 35 else 'valid'


class SelectorTests(unittest.TestCase):
    def test_audit16_fixed_stratification(self):
        with patch.object(paired, 'classify', category):
            rows=paired.make_selection(data(),mode='audit16')
            self.assertEqual(len(rows),16)
            self.assertEqual([c for _,c in rows].count('prefix_only'),12)
            self.assertEqual([c for _,c in rows].count('valid'),4)
            self.assertEqual(set(int(r['subject']) for r,c in rows),{8})

    def test_balanced_90_excludes_audit_16_and_is_deterministic(self):
        with patch.object(paired, 'classify', category):
            audit=paired.make_selection(data(),mode='audit16')
            excluded={r['query_id'] for r,_ in audit}
            one=paired.make_selection(data(),mode='balanced90',excluded=excluded)
            two=paired.make_selection(data(),mode='balanced90',excluded=excluded)
            self.assertEqual([r['query_id'] for r,_ in one],[r['query_id'] for r,_ in two])
            self.assertEqual(len(one),90)
            self.assertFalse(set(r['query_id'] for r,_ in one)&excluded)
            for subject in (7,8,9):
                rows=[c for r,c in one if r['subject']==subject]
                self.assertEqual(len(rows),30)
                self.assertTrue('valid' in rows)

    def test_pair_contract_failure_on_support_change(self):
        base={'query_id':'q','support_ids':['s']*10,'K':10,'draw_id':0,
            'requested_fixation_count':3,'checkpoint':'ckpt','split_manifest_identity':'sha',
            'generation':{'do_sample':False},'canonical_format_valid':False,
            'historical_prefix_bad':True,'starts_with_open':False,'accepted_fixation_count':2,
            'under_generated':True,'fixations':[[1,2,3]],'prompt_token_length':5000,
            'subject':8}
        import copy
        concise=copy.deepcopy(base)
        concise['support_ids'][0]='other'
        with self.assertRaisesRegex(ValueError, 'support'):
            paired.compare_pair(concise,base)
        concise=copy.deepcopy(base)
        concise.update(canonical_format_valid=True,starts_with_open=True,
                       under_generated=False,accepted_fixation_count=3,prompt_token_length=4000)
        row=paired.compare_pair(concise,base)
        self.assertTrue(row['fixed'])
        self.assertFalse(row['broken'])

if __name__=='__main__':unittest.main()
