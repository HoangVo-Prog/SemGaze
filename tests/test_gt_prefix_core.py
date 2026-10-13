"""CPU-only smoke tests: `python -m unittest discover -s tests -p test_gt_prefix_core.py`."""
import unittest
from dataclasses import dataclass
import torch
from gt_prefix_core import native_gt_prefix, suffix_diagnostics, compare_suffix


@dataclass
class FakeNative:
    inputs: dict
    response_start: int
    response_length: int
    response_offsets: tuple


class Tokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, **_):
        return ''.join(self.pieces[i] for i in ids)


class PrefixTest(unittest.TestCase):
    def test_prefix_slices_exactly_at_endfix_and_excludes_suffix(self):
        target = ('[(51, 46, 221)<END_FIX>, (38, 28, 754)<END_FIX>, '
                  '(70, 65, 300)<END_FIX>]')
        # In the real tokenizer END_FIX has the SAME token ID at each occurrence.
        # Simulate this by making the two latter IDs equal to the first END_FIX.
        # Substitute END_FIX token IDs with the shared ID 3 and reassign other IDs.
        tokens = [0,1,2,3,4,5,3,4,8,3,10,11]
        ids=torch.tensor([tokens]); labels=ids.clone();labels[0,0]=-100
        expected=FakeNative({'input_ids':ids,'labels':labels,
              'attention_mask':torch.ones_like(ids),'pixel_values':torch.randn(2,3,3)},
              response_start=1,response_length=11,response_offsets=())
        class CorrectTokenizer:
            pieces={0:'<prompt>',1:'[',2:'(51, 46, 221)',3:'<END_FIX>',4:', ',
                    5:'(38, 28, 754)',8:'(70, 65, 300)',10:']',11:'<eos>'}
            def decode(self, ids, **_):return ''.join(self.pieces[i] for i in ids)
        truncated, forced, count = native_gt_prefix(expected,prefix_count=2,
              end_fix_id=3,tokenizer=CorrectTokenizer(),target=target,
              end_fix_token='<END_FIX>')
        self.assertEqual(count,6)
        self.assertEqual(truncated.inputs['input_ids'].tolist()[0], tokens[:7])
        self.assertEqual(truncated.inputs['labels'].tolist()[0],[-100]*7)
        self.assertEqual(forced,'[(51, 46, 221)<END_FIX>, (38, 28, 754)<END_FIX>')

    def test_only_generated_suffix_is_scored(self):
        gt=[(10,10,100),(20,20,100),(30,30,100),(40,40,100),(50,50,100)]
        free=[(11,11,100),(19,19,100),(20,20,100),(20,20,100),(20,20,100)]
        pref=[(10,10,100),(20,20,100),(30,30,100),(40,40,100),(50,50,100)]
        paired=compare_suffix(gt, free, pref, prefix=2)
        self.assertEqual(paired['paired_positions'],3)
        self.assertEqual(paired['prefix_xy_error'],0)
        self.assertLess(paired['delta_xy_error'],0)
        # suffix consists of three exact repeated points in free, never in pref
        self.assertTrue(suffix_diagnostics(gt,free,2)['collapse3'])
        self.assertFalse(suffix_diagnostics(gt,pref,2)['collapse3'])

    def test_partial_suffix_does_not_fake_success(self):
        gt=[(1,1,100),(2,2,100),(3,3,100),(4,4,100)]
        prefix=[(1,1,100),(2,2,100)]
        diag=suffix_diagnostics(gt,prefix,1)
        self.assertFalse(diag['full_suffix'])
        self.assertEqual(diag['generated_suffix_count'],1)

    def test_invalid_prefix_rejected(self):
        ids=torch.tensor([[0,1,2,3]])
        s=FakeNative({'input_ids':ids,'labels':ids,'attention_mask':torch.ones_like(ids)},0,4,())
        with self.assertRaises(ValueError):
            native_gt_prefix(s,prefix_count=2,end_fix_id=3,tokenizer=Tokenizer([]),
                target='[(1,2,3)<END_FIX>]',end_fix_token='<END_FIX>')


if __name__=='__main__':
    unittest.main()
