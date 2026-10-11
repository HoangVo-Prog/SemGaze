"""Synthetic checks; NOT evidence of real InternVL generation correctness."""
import importlib.util
from pathlib import Path
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / 'tools' / 'audit_where_prefix_runtime.py'
spec = importlib.util.spec_from_file_location('audit_where_prefix_runtime', SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def decode(self, ids, **kwargs):
        return ''.join(self.convert_ids_to_tokens(ids))

    def convert_ids_to_tokens(self, ids):
        return [{0:'_', 1:'P', 2:'Q', 3:'[', 4:'(', 5:'3', 6:'2', 7:',', 8:' ',
                 9:'5', 10:'0', 11:')', 99:'<eos>'}[i] for i in ids]


def test_full_echo_good_slice():
    t = ToyTokenizer()
    r = audit.inspect_one(t, [0, 1, 2], [0, 1, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 99], budget=50)
    assert r['sequence_structure'] == 'FULL_PADDED_PROMPT_ECHO'
    assert r['prefix_width_L'] == 3
    assert r['production_decoded_text'] == '[(32, 50)'
    assert r['first_prompt_mismatch'] is None
    assert r['production_starts_with_open']


def test_raw_never_had_opening():
    t = ToyTokenizer()
    r = audit.inspect_one(t, [0, 1, 2], [0, 1, 1], [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 99], budget=50)
    assert r['full_prompt_prefix_matches']
    assert r['production_decoded_text'] == '32, 50)'
    assert not r['production_starts_with_open']


def test_wrong_offset_detected_without_falsely_fixing():
    t = ToyTokenizer()
    # This synthetic generator echoes only the non-pad input, rather than L=4.
    r = audit.inspect_one(t, [0, 0, 1, 2], [0, 0, 1, 1], [1, 2, 3, 4, 5, 6, 99], budget=50)
    assert not r['full_prompt_prefix_matches']
    assert r['sequence_structure'] == 'UNPADDED_PROMPT_ECHO'
    assert r['candidate_continuation_offset'] == 2
    assert r['candidate_decoded_text'].startswith('[(')
    assert r['production_decoded_text'] == '32'


@pytest.mark.parametrize('mask,ids', [([1, 0, 1], [1, 0, 2]),
                                      ([0, 1, 1], [1, 1, 2]),
                                      ([0, 0, 0], [0, 0, 0])])
def test_left_padding_rejection(mask, ids):
    with pytest.raises(AssertionError):
        audit.validate_left_padding(ids, mask, 0)


def test_temporary_capture_is_identity_and_maps_rows():
    class DummyModel:
        def generate(self, **kwargs):
            return torch.cat([kwargs['input_ids'], torch.tensor([[3, 4, 99], [5, 6, 99]])], axis=1)
    model, tokenizer = DummyModel(), ToyTokenizer()
    old = model.generate
    ids = torch.tensor([[0, 1, 2], [1, 2, 2]])
    masks = torch.tensor([[0, 1, 1], [1, 1, 1]])
    calls = [{'budget': 9, 'query_ids': ['a', 'b'],
              'reference_input_ids': ids.tolist(), 'reference_masks': masks.tolist()}]
    saved = []
    with audit.capture_generate(model, tokenizer, calls, saved):
        output = model.generate(input_ids=ids, attention_mask=masks, max_new_tokens=9)
    assert output.shape == (2, 6)
    assert len(saved) == 2
    assert [r['query_id'] for r in saved] == ['a', 'b']
    assert [r['production_decoded_text'] for r in saved] == ['[(', '32']
    assert model.generate == old


def test_capture_rejects_query_row_swapped():
    class DummyModel:
        def generate(self, **kwargs):
            raise AssertionError('model should never run for misaligned batch')
    model, t = DummyModel(), ToyTokenizer()
    ids = torch.tensor([[0, 1, 2], [1, 2, 2]])
    masks = torch.tensor([[0, 1, 1], [1, 1, 1]])
    calls = [{'budget': 9, 'query_ids': ['a', 'b'],
              'reference_input_ids': ids.flip(0).tolist(), 'reference_masks': masks.tolist()}]
    with pytest.raises(AssertionError, match='wrong query-to-row'):
        with audit.capture_generate(model, t, calls, []):
            model.generate(input_ids=ids, attention_mask=masks, max_new_tokens=9)
