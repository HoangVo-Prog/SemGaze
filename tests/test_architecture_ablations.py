"""Fast ablation contract tests; run from the SemGaze repository root."""
from types import SimpleNamespace
import pytest
import torch
from semgaze.model.config import load_config
from semgaze.where.serialization import serialize_xyd, parse_xyd_output, active_end_fix_token
from semgaze.state.extractor import extract_duration_numeric_states
from semgaze.semantic.flat.prompt import build_flat_prompt
from semgaze.data.semantic import NormalizedSemantic, WhyGroup


def fake_query():
    return SimpleNamespace(record_id='q', task='bottle',
        image_width=100, image_height=100,
        x_px=(52,47,22), y_px=(48,56,48), duration_ms=(231,183,96),
        semantic=NormalizedSemantic(('red bottle','table','shelf'),
            (WhyGroup((1,3),'look for bottle'),WhyGroup((2,),'search table')),
            'inspect scene'))


def make_query():
    # Groups must be chronological by first member and partition 1..N.
    query=fake_query()
    query.semantic=NormalizedSemantic(query.semantic.what,
        (WhyGroup((1,3),'inspect target'), WhyGroup((2,),'inspect alternatives')),
        'inspect scene')
    return query


def test_config_matrix():
    full=load_config('configs/flat_ss_lwce.yaml')
    v1=load_config('configs/ablation_architecture/no_end_fix_token.yaml')
    v2=load_config('configs/ablation_architecture/no_shared_exploration_states.yaml')
    assert full['where']['use_end_fix_token'] and full['semantic']['conditioning']=='shared_states'
    assert not v1['where']['use_end_fix_token']
    assert v1['state']['readout_position']=='last_duration_numeric_token'
    assert v1['state']['projector']['type']=='linear_layernorm'
    assert v2['where']['use_end_fix_token']
    assert v2['semantic']['conditioning']=='explicit_scanpath_text'
    assert v2['state']['projector']['type']=='none'
    assert active_end_fix_token(v1)==''
    assert active_end_fix_token(v2)=='<END_FIX>'
    for a in (v1,v2):
        assert a['training']['scheduled_sampling']==full['training']['scheduled_sampling']
        assert a['training']['late_fixation_ce']==full['training']['late_fixation_ce']
        assert a['training']['lambda_where']==full['training']['lambda_where']
        assert a['training']['lambda_sem']==full['training']['lambda_sem']


def test_v1_codec_numeric_edges():
    values=[(0,0,0),(7,9,7),(52,48,96),(47,56,231),(99,99,999)]
    text=serialize_xyd(values,'')
    assert '<END_FIX>' not in text
    assert text=='[(00, 00, 000), (07, 09, 007), (52, 48, 096), (47, 56, 231), (99, 99, 999)]'
    parsed=parse_xyd_output(text,len(values),'')
    assert parsed['canonical_format_valid'] and parsed['fixations']==values


def test_v1_exact_numeric_readout_and_gradients():
    import re
    text='[(00, 00, 000), (07, 09, 007), (52, 48, 096), (47, 56, 231), (99, 99, 999)]'
    offsets=tuple((i,i+1) for i in range(len(text)))
    native=SimpleNamespace(ablation_query_text=text, response_start=0, response_offsets=offsets,
        inputs={'labels': torch.arange(len(text), dtype=torch.long)[None,:]})
    hidden=torch.randn(1,len(text),4,requires_grad=True)
    states, _, pos=extract_duration_numeric_states(hidden,(native,),[5],return_unpadded=True)
    expected=[m.end(1)-1 for m in re.finditer(r'\(\s*\d{2}\s*,\s*\d{2}\s*,\s*(\d{3})\s*\)',text)]
    assert pos[0].tolist()==expected and states[0].shape==(5,4)
    states[0].sum().backward()
    assert hidden.grad[0,expected].abs().sum()>0
    assert torch.count_nonzero(hidden.grad[0]).item()==5*4


def test_v1_merged_digit_punctuation_is_still_digit_anchored():
    import re
    text='[(52, 48, 231)]'
    offsets=[(i,i+1) for i in range(len(text))]
    digit=list(re.finditer(r'\d{3}',text))[-1].end()-1
    offsets[digit]=(digit,digit+2)
    offsets.pop(digit+1)
    native=SimpleNamespace(ablation_query_text=text,response_start=0,
        response_offsets=tuple(offsets),inputs={'labels':torch.ones((1,len(offsets)),dtype=torch.long)})
    states,_,pos=extract_duration_numeric_states(torch.zeros((1,len(offsets),4)),(native,),[1],return_unpadded=True)
    assert pos[0].tolist()==[digit] and states[0].shape==(1,4)


def test_v2_flat_prompt_is_text_only():
    # Use valid chronological groups (noncontiguous membership is permitted).
    query=make_query()
    prompt=build_flat_prompt(query,conditioning='explicit_scanpath_text')
    assert 'Scanpath:' in prompt.text
    assert 'Fixation 1: (52, 48, 231)' in prompt.text
    assert 'Fixation 3: (22, 48, 096)' in prompt.text
    assert prompt.boundaries==()
    assert 'Gold WHY groups:' in prompt.text
    assert 'Group 1: fixations 1, 3' in prompt.text
    assert 'look for bottle' not in prompt.text
    assert 'inspect target' not in prompt.text
    shared=build_flat_prompt(query)
    assert len(shared.boundaries)==3
    assert 'Exploration states:' in shared.text
