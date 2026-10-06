import copy
import json
from dataclasses import replace
from pathlib import Path
import pytest
from semgaze.data.schema import normalized_episode_from_dict, FlatEpisode
from semgaze.data.semantic import semantic_from_prediction, SemanticAnnotationValidationError
from semgaze.data.fewshot import frozen_episode
from semgaze.model.config import load_config, validate_config, resolve_config
from semgaze.model.build import inspect_adapter
from semgaze.where.conversation import build_where_conversation
from semgaze.where.prompt import build_where_prompt
from semgaze.semantic.flat.prompt import build_flat_prompt, INSTRUCTION

ROOT = Path(__file__).resolve().parents[1]


def fixture_episode():
    return normalized_episode_from_dict(json.loads((ROOT / 'tests/fixtures/flat_episode.json').read_text()))


def test_literal_prompt_contracts():
    spec = (ROOT / 'documents/02_WHERE_SPEC.md').read_text(encoding='utf-8')
    literal = spec.split('`PROMPT_XYD(Q, N)`:', 1)[1].split('```text\n', 1)[1].split('```', 1)[0].rstrip('\n')
    expected = literal.replace('{INTRO(Q)}', 'while searching for a microwave.').replace('{N}', '4').replace(
        '{CONSIDER(Q)}', 'Consider the search target, visual saliency, and how attention naturally flows during visual search.')
    assert build_where_prompt('microwave', 4) == expected
    flat = (ROOT / 'documents/flat/04_SEMANTIC_SINGLE_SPEC.md').read_text(encoding='utf-8')
    expected_flat = flat.split('### 6.1 Literal semantic instruction', 1)[1].split('```text\n', 1)[1].split('```', 1)[0].rstrip('\n')
    assert INSTRUCTION == expected_flat


def test_order_preserved_and_support_semantics_not_exposed():
    e = fixture_episode()
    supports = tuple(replace(e.supports[0], record_id=f's{i}', stimulus_id=f'i{i}', image_path=f'path{i}') for i in (4, 2, 0, 3, 1))
    e = FlatEpisode(supports, e.query)
    messages, paths = build_where_conversation(e)
    assert paths == [s.image_path for s in (*supports, e.query)]
    assert [m['role'] for m in messages] == ['user', 'assistant'] * 5 + ['user']
    assert all(s.semantic.how not in str(messages) for s in supports)
    prompt = build_flat_prompt(e.query)
    assert e.query.task in prompt.text
    assert 'Group 1: fixations 1, 3\nGroup 2: fixations 2, 4' in prompt.text
    assert '<END_FIX>' not in prompt.text and 'fixture::' not in prompt.text
    for t, boundary in enumerate(prompt.boundaries, 1):
        assert prompt.text[:boundary].endswith(f'Fixation {t}:')


def test_raw_prediction_field_mapping():
    e = fixture_episode()
    semantic = e.query.semantic
    prediction = {'fixations': [{'fixation': t, 'what': w} for t, w in reversed(list(enumerate(semantic.what, 1)))],
                  'regions': [{'fixations': list(g.members), 'why': g.why} for g in semantic.why_groups],
                  'how': semantic.how}
    assert semantic_from_prediction(prediction, 4, 'sample') == semantic
    prediction['regions'].reverse()
    with pytest.raises(SemanticAnnotationValidationError): semantic_from_prediction(prediction, 4, 'sample')


def test_frozen_manifest_order_is_used():
    e = fixture_episode()
    query = replace(e.query, subject=7)
    support = replace(e.supports[0], subject=7)
    entry = {'image_name': support.stimulus_id, 'task': support.task, 'resolved_record_id_by_subject': {'7': support.record_id}}
    manifest = {'support_draws': {'1': [[entry]] * 10}, 'test_stimulus_ids': [query.stimulus_id], 'train_stimulus_ids': [support.stimulus_id]}
    got = frozen_episode(query, {support.record_id: support}, manifest, 1, draw_id=3)
    assert got.supports == (support,)


@pytest.mark.parametrize('section,key,value', [('semantic', 'mode', 'other'), ('model', 'merge_adapter_for_training', True),
    ('state', 'detach_where_states', True), ('semantic', 'loss', 'balanced')])
def test_config_rejects_unsupported_contract(section, key, value):
    config = load_config(ROOT / 'configs/flat_single.yaml')
    config[section][key] = value
    with pytest.raises(ValueError): validate_config(config)


def test_engineering_choices_explicit_and_annotation_frame_unresolved():
    config = load_config(ROOT / 'configs/flat_single.yaml')
    resolved = resolve_config(config)
    assert config['training']['weight_decay'] == resolved['training']['weight_decay'] == 0.01
    config['data']['annotation_frame'] = None
    assert resolve_config(config)['data']['annotation_frame'] is None
    config['training']['weight_decay'] = None
    with pytest.raises(ValueError, match='weight_decay'):
        resolve_config(config)  # explicit null must not become a hidden default



def test_adapter_preflight_detects_lfs_without_mutating_reference(tmp_path):
    adapter = ROOT / 'DeepGaze-VL/model/visual_search_adapter'
    (tmp_path / 'adapter_config.json').write_bytes((adapter / 'adapter_config.json').read_bytes())
    for name in ('adapter_model.safetensors', 'tokenizer.json'):
        (tmp_path / name).write_text('version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n')
    result = inspect_adapter(tmp_path)
    assert len(result['blockers']) == 2
    assert all('Git LFS pointer' in b for b in result['blockers'])
