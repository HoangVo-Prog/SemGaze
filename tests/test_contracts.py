import copy
import json
from dataclasses import replace
from pathlib import Path
import pytest
from semgaze.data.schema import normalized_episode_from_dict, FlatEpisode, require_seen_training_episode
from semgaze.data.semantic import SemanticAnnotationValidationError
from semgaze.data.fewshot import TrainingEpisodeSampler
from semgaze.where.coordinates import coordinate_bin
from semgaze.where.duration import duration_bin
from semgaze.where.serialization import serialize_xyd_record, parse_xyd_output
from semgaze.semantic.flat.target import build_flat_target
from semgaze.semantic.flat.parser import parse_flat_output

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def payload():
    return json.loads((ROOT / "tests/fixtures/flat_episode.json").read_text())


def test_goldens(payload):
    episode = normalized_episode_from_dict(payload)
    golden = dict(line.split("=", 1) for line in (ROOT / "tests/fixtures/expected_where.txt").read_text().splitlines())
    assert serialize_xyd_record(episode.supports[0]) == golden['SUPPORT']
    assert serialize_xyd_record(episode.query) == golden['QUERY']
    target = build_flat_target(episode.query.semantic)
    assert target == (ROOT / 'tests/fixtures/expected_flat_target.txt').read_text().rstrip('\n')
    assert parse_flat_output(target, 4, 2)['flat_format_valid']


@pytest.mark.parametrize('field,value', [('y_px', [1]), ('duration_ms', []), ('x_px', [float('nan')]*4)])
def test_scanpath_validation(payload, field, value):
    payload['query'][field] = value
    with pytest.raises(ValueError):
        normalized_episode_from_dict(payload)


@pytest.mark.parametrize('mutation', ['overlap', 'empty', 'order', 'bool', 'what', 'why', 'how'])
def test_semantic_validation(payload, mutation):
    s = payload['query']['semantic']
    if mutation == 'overlap': s['why_groups'][1]['members'] = [1, 4]
    if mutation == 'empty': s['why_groups'][0]['members'] = []
    if mutation == 'order': s['why_groups'].reverse()
    if mutation == 'bool': s['why_groups'][0]['members'][0] = True
    if mutation == 'what': s['what'][0] = ' \n '
    if mutation == 'why': s['why_groups'][0]['why'] = ''
    if mutation == 'how': s['how'] = ''
    with pytest.raises(SemanticAnnotationValidationError, match='fixture::subject1::query'):
        normalized_episode_from_dict(payload)


def test_codecs():
    assert coordinate_bin(2.51, 100) == 2  # two-stage rounding differs from one-stage
    assert [coordinate_bin(v, 100) for v in (-5, 0, 2.5, 3.5, 100)] == [0, 0, 2, 4, 99]
    assert [duration_bin(v) for v in (-4, 0, 2.5, 3.5, 999, 1200)] == [0, 0, 2, 4, 999, 999]
    parsed = parse_xyd_output('prose (-1, +100, 1200) <END_FIX>, (01,02,003)', 2)
    assert parsed['fixations'] == [(0, 99, 999)]
    assert parsed['under_generated'] and not parsed['canonical_format_valid']


@pytest.mark.parametrize('mutation', ['duplicate', 'missing', 'order', 'prose', 'range', 'empty'])
def test_strict_parser(payload, mutation):
    lines = build_flat_target(normalized_episode_from_dict(payload).query.semantic).splitlines()
    if mutation == 'duplicate': lines.append(lines[0])
    if mutation == 'missing': lines.pop(0)
    if mutation == 'order': lines.reverse()
    if mutation == 'prose': lines.append('Here is your explanation.')
    if mutation == 'range': lines[0] = 'WHAT 0: x'
    if mutation == 'empty': lines[-1] = 'HOW:  '
    assert not parse_flat_output('\n'.join(lines), 4, 2)['flat_format_valid']


def test_unseen_and_episode_invariants(payload):
    episode = normalized_episode_from_dict(payload)
    unseen = FlatEpisode(tuple(replace(s, subject=7) for s in episode.supports), replace(episode.query, subject=7))
    with pytest.raises(ValueError, match='unseen'): require_seen_training_episode(unseen)
    with pytest.raises(ValueError): FlatEpisode((episode.query,), episode.query)
    with pytest.raises(ValueError): FlatEpisode((replace(episode.supports[0], subject=2),), episode.query)


def test_sampling_and_rng_resume(payload):
    q = normalized_episode_from_dict(payload).query
    records = [replace(q, record_id=f'{u}:{i}', stimulus_id=str(i), subject=u) for u in (1, 2, 7) for i in range(12)]
    sampler = TrainingEpisodeSampler(records, 42)
    seen_k = set()
    for _ in range(100):
        if sampler.epoch_complete():
            sampler.start_epoch()
        episode = sampler.sample()
        assert episode.query.subject in (1, 2)
        seen_k.add(len(episode.supports))
    assert seen_k == set(range(1, 11))
    state = sampler.state_dict()
    expected = sampler.sample()
    sampler.load_state_dict(state)
    assert sampler.sample() == expected
    with pytest.raises(ValueError, match='11 distinct train images'): TrainingEpisodeSampler(records[:5], 42)
