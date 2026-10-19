"""Reuse persisted split validation and original normalized record adapters."""
from collections import Counter
from pathlib import Path

from semgaze.data.cocosearch18 import CocoSearch18Adapter, read_persisted_splits
from semgaze.data.joint import JointAdapter, unseen_subject_ids
from semgaze.evaluation.test import test_queries
from semgaze.model.config import ROOT, load_config, resolve_config


def load_queries(semgaze_config, *, data_overrides=None, subject_scope="unseen"):
    """Return (eligible train queries, held-out queries, manifest identity, data config).

    Loads persisted records, not fabricated supportless FlatEpisode instances.
    Existing source-manifest and train/test stimulus checks run unchanged.
    """
    config = load_config(Path(semgaze_config))
    if data_overrides:
        config = resolve_config(config, {'data': data_overrides})
    data = config['data']
    raw, manifest, identity = read_persisted_splits(ROOT / data['split_root'], data_config=data)
    adapter = (JointAdapter if data['dataset'] == 'all' else CocoSearch18Adapter)(
        ROOT / data['images_root'], annotation_frame=data['annotation_frame'],
        duration_field=data['duration']['source_field'])
    train = [adapter(row) for row in raw['train']]
    if subject_scope not in ('seen', 'unseen'):
        raise ValueError('baseline query scope must be seen or unseen')
    source_split = 'test_seen' if subject_scope == 'seen' else 'test'
    test = [adapter(row) for row in raw[source_split]]
    if len({r.record_id for r in train + test}) != len(train) + len(test):
        raise ValueError('duplicate query identity across persisted splits')
    if data['dataset'] == 'all':
        for name, records in [('train', train), ('test', test)]:
            if set(Counter(r.dataset for r in records)) != {'AiR', 'COCO-Search18'}:
                raise ValueError(f'{name}: joint loader lost a dataset')
    unseen = unseen_subject_ids(data, manifest)
    eligible = [r for r in train if r.subject not in unseen]
    heldout = list(test_queries({r.record_id: r for r in train}, test, manifest,
                                unseen_subjects=unseen, scope=subject_scope))
    if not eligible:
        raise ValueError('empty seen-subject training population')
    return eligible, heldout, identity, data
