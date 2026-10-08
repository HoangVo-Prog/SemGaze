"""COCO-Search18 query-coverage epochs and stochastic support construction."""
from collections import defaultdict
import math
import random
from .schema import FlatEpisode, UNSEEN_SUBJECTS


class TrainingEpisodeSampler:
    def __init__(self, train_records, seed, *, data_config=None):
        from semgaze.model.config import default_section
        data_config = data_config if data_config is not None else default_section('data')
        if data_config['variant'] != 'all':
            raise ValueError('Q_train must be built from the all-variant train.json')

        self.k_values = tuple(data_config['fewshot']['k_values'])
        self.probabilities = tuple(data_config['fewshot']['train_k_probabilities'])
        # Config validation owns range/normalization; fail closed for direct callers.
        if (not self.k_values or len(self.k_values) != len(self.probabilities) or
                len(set(self.k_values)) != len(self.k_values) or
                any(type(k) is not int or not 1 <= k <= 10 for k in self.k_values) or
                any(not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 for p in self.probabilities) or
                abs(sum(self.probabilities) - 1.0) > 1e-8):
            raise ValueError('invalid configured training K distribution')
        unseen_subjects = data_config['unseen_subjects']
        self.rng = random.Random(seed)
        self.images = defaultdict(lambda: defaultdict(list))
        self.records = []
        if set(unseen_subjects) != {7, 8, 9}:
            raise ValueError("COCO unseen subjects must remain {7,8,9}")
        for r in train_records:
            if getattr(r, 'split', 'train') != 'train' or getattr(r, 'variant', 'all') != 'all':
                raise ValueError('optimization records must come from all/train')
            if r.subject not in unseen_subjects:
                self.images[r.subject][r.stimulus_id].append(r)
                self.records.append(r)
        if not self.records:
            raise ValueError("no seen subjects in training split")
        self.queries = tuple(self.records)
        self.query_ids = tuple(r.record_id for r in self.queries)
        self._query_by_id = {r.record_id: r for r in self.queries}
        if len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError('duplicate optimization query record')
        for subject, images in self.images.items():
            needed = max(self.k_values) + 1
            if len(images) < needed:
                raise ValueError(f'subject {subject} needs {needed} distinct train images for K_train_max={max(self.k_values)}')
        self._epoch = 0
        self._cursor = 0
        self._permutation = None

    @property
    def query_count(self):
        return len(self.queries)

    @property
    def cursor(self):
        return self._cursor

    @property
    def epoch(self):
        return self._epoch

    @property
    def remaining(self):
        return self.query_count - self.cursor

    def start_epoch(self, epoch=None):
        if self._cursor < len(self.queries) and self._permutation is not None:
            raise RuntimeError('cannot reshuffle a partially consumed query-coverage epoch')
        if epoch is not None and epoch != self._epoch + 1:
            raise ValueError('epochs must advance consecutively')
        self._epoch += 1
        self._permutation = list(range(len(self.queries)))
        self.rng.shuffle(self._permutation)
        self._cursor = 0

    def epoch_complete(self):
        return self._permutation is not None and self._cursor == len(self._permutation)

    def next_query(self):
        if self._permutation is None:
            self.start_epoch()
        if self._cursor >= len(self._permutation):
            raise StopIteration
        query = self.queries[self._permutation[self._cursor]]
        self._cursor += 1
        return query

    def sample_for_query(self, query, *, k=None):
        if self._query_by_id.get(query.record_id) != query:
            raise ValueError('support sampling query is outside Q_train')
        if k is not None and (type(k) is not int or k not in self.k_values):
            raise ValueError(f'K_train={k} is not in configured k_values={self.k_values}')
        if k is None:
            # Preserve legacy seeded trajectories for uniform K=1..10.
            if all(p == self.probabilities[0] for p in self.probabilities):
                k = self.rng.choice(self.k_values)
            else:
                k = self.rng.choices(self.k_values, weights=self.probabilities, k=1)[0]
        candidates = [i for i in self.images[query.subject] if i != query.stimulus_id]
        if len(candidates) < k:
            raise RuntimeError(f'query {query.record_id} has only {len(candidates)} eligible support images for K_train={k}')
        images = self.rng.sample(candidates, k)
        supports = [self.rng.choice(self.images[query.subject][i]) for i in images]
        self.rng.shuffle(supports)
        return FlatEpisode(tuple(supports), query)

    def sample(self, *, k=None):
        """Consume exactly one query from the active coverage epoch."""
        if k is not None:
            raise ValueError('K_train is sampled independently per query; forced K is forbidden')
        return self.sample_for_query(self.next_query())

    def state_dict(self):
        return {'version': 1, 'query_ids': self.query_ids, 'rng': self.rng.getstate(),
                'epoch': self.epoch, 'cursor': self.cursor,
                'permutation': list(self._permutation) if self._permutation is not None else None,
                'k_values': self.k_values, 'k_probabilities': self.probabilities}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get('version') != 1:
            raise ValueError('checkpoint lacks query-coverage state; explicitly start a new run')
        if tuple(state['query_ids']) != self.query_ids:
            raise ValueError('checkpoint optimization query universe differs')
        # Legacy v1 checkpoint omitted K because old code always sampled K=1..10 uniformly.
        stored_k = tuple(state.get('k_values', range(1, 11)))
        stored_prob = tuple(state.get('k_probabilities', (0.1,) * 10))
        if stored_k != self.k_values or stored_prob != self.probabilities:
            raise ValueError('checkpoint training K distribution differs from current config; start a new run')
        permutation, cursor, epoch = state['permutation'], state['cursor'], state['epoch']
        if type(cursor) is not int or type(epoch) is not int or epoch < 0:
            raise ValueError('invalid epoch/cursor in checkpoint')
        if permutation is None:
            if cursor != 0 or epoch != 0:
                raise ValueError('invalid unstarted epoch state')
        elif (epoch < 1 or not 0 <= cursor <= self.query_count or
              len(permutation) != self.query_count or any(type(i) is not int for i in permutation) or set(permutation) != set(range(self.query_count))):
            raise ValueError('checkpoint permutation/cursor would duplicate or omit queries')
        self.rng.setstate(state['rng'])
        self._epoch, self._cursor = epoch, cursor
        self._permutation = list(permutation) if permutation is not None else None


def frozen_episode(query, train_by_id, manifest, k, *, draw_id=None, unseen_subjects=None):
    if unseen_subjects is None:
        unseen_subjects = manifest.get("unseen_subject_ids", UNSEEN_SUBJECTS)
    if type(k) is not int or k not in (1,5,10):
        raise ValueError('K_eval must be one of 1/5/10')
    if type(draw_id) is not int:
        raise ValueError('frozen test evaluation requires an explicit integer draw_id')
    else:
        if str(k) not in manifest["support_draws"]:
            raise ValueError(f"no persisted final support draws for K={k}")
        if query.subject not in unseen_subjects or not 0 <= draw_id < len(manifest["support_draws"][str(k)]):
            raise ValueError("final evaluation requires unseen subject and a persisted draw index")
        entries = manifest["support_draws"][str(k)][draw_id]
        ids = [e["resolved_record_id_by_subject"][str(query.subject)] for e in entries]
    if len(ids) != k:
        raise ValueError("frozen support count differs from K")
    if query.stimulus_id not in manifest['test_stimulus_ids']:
        raise ValueError('frozen evaluation query must come from test')
    if any(i not in train_by_id for i in ids):
        raise ValueError('frozen evaluation support must come from train')
    supports = tuple(train_by_id[i] for i in ids)
    if any(s.stimulus_id not in manifest['train_stimulus_ids'] for s in supports):
        raise ValueError('frozen support image is outside train membership')
    for entry, record in zip(entries, supports):
        # The new JSON support entry uses unit_id (task), not task itself.
        task = entry.get('task', entry.get('unit_id'))
        if (entry['image_name'] != record.stimulus_id or task != record.task or
                entry.get('trial_key') != f'{record.task}::{record.stimulus_id}'):
            raise ValueError('frozen trial identity mismatch')
    return FlatEpisode(supports, query, draw_id=draw_id)
