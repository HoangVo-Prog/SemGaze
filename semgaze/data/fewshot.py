"""Normative subject -> K -> query -> image -> record sampling."""
from collections import defaultdict
import random
from .schema import FlatEpisode, UNSEEN_SUBJECTS


class TrainingEpisodeSampler:
    def __init__(self, train_records, seed, *, data_config=None):
        from semgaze.model.config import default_section
        data_config = data_config if data_config is not None else default_section('data')
        self.k_values = data_config['fewshot']['k_values']
        self.probabilities = data_config['fewshot']['train_k_probabilities']
        unseen_subjects = data_config['unseen_subjects']
        self.rng = random.Random(seed)
        self.images = defaultdict(lambda: defaultdict(list))
        for r in train_records:
            if r.subject not in unseen_subjects:
                self.images[r.subject][r.stimulus_id].append(r)
        self.subjects = sorted(self.images)
        if not self.subjects:
            raise ValueError("no seen subjects in training split")
        self.queries = {}
        for subject in self.subjects:
            images = self.images[subject]
            for k, probability in zip(self.k_values, self.probabilities):
                if probability == 0:
                    continue
                pool = [r for rows in images.values() for r in rows if len(images) - 1 >= k]
                if not pool:
                    raise ValueError(f"empty valid query pool for subject={subject}, K={k}")
                self.queries[subject, k] = pool

    def sample(self):
        u = self.rng.choice(self.subjects)
        k = self.rng.choices(self.k_values, weights=self.probabilities, k=1)[0]
        query = self.rng.choice(self.queries[u, k])
        images = self.rng.sample([i for i in self.images[u] if i != query.stimulus_id], k)
        supports = [self.rng.choice(self.images[u][i]) for i in images]
        self.rng.shuffle(supports)
        return FlatEpisode(tuple(supports), query)

    def state_dict(self):
        return self.rng.getstate()

    def load_state_dict(self, state):
        self.rng.setstate(state)


def frozen_episode(query, train_by_id, manifest, k, *, draw_id=None, unseen_subjects=None):
    if unseen_subjects is None:
        unseen_subjects = manifest.get("unseen_subject_ids", UNSEEN_SUBJECTS)
    if type(k) is not int or k < 1:
        raise ValueError("K must be positive")
    if draw_id is None:
        if query.subject in unseen_subjects:
            raise ValueError("unseen subject cannot enter validation")
        blocks = manifest["validation_supports"][str(query.subject)]
        if str(k) not in blocks:
            raise ValueError(f"no persisted validation supports for subject={query.subject}, K={k}")
        entries = blocks[str(k)]
        ids = [e["record_id"] for e in entries]
    else:
        if str(k) not in manifest["support_draws"]:
            raise ValueError(f"no persisted final support draws for K={k}")
        if query.subject not in unseen_subjects or not 0 <= draw_id < len(manifest["support_draws"][str(k)]):
            raise ValueError("final evaluation requires unseen subject and a persisted draw index")
        entries = manifest["support_draws"][str(k)][draw_id]
        ids = [e["resolved_record_id_by_subject"][str(query.subject)] for e in entries]
    if len(ids) != k:
        raise ValueError("frozen support count differs from K")
    supports = tuple(train_by_id[i] for i in ids)
    for entry, record in zip(entries, supports):
        if entry["image_name"] != record.stimulus_id or entry["task"] != record.task:
            raise ValueError("frozen trial identity mismatch")
    return FlatEpisode(supports, query)
