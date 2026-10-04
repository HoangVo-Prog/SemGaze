"""Normative subject -> K -> query -> image -> record sampling."""
from collections import defaultdict
import random
from .schema import FlatEpisode, K_VALUES, UNSEEN_SUBJECTS


class TrainingEpisodeSampler:
    def __init__(self, train_records, seed):
        self.rng = random.Random(seed)
        self.images = defaultdict(lambda: defaultdict(list))
        for r in train_records:
            if r.subject not in UNSEEN_SUBJECTS:
                self.images[r.subject][r.stimulus_id].append(r)
        self.subjects = sorted(self.images)
        if not self.subjects:
            raise ValueError("no seen subjects in training split")
        self.queries = {}
        for subject in self.subjects:
            images = self.images[subject]
            for k in K_VALUES:
                pool = [r for rows in images.values() for r in rows if len(images) - 1 >= k]
                if not pool:
                    raise ValueError(f"empty valid query pool for subject={subject}, K={k}")
                self.queries[subject, k] = pool

    def sample(self):
        u = self.rng.choice(self.subjects)
        k = self.rng.choice(K_VALUES)
        query = self.rng.choice(self.queries[u, k])
        images = self.rng.sample([i for i in self.images[u] if i != query.stimulus_id], k)
        supports = [self.rng.choice(self.images[u][i]) for i in images]
        self.rng.shuffle(supports)
        return FlatEpisode(tuple(supports), query)

    def state_dict(self):
        return self.rng.getstate()

    def load_state_dict(self, state):
        self.rng.setstate(state)


def frozen_episode(query, train_by_id, manifest, k, *, draw_id=None):
    if k not in K_VALUES:
        raise ValueError("unsupported K")
    if draw_id is None:
        if query.subject in UNSEEN_SUBJECTS:
            raise ValueError("unseen subject cannot enter validation")
        entries = manifest["validation_supports"][str(query.subject)][str(k)]
        ids = [e["record_id"] for e in entries]
    else:
        if query.subject not in UNSEEN_SUBJECTS or not 0 <= draw_id < 10:
            raise ValueError("final evaluation requires unseen subject and draw 0..9")
        entries = manifest["support_draws"][str(k)][draw_id]
        ids = [e["resolved_record_id_by_subject"][str(query.subject)] for e in entries]
    if len(ids) != k:
        raise ValueError("frozen support count differs from K")
    supports = tuple(train_by_id[i] for i in ids)
    for entry, record in zip(entries, supports):
        if entry["image_name"] != record.stimulus_id or entry["task"] != record.task:
            raise ValueError("frozen trial identity mismatch")
    return FlatEpisode(supports, query)
