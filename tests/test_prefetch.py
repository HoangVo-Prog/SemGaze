"""Deterministic producer/consumer and consumed-state checkpoint contracts."""
from types import SimpleNamespace
import copy
import time
import pytest

from semgaze.training import prefetch as prefetch_module
from semgaze.training.prefetch import OptimizerWindowPrefetcher


class FakeEpisode:
    def __init__(self, value):
        self.query = SimpleNamespace(record_id=value)


class FakeSampler:
    def __init__(self, count=8):
        self.query_count = count
        self.cursor = 0
        self.epoch = 0

    @property
    def remaining(self):
        return self.query_count - self.cursor

    def start_epoch(self):
        assert self.epoch == 0 or self.epoch_complete()
        self.epoch += 1
        self.cursor = 0

    def epoch_complete(self):
        return self.epoch > 0 and self.cursor == self.query_count

    def sample(self):
        if self.epoch == 0 or self.epoch_complete():
            self.start_epoch()
        value = (self.epoch, self.cursor)
        self.cursor += 1
        return value

    def state_dict(self):
        return (self.epoch, self.cursor)

    def load_state_dict(self, state):
        self.epoch, self.cursor = state


def _fake_sample(bundle, sampler):
    count = min(bundle.config['training']['per_device_train_batch_size'] *
                bundle.config['training']['gradient_accumulation_steps'], sampler.remaining)
    episodes = [FakeEpisode(sampler.sample()) for _ in range(count)]
    batch = SimpleNamespace(episodes=episodes, image_cache={}, semantic_samples=(),
                            pin_memory_=lambda enabled=True: batch)
    return [batch], episodes, 0


def _collect(prefetcher, n):
    values = []
    states = []
    for _ in range(n):
        item = prefetcher.get()
        values.extend(ep.query.record_id for ep in item.sampled_episodes)
        states.append((item.before_state, item.after_state))
    return values, states


def test_prefetch_matches_synchronous_sampler_order(monkeypatch):
    monkeypatch.setattr(prefetch_module, 'sample_optimizer_batches', _fake_sample)
    monkeypatch.setattr(prefetch_module, 'prepare_semantic_native', lambda *args, **kwargs: object())
    bundle = SimpleNamespace(config={'training': {'per_device_train_batch_size': 2,
                                                   'gradient_accumulation_steps': 1}})
    synchronous = FakeSampler()
    expected = []
    for _ in range(4):
        if synchronous.epoch == 0 or synchronous.epoch_complete():
            synchronous.start_epoch()
        _, episodes, _ = _fake_sample(bundle, synchronous)
        expected.extend(ep.query.record_id for ep in episodes)

    producer = OptimizerWindowPrefetcher(bundle, FakeSampler(), depth=2, pin_memory=False, max_windows=4)
    producer.start()
    try:
        actual, states = _collect(producer, 4)
    finally:
        producer.close()
    assert actual == expected
    assert all(after == states[i + 1][0] for i, (_, after) in enumerate(states[:-1]))


def test_checkpoint_uses_consumed_state_when_producer_is_ahead(monkeypatch):
    monkeypatch.setattr(prefetch_module, 'sample_optimizer_batches', _fake_sample)
    monkeypatch.setattr(prefetch_module, 'prepare_semantic_native', lambda *args, **kwargs: object())
    bundle = SimpleNamespace(config={'training': {'per_device_train_batch_size': 1,
                                                   'gradient_accumulation_steps': 1}})
    live = FakeSampler()
    producer = OptimizerWindowPrefetcher(bundle, live, depth=2, pin_memory=False, max_windows=4)
    producer.start()
    try:
        first = producer.get()
        producer.assert_consumer_state(live, first)
        live.load_state_dict(first.after_state)
        consumed_state = copy.deepcopy(live.state_dict())
        # Let the bounded producer fill at least one future slot when possible.
        time.sleep(0.02)
        assert producer._queue.qsize() >= 1
    finally:
        producer.close()

    resumed = OptimizerWindowPrefetcher(bundle, live, depth=2, pin_memory=False, max_windows=3)
    resumed.start()
    try:
        next_window = resumed.get()
    finally:
        resumed.close()
    assert live.state_dict() == consumed_state
    assert next_window.before_state == consumed_state


def test_producer_exception_reaches_consumer_and_closes(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError('preprocessing failed')
    monkeypatch.setattr(prefetch_module, 'sample_optimizer_batches', fail)
    bundle = SimpleNamespace(config={'training': {'per_device_train_batch_size': 1,
                                                   'gradient_accumulation_steps': 1}})
    producer = OptimizerWindowPrefetcher(bundle, FakeSampler(), pin_memory=False, max_windows=1)
    producer.start()
    with pytest.raises(RuntimeError, match='producer failed'):
        producer.get()
    producer.close()
