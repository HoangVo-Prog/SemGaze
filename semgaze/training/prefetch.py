"""Deterministic, bounded CPU preparation for training optimizer windows.

The producer owns a deepcopy of the sampler. The live sampler belongs to the
consumer and is advanced only by loading the consumed window's after-state.
Consequently a checkpoint always records consumed progress even when the
producer has already prepared future windows.
"""
from dataclasses import dataclass
import copy
import queue
import threading
import torch

from semgaze.semantic.flat.forward import prepare_semantic_native
from semgaze.training.batching import sample_optimizer_batches


@dataclass
class PreparedWindow:
    batches: object
    sampled_episodes: tuple
    rejected: int
    before_state: dict
    after_state: dict


@dataclass
class _Failure:
    error: BaseException


class OptimizerWindowPrefetcher:
    """One authoritative CPU producer with a bounded queue."""

    def __init__(self, bundle, sampler, *, depth=2, pin_memory=True, max_windows=None):
        if type(depth) is not int or depth < 1:
            raise ValueError('prefetch depth must be a positive integer')
        # The producer is the sole owner of its tokenizer/processor instance;
        # evaluation and checkpoint code may use the consumer's instance while
        # a future window is being prepared.
        self.bundle = copy.copy(bundle)
        if hasattr(bundle, 'processor'):
            try:
                self.bundle.processor = copy.deepcopy(bundle.processor)
            except Exception as error:
                raise RuntimeError('training processor cannot be safely cloned for CPU prefetch') from error
        # CPU preparation must not touch model/CUDA state.  The shallow bundle
        # carries configuration and tokenizer metadata only.
        self.bundle.model = None
        self.bundle.projector = None
        self.bundle.optimizer = None
        self.bundle.scheduler = None
        self.sampler = copy.deepcopy(sampler)
        self.depth = depth
        self.pin_memory = bool(pin_memory)
        self.max_windows = max_windows
        self._queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            raise RuntimeError('prefetcher already started')
        self._thread = threading.Thread(target=self._produce, name='semgaze-cpu-producer', daemon=True)
        self._thread.start()

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _prepare_window(self):
        # Snapshot the producer's logical starting point before any implicit
        # epoch transition.  This is the state the consumer must own before
        # committing this window's after-state.
        before = copy.deepcopy(self.sampler.state_dict())
        if self.sampler.epoch == 0 or self.sampler.epoch_complete():
            self.sampler.start_epoch()
        batches, accepted, rejected = sample_optimizer_batches(self.bundle, self.sampler)
        for batch in batches:
            batch.semantic_samples = tuple(
                prepare_semantic_native(self.bundle, episode.query, image_cache=batch.image_cache)
                for episode in batch.episodes)
            if self.pin_memory:
                batch.pin_memory_(torch.cuda.is_available(), pin_semantic_pixels=not bool(
                    self.bundle.config['training'].get('reuse_query_vision', True)))
        after = copy.deepcopy(self.sampler.state_dict())
        return PreparedWindow(batches, tuple(accepted), rejected, before, after)

    def _produce(self):
        produced = 0
        try:
            while self.max_windows is None or produced < self.max_windows:
                if self._stop.is_set():
                    return
                item = self._prepare_window()
                if not self._put(item):
                    return
                produced += 1
            self._put(None)
        except BaseException as error:
            self._put(_Failure(error))

    def get(self):
        if self._thread is None:
            raise RuntimeError('prefetcher is not started')
        while True:
            try:
                item = self._queue.get(timeout=0.1)
                break
            except queue.Empty:
                if not self._thread.is_alive() and self._queue.empty():
                    raise RuntimeError('CPU prefetch producer exited without a result')
        if isinstance(item, _Failure):
            raise RuntimeError('CPU prefetch producer failed') from item.error
        if item is None:
            raise StopIteration
        return item

    @staticmethod
    def assert_consumer_state(sampler, prepared):
        """Ensure a queued window is the next logical window to consume."""
        if sampler.state_dict() != prepared.before_state:
            raise RuntimeError('prefetched window is out of order relative to consumed sampler state')

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            if self._thread.is_alive():
                raise RuntimeError('CPU prefetch producer did not shut down')
            self._thread = None


class PrefetchLifecycle:
    """Context manager that always stops the producer on loop/eval failure."""
    def __init__(self, prefetcher):
        self.prefetcher = prefetcher

    def __enter__(self):
        return self.prefetcher

    def __exit__(self, exc_type, exc, tb):
        if self.prefetcher is not None:
            self.prefetcher.close()
        return False
