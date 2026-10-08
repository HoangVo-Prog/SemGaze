"""Canonical evaluation identity and metrics artifact helpers."""
from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path
from typing import Iterable, Mapping


def implementation_head():
    """Return the checkout revision used for an artifact, when available."""
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ['git', '-c', f'safe.directory={root.as_posix()}', 'rev-parse', 'HEAD'],
            cwd=root, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'
    return result.stdout.strip()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def episode_key(row: Mapping):
    query_id = row.get("query_id", row.get("record_id"))
    k = row.get("K", row.get("k"))
    draw_id = row.get("draw_id")
    if query_id is None or k is None or draw_id is None:
        raise KeyError("canonical episode requires (query_id, K, draw_id)")
    return (str(query_id), int(k), int(draw_id))


def index_by_episode(rows: Iterable[Mapping]):
    indexed = {}
    for row in rows:
        key = episode_key(row)
        if key in indexed:
            raise ValueError(f"duplicate logical episode: {key}")
        indexed[key] = row
    return indexed


def assert_key_sets_equal(*sources):
    indexed = [index_by_episode(source) for source in sources]
    if not indexed:
        return set()
    expected = set(indexed[0])
    for index in indexed[1:]:
        if set(index) != expected:
            missing, extra = expected - set(index), set(index) - expected
            raise ValueError(f"evaluation key mismatch: missing={sorted(missing)!r}, extra={sorted(extra)!r}")
    return expected


def expected_episode_keys(query_ids, k_values, draw_counts):
    """Return the required logical episode identities for a test evaluation."""
    query_ids = [str(q) for q in query_ids]
    keys = set()
    for k in k_values:
        count = draw_counts[str(k)] if str(k) in draw_counts else draw_counts[k]
        count = len(count) if not isinstance(count, int) else count
        for draw_id in range(int(count)):
            for query_id in query_ids:
                key = (query_id, int(k), draw_id)
                if key in keys:
                    raise ValueError(f'duplicate expected logical episode: {key}')
                keys.add(key)
    return keys


def resolve_evaluation_draw_counts(draw_count, manifest, k_values):
    """Resolve one configured traversal count against persisted support draws.

    The persisted COCO-Search18 protocol contains the canonical ordered support
    draws.  Evaluation may execute a prefix of that order, but it must not
    invent new draws or silently execute fewer than configured.
    """
    if type(draw_count) is not int or draw_count < 1:
        raise ValueError('evaluation.draw must be an integer >= 1')
    counts = {}
    for k in k_values:
        key = str(k)
        if key not in manifest.get('support_draws', {}):
            raise ValueError(f'persisted support draws are missing K={k}')
        available = len(manifest['support_draws'][key])
        if draw_count > available:
            raise ValueError(f'evaluation.draw={draw_count} exceeds the {available} persisted support draws for K={k}')
        counts[key] = draw_count
    return counts


def resolve_semantic_draw_counts(where_draw_counts, semantic_use_draws=True):
    """Resolve semantic prediction traversals from the WHERE traversal plan.

    WHERE evaluation always owns validation against the persisted support draws.
    When semantic draws are disabled, draw zero is still used as the normal
    support realization for every K; no synthetic or unsampled episode is
    introduced.
    """
    if type(semantic_use_draws) is not bool:
        raise ValueError('evaluation.semantic_use_draws must be boolean')
    if semantic_use_draws:
        return dict(where_draw_counts)
    return {str(k): 1 for k in where_draw_counts}


def write_metrics_artifact(path, *, checkpoint, config, split_manifest_identity, by_k,
                           provenance, implementation_head, split="test", evaluation_draw=None):
    if split != "test":
        raise ValueError("canonical metrics artifacts must use split=test")
    payload = {"schema_version": 1, "baseline_commit": "b84e9c752124457244ee34b93e70f72be5690914",
               "implementation_head": implementation_head, "checkpoint": str(checkpoint),
               "split_manifest_identity": split_manifest_identity, "config": str(config),
               "split": split, "evaluation_draw": evaluation_draw,
               "metric_provenance": provenance, "by_k": by_k}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload
