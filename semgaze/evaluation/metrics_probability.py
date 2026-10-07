"""DeepGaze-fast LL/IG primitives.

The functions here only score teacher-forced GT transitions.  They do not
decode bins and do not call ``generate``.  A caller must supply logits produced
under the exact SemGaze episode context (query, support draw, and GT prefix).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import pickle
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import re
import torch


class ProbabilityProtocolError(ValueError):
    pass


def audit_digit_tokenization(tokenizer, strings=("(00, 00, 000)<END_FIX>", "(09, 09, 009)<END_FIX>",
                                                 "(10, 10, 010)<END_FIX>", "(99, 99, 999)<END_FIX>")):
    """Return character, token, ID, and offset records for the four audit values."""
    result = []
    for text in strings:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(encoded["input_ids"])
        tokens = tokenizer.convert_ids_to_tokens(ids)
        result.append({"text": text, "tokens": tokens, "token_ids": ids,
                       "offsets": [tuple(x) for x in encoded["offset_mapping"]],
                       "characters": list(text)})
    return result


def digit_logprob(logits, target_digit: int, digit_ids: Sequence[int]):
    """Normalize exactly over the ten digit alternatives using natural logs."""
    if type(target_digit) is not int or not 0 <= target_digit <= 9 or len(digit_ids) != 10:
        raise ProbabilityProtocolError("target digit and ten digit token IDs are required")
    if hasattr(logits, "detach"):
        values = logits.detach().float()
        selected = values[list(digit_ids)]
        if not bool(torch.isfinite(selected).all()):
            raise ProbabilityProtocolError("digit logits must be finite")
        return float(values[digit_ids[target_digit]] - np.logaddexp.reduce(selected.cpu().numpy()))
    values = np.asarray([float(logits[i]) for i in digit_ids], dtype=float)
    if not np.isfinite(values).all():
        raise ProbabilityProtocolError("digit logits must be finite")
    m = float(values.max())
    return float(values[target_digit] - (m + math.log(float(np.exp(values - m).sum()))))


def score_gt_transition(x: int, y: int, phase_logits: Sequence[object], digit_ids: Sequence[int]):
    """Score x/y as four causal digit phases: x1, x2, y1, y2."""
    if type(x) is not int or type(y) is not int or not (0 <= x <= 99 and 0 <= y <= 99):
        raise ProbabilityProtocolError("GT reduced coordinates must be integers in 0..99")
    if len(phase_logits) != 4:
        raise ProbabilityProtocolError("four causal phase logits are required")
    digits = (x // 10, x % 10, y // 10, y % 10)
    terms = [digit_logprob(logits, digit, digit_ids) for logits, digit in zip(phase_logits, digits)]
    return {"ll_fix": float(sum(terms)), "digit_log_probs": terms, "x": x, "y": y}


def load_canonical_centerbias(path: str | Path, *, resolution: int = 100):
    """Load a DeepGaze pickle and reproduce its exp/resize/normalize/log path."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"canonical center-bias asset missing: {path}")
    with path.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, dict) or "centerbias" not in data:
        raise ProbabilityProtocolError("center-bias pickle must contain data['centerbias']")
    centerbias = np.asarray(data["centerbias"], dtype=float)
    if centerbias.ndim != 2 or not np.isfinite(centerbias).all():
        raise ProbabilityProtocolError("center-bias map must be a finite 2-D array")
    density = np.exp(centerbias - centerbias.max())
    try:
        from scipy.ndimage import zoom
        density = zoom(density, (resolution / density.shape[0], resolution / density.shape[1]), order=1)
    except ImportError as exc:
        raise RuntimeError("scipy is required for canonical center-bias resizing") from exc
    density = np.asarray(density, dtype=float)
    density /= density.sum()
    return np.log(np.clip(density, 1e-10, None))


def centerbias_logprob(log_density, x: int, y: int):
    if not (0 <= x < log_density.shape[1] and 0 <= y < log_density.shape[0]):
        raise ProbabilityProtocolError("center-bias coordinate is outside the reduced grid")
    return float(log_density[y, x])


def score_probability_query(transitions: Sequence[Mapping], centerbias_log_density=None):
    """Aggregate indices 1..N-1; index 0 is structurally undefined."""
    if not transitions:
        return {"ll_query": None, "ig_query": None, "transition_count": 0, "defined": False}
    ll = [float(t["ll_fix"]) for t in transitions]
    if not all(math.isfinite(v) for v in ll):
        raise ProbabilityProtocolError("nonfinite LL transition")
    ig = None
    if centerbias_log_density is not None:
        ig = [(value - centerbias_logprob(centerbias_log_density, int(t["x"]), int(t["y"]))) / math.log(2)
              for value, t in zip(ll, transitions)]
    return {"ll_query": sum(ll) / len(ll), "ig_query": None if ig is None else sum(ig) / len(ig),
            "transition_count": len(ll), "defined": True}


def aggregate_probability_draw(rows: Sequence[Mapping]):
    defined = [r for r in rows if r.get("defined")]
    undefined = len(rows) - len(defined)
    return {"eval_where_ll": None if not defined else sum(r["ll_query"] for r in defined) / len(defined),
            "eval_where_ig": None if not defined or any(r.get("ig_query") is None for r in defined)
                else sum(r["ig_query"] for r in defined) / len(defined),
            "ll_defined_query_count": sum(r.get("ll_query") is not None for r in rows),
            "ig_defined_query_count": sum(r.get("ig_query") is not None for r in rows),
            "probability_undefined_query_count": undefined,
            "transition_count": sum(int(r.get("transition_count", 0)) for r in rows)}


def aggregate_probability_k(draws: Sequence[Mapping]):
    """Mean only defined complete draws; undefined draws remain accounted for."""
    ll = [d["eval_where_ll"] for d in draws if d.get("eval_where_ll") is not None]
    ig = [d["eval_where_ig"] for d in draws if d.get("eval_where_ig") is not None]
    return {"eval_where_ll": None if not ll else sum(ll) / len(ll),
            "eval_where_ig": None if not ig else sum(ig) / len(ig),
            # ``defined_draw_count`` retains the historical LL alias; the
            # explicit per-metric fields remove ambiguity for IG.
            "defined_draw_count": len(ll), "undefined_draw_count": len(draws) - len(ll),
            "ll_defined_draw_count": len(ll), "ll_undefined_draw_count": len(draws) - len(ll),
            "ig_defined_draw_count": len(ig), "ig_undefined_draw_count": len(draws) - len(ig)}


@torch.inference_mode()
def score_probability_batch(bundle, episodes, *, visual_cache=None, centerbias_by_query=None):
    """Outcome-B exact-context scoring from one teacher-forced WHERE forward.

    The decorator is applied lazily below so importing this module remains
    possible in environments that do not install PyTorch. The function uses
    the exact SemGaze serialization and episode context; it never calls
    ``generate``.
    """
    from semgaze.where.collator import collate_where_batch, to_model_device
    from semgaze.model.selected_loss import forward_backbone
    from semgaze.where.serialization import serialize_xyd_record

    if not episodes:
        return []
    samples = collate_where_batch(bundle.processor, episodes, bundle.end_fix_id,
                                  bundle.context_limit, config=bundle.config,
                                  image_cache=visual_cache)
    inputs = {k: v for k, v in samples.inputs.items() if k != "labels"}
    host_pixels = inputs.get("pixel_values")
    cached_features = visual_cache is not None and visual_cache.features
    inputs = to_model_device({k: v for k, v in inputs.items()
                              if not (cached_features and k == "pixel_values")}, bundle.model)
    if visual_cache is not None:
        inputs, _ = visual_cache.fuse(bundle, inputs,
                                      [p for s in samples.samples for p in s.image_paths],
                                      host_pixel_values=host_pixels)
    outputs = forward_backbone(bundle.model, inputs, use_cache=False)
    head = bundle.model.get_output_embeddings()
    digit_ids = [bundle.processor.tokenizer.convert_tokens_to_ids(str(i)) for i in range(10)]
    if any(type(i) is not int or i < 0 for i in digit_ids) or len(set(digit_ids)) != 10:
        raise ProbabilityProtocolError("checkpoint tokenizer lacks atomic digit tokens")
    if any(bundle.processor.tokenizer.convert_ids_to_tokens(i) != str(d) for d, i in enumerate(digit_ids)):
        raise ProbabilityProtocolError("checkpoint tokenizer digit IDs are not exact atomic digits")
    rows = []
    token = bundle.config["where"]["end_fix_token"]
    for b, (episode, sample) in enumerate(zip(episodes, samples.samples)):
        target = serialize_xyd_record(episode.query, token)
        matches = list(re.finditer(
            r"\((\d{2}), (\d{2}), \d{3}\)" + re.escape(token), target))
        if len(matches) != len(episode.query.x_px):
            raise ProbabilityProtocolError("serialized GT WHERE target cannot be audited")
        transitions = []
        for fixation_index, match in enumerate(matches):
            if fixation_index == 0:
                continue
            positions = [match.start(1), match.start(1) + 1,
                         match.start(2), match.start(2) + 1]
            phases = []
            for char_position in positions:
                token_indices = [i for i, (start, end) in enumerate(sample.response_offsets)
                                 if start <= char_position < end]
                if len(token_indices) != 1 or sample.response_offsets[token_indices[0]][1] - sample.response_offsets[token_indices[0]][0] != 1:
                    raise ProbabilityProtocolError("coordinate digit is not an atomic predictor token in full context")
                target_token_index = token_indices[0]
                hidden_index = sample.response_start + target_token_index - 1
                logits = head(outputs.last_hidden_state[b, hidden_index])
                phases.append(logits)
            x = int(match.group(1)); y = int(match.group(2))
            transitions.append(score_gt_transition(x, y, phases, digit_ids))
        cb = None if centerbias_by_query is None else centerbias_by_query.get(episode.query.record_id)
        scored = score_probability_query(transitions, cb)
        scored.update(query_id=episode.query.record_id, K=len(episode.supports),
                      draw_id=episode.draw_id, outcome="B")
        rows.append(scored)
    return rows
