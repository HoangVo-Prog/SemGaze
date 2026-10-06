from dataclasses import dataclass
import math
from .semantic import NormalizedSemantic, WhyGroup, semantic_from_dict, validate_semantic

from semgaze.model.config import default_section

# Backward-compatible direct-call defaults; runtime always supplies resolved data config.
UNSEEN_SUBJECTS = frozenset(default_section('data')['unseen_subjects'])
K_VALUES = tuple(default_section('data')['fewshot']['k_values'])


@dataclass(frozen=True)
class NormalizedRecord:
    record_id: str
    stimulus_id: str
    subject: int
    image_path: str
    image_width: int
    image_height: int
    task: str
    condition: str
    x_px: tuple[float, ...]
    y_px: tuple[float, ...]
    duration_ms: tuple[float, ...]
    semantic: NormalizedSemantic

    def __post_init__(self):
        n = len(self.x_px)
        if not n or len(self.y_px) != n or len(self.duration_ms) != n:
            raise ValueError(f"{self.record_id}: len(X) == len(Y) == len(T) > 0 required")
        if not all(type(v) in (float, int) and math.isfinite(v)
                   for seq in (self.x_px, self.y_px, self.duration_ms) for v in seq):
            raise ValueError(f"{self.record_id}: finite numeric fixation fields required")
        if any(type(v) is not int or v <= 0 for v in (self.image_width, self.image_height)):
            raise ValueError(f"{self.record_id}: positive annotation image dimensions required")
        if type(self.subject) is not int or self.condition not in ("present", "absent"):
            raise ValueError(f"{self.record_id}: invalid subject/condition")
        if not all(isinstance(v, str) and v.strip() for v in
                   (self.record_id, self.stimulus_id, self.image_path, self.task)):
            raise ValueError("non-empty record identity, image path and task required")
        validate_semantic(self.semantic, n, self.record_id)


@dataclass(frozen=True)
class FlatEpisode:
    supports: tuple[NormalizedRecord, ...]
    query: NormalizedRecord
    draw_id: int | None = None

    def __post_init__(self):
        if not self.supports:
            raise ValueError("at least one support is required")
        if any(s.subject != self.query.subject for s in self.supports):
            raise ValueError("supports and query must have the same subject")
        stimuli = [s.stimulus_id for s in self.supports]
        if len(set(stimuli)) != len(stimuli) or self.query.stimulus_id in stimuli:
            raise ValueError("support stimuli must be distinct and exclude the query stimulus")


def normalized_record_from_dict(payload):
    values = dict(payload)
    for key in ("x_px", "y_px", "duration_ms"):
        values[key] = tuple(values[key])
    values["semantic"] = semantic_from_dict(values["semantic"], len(values["x_px"]), values["record_id"])
    return NormalizedRecord(**values)


def normalized_episode_from_dict(payload):
    return FlatEpisode(tuple(normalized_record_from_dict(s) for s in payload["supports"]),
                       normalized_record_from_dict(payload["query"]))


def require_seen_training_episode(episode, unseen_subjects=UNSEEN_SUBJECTS):
    if episode.query.subject in unseen_subjects:
        raise ValueError(f"{episode.query.record_id}: unseen subjects never contribute training gradients")
