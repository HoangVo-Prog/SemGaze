"""Shared normalized annotation validation; no annotation repair."""
from dataclasses import dataclass


class SemanticAnnotationValidationError(ValueError):
    def __init__(self, sample_id, invariant, value):
        self.sample_id, self.invariant, self.value = sample_id, invariant, value
        super().__init__(f"{sample_id}: {invariant}; offending value={value!r}")


@dataclass(frozen=True)
class WhyGroup:
    members: tuple[int, ...]
    why: str


@dataclass(frozen=True)
class NormalizedSemantic:
    what: tuple[str, ...]
    why_groups: tuple[WhyGroup, ...]
    how: str


def validate_semantic(semantic, n, sample_id):
    def require(ok, invariant, value):
        if not ok:
            raise SemanticAnnotationValidationError(sample_id, invariant, value)
    require(n > 0 and len(semantic.what) == n, "len(what) == N > 0", semantic.what)
    for value in (*semantic.what, semantic.how):
        require(isinstance(value, str) and bool(value.strip()), "non-empty semantic text", value)
    require(bool(semantic.why_groups), "non-empty WHY groups", semantic.why_groups)
    members, minima = [], []
    for group in semantic.why_groups:
        require(bool(group.members), "non-empty group", group.members)
        require(all(type(t) is int and 1 <= t <= n for t in group.members),
                "1-based integer members in [1,N]", group.members)
        require(all(a < b for a, b in zip(group.members, group.members[1:])),
                "strictly increasing group members", group.members)
        require(isinstance(group.why, str) and bool(group.why.strip()), "non-empty WHY", group.why)
        members.extend(group.members)
        minima.append(group.members[0])
    require(sorted(members) == list(range(1, n + 1)), "groups partition 1..N exactly", members)
    require(all(a < b for a, b in zip(minima, minima[1:])), "groups ordered by minimum", minima)
    return semantic


def semantic_from_dict(payload, n, sample_id):
    try:
        result = NormalizedSemantic(tuple(payload["what"]), tuple(
            WhyGroup(tuple(g["members"]), g["why"]) for g in payload["why_groups"]), payload["how"])
    except (KeyError, TypeError) as exc:
        raise SemanticAnnotationValidationError(sample_id, "normalized semantic schema", payload) from exc
    return validate_semantic(result, n, sample_id)


def semantic_from_prediction(payload, n, sample_id):
    """Rename persisted fixations/regions fields; indices are already 1-based.

    Explicit fixation IDs define WHAT chronology, independently of JSON row order.
    Group/member order is validated, never repaired.
    """
    try:
        fixations = payload["fixations"]
        ids = [f["fixation"] for f in fixations]
        if any(type(i) is not int for i in ids) or sorted(ids) != list(range(1, n + 1)):
            raise SemanticAnnotationValidationError(sample_id, "WHAT IDs partition 1..N", ids)
        by_id = {f["fixation"]: f["what"] for f in fixations}
        return semantic_from_dict({"what": [by_id[t] for t in range(1, n + 1)],
            "why_groups": [{"members": g["fixations"], "why": g["why"]} for g in payload["regions"]],
            "how": payload["how"]}, n, sample_id)
    except (KeyError, TypeError) as exc:
        raise SemanticAnnotationValidationError(sample_id, "persisted prediction schema", payload) from exc
