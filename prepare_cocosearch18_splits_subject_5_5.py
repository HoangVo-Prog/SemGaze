#!/usr/bin/env python3
"""Build SemGaze COCO-Search18 subject-stratified 5%/5% evaluation splits.

Protocol
--------
- Subjects 7/8/9 are the unseen-subject group; all others are seen subjects.
- test.json contains unseen-subject records only.
- test_seen.json contains seen-subject records only.
- Each test file targets 5% of ALL records in its variant, not 5% of its
  corresponding subject pool. Example: for 20k records, each target is ~1k.
- Split safety is enforced at stimulus/image level: every image used by either
  test file belongs to one shared held-out evaluation-image pool, and NO record
  from those images may appear in train.json.
- test.json and test_seen.json intentionally share the same held-out image pool.
  This controls stimulus difficulty when comparing unseen vs seen subjects and
  avoids creating two unnecessarily large, disjoint test-image pools.
- Because the held-out pool must contain enough unseen records to reach 5% of
  the full dataset, its image fraction can be >10%. This is expected. The 5/5
  targets are RECORD targets, not image-ratio targets.
- If the held-out pool contains more eligible records than needed, deterministic
  role-specific subsampling selects exactly the requested test counts; unused
  held-out records are excluded rather than leaked back into training.
- There is NO validation split. Final test/test_seen results must not be used
  for checkpoint or hyperparameter selection.
- tp_only / ta_only / all use the SAME master train/eval image partition.
- Final few-shot supports for unseen subjects come only from TRAIN images.
- Unseen subjects are 7/8/9; K in {1,5,10}; 10 frozen support draws per K.

Run from repository root:

    python prepare_cocosearch18_splits_subject_5_5.py

Default input:
    data/COCO_Search18/COCOSearch-18.json

Default output:
    data/COCO_Search18/split_subject_5_5/{tp_only,ta_only,all}/...

The script is stdlib-only and deterministic.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


PROTOCOL_VERSION = "cocosearch18_semgaze_subject_stratified_5_5_shared_eval_v1"
DEFAULT_INPUT = Path("data/COCO_Search18/COCOSearch-18.json")
DEFAULT_OUTPUT_ROOT = Path("data/COCO_Search18/split_subject_5_5")

UNSEEN_SUBJECTS = (7, 8, 9)
KS = (1, 5, 10)
NUM_DRAWS = 10
TEST_UNSEEN_RATIO_OF_TOTAL = 0.05
TEST_SEEN_RATIO_OF_TOTAL = 0.05
MAX_ATTEMPTS = 2_000

# Distribution-search weights. Lower total score is better.
WEIGHT_TASK_TV = 4.0
WEIGHT_SUBJECT_TV = 2.0
WEIGHT_CONDITION_TV = 2.0
WEIGHT_HOLDOUT_IMAGE_RATIO = 2.0
WEIGHT_EXCLUDED_RECORD_RATIO = 4.0

VARIANTS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "tp_only": lambda r: r["condition"] == "present",
    "ta_only": lambda r: r["condition"] == "absent",
    "all": lambda r: True,
}

REQUIRED_FIELDS = (
    "name",
    "subject",
    "task",
    "condition",
    "X",
    "Y",
    "T",
    "prediction",
)


class SplitBuildError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dump_json(path: Path, obj: Any, *, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        else:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    tmp.replace(path)


def canonical_record_id(record: dict[str, Any]) -> str:
    return f"coco_semgaze::{record['subject']}::{record['task']}::{record['name']}"


def canonical_trial_key(record: dict[str, Any]) -> str:
    return f"{record['task']}::{record['name']}"


def validate_record(record: dict[str, Any], index: int) -> None:
    missing = [k for k in REQUIRED_FIELDS if k not in record]
    if missing:
        raise SplitBuildError(f"record[{index}] missing required fields: {missing}")

    if not isinstance(record["name"], str) or not record["name"].strip():
        raise SplitBuildError(f"record[{index}] has invalid name")
    if not isinstance(record["task"], str) or not record["task"].strip():
        raise SplitBuildError(f"record[{index}] has invalid task")
    if record["condition"] not in {"present", "absent"}:
        raise SplitBuildError(
            f"record[{index}] condition must be 'present' or 'absent', "
            f"got {record['condition']!r}"
        )

    x, y, t = record["X"], record["Y"], record["T"]
    if not all(isinstance(v, list) for v in (x, y, t)):
        raise SplitBuildError(f"record[{index}] X/Y/T must be lists")
    if not (len(x) == len(y) == len(t) and len(x) > 0):
        raise SplitBuildError(
            f"record[{index}] requires len(X)==len(Y)==len(T)>0; "
            f"got {len(x)}, {len(y)}, {len(t)}"
        )

    pred = record["prediction"]
    if not isinstance(pred, dict):
        raise SplitBuildError(f"record[{index}] prediction must be an object")

    fixations = pred.get("fixations")
    regions = pred.get("regions")
    how = pred.get("how")
    n = len(x)

    if not isinstance(fixations, list) or len(fixations) != n:
        raise SplitBuildError(
            f"record[{index}] prediction.fixations must have exactly {n} items"
        )
    if not isinstance(regions, list) or not regions:
        raise SplitBuildError(
            f"record[{index}] prediction.regions must be a non-empty list"
        )
    if not isinstance(how, str) or not how.strip():
        raise SplitBuildError(f"record[{index}] prediction.how must be non-empty")

    fixation_ids: list[int] = []
    for j, fx in enumerate(fixations):
        if not isinstance(fx, dict):
            raise SplitBuildError(f"record[{index}] fixation[{j}] is not an object")
        fid = fx.get("fixation")
        what = fx.get("what")
        if not isinstance(fid, int):
            raise SplitBuildError(f"record[{index}] fixation[{j}].fixation must be int")
        if not isinstance(what, str) or not what.strip():
            raise SplitBuildError(f"record[{index}] fixation[{j}].what must be non-empty")
        fixation_ids.append(fid)

    expected = list(range(1, n + 1))
    if sorted(fixation_ids) != expected or len(set(fixation_ids)) != n:
        raise SplitBuildError(
            f"record[{index}] fixation IDs must cover 1..{n} exactly once; "
            f"got {fixation_ids}"
        )

    region_membership: list[int] = []
    for j, region in enumerate(regions):
        if not isinstance(region, dict):
            raise SplitBuildError(f"record[{index}] region[{j}] is not an object")
        ids = region.get("fixations")
        why = region.get("why")
        if not isinstance(ids, list) or not ids:
            raise SplitBuildError(f"record[{index}] region[{j}].fixations must be non-empty")
        if not all(isinstance(fid, int) for fid in ids):
            raise SplitBuildError(
                f"record[{index}] region[{j}].fixations must contain ints"
            )
        if not isinstance(why, str) or not why.strip():
            raise SplitBuildError(f"record[{index}] region[{j}].why must be non-empty")
        region_membership.extend(ids)

    if sorted(region_membership) != expected or len(set(region_membership)) != n:
        raise SplitBuildError(
            f"record[{index}] regions must partition fixation IDs 1..{n} exactly once; "
            f"got {region_membership}"
        )


def load_and_validate(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SplitBuildError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list) or not records:
        raise SplitBuildError("Curated input must be a non-empty JSON list")

    seen_record_ids: set[str] = set()
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise SplitBuildError(f"record[{i}] is not a JSON object")
        validate_record(record, i)
        rid = canonical_record_id(record)
        if rid in seen_record_ids:
            raise SplitBuildError(f"Duplicate canonical record_id: {rid}")
        seen_record_ids.add(rid)
    return records


def variant_records(
    records: list[dict[str, Any]], variant: str
) -> list[dict[str, Any]]:
    pred = VARIANTS[variant]
    filtered = [r for r in records if pred(r)]
    if not filtered:
        raise SplitBuildError(f"Variant {variant!r} contains no eligible records")
    return filtered


def index_by_subject(
    records: Iterable[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        out[int(record["subject"])].append(record)
    return dict(out)


def shared_support_candidates_without_split(
    records: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Eligible shared (task,image) support units for unseen subjects 7/8/9."""
    by_trial_subject: dict[tuple[str, str], dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for record in records:
        subject = int(record["subject"])
        if subject not in UNSEEN_SUBJECTS:
            continue
        key = (record["task"], record["name"])
        by_trial_subject[key][subject] += 1

    eligible: list[tuple[str, str]] = []
    for key, per_subject in by_trial_subject.items():
        if all(per_subject.get(subject, 0) == 1 for subject in UNSEEN_SUBJECTS):
            eligible.append(key)
    return sorted(eligible)


def shared_support_candidates(
    records: list[dict[str, Any]], image_split: dict[str, str]
) -> list[tuple[str, str]]:
    """Eligible TRAIN shared (task,image) support units for unseen subjects."""
    return [
        key
        for key in shared_support_candidates_without_split(records)
        if image_split[key[1]] == "train"
    ]


def total_variation(global_counts: Counter[Any], test_counts: Counter[Any]) -> float:
    global_total = sum(global_counts.values())
    test_total = sum(test_counts.values())
    if global_total <= 0 or test_total <= 0:
        return math.inf
    keys = set(global_counts) | set(test_counts)
    return 0.5 * sum(
        abs(global_counts.get(k, 0) / global_total - test_counts.get(k, 0) / test_total)
        for k in keys
    )



def build_search_index(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Precompute variant/image summaries used by deterministic split search."""
    images = sorted({r["name"] for r in records})
    per_variant: dict[str, dict[str, Any]] = {}

    for variant in VARIANTS:
        rows = variant_records(records, variant)
        seen_subjects = sorted(
            {int(r["subject"]) for r in rows} - set(UNSEEN_SUBJECTS)
        )
        target_unseen = round(TEST_UNSEEN_RATIO_OF_TOTAL * len(rows))
        target_seen = round(TEST_SEEN_RATIO_OF_TOTAL * len(rows))
        if target_unseen <= 0 or target_seen <= 0:
            raise SplitBuildError(
                f"Variant {variant!r} is too small for 5%/5% evaluation targets"
            )

        image_unseen_count: Counter[str] = Counter()
        image_seen_count: Counter[str] = Counter()
        global_unseen_tasks: Counter[str] = Counter()
        global_unseen_subjects: Counter[int] = Counter()
        global_unseen_conditions: Counter[str] = Counter()
        global_seen_tasks: Counter[str] = Counter()
        global_seen_subjects: Counter[int] = Counter()
        global_seen_conditions: Counter[str] = Counter()

        for r in rows:
            image = r["name"]
            subject = int(r["subject"])
            if subject in UNSEEN_SUBJECTS:
                image_unseen_count[image] += 1
                global_unseen_tasks[r["task"]] += 1
                global_unseen_subjects[subject] += 1
                global_unseen_conditions[r["condition"]] += 1
            else:
                image_seen_count[image] += 1
                global_seen_tasks[r["task"]] += 1
                global_seen_subjects[subject] += 1
                global_seen_conditions[r["condition"]] += 1

        total_unseen = sum(global_unseen_subjects.values())
        total_seen = sum(global_seen_subjects.values())
        if target_unseen > total_unseen:
            raise SplitBuildError(
                f"Variant {variant!r}: unseen target {target_unseen} exceeds "
                f"available unseen records {total_unseen}"
            )
        if target_seen > total_seen:
            raise SplitBuildError(
                f"Variant {variant!r}: seen target {target_seen} exceeds "
                f"available seen records {total_seen}"
            )

        shared_trials = shared_support_candidates_without_split(rows)
        shared_images = {image for _, image in shared_trials}
        per_variant[variant] = {
            "rows": rows,
            "seen_subject_ids": seen_subjects,
            "total_records": len(rows),
            "target_test_records": target_unseen,
            "target_test_seen_records": target_seen,
            "total_unseen_records": total_unseen,
            "total_seen_records": total_seen,
            "image_unseen_count": image_unseen_count,
            "image_seen_count": image_seen_count,
            "global_unseen_tasks": global_unseen_tasks,
            "global_unseen_subjects": global_unseen_subjects,
            "global_unseen_conditions": global_unseen_conditions,
            "global_seen_tasks": global_seen_tasks,
            "global_seen_subjects": global_seen_subjects,
            "global_seen_conditions": global_seen_conditions,
            "shared_support_images": shared_images,
        }

    return {"images": images, "variants": per_variant}


def candidate_eval_images(
    search_index: dict[str, Any], attempt_id: int
) -> set[str]:
    """Return the smallest hash-ranked shared eval-image prefix with enough capacity.

    Capacity is checked for BOTH subject roles and ALL output variants. Therefore
    exact 5%-of-total record subsampling can be performed later without allowing
    any held-out image back into training.
    """
    images: list[str] = search_index["images"]
    ranked = sorted(
        images,
        key=lambda name: stable_digest(
            f"{PROTOCOL_VERSION}|master|{attempt_id}|image::{name}"
        ),
    )

    unseen_capacity = {variant: 0 for variant in VARIANTS}
    seen_capacity = {variant: 0 for variant in VARIANTS}
    eval_images: set[str] = set()

    for image in ranked:
        eval_images.add(image)
        for variant, idx in search_index["variants"].items():
            unseen_capacity[variant] += idx["image_unseen_count"].get(image, 0)
            seen_capacity[variant] += idx["image_seen_count"].get(image, 0)

        enough = all(
            unseen_capacity[variant] >= idx["target_test_records"]
            and seen_capacity[variant] >= idx["target_test_seen_records"]
            for variant, idx in search_index["variants"].items()
        )
        if enough:
            if len(eval_images) >= len(images):
                raise SplitBuildError("Evaluation image pool consumed all images")
            return eval_images

    raise SplitBuildError(
        "Could not build an evaluation-image pool with enough seen/unseen capacity"
    )


def select_eval_rows(
    rows: list[dict[str, Any]],
    eval_images: set[str],
    *,
    unseen: bool,
    target: int,
    attempt_id: int,
    variant: str,
) -> list[dict[str, Any]]:
    """Select exactly target rows from the held-out image pool deterministically."""
    role = "unseen" if unseen else "seen"
    candidates = [
        r
        for r in rows
        if r["name"] in eval_images
        and ((int(r["subject"]) in UNSEEN_SUBJECTS) == unseen)
    ]
    if len(candidates) < target:
        raise SplitBuildError(
            f"variant={variant} role={role}: need {target} held-out records, "
            f"found {len(candidates)}"
        )
    ranked = sorted(
        candidates,
        key=lambda r: stable_digest(
            f"{PROTOCOL_VERSION}|record|{attempt_id}|{variant}|{role}|"
            f"{canonical_record_id(r)}"
        ),
    )
    return ranked[:target]


def summarize_selected_rows(rows: list[dict[str, Any]]) -> dict[str, Counter[Any]]:
    return {
        "tasks": Counter(r["task"] for r in rows),
        "subjects": Counter(int(r["subject"]) for r in rows),
        "conditions": Counter(r["condition"] for r in rows),
    }


def build_candidate_summary(
    search_index: dict[str, Any], eval_images: set[str], attempt_id: int
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for variant, idx in search_index["variants"].items():
        rows = idx["rows"]
        test_rows = select_eval_rows(
            rows,
            eval_images,
            unseen=True,
            target=idx["target_test_records"],
            attempt_id=attempt_id,
            variant=variant,
        )
        test_seen_rows = select_eval_rows(
            rows,
            eval_images,
            unseen=False,
            target=idx["target_test_seen_records"],
            attempt_id=attempt_id,
            variant=variant,
        )
        test_stats = summarize_selected_rows(test_rows)
        test_seen_stats = summarize_selected_rows(test_seen_rows)

        eval_unseen_pool = sum(
            idx["image_unseen_count"].get(image, 0) for image in eval_images
        )
        eval_seen_pool = sum(
            idx["image_seen_count"].get(image, 0) for image in eval_images
        )
        excluded = (
            eval_unseen_pool
            + eval_seen_pool
            - len(test_rows)
            - len(test_seen_rows)
        )
        out[variant] = {
            "test_rows": test_rows,
            "test_seen_rows": test_seen_rows,
            "test_tasks": test_stats["tasks"],
            "test_subjects": test_stats["subjects"],
            "test_conditions": test_stats["conditions"],
            "test_seen_tasks": test_seen_stats["tasks"],
            "test_seen_subjects": test_seen_stats["subjects"],
            "test_seen_conditions": test_seen_stats["conditions"],
            "eval_unseen_candidate_records": eval_unseen_pool,
            "eval_seen_candidate_records": eval_seen_pool,
            "excluded_heldout_records": excluded,
            "train_shared_support_distinct_images": len(
                idx["shared_support_images"] - eval_images
            ),
        }
    return out


def candidate_errors(
    search_index: dict[str, Any],
    eval_images: set[str],
    summary: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    required_support_images = NUM_DRAWS * max(KS)

    if not eval_images:
        errors.append("MASTER: empty evaluation-image pool")
    if len(eval_images) >= len(search_index["images"]):
        errors.append("MASTER: no train images remain")

    for variant, idx in search_index["variants"].items():
        s = summary[variant]
        if len(s["test_rows"]) != idx["target_test_records"]:
            errors.append(f"{variant}: incorrect unseen test record count")
        if len(s["test_seen_rows"]) != idx["target_test_seen_records"]:
            errors.append(f"{variant}: incorrect seen test record count")

        missing_unseen_tasks = set(idx["global_unseen_tasks"]) - set(s["test_tasks"])
        if missing_unseen_tasks:
            errors.append(
                f"{variant}: test misses unseen tasks={sorted(missing_unseen_tasks)}"
            )
        for subject in UNSEEN_SUBJECTS:
            if idx["global_unseen_subjects"].get(subject, 0) > 0 and s[
                "test_subjects"
            ].get(subject, 0) < 1:
                errors.append(
                    f"{variant}: unseen subject {subject} has no test record"
                )

        missing_seen_tasks = set(idx["global_seen_tasks"]) - set(
            s["test_seen_tasks"]
        )
        if missing_seen_tasks:
            errors.append(
                f"{variant}: test_seen misses seen tasks={sorted(missing_seen_tasks)}"
            )
        for subject in idx["seen_subject_ids"]:
            if s["test_seen_subjects"].get(subject, 0) < 1:
                errors.append(
                    f"{variant}: seen subject {subject} has no test_seen record"
                )

        if s["train_shared_support_distinct_images"] < required_support_images:
            errors.append(
                f"{variant}: insufficient shared train support images: "
                f"need >= {required_support_images}, "
                f"got {s['train_shared_support_distinct_images']}"
            )
    return errors


def candidate_score(
    search_index: dict[str, Any],
    eval_images: set[str],
    summary: dict[str, dict[str, Any]],
) -> tuple[float, dict[str, dict[str, float]]]:
    total = 0.0
    diagnostics: dict[str, dict[str, float]] = {}
    holdout_image_ratio = len(eval_images) / len(search_index["images"])

    for variant, idx in search_index["variants"].items():
        s = summary[variant]
        unseen_task_tv = total_variation(
            idx["global_unseen_tasks"], s["test_tasks"]
        )
        unseen_subject_tv = total_variation(
            idx["global_unseen_subjects"], s["test_subjects"]
        )
        unseen_condition_tv = total_variation(
            idx["global_unseen_conditions"], s["test_conditions"]
        )
        seen_task_tv = total_variation(
            idx["global_seen_tasks"], s["test_seen_tasks"]
        )
        seen_subject_tv = total_variation(
            idx["global_seen_subjects"], s["test_seen_subjects"]
        )
        seen_condition_tv = total_variation(
            idx["global_seen_conditions"], s["test_seen_conditions"]
        )
        excluded_ratio = s["excluded_heldout_records"] / idx["total_records"]
        condition_term = (
            unseen_condition_tv + seen_condition_tv if variant == "all" else 0.0
        )
        variant_score = (
            WEIGHT_TASK_TV * (unseen_task_tv + seen_task_tv)
            + WEIGHT_SUBJECT_TV * (unseen_subject_tv + seen_subject_tv)
            + WEIGHT_CONDITION_TV * condition_term
            + WEIGHT_HOLDOUT_IMAGE_RATIO * holdout_image_ratio
            + WEIGHT_EXCLUDED_RECORD_RATIO * excluded_ratio
        )
        total += variant_score
        diagnostics[variant] = {
            "score": variant_score,
            "unseen_test_task_tv": unseen_task_tv,
            "unseen_test_subject_tv": unseen_subject_tv,
            "unseen_test_condition_tv": unseen_condition_tv,
            "seen_test_task_tv": seen_task_tv,
            "seen_test_subject_tv": seen_subject_tv,
            "seen_test_condition_tv": seen_condition_tv,
            "heldout_image_ratio": holdout_image_ratio,
            "excluded_heldout_record_ratio": excluded_ratio,
        }
    return total, diagnostics


def choose_master_split(
    records: list[dict[str, Any]], max_attempts: int
) -> tuple[int, dict[str, str], float, dict[str, Any]]:
    search_index = build_search_index(records)
    best: tuple[int, set[str], float, dict[str, Any]] | None = None
    valid_attempts = 0
    last_errors: list[str] = []

    for attempt_id in range(max_attempts):
        eval_images = candidate_eval_images(search_index, attempt_id)
        summary = build_candidate_summary(search_index, eval_images, attempt_id)
        errors = candidate_errors(search_index, eval_images, summary)
        if errors:
            last_errors = errors
            continue

        valid_attempts += 1
        score, diagnostics = candidate_score(search_index, eval_images, summary)
        payload = {
            "valid_attempts_seen": valid_attempts,
            "eval_image_count": len(eval_images),
            "eval_image_ratio": len(eval_images) / len(search_index["images"]),
            "candidate_summary": {
                variant: {
                    "target_test_records": idx["target_test_records"],
                    "target_test_seen_records": idx["target_test_seen_records"],
                    "eval_unseen_candidate_records": summary[variant][
                        "eval_unseen_candidate_records"
                    ],
                    "eval_seen_candidate_records": summary[variant][
                        "eval_seen_candidate_records"
                    ],
                    "excluded_heldout_records": summary[variant][
                        "excluded_heldout_records"
                    ],
                    "train_shared_support_distinct_images": summary[variant][
                        "train_shared_support_distinct_images"
                    ],
                    "test_task_counts": dict(
                        sorted(summary[variant]["test_tasks"].items())
                    ),
                    "test_subject_counts": {
                        str(k): v
                        for k, v in sorted(
                            summary[variant]["test_subjects"].items()
                        )
                    },
                    "test_seen_task_counts": dict(
                        sorted(summary[variant]["test_seen_tasks"].items())
                    ),
                    "test_seen_subject_counts": {
                        str(k): v
                        for k, v in sorted(
                            summary[variant]["test_seen_subjects"].items()
                        )
                    },
                }
                for variant, idx in search_index["variants"].items()
            },
            "distribution_diagnostics": diagnostics,
        }
        if best is None or score < best[2]:
            best = (attempt_id, eval_images, score, payload)

    if best is None:
        raise SplitBuildError(
            f"No valid subject-stratified 5%/5% split found in attempts "
            f"[0, {max_attempts - 1}]. Last errors: {last_errors}"
        )

    attempt_id, eval_images, score, payload = best
    image_split = {
        image: ("eval" if image in eval_images else "train")
        for image in search_index["images"]
    }
    payload["num_valid_candidates"] = valid_attempts
    payload["selected_attempt_id"] = attempt_id
    payload["selected_score"] = score
    return attempt_id, image_split, score, payload


def materialized_record(
    record: dict[str, Any], variant: str, split: str
) -> dict[str, Any]:
    out = copy.deepcopy(record)
    out["record_id"] = canonical_record_id(record)
    out["stimulus_id"] = record["name"]
    out["trial_key"] = canonical_trial_key(record)
    out["split"] = split
    out["variant"] = variant
    return out


def build_final_support_draws(
    records: list[dict[str, Any]], image_split: dict[str, str]
) -> dict[str, list[list[dict[str, Any]]]]:
    resolver: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    counts: Counter[tuple[tuple[str, str], int]] = Counter()

    for row in records:
        if image_split[row["name"]] != "train":
            continue
        subject = int(row["subject"])
        if subject not in UNSEEN_SUBJECTS:
            continue
        key = (row["task"], row["name"])
        counts[(key, subject)] += 1
        resolver[key][subject] = row

    eligible: list[tuple[str, str]] = []
    for key in sorted(resolver):
        if all(counts[(key, subject)] == 1 for subject in UNSEEN_SUBJECTS):
            eligible.append(key)

    draws_by_k: dict[str, list[list[dict[str, Any]]]] = {}
    for k in KS:
        units = list(eligible)
        rng = random.Random(k)
        rng.shuffle(units)
        retained: list[tuple[str, str]] = []
        used_images: set[str] = set()
        for task, image_name in units:
            if image_name in used_images:
                continue
            used_images.add(image_name)
            retained.append((task, image_name))
            if len(retained) == NUM_DRAWS * k:
                break

        required = NUM_DRAWS * k
        if len(retained) < required:
            raise SplitBuildError(
                f"Need {required} exclusive shared support images for K={k}, "
                f"found only {len(retained)}"
            )

        k_draws: list[list[dict[str, Any]]] = []
        for draw_id in range(NUM_DRAWS):
            block = retained[draw_id * k : (draw_id + 1) * k]
            entries: list[dict[str, Any]] = []
            for task, image_name in block:
                key = (task, image_name)
                entries.append(
                    {
                        "task": task,
                        "image_name": image_name,
                        "trial_key": f"{task}::{image_name}",
                        "resolved_record_id_by_subject": {
                            str(subject): canonical_record_id(resolver[key][subject])
                            for subject in UNSEEN_SUBJECTS
                        },
                    }
                )
            k_draws.append(entries)
        draws_by_k[str(k)] = k_draws
    return draws_by_k


def compute_variant_selection(
    records: list[dict[str, Any]],
    image_split: dict[str, str],
    attempt_id: int,
    variant: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eval_images = {image for image, split in image_split.items() if split == "eval"}
    target = round(TEST_UNSEEN_RATIO_OF_TOTAL * len(records))
    target_seen = round(TEST_SEEN_RATIO_OF_TOTAL * len(records))
    test_rows = select_eval_rows(
        records,
        eval_images,
        unseen=True,
        target=target,
        attempt_id=attempt_id,
        variant=variant,
    )
    test_seen_rows = select_eval_rows(
        records,
        eval_images,
        unseen=False,
        target=target_seen,
        attempt_id=attempt_id,
        variant=variant,
    )
    return test_rows, test_seen_rows


def assert_materialization_contract(
    records: list[dict[str, Any]],
    image_split: dict[str, str],
    split_rows: dict[str, list[dict[str, Any]]],
) -> None:
    train_ids = {r["record_id"] for r in split_rows["train"]}
    test_ids = {r["record_id"] for r in split_rows["test"]}
    test_seen_ids = {r["record_id"] for r in split_rows["test_seen"]}
    if train_ids & test_ids or train_ids & test_seen_ids or test_ids & test_seen_ids:
        raise SplitBuildError("A runtime record appears in more than one split")

    eval_images = {image for image, split in image_split.items() if split == "eval"}
    train_images = {r["stimulus_id"] for r in split_rows["train"]}
    if train_images & eval_images:
        raise SplitBuildError("Held-out evaluation image leaked into train.json")

    for row in split_rows["test"]:
        if int(row["subject"]) not in UNSEEN_SUBJECTS:
            raise SplitBuildError("test.json contains a seen-subject record")
        if row["stimulus_id"] not in eval_images:
            raise SplitBuildError("test.json contains a non-held-out image")
    for row in split_rows["test_seen"]:
        if int(row["subject"]) in UNSEEN_SUBJECTS:
            raise SplitBuildError("test_seen.json contains an unseen-subject record")
        if row["stimulus_id"] not in eval_images:
            raise SplitBuildError("test_seen.json contains a non-held-out image")

    expected_train = {
        canonical_record_id(r)
        for r in records
        if image_split[r["name"]] == "train"
    }
    if train_ids != expected_train:
        raise SplitBuildError(
            "train.json must contain every source record on train images and no others"
        )


def coverage_errors(
    records: list[dict[str, Any]],
    image_split: dict[str, str],
    attempt_id: int,
    variant: str,
) -> list[str]:
    errors: list[str] = []
    test_rows, test_seen_rows = compute_variant_selection(
        records, image_split, attempt_id, variant
    )
    target = round(TEST_UNSEEN_RATIO_OF_TOTAL * len(records))
    target_seen = round(TEST_SEEN_RATIO_OF_TOTAL * len(records))
    if len(test_rows) != target:
        errors.append(f"test has {len(test_rows)} records; expected {target}")
    if len(test_seen_rows) != target_seen:
        errors.append(
            f"test_seen has {len(test_seen_rows)} records; expected {target_seen}"
        )

    unseen_tasks = {
        r["task"] for r in records if int(r["subject"]) in UNSEEN_SUBJECTS
    }
    if unseen_tasks - {r["task"] for r in test_rows}:
        errors.append(
            f"test misses unseen tasks={sorted(unseen_tasks - {r['task'] for r in test_rows})}"
        )
    for subject in UNSEEN_SUBJECTS:
        if any(int(r["subject"]) == subject for r in records) and not any(
            int(r["subject"]) == subject for r in test_rows
        ):
            errors.append(f"unseen subject {subject} has no test query record")

    seen_subjects = sorted(
        {int(r["subject"]) for r in records} - set(UNSEEN_SUBJECTS)
    )
    seen_tasks = {
        r["task"] for r in records if int(r["subject"]) not in UNSEEN_SUBJECTS
    }
    if seen_tasks - {r["task"] for r in test_seen_rows}:
        errors.append(
            f"test_seen misses seen tasks={sorted(seen_tasks - {r['task'] for r in test_seen_rows})}"
        )
    for subject in seen_subjects:
        if not any(int(r["subject"]) == subject for r in test_seen_rows):
            errors.append(f"seen subject {subject} has no test_seen query record")

    shared = shared_support_candidates(records, image_split)
    required = NUM_DRAWS * max(KS)
    shared_distinct_images = len({image_name for _, image_name in shared})
    if shared_distinct_images < required:
        errors.append(
            "insufficient shared train support coverage: "
            f"need >= {required} distinct images, got {shared_distinct_images}"
        )
    return errors


def make_report(
    variant: str,
    records: list[dict[str, Any]],
    image_split: dict[str, str],
    attempt_id: int,
    selected_score: float,
) -> dict[str, Any]:
    test_rows, test_seen_rows = compute_variant_selection(
        records, image_split, attempt_id, variant
    )
    test_ids = {canonical_record_id(r) for r in test_rows}
    test_seen_ids = {canonical_record_id(r) for r in test_seen_rows}
    eval_images = {image for image, split in image_split.items() if split == "eval"}
    train_rows = [r for r in records if image_split[r["name"]] == "train"]
    heldout_rows = [r for r in records if r["name"] in eval_images]
    excluded_rows = [
        r
        for r in heldout_rows
        if canonical_record_id(r) not in test_ids | test_seen_ids
    ]
    total_unseen = sum(
        1 for r in records if int(r["subject"]) in UNSEEN_SUBJECTS
    )
    total_seen = len(records) - total_unseen
    shared = shared_support_candidates(records, image_split)

    def counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
        c = Counter(str(r[key]) for r in rows)
        return dict(sorted(c.items()))

    return {
        "protocol_version": PROTOCOL_VERSION,
        "variant": variant,
        "master_attempt_id": attempt_id,
        "master_selected_score": selected_score,
        "total_source_records": len(records),
        "target_record_ratio_over_source": {
            "test_unseen": TEST_UNSEEN_RATIO_OF_TOTAL,
            "test_seen": TEST_SEEN_RATIO_OF_TOTAL,
        },
        "target_record_counts": {
            "test": round(TEST_UNSEEN_RATIO_OF_TOTAL * len(records)),
            "test_seen": round(TEST_SEEN_RATIO_OF_TOTAL * len(records)),
        },
        "runtime_record_counts_by_split": {
            "train": len(train_rows),
            "test": len(test_rows),
            "test_seen": len(test_seen_rows),
        },
        "runtime_record_ratio_by_split_over_source": {
            "train": len(train_rows) / len(records),
            "test": len(test_rows) / len(records),
            "test_seen": len(test_seen_rows) / len(records),
        },
        "test_unseen_ratio_within_unseen_pool": (
            len(test_rows) / total_unseen if total_unseen else 0.0
        ),
        "test_seen_ratio_within_seen_pool": (
            len(test_seen_rows) / total_seen if total_seen else 0.0
        ),
        "master_image_counts": {
            "train": sum(1 for split in image_split.values() if split == "train"),
            "eval": len(eval_images),
        },
        "master_eval_image_ratio": len(eval_images) / len(image_split),
        "test_and_test_seen_share_eval_image_pool": True,
        "heldout_candidate_record_counts": {
            "unseen": sum(
                1
                for r in heldout_rows
                if int(r["subject"]) in UNSEEN_SUBJECTS
            ),
            "seen": sum(
                1
                for r in heldout_rows
                if int(r["subject"]) not in UNSEEN_SUBJECTS
            ),
        },
        "excluded_heldout_records": {
            "count": len(excluded_rows),
            "ratio_over_source": len(excluded_rows) / len(records),
        },
        "subject_record_counts_by_runtime_split": {
            "train": counts(train_rows, "subject"),
            "test": counts(test_rows, "subject"),
            "test_seen": counts(test_seen_rows, "subject"),
        },
        "task_record_counts_by_runtime_split": {
            "train": counts(train_rows, "task"),
            "test": counts(test_rows, "task"),
            "test_seen": counts(test_seen_rows, "task"),
        },
        "condition_record_counts_by_runtime_split": {
            "train": counts(train_rows, "condition"),
            "test": counts(test_rows, "condition"),
            "test_seen": counts(test_seen_rows, "condition"),
        },
        "shared_train_support_trial_keys": len(shared),
        "shared_train_support_distinct_images": len({image for _, image in shared}),
        "coverage_errors": coverage_errors(
            records, image_split, attempt_id, variant
        ),
    }


def build_variant(
    source_records: list[dict[str, Any]],
    source_path: Path,
    output_root: Path,
    variant: str,
    image_split: dict[str, str],
    attempt_id: int,
    selected_score: float,
    force: bool,
) -> dict[str, Any]:
    records = variant_records(source_records, variant)
    errors = coverage_errors(records, image_split, attempt_id, variant)
    if errors:
        raise SplitBuildError(f"Master split invalid for variant={variant}: {errors}")

    variant_dir = output_root / variant
    if variant_dir.exists() and any(variant_dir.iterdir()):
        if not force:
            raise SplitBuildError(
                f"Output directory is not empty: {variant_dir}. Use --force to regenerate."
            )
        shutil.rmtree(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)

    test_source_rows, test_seen_source_rows = compute_variant_selection(
        records, image_split, attempt_id, variant
    )
    test_ids = {canonical_record_id(r) for r in test_source_rows}
    test_seen_ids = {canonical_record_id(r) for r in test_seen_source_rows}

    split_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "test": [],
        "test_seen": [],
    }
    for record in records:
        rid = canonical_record_id(record)
        if image_split[record["name"]] == "train":
            split_rows["train"].append(materialized_record(record, variant, "train"))
        elif rid in test_ids:
            split_rows["test"].append(materialized_record(record, variant, "test"))
        elif rid in test_seen_ids:
            split_rows["test_seen"].append(
                materialized_record(record, variant, "test_seen")
            )

    assert_materialization_contract(records, image_split, split_rows)
    support_draws = build_final_support_draws(records, image_split)

    split_paths = {
        split: variant_dir / f"{split}.json"
        for split in ("train", "test", "test_seen")
    }
    for split, path in split_paths.items():
        dump_json(path, split_rows[split], pretty=False)
    split_hashes = {split: sha256_file(path) for split, path in split_paths.items()}

    source_hash = sha256_file(source_path)
    seen_subjects = sorted(
        {int(r["subject"]) for r in records} - set(UNSEEN_SUBJECTS)
    )
    eval_images = sorted(
        image for image, split in image_split.items() if split == "eval"
    )
    train_images = sorted(
        image for image, split in image_split.items() if split == "train"
    )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "COCO-Search18",
        "benchmark_kind": "SemGaze-curated",
        "variant": variant,
        "condition_filter": {
            "tp_only": "condition == 'present'",
            "ta_only": "condition == 'absent'",
            "all": "all curated records",
        }[variant],
        "curated_source_file": source_path.as_posix(),
        "curated_source_sha256": source_hash,
        "split_generation": {
            "algorithm": (
                "deterministic searched shared held-out image pool + "
                "deterministic subject-role record subsampling"
            ),
            "master_attempt_id": attempt_id,
            "master_selected_score": selected_score,
            "target_record_ratio_over_source": {
                "test": TEST_UNSEEN_RATIO_OF_TOTAL,
                "test_seen": TEST_SEEN_RATIO_OF_TOTAL,
            },
            "target_is_record_ratio_not_image_ratio": True,
            "master_split_unit": "stimulus_id=name",
            "master_image_partition": ["train", "eval"],
            "train_eval_image_disjoint": True,
            "test_and_test_seen_share_eval_image_pool": True,
            "has_validation_split": False,
            "variant_split_independently": False,
            "heldout_unselected_record_policy": "exclude; never return to train",
            "runtime_subject_filter": {
                "test": f"subject in {list(UNSEEN_SUBJECTS)}",
                "test_seen": f"subject not in {list(UNSEEN_SUBJECTS)}",
            },
        },
        "split_files": {split: path.as_posix() for split, path in split_paths.items()},
        "split_file_sha256": split_hashes,
        "train_stimulus_ids": train_images,
        "eval_stimulus_ids": eval_images,
        "test_stimulus_ids": sorted({r["stimulus_id"] for r in split_rows["test"]}),
        "test_seen_stimulus_ids": sorted(
            {r["stimulus_id"] for r in split_rows["test_seen"]}
        ),
        "seen_subject_ids": seen_subjects,
        "unseen_subject_ids": list(UNSEEN_SUBJECTS),
        "support_sampling": {
            "unit": "trial_key=(task,image_name)",
            "source_split": "train_images_only",
            "same_trial_keys_for_unseen_subjects": True,
            "distinct_image_within_block": True,
            "exclusive_across_draws_within_k": True,
            "num_draws_per_k": NUM_DRAWS,
            "python_random_seed_by_k": {str(k): k for k in KS},
            "nested_k": False,
        },
        "support_draws": support_draws,
    }

    report = make_report(
        variant=variant,
        records=records,
        image_split=image_split,
        attempt_id=attempt_id,
        selected_score=selected_score,
    )
    report["split_file_sha256"] = split_hashes
    dump_json(variant_dir / "split_manifest.json", manifest, pretty=True)
    dump_json(variant_dir / "preprocess_report.json", report, pretty=True)

    return {
        "variant": variant,
        "source_records": len(records),
        "runtime_record_counts": {
            split: len(rows) for split, rows in split_rows.items()
        },
        "eval_image_count": len(eval_images),
        "eval_image_ratio": len(eval_images) / len(image_split),
        "excluded_heldout_records": report["excluded_heldout_records"],
        "shared_support_distinct_images": report[
            "shared_train_support_distinct_images"
        ],
        "output_dir": variant_dir.as_posix(),
        "manifest_sha256": sha256_file(variant_dir / "split_manifest.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic SemGaze COCO-Search18 subject-stratified "
            "5%-unseen / 5%-seen evaluation splits with one shared held-out "
            "image pool and no train/eval image leakage."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and regenerate non-empty variant output directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_attempts <= 0:
        raise SplitBuildError("--max-attempts must be > 0")

    source_records = load_and_validate(args.input)
    args.output_root.mkdir(parents=True, exist_ok=True)
    attempt_id, image_split, selected_score, search_report = choose_master_split(
        source_records, args.max_attempts
    )

    master_images = sorted(image_split)
    train_images = [name for name in master_images if image_split[name] == "train"]
    eval_images = [name for name in master_images if image_split[name] == "eval"]
    master_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "COCO-Search18",
        "curated_source_file": args.input.as_posix(),
        "curated_source_sha256": sha256_file(args.input),
        "split_unit": "stimulus_id=name",
        "target_record_ratio_over_source": {
            "test_unseen": TEST_UNSEEN_RATIO_OF_TOTAL,
            "test_seen": TEST_SEEN_RATIO_OF_TOTAL,
        },
        "target_is_record_ratio_not_image_ratio": True,
        "master_image_partition": ["train", "eval"],
        "train_eval_image_disjoint": True,
        "test_and_test_seen_share_eval_image_pool": True,
        "actual_master_image_counts": {
            "train": len(train_images),
            "eval": len(eval_images),
        },
        "actual_master_image_ratios": {
            "train": len(train_images) / len(master_images),
            "eval": len(eval_images) / len(master_images),
        },
        "selected_attempt_id": attempt_id,
        "selected_score": selected_score,
        "max_attempts": args.max_attempts,
        "search_report": search_report,
        "train_stimulus_ids": train_images,
        "eval_stimulus_ids": eval_images,
    }
    dump_json(args.output_root / "master_split_manifest.json", master_manifest, pretty=True)

    summaries = []
    for variant in ("tp_only", "ta_only", "all"):
        summary = build_variant(
            source_records=source_records,
            source_path=args.input,
            output_root=args.output_root,
            variant=variant,
            image_split=image_split,
            attempt_id=attempt_id,
            selected_score=selected_score,
            force=args.force,
        )
        summaries.append(summary)
        print(
            f"[{variant}] source_records={summary['source_records']} "
            f"runtime_record_counts={summary['runtime_record_counts']} "
            f"eval_images={summary['eval_image_count']} "
            f"eval_image_ratio={summary['eval_image_ratio']:.4f} "
            f"excluded_heldout_records={summary['excluded_heldout_records']} "
            f"shared_support_images={summary['shared_support_distinct_images']}"
        )

    index = {
        "protocol_version": PROTOCOL_VERSION,
        "source_file": args.input.as_posix(),
        "source_sha256": sha256_file(args.input),
        "master_split_manifest": (
            args.output_root / "master_split_manifest.json"
        ).as_posix(),
        "master_attempt_id": attempt_id,
        "master_selected_score": selected_score,
        "master_image_counts": {
            "train": len(train_images),
            "eval": len(eval_images),
        },
        "unseen_subject_ids": list(UNSEEN_SUBJECTS),
        "k_shot": list(KS),
        "num_exclusive_draws_per_k": NUM_DRAWS,
        "variants": summaries,
    }
    dump_json(args.output_root / "split_index.json", index, pretty=True)

    print(
        f"[master] attempt={attempt_id} score={selected_score:.8f} "
        f"images={len(master_images)} train={len(train_images)} eval={len(eval_images)}"
    )
    print(f"Wrote split index: {args.output_root / 'split_index.json'}")


if __name__ == "__main__":
    try:
        main()
    except SplitBuildError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
