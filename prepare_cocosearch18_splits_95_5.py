#!/usr/bin/env python3
"""Build SemGaze COCO-Search18 master 95/5 train-test splits.

Protocol
--------
- One MASTER split is created over unique stimulus_id/name values.
- 95% of images are train, 5% are test; there is NO validation split.
- tp_only / ta_only / all are derived from the SAME master image split.
- Candidate master splits are searched deterministically and the most
  distribution-preserving valid candidate is selected.
- Final few-shot support examples come only from TRAIN images.
- Unseen subjects are 7/8/9; K in {1,5,10}; 10 frozen support draws per K.

Run from repository root:

    python prepare_cocosearch18_splits_95_5.py

Default input:
    data/COCO_Search18/COCOSearch-18.json

Default output:
    data/COCO_Search18/split_95_5/{tp_only,ta_only,all}/...

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


PROTOCOL_VERSION = "cocosearch18_semgaze_master_955_v1"
DEFAULT_INPUT = Path("data/COCO_Search18/COCOSearch-18.json")
DEFAULT_OUTPUT_ROOT = Path("data/COCO_Search18/split_95_5")

UNSEEN_SUBJECTS = (7, 8, 9)
KS = (1, 5, 10)
NUM_DRAWS = 10
TEST_RATIO = 0.05
MAX_ATTEMPTS = 10_000

# Distribution-search weights. Lower total score is better.
WEIGHT_TASK_TV = 4.0
WEIGHT_SUBJECT_TV = 2.0
WEIGHT_CONDITION_TV = 2.0
WEIGHT_RECORD_RATIO = 8.0

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
    """Precompute per-image summaries so 10k split attempts remain cheap."""
    images = sorted({r["name"] for r in records})

    per_variant: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        rows = variant_records(records, variant)
        image_record_count: Counter[str] = Counter()
        image_tasks: dict[str, Counter[str]] = defaultdict(Counter)
        image_subjects: dict[str, Counter[int]] = defaultdict(Counter)
        image_conditions: dict[str, Counter[str]] = defaultdict(Counter)

        global_tasks: Counter[str] = Counter()
        global_subjects: Counter[int] = Counter()
        global_conditions: Counter[str] = Counter()

        for r in rows:
            image = r["name"]
            task = r["task"]
            subject = int(r["subject"])
            condition = r["condition"]

            image_record_count[image] += 1
            image_tasks[image][task] += 1
            image_subjects[image][subject] += 1
            image_conditions[image][condition] += 1

            global_tasks[task] += 1
            global_subjects[subject] += 1
            global_conditions[condition] += 1

        shared_trials = shared_support_candidates_without_split(rows)
        shared_images = {image for _, image in shared_trials}

        per_variant[variant] = {
            "total_records": len(rows),
            "image_record_count": image_record_count,
            "image_tasks": image_tasks,
            "image_subjects": image_subjects,
            "image_conditions": image_conditions,
            "global_tasks": global_tasks,
            "global_subjects": global_subjects,
            "global_conditions": global_conditions,
            "shared_support_images": shared_images,
        }

    return {"images": images, "variants": per_variant}


def candidate_test_images(images: list[str], attempt_id: int) -> set[str]:
    n = len(images)
    if n < 2:
        raise SplitBuildError(f"Need at least 2 unique images for 95/5 split; got {n}")

    # round() gives 213 test images for N=4261; train is the exact complement (4048).
    n_test = max(1, min(n - 1, round(TEST_RATIO * n)))
    ranked = sorted(
        images,
        key=lambda name: stable_digest(
            f"{PROTOCOL_VERSION}|master|{attempt_id}|image::{name}"
        ),
    )
    return set(ranked[:n_test])


def summarize_test_candidate(
    search_index: dict[str, Any], test_images: set[str]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for variant, idx in search_index["variants"].items():
        test_records = 0
        tasks: Counter[str] = Counter()
        subjects: Counter[int] = Counter()
        conditions: Counter[str] = Counter()

        for image in test_images:
            test_records += idx["image_record_count"].get(image, 0)
            tasks.update(idx["image_tasks"].get(image, {}))
            subjects.update(idx["image_subjects"].get(image, {}))
            conditions.update(idx["image_conditions"].get(image, {}))

        out[variant] = {
            "test_records": test_records,
            "tasks": tasks,
            "subjects": subjects,
            "conditions": conditions,
            "train_shared_support_distinct_images": len(
                idx["shared_support_images"] - test_images
            ),
        }
    return out


def candidate_errors(
    search_index: dict[str, Any], test_images: set[str], summary: dict[str, dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    required_support_images = NUM_DRAWS * max(KS)

    for variant, idx in search_index["variants"].items():
        s = summary[variant]
        if s["test_records"] <= 0:
            errors.append(f"{variant}: no test records")
            continue

        # Every target category available in the variant must be represented in test.
        missing_tasks = set(idx["global_tasks"]) - set(s["tasks"])
        if missing_tasks:
            errors.append(
                f"{variant}: test misses tasks={sorted(missing_tasks)}"
            )

        # Final evaluation is on unseen subjects 7/8/9; each must be represented.
        for subject in UNSEEN_SUBJECTS:
            if s["subjects"].get(subject, 0) < 1:
                errors.append(f"{variant}: unseen subject {subject} has no test record")

        # Keep the current strong few-shot requirement: 10 exclusive K=10 draws.
        if s["train_shared_support_distinct_images"] < required_support_images:
            errors.append(
                f"{variant}: insufficient shared train support images: "
                f"need >= {required_support_images}, "
                f"got {s['train_shared_support_distinct_images']}"
            )

    return errors


def candidate_score(
    search_index: dict[str, Any], summary: dict[str, dict[str, Any]]
) -> tuple[float, dict[str, dict[str, float]]]:
    total = 0.0
    diagnostics: dict[str, dict[str, float]] = {}

    for variant, idx in search_index["variants"].items():
        s = summary[variant]
        task_tv = total_variation(idx["global_tasks"], s["tasks"])
        subject_tv = total_variation(idx["global_subjects"], s["subjects"])
        condition_tv = total_variation(idx["global_conditions"], s["conditions"])
        record_ratio = s["test_records"] / idx["total_records"]
        record_ratio_error = abs(record_ratio - TEST_RATIO)

        # Condition TV is only informative for the mixed `all` variant.
        condition_term = condition_tv if variant == "all" else 0.0

        variant_score = (
            WEIGHT_TASK_TV * task_tv
            + WEIGHT_SUBJECT_TV * subject_tv
            + WEIGHT_CONDITION_TV * condition_term
            + WEIGHT_RECORD_RATIO * record_ratio_error
        )
        total += variant_score
        diagnostics[variant] = {
            "score": variant_score,
            "task_tv": task_tv,
            "subject_tv": subject_tv,
            "condition_tv": condition_tv,
            "test_record_ratio": record_ratio,
            "test_record_ratio_abs_error": record_ratio_error,
        }

    return total, diagnostics


def choose_master_split(
    records: list[dict[str, Any]], max_attempts: int
) -> tuple[int, dict[str, str], float, dict[str, Any]]:
    search_index = build_search_index(records)
    images: list[str] = search_index["images"]

    best: tuple[int, set[str], float, dict[str, Any], dict[str, Any]] | None = None
    valid_attempts = 0
    last_errors: list[str] = []

    for attempt_id in range(max_attempts):
        test_images = candidate_test_images(images, attempt_id)
        summary = summarize_test_candidate(search_index, test_images)
        errors = candidate_errors(search_index, test_images, summary)
        if errors:
            last_errors = errors
            continue

        valid_attempts += 1
        score, diagnostics = candidate_score(search_index, summary)
        payload = {
            "valid_attempts_seen": valid_attempts,
            "candidate_summary": {
                variant: {
                    "test_records": s["test_records"],
                    "train_shared_support_distinct_images": s[
                        "train_shared_support_distinct_images"
                    ],
                    "test_task_counts": dict(sorted(s["tasks"].items())),
                    "test_subject_counts": {
                        str(k): v for k, v in sorted(s["subjects"].items())
                    },
                    "test_condition_counts": dict(sorted(s["conditions"].items())),
                }
                for variant, s in summary.items()
            },
            "distribution_diagnostics": diagnostics,
        }

        if best is None or score < best[2]:
            best = (attempt_id, test_images, score, payload, diagnostics)

    if best is None:
        raise SplitBuildError(
            f"No valid master 95/5 split found in attempts [0, {max_attempts - 1}]. "
            f"Last errors: {last_errors}"
        )

    attempt_id, test_images, score, payload, _ = best
    image_split = {
        image: ("test" if image in test_images else "train") for image in images
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
    # Resolve each shared TRAIN trial to exactly one record per unseen subject.
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

        # Strong exclusivity: no image reused anywhere across the 10 draws
        # of a given K family, even if multiple tasks exist for that image.
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


def assert_exact_partition(
    original: list[dict[str, Any]], split_rows: dict[str, list[dict[str, Any]]]
) -> None:
    expected = {canonical_record_id(r) for r in original}
    actual_lists = {
        split: [r["record_id"] for r in rows] for split, rows in split_rows.items()
    }
    actual = set().union(*(set(ids) for ids in actual_lists.values()))
    if actual != expected:
        raise SplitBuildError(
            "Generated records do not exactly partition the variant: "
            f"missing={len(expected - actual)}, extra={len(actual - expected)}"
        )
    total = sum(len(ids) for ids in actual_lists.values())
    if total != len(expected):
        raise SplitBuildError("A record appears in more than one generated split")


def verify_image_disjointness(split_rows: dict[str, list[dict[str, Any]]]) -> None:
    train_images = {r["stimulus_id"] for r in split_rows["train"]}
    test_images = {r["stimulus_id"] for r in split_rows["test"]}
    overlap = train_images & test_images
    if overlap:
        raise SplitBuildError(
            f"Image leakage between train and test; example={next(iter(overlap))}"
        )


def coverage_errors(
    records: list[dict[str, Any]], image_split: dict[str, str]
) -> list[str]:
    errors: list[str] = []
    tasks = {r["task"] for r in records}
    test_rows = [r for r in records if image_split[r["name"]] == "test"]
    test_tasks = {r["task"] for r in test_rows}

    if tasks - test_tasks:
        errors.append(f"test misses tasks={sorted(tasks - test_tasks)}")

    by_subject = index_by_subject(records)
    for subject in UNSEEN_SUBJECTS:
        rows = by_subject.get(subject, [])
        test_count = sum(1 for r in rows if image_split[r["name"]] == "test")
        if test_count < 1:
            errors.append(f"unseen subject {subject} has no test query record")

    shared = shared_support_candidates(records, image_split)
    shared_distinct_images = len({image_name for _, image_name in shared})
    required = NUM_DRAWS * max(KS)
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
    record_counts = Counter(image_split[r["name"]] for r in records)
    variant_images = {r["name"] for r in records}
    image_counts = Counter(image_split[name] for name in variant_images)
    condition_counts = Counter(r["condition"] for r in records)
    subject_counts = Counter(str(r["subject"]) for r in records)
    task_counts = Counter(r["task"] for r in records)

    split_subject_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "test": Counter(),
    }
    split_task_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "test": Counter(),
    }
    split_condition_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "test": Counter(),
    }

    for r in records:
        split = image_split[r["name"]]
        split_subject_counts[split][str(r["subject"])] += 1
        split_task_counts[split][r["task"]] += 1
        split_condition_counts[split][r["condition"]] += 1

    shared = shared_support_candidates(records, image_split)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "variant": variant,
        "master_attempt_id": attempt_id,
        "master_selected_score": selected_score,
        "total_records": len(records),
        "total_unique_images": len(variant_images),
        "condition_counts": dict(sorted(condition_counts.items())),
        "task_record_counts": dict(sorted(task_counts.items())),
        "subject_record_counts": dict(
            sorted(subject_counts.items(), key=lambda kv: int(kv[0]))
        ),
        "record_counts_by_split": dict(record_counts),
        "record_ratio_by_split": {
            split: count / len(records) for split, count in sorted(record_counts.items())
        },
        "image_counts_by_split": dict(image_counts),
        "image_ratio_by_split": {
            split: count / len(variant_images)
            for split, count in sorted(image_counts.items())
        },
        "subject_record_counts_by_split": {
            split: dict(sorted(counts.items(), key=lambda kv: int(kv[0])))
            for split, counts in split_subject_counts.items()
        },
        "task_record_counts_by_split": {
            split: dict(sorted(counts.items())) for split, counts in split_task_counts.items()
        },
        "condition_record_counts_by_split": {
            split: dict(sorted(counts.items()))
            for split, counts in split_condition_counts.items()
        },
        "shared_train_support_trial_keys": len(shared),
        "shared_train_support_distinct_images": len({image for _, image in shared}),
        "coverage_errors": coverage_errors(records, image_split),
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
    errors = coverage_errors(records, image_split)
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

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    for record in records:
        split = image_split[record["name"]]
        split_rows[split].append(materialized_record(record, variant, split))

    assert_exact_partition(records, split_rows)
    verify_image_disjointness(split_rows)
    support_draws = build_final_support_draws(records, image_split)

    split_paths = {
        split: variant_dir / f"{split}.json" for split in ("train", "test")
    }
    for split, path in split_paths.items():
        dump_json(path, split_rows[split], pretty=False)

    split_hashes = {split: sha256_file(path) for split, path in split_paths.items()}
    source_hash = sha256_file(source_path)
    seen_subjects = sorted(
        {int(r["subject"]) for r in records} - set(UNSEEN_SUBJECTS)
    )
    variant_images = {r["name"] for r in records}

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
            "algorithm": "deterministic searched SHA256 ranking over MASTER unique stimulus_id=name",
            "master_attempt_id": attempt_id,
            "master_selected_score": selected_score,
            "target_test_image_ratio": TEST_RATIO,
            "split_unit": "stimulus_id=name",
            "has_validation_split": False,
            "variant_split_independently": False,
            "derived_from_shared_master_split": True,
        },
        "split_files": {split: path.as_posix() for split, path in split_paths.items()},
        "split_file_sha256": split_hashes,
        "train_stimulus_ids": sorted(
            name for name in variant_images if image_split[name] == "train"
        ),
        "test_stimulus_ids": sorted(
            name for name in variant_images if image_split[name] == "test"
        ),
        "seen_subject_ids": seen_subjects,
        "unseen_subject_ids": list(UNSEEN_SUBJECTS),
        "support_sampling": {
            "unit": "trial_key=(task,image_name)",
            "source_split": "train_only",
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
        "records": len(records),
        "images": len(variant_images),
        "image_counts": dict(
            Counter(image_split[name] for name in variant_images)
        ),
        "record_counts": dict(Counter(image_split[r["name"]] for r in records)),
        "shared_support_distinct_images": report[
            "shared_train_support_distinct_images"
        ],
        "output_dir": variant_dir.as_posix(),
        "manifest_sha256": sha256_file(variant_dir / "split_manifest.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic distribution-balanced SemGaze COCO-Search18 "
            "MASTER 95/5 train-test image split for tp_only, ta_only, and all variants."
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
    master_counts = Counter(image_split.values())
    master_manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "COCO-Search18",
        "curated_source_file": args.input.as_posix(),
        "curated_source_sha256": sha256_file(args.input),
        "split_unit": "stimulus_id=name",
        "target_ratio": {"train": 1.0 - TEST_RATIO, "test": TEST_RATIO},
        "actual_master_image_counts": dict(master_counts),
        "actual_master_image_ratios": {
            split: count / len(master_images)
            for split, count in sorted(master_counts.items())
        },
        "selected_attempt_id": attempt_id,
        "selected_score": selected_score,
        "max_attempts": args.max_attempts,
        "search_report": search_report,
        "train_stimulus_ids": [
            name for name in master_images if image_split[name] == "train"
        ],
        "test_stimulus_ids": [
            name for name in master_images if image_split[name] == "test"
        ],
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
            f"[{variant}] records={summary['records']} images={summary['images']} "
            f"image_counts={summary['image_counts']} "
            f"record_counts={summary['record_counts']} "
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
        "master_image_counts": dict(master_counts),
        "unseen_subject_ids": list(UNSEEN_SUBJECTS),
        "k_shot": list(KS),
        "num_exclusive_draws_per_k": NUM_DRAWS,
        "variants": summaries,
    }
    dump_json(args.output_root / "split_index.json", index, pretty=True)

    print(
        f"[master] attempt={attempt_id} score={selected_score:.8f} "
        f"images={len(master_images)} counts={dict(master_counts)}"
    )
    print(f"Wrote split index: {args.output_root / 'split_index.json'}")


if __name__ == "__main__":
    try:
        main()
    except SplitBuildError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
