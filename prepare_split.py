#!/usr/bin/env python3
"""Build leakage-safe SemGaze splits for AiR, COCO-Search18, and their union.

Repository layout expected by default:

    data/
    ├── images/
    └── split/
        ├── AiR/
        │   ├── AiR.json
        │   └── split/all/
        ├── COCO_Search18/
        │   ├── COCOSearch-18.json
        │   └── split/{all,tp,ta}/
        └── all/

Protocol
--------
1. Split UNIT is the image (`name`), never an individual gaze record or qid.
   This prevents the same visual stimulus from appearing in train and test.
2. Each dataset gets its own deterministic searched 90/10 MASTER image split.
3. On MASTER test images:
      * unseen subjects -> test.json
      * seen subjects   -> test_seen.json
4. On MASTER train images every record is retained in train.json. Runtime
   training must optimize seen-subject queries only; unseen-subject rows on
   train images are support-only.
5. No source record is discarded.
6. COCO-Search18 keeps one shared MASTER split for `all`, `tp`, and `ta`, so
   the three variants cannot disagree about whether an image is train/test.
7. AiR uses qid as the support trial identity because one image may host
   multiple VQA questions. COCO-Search18 uses (task, image) as before.
8. The combined `data/split/all` split is constructed ONLY by concatenating
   the already-materialized AiR/all and COCO_Search18/all memberships. It does
   not resplit or reshuffle either dataset.
9. Support examples are always drawn from TRAIN images. For each dataset, 10
   frozen exclusive draws are created for K in {1,5,10}; within a K family an
   image is never reused across the 10 draws.

Default unseen subjects
-----------------------
COCO-Search18: 7, 8, 9 (existing SemGaze/ISP-SENet-compatible protocol)
AiR: JY, SC, YN (SemGaze protocol choice; overrideable via CLI). This trio is
chosen because the curated AiR file has strong and balanced subject coverage
and abundant shared qid/image support for K={1,5,10} x 10 draws.

Run from repository root:

    python prepare_semgaze_splits.py --force

The script is stdlib-only and deterministic.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


PROTOCOL_VERSION = "semgaze_air_coco_master_9010_subject_eval_v1"
COCO_PROTOCOL_VERSION = "cocosearch18_semgaze_master_9010_subject_eval_v1"
AIR_PROTOCOL_VERSION = "air_semgaze_master_9010_subject_eval_v1"
TEST_IMAGE_RATIO = 0.10
MAX_ATTEMPTS = 10_000
KS = (1, 5, 10)
NUM_DRAWS = 10

DEFAULT_AIR_INPUT = Path("data/split/AiR/AiR.json")
DEFAULT_COCO_INPUT = Path("data/split/COCO_Search18/COCOSearch-18.json")
DEFAULT_AIR_OUTPUT = Path("data/split/AiR/split")
DEFAULT_COCO_OUTPUT = Path("data/split/COCO_Search18/split")
DEFAULT_COMBINED_OUTPUT = Path("data/split/all")

DEFAULT_AIR_UNSEEN = ("JY", "SC", "YN")
DEFAULT_COCO_UNSEEN = (7, 8, 9)

COCO_VARIANTS: dict[str, Callable[[dict[str, Any]], bool]] = {
    "all": lambda r: True,
    "tp": lambda r: r["condition"] == "present",
    "ta": lambda r: r["condition"] == "absent",
}

COMMON_REQUIRED_FIELDS = (
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


def dump_json(path: Path, obj: Any, *, pretty: bool = False) -> None:
    """Write JSON safely, with a WSL/DrvFS fallback for denied atomic replace.

    On native Linux filesystems we keep the normal temp-file + os.replace behavior.
    Some Windows-mounted paths (for example /mnt/d under WSL) can allow creating
    and writing a temp file while denying rename/replace because of Windows ACLs
    or an open handle. In that case, fall back to copying the completed temp file
    into the destination and then remove the temp file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    # Remove a stale temp file from an interrupted previous run.
    if tmp.exists():
        try:
            tmp.unlink()
        except PermissionError as exc:
            raise SplitBuildError(
                f"Cannot remove stale temporary file: {tmp}. "
                "Close programs using it or fix its Windows/WSL permissions."
            ) from exc

    with tmp.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        else:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")

    try:
        tmp.replace(path)
        return
    except PermissionError as replace_exc:
        # WSL DrvFS/NTFS can reject rename/replace while still permitting a
        # normal file write. Preserve the fully-written temp file until the
        # fallback succeeds so a failed write does not destroy the output.
        try:
            with tmp.open("rb") as src, path.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
                dst.flush()
            tmp.unlink()
            print(
                f"[WARN] Atomic replace denied for {path}; "
                "used direct-write fallback (common on /mnt/* under WSL)."
            )
            return
        except PermissionError as direct_exc:
            raise SplitBuildError(
                f"Permission denied writing {path}. Temporary file was created at "
                f"{tmp}, but both atomic replace and direct destination write failed. "
                "Check Windows read-only/ACL settings and whether another program "
                "has the destination file open."
            ) from direct_exc
        except OSError as direct_exc:
            raise SplitBuildError(
                f"Could not finalize JSON output {path} after os.replace failed: "
                f"{direct_exc}. Original replace error: {replace_exc}"
            ) from direct_exc


def normalize_subject(value: Any) -> str:
    return str(value)


def canonical_record_id(dataset: str, record: dict[str, Any]) -> str:
    if dataset == "AiR":
        return f"air_semgaze::{record['subject']}::{record['qid']}::{record['name']}"
    return f"coco_semgaze::{record['subject']}::{record['task']}::{record['name']}"


def trial_key(dataset: str, record: dict[str, Any]) -> str:
    if dataset == "AiR":
        return f"{record['qid']}::{record['name']}"
    return f"{record['task']}::{record['name']}"


def question_family(task: str) -> str:
    """Coarse AiR question family used only for distribution balancing."""
    m = re.match(r"\s*([A-Za-z]+)", task)
    return m.group(1).lower() if m else "other"


def validate_semantic_record(record: dict[str, Any], index: int, dataset: str) -> None:
    missing = [k for k in COMMON_REQUIRED_FIELDS if k not in record]
    if dataset == "AiR" and "qid" not in record:
        missing.append("qid")
    if missing:
        raise SplitBuildError(f"{dataset} record[{index}] missing fields: {missing}")

    if not isinstance(record["name"], str) or not record["name"].strip():
        raise SplitBuildError(f"{dataset} record[{index}] has invalid name")
    if not isinstance(record["task"], str) or not record["task"].strip():
        raise SplitBuildError(f"{dataset} record[{index}] has invalid task")

    if dataset == "AiR":
        if str(record["condition"]).lower() != "vqa":
            raise SplitBuildError(
                f"AiR record[{index}] condition must be 'vqa', got {record['condition']!r}"
            )
        if not str(record["qid"]).strip():
            raise SplitBuildError(f"AiR record[{index}] has invalid qid")
    else:
        if record["condition"] not in {"present", "absent"}:
            raise SplitBuildError(
                f"COCO record[{index}] condition must be present/absent, "
                f"got {record['condition']!r}"
            )

    x, y, t = record["X"], record["Y"], record["T"]
    if not all(isinstance(v, list) for v in (x, y, t)):
        raise SplitBuildError(f"{dataset} record[{index}] X/Y/T must be lists")
    if not (len(x) == len(y) == len(t) and len(x) > 0):
        raise SplitBuildError(
            f"{dataset} record[{index}] needs len(X)==len(Y)==len(T)>0; "
            f"got {len(x)}, {len(y)}, {len(t)}"
        )

    pred = record["prediction"]
    if not isinstance(pred, dict):
        raise SplitBuildError(f"{dataset} record[{index}] prediction must be object")
    fixations = pred.get("fixations")
    regions = pred.get("regions")
    how = pred.get("how")
    n = len(x)

    if not isinstance(fixations, list) or len(fixations) != n:
        raise SplitBuildError(
            f"{dataset} record[{index}] prediction.fixations must have {n} items"
        )
    if not isinstance(regions, list) or not regions:
        raise SplitBuildError(
            f"{dataset} record[{index}] prediction.regions must be a non-empty list"
        )
    if not isinstance(how, str) or not how.strip():
        raise SplitBuildError(f"{dataset} record[{index}] prediction.how empty")

    fixation_ids: list[int] = []
    for j, fx in enumerate(fixations):
        if not isinstance(fx, dict):
            raise SplitBuildError(f"{dataset} record[{index}] fixation[{j}] invalid")
        fid = fx.get("fixation")
        what = fx.get("what")
        if not isinstance(fid, int):
            raise SplitBuildError(f"{dataset} record[{index}] fixation id must be int")
        if not isinstance(what, str) or not what.strip():
            raise SplitBuildError(f"{dataset} record[{index}] fixation.what empty")
        fixation_ids.append(fid)

    expected = list(range(1, n + 1))
    if sorted(fixation_ids) != expected or len(set(fixation_ids)) != n:
        raise SplitBuildError(
            f"{dataset} record[{index}] fixation IDs must cover 1..{n} exactly once"
        )

    membership: list[int] = []
    for j, region in enumerate(regions):
        if not isinstance(region, dict):
            raise SplitBuildError(f"{dataset} record[{index}] region[{j}] invalid")
        ids = region.get("fixations")
        why = region.get("why")
        if not isinstance(ids, list) or not ids or not all(isinstance(v, int) for v in ids):
            raise SplitBuildError(
                f"{dataset} record[{index}] region[{j}].fixations invalid"
            )
        if not isinstance(why, str) or not why.strip():
            raise SplitBuildError(f"{dataset} record[{index}] region[{j}].why empty")
        membership.extend(ids)

    if sorted(membership) != expected or len(set(membership)) != n:
        raise SplitBuildError(
            f"{dataset} record[{index}] regions must partition fixation IDs 1..{n}"
        )


def load_records(path: Path, dataset: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SplitBuildError(f"Input not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or not rows:
        raise SplitBuildError(f"{dataset} input must be a non-empty JSON list")

    seen_ids: set[str] = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SplitBuildError(f"{dataset} record[{i}] is not an object")
        validate_semantic_record(row, i, dataset)
        rid = canonical_record_id(dataset, row)
        if rid in seen_ids:
            raise SplitBuildError(f"Duplicate canonical record id: {rid}")
        seen_ids.add(rid)
    return rows


def total_variation(global_counts: Counter[Any], test_counts: Counter[Any]) -> float:
    a = sum(global_counts.values())
    b = sum(test_counts.values())
    if a <= 0 or b <= 0:
        return math.inf
    keys = set(global_counts) | set(test_counts)
    return 0.5 * sum(
        abs(global_counts.get(k, 0) / a - test_counts.get(k, 0) / b) for k in keys
    )


def candidate_test_images(
    images: Sequence[str], dataset: str, attempt_id: int, ratio: float
) -> set[str]:
    n = len(images)
    if n < 2:
        raise SplitBuildError(f"{dataset}: need >=2 unique images, got {n}")
    n_test = max(1, min(n - 1, round(ratio * n)))
    if dataset == "COCO-Search18":
        # Preserve the exact candidate ranking of the previous COCO-only splitter.
        def rank_key(name: str) -> str:
            return stable_digest(
                f"{COCO_PROTOCOL_VERSION}|master|{attempt_id}|image::{name}"
            )
    else:
        def rank_key(name: str) -> str:
            return stable_digest(
                f"{AIR_PROTOCOL_VERSION}|master|{attempt_id}|image::{name}"
            )

    ranked = sorted(images, key=rank_key)
    return set(ranked[:n_test])


def role_rows(
    rows: Iterable[dict[str, Any]],
    test_images: set[str],
    unseen_subjects: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    test_seen: list[dict[str, Any]] = []
    for row in rows:
        if row["name"] not in test_images:
            train.append(row)
        elif normalize_subject(row["subject"]) in unseen_subjects:
            test.append(row)
        else:
            test_seen.append(row)
    return train, test, test_seen


def shared_support_units(
    dataset: str,
    rows: list[dict[str, Any]],
    unseen_subjects: Sequence[str],
    train_images: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Return support units shared exactly once by every unseen subject.

    AiR unit: (qid, image)
    COCO unit: (task, image)
    """
    target = set(unseen_subjects)
    per_unit: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        if train_images is not None and row["name"] not in train_images:
            continue
        subject = normalize_subject(row["subject"])
        if subject not in target:
            continue
        key = (
            str(row["qid"]) if dataset == "AiR" else str(row["task"]),
            str(row["name"]),
        )
        per_unit[key][subject] += 1

    return sorted(
        key
        for key, counts in per_unit.items()
        if all(counts.get(subject, 0) == 1 for subject in target)
    )


def support_image_capacity(
    dataset: str,
    rows: list[dict[str, Any]],
    unseen_subjects: Sequence[str],
    train_images: set[str],
) -> int:
    units = shared_support_units(dataset, rows, unseen_subjects, train_images)
    return len({image for _, image in units})


def validate_unseen_subjects(
    dataset: str, rows: list[dict[str, Any]], unseen_subjects: Sequence[str]
) -> None:
    available = {normalize_subject(r["subject"]) for r in rows}
    missing = set(unseen_subjects) - available
    if missing:
        raise SplitBuildError(
            f"{dataset}: unseen subjects not present in source: {sorted(missing)}; "
            f"available={sorted(available)}"
        )
    if len(set(unseen_subjects)) != len(unseen_subjects):
        raise SplitBuildError(f"{dataset}: duplicate unseen subject IDs")


def coco_variant_rows(rows: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    out = [r for r in rows if COCO_VARIANTS[variant](r)]
    if not out:
        raise SplitBuildError(f"COCO variant {variant!r} is empty")
    return out


def counter(rows: Iterable[dict[str, Any]], field: str) -> Counter[Any]:
    return Counter(r[field] for r in rows)


def coco_candidate_errors_and_score(
    rows: list[dict[str, Any]],
    test_images: set[str],
    unseen_subjects: Sequence[str],
    ratio: float,
) -> tuple[list[str], float, dict[str, Any]]:
    errors: list[str] = []
    total_score = 0.0
    diagnostics: dict[str, Any] = {}
    unseen = set(unseen_subjects)
    required_support_images = NUM_DRAWS * max(KS)

    for variant in ("all", "tp", "ta"):
        vr = coco_variant_rows(rows, variant)
        train, test, test_seen = role_rows(vr, test_images, unseen)
        variant_images = {r["name"] for r in vr}
        vt_images = variant_images & test_images

        if not test:
            errors.append(f"COCO/{variant}: no unseen-subject test rows")
        if not test_seen:
            errors.append(f"COCO/{variant}: no seen-subject test_seen rows")
        if not train:
            errors.append(f"COCO/{variant}: no train rows")
        if errors:
            continue

        global_unseen = [r for r in vr if normalize_subject(r["subject"]) in unseen]
        global_seen = [r for r in vr if normalize_subject(r["subject"]) not in unseen]

        missing_unseen_tasks = set(r["task"] for r in global_unseen) - set(
            r["task"] for r in test
        )
        missing_seen_tasks = set(r["task"] for r in global_seen) - set(
            r["task"] for r in test_seen
        )
        if missing_unseen_tasks:
            errors.append(
                f"COCO/{variant}: unseen test misses tasks={sorted(missing_unseen_tasks)}"
            )
        if missing_seen_tasks:
            errors.append(
                f"COCO/{variant}: seen test misses tasks={sorted(missing_seen_tasks)}"
            )

        all_subjects = {normalize_subject(r["subject"]) for r in vr}
        for s in unseen:
            if s in all_subjects and not any(normalize_subject(r["subject"]) == s for r in test):
                errors.append(f"COCO/{variant}: unseen subject {s} absent from test")
        for s in sorted(all_subjects - unseen):
            if not any(normalize_subject(r["subject"]) == s for r in test_seen):
                errors.append(f"COCO/{variant}: seen subject {s} absent from test_seen")

        train_images = variant_images - test_images
        capacity = support_image_capacity(
            "COCO-Search18", vr, unseen_subjects, train_images
        )
        if capacity < required_support_images:
            errors.append(
                f"COCO/{variant}: support capacity {capacity} < {required_support_images}"
            )

        if errors:
            continue

        # Match the existing COCO split objective: task/subject/condition parity
        # plus image- and record-ratio parity, scored for all/unseen/seen roles.
        score = 0.0
        role_specs = [
            ("all", vr, test + test_seen),
            ("unseen", global_unseen, test),
            ("seen", global_seen, test_seen),
        ]
        role_diag: dict[str, Any] = {}
        for role, global_rows, test_rows in role_specs:
            task_tv = total_variation(counter(global_rows, "task"), counter(test_rows, "task"))
            subj_tv = total_variation(
                Counter(normalize_subject(r["subject"]) for r in global_rows),
                Counter(normalize_subject(r["subject"]) for r in test_rows),
            )
            cond_tv = total_variation(
                counter(global_rows, "condition"), counter(test_rows, "condition")
            )
            record_ratio = len(test_rows) / len(global_rows)
            ratio_error = abs(record_ratio - ratio)
            score += 4.0 * task_tv + 2.0 * subj_tv + 8.0 * ratio_error
            if variant == "all":
                score += 2.0 * cond_tv
            role_diag[role] = {
                "task_tv": task_tv,
                "subject_tv": subj_tv,
                "condition_tv": cond_tv,
                "record_ratio": record_ratio,
                "record_ratio_abs_error": ratio_error,
            }

        image_ratio = len(vt_images) / len(variant_images)
        image_error = abs(image_ratio - ratio)
        score += 4.0 * image_error
        total_score += score
        diagnostics[variant] = {
            "score": score,
            "image_ratio": image_ratio,
            "image_ratio_abs_error": image_error,
            "train_support_distinct_images": capacity,
            "roles": role_diag,
        }

    return errors, total_score, diagnostics


def air_candidate_errors_and_score(
    rows: list[dict[str, Any]],
    test_images: set[str],
    unseen_subjects: Sequence[str],
    ratio: float,
) -> tuple[list[str], float, dict[str, Any]]:
    unseen = set(unseen_subjects)
    errors: list[str] = []
    train, test, test_seen = role_rows(rows, test_images, unseen)

    if not train:
        errors.append("AiR: no train rows")
    if not test:
        errors.append("AiR: no unseen-subject test rows")
    if not test_seen:
        errors.append("AiR: no seen-subject test_seen rows")
    if errors:
        return errors, math.inf, {}

    global_unseen = [r for r in rows if normalize_subject(r["subject"]) in unseen]
    global_seen = [r for r in rows if normalize_subject(r["subject"]) not in unseen]
    available_subjects = {normalize_subject(r["subject"]) for r in rows}

    for s in unseen:
        if not any(normalize_subject(r["subject"]) == s for r in test):
            errors.append(f"AiR: unseen subject {s} absent from test")
    for s in sorted(available_subjects - unseen):
        if not any(normalize_subject(r["subject"]) == s for r in test_seen):
            errors.append(f"AiR: seen subject {s} absent from test_seen")

    all_images = {r["name"] for r in rows}
    train_images = all_images - test_images
    capacity = support_image_capacity("AiR", rows, unseen_subjects, train_images)
    required_support_images = NUM_DRAWS * max(KS)
    if capacity < required_support_images:
        errors.append(
            f"AiR: support capacity {capacity} < {required_support_images} distinct images"
        )
    if errors:
        return errors, math.inf, {}

    # AiR questions are mostly unique strings, so unlike COCO we do NOT require
    # every task string to occur in test. Instead preserve subject distribution,
    # coarse question-family distribution, unique-qid ratio, image ratio and
    # record ratio for all/unseen/seen populations.
    def fam_counts(rr: Iterable[dict[str, Any]]) -> Counter[str]:
        return Counter(question_family(str(r["task"])) for r in rr)

    score = 0.0
    role_diag: dict[str, Any] = {}
    for role, global_rows, test_rows in [
        ("all", rows, test + test_seen),
        ("unseen", global_unseen, test),
        ("seen", global_seen, test_seen),
    ]:
        subj_tv = total_variation(
            Counter(normalize_subject(r["subject"]) for r in global_rows),
            Counter(normalize_subject(r["subject"]) for r in test_rows),
        )
        fam_tv = total_variation(fam_counts(global_rows), fam_counts(test_rows))
        rec_ratio = len(test_rows) / len(global_rows)
        rec_err = abs(rec_ratio - ratio)
        global_qids = {str(r["qid"]) for r in global_rows}
        test_qids = {str(r["qid"]) for r in test_rows}
        qid_ratio = len(test_qids) / len(global_qids)
        qid_err = abs(qid_ratio - ratio)

        score += 3.0 * subj_tv + 3.0 * fam_tv + 8.0 * rec_err + 5.0 * qid_err
        role_diag[role] = {
            "subject_tv": subj_tv,
            "question_family_tv": fam_tv,
            "record_ratio": rec_ratio,
            "record_ratio_abs_error": rec_err,
            "qid_ratio": qid_ratio,
            "qid_ratio_abs_error": qid_err,
        }

    image_ratio = len(test_images) / len(all_images)
    image_err = abs(image_ratio - ratio)
    score += 5.0 * image_err

    return [], score, {
        "score": score,
        "image_ratio": image_ratio,
        "image_ratio_abs_error": image_err,
        "train_support_distinct_images": capacity,
        "roles": role_diag,
    }


def build_fast_search_index(
    dataset: str,
    rows: list[dict[str, Any]],
    unseen_subjects: Sequence[str],
) -> dict[str, Any]:
    """Precompute per-image summaries so 10k candidate search stays cheap."""
    unseen = set(unseen_subjects)
    variants = ["all"] if dataset == "AiR" else ["all", "tp", "ta"]
    out: dict[str, Any] = {"images": sorted({str(r["name"]) for r in rows}), "variants": {}}

    for variant in variants:
        vr = rows if dataset == "AiR" else coco_variant_rows(rows, variant)
        roles = {
            "all": vr,
            "unseen": [r for r in vr if normalize_subject(r["subject"]) in unseen],
            "seen": [r for r in vr if normalize_subject(r["subject"]) not in unseen],
        }
        v: dict[str, Any] = {
            "variant_images": {str(r["name"]) for r in vr},
            "support_images": {
                image for _, image in shared_support_units(dataset, vr, unseen_subjects, None)
            },
            "roles": {},
        }
        for role, rr in roles.items():
            info: dict[str, Any] = {
                "total_records": len(rr),
                "global_subjects": Counter(normalize_subject(r["subject"]) for r in rr),
                "image_count": Counter(),
                "image_subjects": defaultdict(Counter),
            }
            if dataset == "AiR":
                info.update(
                    {
                        "global_families": Counter(question_family(str(r["task"])) for r in rr),
                        "global_qids": {str(r["qid"]) for r in rr},
                        "image_families": defaultdict(Counter),
                        "image_qids": defaultdict(set),
                    }
                )
            else:
                info.update(
                    {
                        "global_tasks": Counter(str(r["task"]) for r in rr),
                        "global_conditions": Counter(str(r["condition"]) for r in rr),
                        "image_tasks": defaultdict(Counter),
                        "image_conditions": defaultdict(Counter),
                    }
                )

            for r in rr:
                image = str(r["name"])
                subject = normalize_subject(r["subject"])
                info["image_count"][image] += 1
                info["image_subjects"][image][subject] += 1
                if dataset == "AiR":
                    info["image_families"][image][question_family(str(r["task"]))] += 1
                    info["image_qids"][image].add(str(r["qid"]))
                else:
                    info["image_tasks"][image][str(r["task"])] += 1
                    info["image_conditions"][image][str(r["condition"])] += 1
            v["roles"][role] = info
        out["variants"][variant] = v
    return out


def _aggregate_test_role(info: dict[str, Any], test_images: set[str], dataset: str) -> dict[str, Any]:
    records = 0
    subjects: Counter[str] = Counter()
    for image in test_images:
        records += info["image_count"].get(image, 0)
        subjects.update(info["image_subjects"].get(image, {}))
    out: dict[str, Any] = {"records": records, "subjects": subjects}
    if dataset == "AiR":
        families: Counter[str] = Counter()
        qids: set[str] = set()
        for image in test_images:
            families.update(info["image_families"].get(image, {}))
            qids.update(info["image_qids"].get(image, set()))
        out.update({"families": families, "qids": qids})
    else:
        tasks: Counter[str] = Counter()
        conditions: Counter[str] = Counter()
        for image in test_images:
            tasks.update(info["image_tasks"].get(image, {}))
            conditions.update(info["image_conditions"].get(image, {}))
        out.update({"tasks": tasks, "conditions": conditions})
    return out


def fast_candidate_eval(
    dataset: str,
    index: dict[str, Any],
    test_images: set[str],
    unseen_subjects: Sequence[str],
    ratio: float,
) -> tuple[list[str], float, dict[str, Any]]:
    unseen = set(unseen_subjects)
    required_support_images = NUM_DRAWS * max(KS)
    errors: list[str] = []
    total_score = 0.0
    diagnostics: dict[str, Any] = {}

    for variant, v in index["variants"].items():
        test_variant_images = v["variant_images"] & test_images
        support_capacity = len(v["support_images"] - test_images)
        if support_capacity < required_support_images:
            errors.append(
                f"{dataset}/{variant}: support capacity {support_capacity} < {required_support_images}"
            )

        agg = {
            role: _aggregate_test_role(info, test_images, dataset)
            for role, info in v["roles"].items()
        }
        if agg["all"]["records"] <= 0:
            errors.append(f"{dataset}/{variant}: no test records")
        if agg["unseen"]["records"] <= 0:
            errors.append(f"{dataset}/{variant}: no unseen-subject test records")
        if agg["seen"]["records"] <= 0:
            errors.append(f"{dataset}/{variant}: no seen-subject test_seen records")

        if dataset == "AiR":
            all_subjects = set(v["roles"]["all"]["global_subjects"])
            for s in unseen:
                if v["roles"]["unseen"]["global_subjects"].get(s, 0) and agg["unseen"]["subjects"].get(s, 0) < 1:
                    errors.append(f"AiR: unseen subject {s} absent from test")
            for s in sorted(all_subjects - unseen):
                if v["roles"]["seen"]["global_subjects"].get(s, 0) and agg["seen"]["subjects"].get(s, 0) < 1:
                    errors.append(f"AiR: seen subject {s} absent from test_seen")
        else:
            # Preserve every COCO task for both subject roles, matching the prior splitter.
            missing_unseen_tasks = set(v["roles"]["unseen"]["global_tasks"]) - set(agg["unseen"]["tasks"])
            missing_seen_tasks = set(v["roles"]["seen"]["global_tasks"]) - set(agg["seen"]["tasks"])
            if missing_unseen_tasks:
                errors.append(
                    f"COCO/{variant}: unseen test misses tasks={sorted(missing_unseen_tasks)}"
                )
            if missing_seen_tasks:
                errors.append(
                    f"COCO/{variant}: seen test misses tasks={sorted(missing_seen_tasks)}"
                )
            for s in unseen:
                if v["roles"]["unseen"]["global_subjects"].get(s, 0) and agg["unseen"]["subjects"].get(s, 0) < 1:
                    errors.append(f"COCO/{variant}: unseen subject {s} absent from test")
            for s in sorted(set(v["roles"]["seen"]["global_subjects"])):
                if agg["seen"]["subjects"].get(s, 0) < 1:
                    errors.append(f"COCO/{variant}: seen subject {s} absent from test_seen")

        if errors:
            continue

        role_diag: dict[str, Any] = {}
        score = 0.0
        for role in ("all", "unseen", "seen"):
            info = v["roles"][role]
            a = agg[role]
            subj_tv = total_variation(info["global_subjects"], a["subjects"])
            rec_ratio = a["records"] / info["total_records"]
            rec_err = abs(rec_ratio - ratio)

            if dataset == "AiR":
                fam_tv = total_variation(info["global_families"], a["families"])
                qid_ratio = len(a["qids"]) / len(info["global_qids"])
                qid_err = abs(qid_ratio - ratio)
                score += 3.0 * subj_tv + 3.0 * fam_tv + 8.0 * rec_err + 5.0 * qid_err
                role_diag[role] = {
                    "subject_tv": subj_tv,
                    "question_family_tv": fam_tv,
                    "record_ratio": rec_ratio,
                    "record_ratio_abs_error": rec_err,
                    "qid_ratio": qid_ratio,
                    "qid_ratio_abs_error": qid_err,
                }
            else:
                task_tv = total_variation(info["global_tasks"], a["tasks"])
                cond_tv = total_variation(info["global_conditions"], a["conditions"])
                score += 4.0 * task_tv + 2.0 * subj_tv + 8.0 * rec_err
                if variant == "all":
                    score += 2.0 * cond_tv
                role_diag[role] = {
                    "task_tv": task_tv,
                    "subject_tv": subj_tv,
                    "condition_tv": cond_tv,
                    "record_ratio": rec_ratio,
                    "record_ratio_abs_error": rec_err,
                }

        image_ratio = len(test_variant_images) / len(v["variant_images"])
        image_err = abs(image_ratio - ratio)
        score += (5.0 if dataset == "AiR" else 4.0) * image_err
        total_score += score
        diagnostics[variant] = {
            "score": score,
            "image_ratio": image_ratio,
            "image_ratio_abs_error": image_err,
            "train_support_distinct_images": support_capacity,
            "roles": role_diag,
        }

    return errors, total_score, diagnostics


def choose_master_split(
    dataset: str,
    rows: list[dict[str, Any]],
    unseen_subjects: Sequence[str],
    max_attempts: int,
    ratio: float,
) -> tuple[int, dict[str, str], float, dict[str, Any]]:
    search_index = build_fast_search_index(dataset, rows, unseen_subjects)
    images: list[str] = search_index["images"]
    best: tuple[int, set[str], float, dict[str, Any]] | None = None
    valid = 0
    last_errors: list[str] = []

    for attempt in range(max_attempts):
        test_images = candidate_test_images(images, dataset, attempt, ratio)
        errors, score, diagnostics = fast_candidate_eval(
            dataset, search_index, test_images, unseen_subjects, ratio
        )
        if errors:
            last_errors = errors
            continue
        valid += 1
        payload = {"valid_attempts_seen": valid, "diagnostics": diagnostics}
        if best is None or score < best[2]:
            best = (attempt, test_images, score, payload)

    if best is None:
        raise SplitBuildError(
            f"{dataset}: no valid 90/10 master split in {max_attempts} attempts. "
            f"Last errors: {last_errors}"
        )

    attempt, test_images, score, payload = best
    payload["num_valid_candidates"] = valid
    payload["selected_attempt_id"] = attempt
    payload["selected_score"] = score
    image_split = {
        image: ("test" if image in test_images else "train") for image in images
    }
    return attempt, image_split, score, payload

def materialize(
    dataset: str,
    row: dict[str, Any],
    variant: str,
    split: str,
) -> dict[str, Any]:
    out = copy.deepcopy(row)
    out["record_id"] = canonical_record_id(dataset, row)
    out["dataset"] = dataset
    out["stimulus_id"] = row["name"]
    out["global_stimulus_id"] = f"{dataset}::{row['name']}"
    out["trial_key"] = trial_key(dataset, row)
    out["global_subject_id"] = f"{dataset}::{row['subject']}"
    out["split"] = split
    out["variant"] = variant
    return out


def build_support_draws(
    dataset: str,
    rows: list[dict[str, Any]],
    image_split: dict[str, str],
    unseen_subjects: Sequence[str],
) -> dict[str, list[list[dict[str, Any]]]]:
    train_images = {name for name, split in image_split.items() if split == "train"}
    units = shared_support_units(dataset, rows, unseen_subjects, train_images)

    # Resolve each unit to one row per unseen subject.
    resolver: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["name"] not in train_images:
            continue
        subject = normalize_subject(row["subject"])
        if subject not in set(unseen_subjects):
            continue
        key = (
            str(row["qid"]) if dataset == "AiR" else str(row["task"]),
            str(row["name"]),
        )
        resolver[key][subject] = row

    draws_by_k: dict[str, list[list[dict[str, Any]]]] = {}
    for k in KS:
        shuffled = list(units)
        # Preserve exact legacy COCO support draws; AiR gets its own stable seed.
        rng = (
            random.Random(k)
            if dataset == "COCO-Search18"
            else random.Random(f"{AIR_PROTOCOL_VERSION}|K={k}")
        )
        rng.shuffle(shuffled)

        retained: list[tuple[str, str]] = []
        used_images: set[str] = set()
        for unit in shuffled:
            image = unit[1]
            if image in used_images:
                continue
            used_images.add(image)
            retained.append(unit)
            if len(retained) == NUM_DRAWS * k:
                break

        required = NUM_DRAWS * k
        if len(retained) < required:
            raise SplitBuildError(
                f"{dataset}: need {required} exclusive support images for K={k}, "
                f"found {len(retained)}"
            )

        blocks: list[list[dict[str, Any]]] = []
        for draw_id in range(NUM_DRAWS):
            block = retained[draw_id * k : (draw_id + 1) * k]
            entries: list[dict[str, Any]] = []
            for unit_id, image_name in block:
                key = (unit_id, image_name)
                entries.append(
                    {
                        "dataset": dataset,
                        "unit_id": unit_id,
                        "image_name": image_name,
                        "trial_key": (
                            f"{unit_id}::{image_name}"
                        ),
                        "resolved_record_id_by_subject": {
                            s: canonical_record_id(dataset, resolver[key][s])
                            for s in unseen_subjects
                        },
                    }
                )
            blocks.append(entries)
        draws_by_k[str(k)] = blocks
    return draws_by_k


def exact_partition_check(
    dataset: str,
    original: list[dict[str, Any]],
    split_rows: dict[str, list[dict[str, Any]]],
) -> None:
    expected = {canonical_record_id(dataset, r) for r in original}
    lists = [[r["record_id"] for r in split_rows[k]] for k in ("train", "test", "test_seen")]
    actual = set().union(*(set(x) for x in lists))
    if actual != expected:
        raise SplitBuildError(
            f"{dataset}: partition mismatch missing={len(expected-actual)} extra={len(actual-expected)}"
        )
    if sum(len(x) for x in lists) != len(expected):
        raise SplitBuildError(f"{dataset}: a record appears in more than one split")


def split_contract_check(
    dataset: str,
    split_rows: dict[str, list[dict[str, Any]]],
    unseen_subjects: Sequence[str],
) -> None:
    unseen = set(unseen_subjects)
    train_images = {r["global_stimulus_id"] for r in split_rows["train"]}
    test_images = {
        r["global_stimulus_id"] for r in split_rows["test"] + split_rows["test_seen"]
    }
    if train_images & test_images:
        raise SplitBuildError(f"{dataset}: image leakage between train and test")
    if any(normalize_subject(r["subject"]) not in unseen for r in split_rows["test"]):
        raise SplitBuildError(f"{dataset}: test.json contains a seen subject")
    if any(normalize_subject(r["subject"]) in unseen for r in split_rows["test_seen"]):
        raise SplitBuildError(f"{dataset}: test_seen.json contains an unseen subject")


def build_one_variant(
    *,
    dataset: str,
    source_rows: list[dict[str, Any]],
    source_path: Path,
    output_dir: Path,
    variant: str,
    image_split: dict[str, str],
    unseen_subjects: Sequence[str],
    attempt_id: int,
    selected_score: float,
    search_report: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    rows = (
        source_rows
        if dataset == "AiR"
        else coco_variant_rows(source_rows, variant)
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise SplitBuildError(
                f"Output directory is non-empty: {output_dir}; use --force"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unseen = set(unseen_subjects)
    split_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "test": [],
        "test_seen": [],
    }
    for row in rows:
        if image_split[row["name"]] == "train":
            split = "train"
        elif normalize_subject(row["subject"]) in unseen:
            split = "test"
        else:
            split = "test_seen"
        item = materialize(dataset, row, variant, split)
        item["subject_role"] = (
            "unseen" if normalize_subject(row["subject"]) in unseen else "seen"
        )
        item["train_query_eligible"] = (split == "train" and item["subject_role"] == "seen")
        split_rows[split].append(item)

    exact_partition_check(dataset, rows, split_rows)
    split_contract_check(dataset, split_rows, unseen_subjects)
    support_draws = build_support_draws(dataset, rows, image_split, unseen_subjects)

    split_paths = {
        split: output_dir / f"{split}.json" for split in ("train", "test", "test_seen")
    }
    for split, path in split_paths.items():
        dump_json(path, split_rows[split], pretty=False)

    images = {r["name"] for r in rows}
    train_images = sorted(name for name in images if image_split[name] == "train")
    test_images = sorted(name for name in images if image_split[name] == "test")
    subjects = sorted({normalize_subject(r["subject"]) for r in rows})
    seen_subjects = sorted(set(subjects) - set(unseen_subjects))

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "variant": variant,
        "source_file": source_path.as_posix(),
        "source_sha256": sha256_file(source_path),
        "split_unit": "name (image)",
        "target_test_image_ratio": TEST_IMAGE_RATIO,
        "master_attempt_id": attempt_id,
        "master_selected_score": selected_score,
        "master_search_report": search_report,
        "seen_subject_ids": seen_subjects,
        "unseen_subject_ids": list(unseen_subjects),
        "runtime_subject_role": {
            "train": "all subjects on train images; optimize seen-subject queries only; unseen rows support-only",
            "test": "unseen subjects on test images",
            "test_seen": "seen subjects on the same test-image pool",
        },
        "all_source_records_materialized": True,
        "discarded_records": 0,
        "train_stimulus_ids": train_images,
        "test_stimulus_ids": test_images,
        "support_sampling": {
            "source_split": "train_only",
            "unit": "qid+image" if dataset == "AiR" else "task+image",
            "same_unit_for_all_unseen_subjects": True,
            "distinct_image_within_draw": True,
            "exclusive_across_draws_within_k": True,
            "k_values": list(KS),
            "num_draws_per_k": NUM_DRAWS,
        },
        "support_draws": support_draws,
        "split_files": {k: v.as_posix() for k, v in split_paths.items()},
        "split_file_sha256": {k: sha256_file(v) for k, v in split_paths.items()},
    }
    dump_json(output_dir / "split_manifest.json", manifest, pretty=True)

    def count_field(rr: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
        return dict(sorted(Counter(str(r[field]) for r in rr).items()))

    report = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "variant": variant,
        "total_records": len(rows),
        "total_unique_images": len(images),
        "record_counts": {k: len(v) for k, v in split_rows.items()},
        "record_ratios": {k: len(v) / len(rows) for k, v in split_rows.items()},
        "image_counts": {"train": len(train_images), "test": len(test_images)},
        "image_ratios": {
            "train": len(train_images) / len(images),
            "test": len(test_images) / len(images),
        },
        "subject_counts": {
            k: count_field(v, "subject") for k, v in split_rows.items()
        },
        "condition_counts": {
            k: count_field(v, "condition") for k, v in split_rows.items()
        },
        "discarded_records": 0,
        "all_source_records_materialized": (
            sum(len(v) for v in split_rows.values()) == len(rows)
        ),
        "shared_train_support_distinct_images": support_image_capacity(
            dataset, rows, unseen_subjects, set(train_images)
        ),
    }
    if dataset == "AiR":
        report["qid_counts"] = {
            k: len({str(r["qid"]) for r in v}) for k, v in split_rows.items()
        }
        report["question_family_counts"] = {
            k: dict(sorted(Counter(question_family(r["task"]) for r in v).items()))
            for k, v in split_rows.items()
        }
    else:
        report["task_counts"] = {
            k: count_field(v, "task") for k, v in split_rows.items()
        }
    dump_json(output_dir / "preprocess_report.json", report, pretty=True)

    return {
        "dataset": dataset,
        "variant": variant,
        "output_dir": output_dir.as_posix(),
        "records": len(rows),
        "images": len(images),
        "record_counts": report["record_counts"],
        "image_counts": report["image_counts"],
        "support_images": report["shared_train_support_distinct_images"],
    }


def write_master_manifest(
    *,
    dataset: str,
    rows: list[dict[str, Any]],
    source_path: Path,
    output_root: Path,
    image_split: dict[str, str],
    unseen_subjects: Sequence[str],
    attempt_id: int,
    selected_score: float,
    search_report: dict[str, Any],
) -> None:
    images = sorted(image_split)
    counts = Counter(image_split.values())
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "source_file": source_path.as_posix(),
        "source_sha256": sha256_file(source_path),
        "split_unit": "name (image)",
        "target_ratio": {
            "train_images": 1.0 - TEST_IMAGE_RATIO,
            "test_images": TEST_IMAGE_RATIO,
        },
        "actual_master_image_counts": dict(counts),
        "actual_master_image_ratios": {
            k: v / len(images) for k, v in sorted(counts.items())
        },
        "selected_attempt_id": attempt_id,
        "selected_score": selected_score,
        "search_report": search_report,
        "train_stimulus_ids": [x for x in images if image_split[x] == "train"],
        "test_stimulus_ids": [x for x in images if image_split[x] == "test"],
        "unseen_subject_ids": list(unseen_subjects),
        "test_subject_partition": {
            "test": "subject in unseen_subject_ids",
            "test_seen": "subject not in unseen_subject_ids",
        },
        "all_source_records_materialized": True,
        "discarded_records": 0,
    }
    dump_json(output_root / "master_split_manifest.json", manifest, pretty=True)


def build_dataset(
    *,
    dataset: str,
    source_path: Path,
    output_root: Path,
    unseen_subjects: Sequence[str],
    max_attempts: int,
    ratio: float,
    force: bool,
) -> tuple[list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    rows = load_records(source_path, dataset)
    validate_unseen_subjects(dataset, rows, unseen_subjects)
    output_root.mkdir(parents=True, exist_ok=True)

    attempt, image_split, score, search_report = choose_master_split(
        dataset, rows, unseen_subjects, max_attempts, ratio
    )
    write_master_manifest(
        dataset=dataset,
        rows=rows,
        source_path=source_path,
        output_root=output_root,
        image_split=image_split,
        unseen_subjects=unseen_subjects,
        attempt_id=attempt,
        selected_score=score,
        search_report=search_report,
    )

    variants = ["all"] if dataset == "AiR" else ["all", "tp", "ta"]
    summaries: list[dict[str, Any]] = []
    for variant in variants:
        summary = build_one_variant(
            dataset=dataset,
            source_rows=rows,
            source_path=source_path,
            output_dir=output_root / variant,
            variant=variant,
            image_split=image_split,
            unseen_subjects=unseen_subjects,
            attempt_id=attempt,
            selected_score=score,
            search_report=search_report,
            force=force,
        )
        summaries.append(summary)
        print(
            f"[{dataset}/{variant}] records={summary['records']} images={summary['images']} "
            f"image_counts={summary['image_counts']} record_counts={summary['record_counts']} "
            f"support_images={summary['support_images']}"
        )

    index = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "source_file": source_path.as_posix(),
        "source_sha256": sha256_file(source_path),
        "master_split_manifest": (output_root / "master_split_manifest.json").as_posix(),
        "master_attempt_id": attempt,
        "master_selected_score": score,
        "unseen_subject_ids": list(unseen_subjects),
        "k_values": list(KS),
        "num_draws_per_k": NUM_DRAWS,
        "variants": summaries,
    }
    dump_json(output_root / "split_index.json", index, pretty=True)
    return rows, image_split, summaries


def read_materialized_split(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SplitBuildError(f"Expected generated split missing: {path}")
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise SplitBuildError(f"Expected JSON list: {path}")
    return rows


def build_combined(
    *,
    air_output_root: Path,
    coco_output_root: Path,
    combined_output: Path,
    force: bool,
) -> None:
    """Combine dataset-specific `all` memberships without resplitting."""
    if combined_output.exists() and any(combined_output.iterdir()):
        if not force:
            raise SplitBuildError(
                f"Combined output is non-empty: {combined_output}; use --force"
            )
        shutil.rmtree(combined_output)
    combined_output.mkdir(parents=True, exist_ok=True)

    combined: dict[str, list[dict[str, Any]]] = {}
    per_dataset_counts: dict[str, dict[str, int]] = {"AiR": {}, "COCO-Search18": {}}
    for split in ("train", "test", "test_seen"):
        air_rows = read_materialized_split(air_output_root / "all" / f"{split}.json")
        coco_rows = read_materialized_split(coco_output_root / "all" / f"{split}.json")
        per_dataset_counts["AiR"][split] = len(air_rows)
        per_dataset_counts["COCO-Search18"][split] = len(coco_rows)
        merged = air_rows + coco_rows
        merged.sort(key=lambda r: (r["dataset"], r["record_id"]))
        combined[split] = merged
        dump_json(combined_output / f"{split}.json", merged, pretty=False)

    # Strong combined leakage check uses namespaced image IDs.
    train_images = {r["global_stimulus_id"] for r in combined["train"]}
    eval_images = {
        r["global_stimulus_id"]
        for r in combined["test"] + combined["test_seen"]
    }
    overlap = train_images & eval_images
    if overlap:
        raise SplitBuildError(
            f"Combined split has train/eval image leakage; example={next(iter(overlap))}"
        )

    record_ids = [r["record_id"] for rows in combined.values() for r in rows]
    if len(record_ids) != len(set(record_ids)):
        raise SplitBuildError("Combined split has duplicate record_id values")

    with (air_output_root / "all" / "split_manifest.json").open("r", encoding="utf-8") as f:
        air_manifest = json.load(f)
    with (coco_output_root / "all" / "split_manifest.json").open("r", encoding="utf-8") as f:
        coco_manifest = json.load(f)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "all",
        "composition": "AiR/all + COCO-Search18/all; memberships preserved exactly",
        "resplit_after_merge": False,
        "split_unit": "dataset::name (namespaced image)",
        "all_source_records_materialized": True,
        "discarded_records": 0,
        "per_dataset_unseen_subject_ids": {
            "AiR": air_manifest["unseen_subject_ids"],
            "COCO-Search18": coco_manifest["unseen_subject_ids"],
        },
        "support_sampling": {
            "rule": "support must match query dataset and subject",
            "per_dataset_support_draws": {
                "AiR": air_manifest["support_draws"],
                "COCO-Search18": coco_manifest["support_draws"],
            },
        },
        "record_counts": {k: len(v) for k, v in combined.items()},
        "per_dataset_record_counts": per_dataset_counts,
        "split_files": {
            k: (combined_output / f"{k}.json").as_posix() for k in combined
        },
        "split_file_sha256": {
            k: sha256_file(combined_output / f"{k}.json") for k in combined
        },
        "source_manifests": {
            "AiR": (air_output_root / "all" / "split_manifest.json").as_posix(),
            "COCO-Search18": (
                coco_output_root / "all" / "split_manifest.json"
            ).as_posix(),
        },
    }
    dump_json(combined_output / "split_manifest.json", manifest, pretty=True)

    report = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "all",
        "record_counts": {k: len(v) for k, v in combined.items()},
        "per_dataset_record_counts": per_dataset_counts,
        "unique_global_images": {
            k: len({r["global_stimulus_id"] for r in v}) for k, v in combined.items()
        },
        "dataset_counts": {
            k: dict(sorted(Counter(r["dataset"] for r in v).items()))
            for k, v in combined.items()
        },
        "discarded_records": 0,
        "train_eval_image_overlap": 0,
    }
    dump_json(combined_output / "preprocess_report.json", report, pretty=True)

    print(
        f"[all combined] record_counts={manifest['record_counts']} "
        f"per_dataset={per_dataset_counts}"
    )


def parse_subject_csv(value: str) -> tuple[str, ...]:
    parts = tuple(x.strip() for x in value.split(",") if x.strip())
    if not parts:
        raise argparse.ArgumentTypeError("subject list cannot be empty")
    return parts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build SemGaze AiR, COCO-Search18, and combined leakage-safe splits"
    )
    p.add_argument("--air-input", type=Path, default=DEFAULT_AIR_INPUT)
    p.add_argument("--coco-input", type=Path, default=DEFAULT_COCO_INPUT)
    p.add_argument("--air-output-root", type=Path, default=DEFAULT_AIR_OUTPUT)
    p.add_argument("--coco-output-root", type=Path, default=DEFAULT_COCO_OUTPUT)
    p.add_argument("--combined-output", type=Path, default=DEFAULT_COMBINED_OUTPUT)
    p.add_argument(
        "--air-unseen-subjects",
        type=parse_subject_csv,
        default=DEFAULT_AIR_UNSEEN,
        help="comma-separated, default: JY,SC,YN",
    )
    p.add_argument(
        "--coco-unseen-subjects",
        type=parse_subject_csv,
        default=tuple(str(x) for x in DEFAULT_COCO_UNSEEN),
        help="comma-separated, default: 7,8,9",
    )
    p.add_argument("--test-image-ratio", type=float, default=TEST_IMAGE_RATIO)
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    p.add_argument(
        "--only",
        choices=("all", "air", "coco"),
        default="all",
        help=(
            "all=build AiR + COCO + combined; air=AiR only; "
            "coco=COCO only"
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="delete/regenerate non-empty generated split directories",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_attempts <= 0:
        raise SplitBuildError("--max-attempts must be > 0")
    if not (0.0 < args.test_image_ratio < 1.0):
        raise SplitBuildError("--test-image-ratio must be in (0,1)")
    if abs(args.test_image_ratio - TEST_IMAGE_RATIO) > 1e-12:
        # The protocol is designed for 90/10. We permit override, but make it explicit.
        print(
            f"WARNING: overriding protocol test image ratio from {TEST_IMAGE_RATIO} "
            f"to {args.test_image_ratio}"
        )

    if args.only in {"all", "air"}:
        build_dataset(
            dataset="AiR",
            source_path=args.air_input,
            output_root=args.air_output_root,
            unseen_subjects=tuple(args.air_unseen_subjects),
            max_attempts=args.max_attempts,
            ratio=args.test_image_ratio,
            force=args.force,
        )

    if args.only in {"all", "coco"}:
        build_dataset(
            dataset="COCO-Search18",
            source_path=args.coco_input,
            output_root=args.coco_output_root,
            unseen_subjects=tuple(args.coco_unseen_subjects),
            max_attempts=args.max_attempts,
            ratio=args.test_image_ratio,
            force=args.force,
        )

    if args.only == "all":
        build_combined(
            air_output_root=args.air_output_root,
            coco_output_root=args.coco_output_root,
            combined_output=args.combined_output,
            force=args.force,
        )


if __name__ == "__main__":
    try:
        main()
    except SplitBuildError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
