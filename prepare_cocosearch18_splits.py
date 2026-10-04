#!/usr/bin/env python3
"""Materialize SemGaze COCO-Search18 81/9/10 splits and frozen few-shot draws.

Run from repository root:

    python prepare_cocosearch18_splits.py

Default input:
    data/COCO_Search18/COCOSearch-18.json

Default output:
    data/COCO_Search18/split/{tp_only,ta_only,all}/...

The script is intentionally stdlib-only and deterministic.
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


PROTOCOL_VERSION = "cocosearch18_semgaze_81910_v1"
DEFAULT_INPUT = Path("data/COCO_Search18/COCOSearch-18.json")
DEFAULT_OUTPUT_ROOT = Path("data/COCO_Search18/split")
UNSEEN_SUBJECTS = (7, 8, 9)
KS = (1, 5, 10)
NUM_DRAWS = 10
TRAIN_RATIO = 0.81
VAL_RATIO = 0.09
MAX_ATTEMPTS = 10000

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
            f"record[{index}] condition must be 'present' or 'absent', got {record['condition']!r}"
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
        raise SplitBuildError(f"record[{index}] prediction.regions must be a non-empty list")
    if not isinstance(how, str) or not how.strip():
        raise SplitBuildError(f"record[{index}] prediction.how must be non-empty")

    fixation_ids: list[int] = []
    for j, fx in enumerate(fixations):
        if not isinstance(fx, dict):
            raise SplitBuildError(f"record[{index}] fixation[{j}] must be an object")
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
            f"record[{index}] fixation IDs must cover 1..{n} exactly once; got {fixation_ids}"
        )

    region_membership: list[int] = []
    for j, region in enumerate(regions):
        if not isinstance(region, dict):
            raise SplitBuildError(f"record[{index}] region[{j}] must be an object")
        ids = region.get("fixations")
        why = region.get("why")
        if not isinstance(ids, list) or not ids:
            raise SplitBuildError(f"record[{index}] region[{j}].fixations must be non-empty")
        if not all(isinstance(fid, int) for fid in ids):
            raise SplitBuildError(f"record[{index}] region[{j}].fixations must contain ints")
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


def split_images_for_attempt(
    records: list[dict[str, Any]], variant: str, attempt_id: int
) -> dict[str, str]:
    images = sorted({r["name"] for r in records})
    images.sort(
        key=lambda name: stable_digest(
            f"{PROTOCOL_VERSION}|{variant}|{attempt_id}|image::{name}"
        )
    )

    n = len(images)
    n_train = math.floor(TRAIN_RATIO * n)
    n_val = math.floor(VAL_RATIO * n)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise SplitBuildError(
            f"Variant {variant!r} has too few images for 81/9/10: N={n}, "
            f"counts=({n_train}, {n_val}, {n_test})"
        )

    out: dict[str, str] = {}
    for name in images[:n_train]:
        out[name] = "train"
    for name in images[n_train : n_train + n_val]:
        out[name] = "validation"
    for name in images[n_train + n_val :]:
        out[name] = "test"
    return out


def index_by_subject(records: Iterable[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        out[int(record["subject"])].append(record)
    return dict(out)


def shared_support_candidates(
    records: list[dict[str, Any]], image_split: dict[str, str]
) -> list[tuple[str, str]]:
    """Return eligible shared (task, image_name) support units for subjects 7/8/9."""
    by_trial_subject: dict[tuple[str, str], dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        if image_split[record["name"]] != "train":
            continue
        subject = int(record["subject"])
        if subject not in UNSEEN_SUBJECTS:
            continue
        key = (record["task"], record["name"])
        by_trial_subject[key][subject].append(record)

    eligible: list[tuple[str, str]] = []
    for key, per_subject in by_trial_subject.items():
        if all(len(per_subject.get(subject, [])) == 1 for subject in UNSEEN_SUBJECTS):
            eligible.append(key)
    return sorted(eligible)


def coverage_errors(
    records: list[dict[str, Any]], image_split: dict[str, str]
) -> list[str]:
    errors: list[str] = []
    by_subject = index_by_subject(records)
    present_subjects = set(by_subject)

    missing_unseen = set(UNSEEN_SUBJECTS) - present_subjects
    if missing_unseen:
        errors.append(f"missing unseen subjects: {sorted(missing_unseen)}")

    seen_subjects = sorted(present_subjects - set(UNSEEN_SUBJECTS))
    if not seen_subjects:
        errors.append("no seen subjects present")

    for subject in seen_subjects:
        rows = by_subject[subject]
        train_images = {
            r["name"] for r in rows if image_split[r["name"]] == "train"
        }
        val_count = sum(1 for r in rows if image_split[r["name"]] == "validation")
        if len(train_images) < 11:
            errors.append(
                f"seen subject {subject} has only {len(train_images)} distinct train images"
            )
        if val_count < 1:
            errors.append(f"seen subject {subject} has no validation query record")

    for subject in UNSEEN_SUBJECTS:
        rows = by_subject.get(subject, [])
        test_count = sum(1 for r in rows if image_split[r["name"]] == "test")
        if test_count < 1:
            errors.append(f"unseen subject {subject} has no test query record")

    shared = shared_support_candidates(records, image_split)
    shared_distinct_images = len({image_name for _, image_name in shared})
    if shared_distinct_images < NUM_DRAWS * max(KS):
        errors.append(
            "insufficient shared support coverage: "
            f"need >= {NUM_DRAWS * max(KS)} distinct images, "
            f"got {shared_distinct_images}"
        )

    return errors


def choose_split(
    records: list[dict[str, Any]], variant: str, max_attempts: int
) -> tuple[int, dict[str, str]]:
    last_errors: list[str] = []
    for attempt_id in range(max_attempts):
        image_split = split_images_for_attempt(records, variant, attempt_id)
        errors = coverage_errors(records, image_split)
        if not errors:
            return attempt_id, image_split
        last_errors = errors
    raise SplitBuildError(
        f"No valid split for variant={variant!r} in attempts [0, {max_attempts - 1}]. "
        f"Last errors: {last_errors}"
    )


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


def build_validation_supports(
    records: list[dict[str, Any]], image_split: dict[str, str]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    by_subject = index_by_subject(records)
    seen_subjects = sorted(set(by_subject) - set(UNSEEN_SUBJECTS))
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for subject in seen_subjects:
        train_rows = [
            r for r in by_subject[subject] if image_split[r["name"]] == "train"
        ]
        by_trial: dict[str, dict[str, Any]] = {}
        for row in train_rows:
            trial_key = canonical_trial_key(row)
            if trial_key in by_trial:
                raise SplitBuildError(
                    f"Validation support trial_key not unique for subject={subject}: {trial_key}"
                )
            by_trial[trial_key] = row

        result[str(subject)] = {}
        for k in KS:
            units = sorted(by_trial)
            rng = random.Random(1000 + k)
            rng.shuffle(units)
            selected: list[dict[str, Any]] = []
            used_images: set[str] = set()
            for trial_key in units:
                row = by_trial[trial_key]
                if row["name"] in used_images:
                    continue
                used_images.add(row["name"])
                selected.append(
                    {
                        "task": row["task"],
                        "image_name": row["name"],
                        "trial_key": trial_key,
                        "record_id": canonical_record_id(row),
                    }
                )
                if len(selected) == k:
                    break
            if len(selected) != k:
                raise SplitBuildError(
                    f"Cannot construct validation {k}-shot support for seen subject {subject}"
                )
            result[str(subject)][str(k)] = selected
    return result


def build_final_support_draws(
    records: list[dict[str, Any]], image_split: dict[str, str]
) -> dict[str, list[list[dict[str, Any]]]]:
    # Resolve each shared trial to exactly one record per unseen subject.
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

    eligible = []
    for key in sorted(resolver):
        if all(counts[(key, subject)] == 1 for subject in UNSEEN_SUBJECTS):
            eligible.append(key)

    draws_by_k: dict[str, list[list[dict[str, Any]]]] = {}
    for k in KS:
        units = list(eligible)
        rng = random.Random(k)
        rng.shuffle(units)

        # Strong exclusivity: no image is reused anywhere across the 10 draws
        # of this K family, even if multiple tasks exist for the image.
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
            f"Generated records do not exactly partition the variant: "
            f"missing={len(expected - actual)}, extra={len(actual - expected)}"
        )
    total = sum(len(ids) for ids in actual_lists.values())
    if total != len(expected):
        raise SplitBuildError("A record appears in more than one generated split")


def verify_image_disjointness(split_rows: dict[str, list[dict[str, Any]]]) -> None:
    image_sets = {
        split: {r["stimulus_id"] for r in rows} for split, rows in split_rows.items()
    }
    names = list(image_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = image_sets[names[i]] & image_sets[names[j]]
            if overlap:
                example = next(iter(overlap))
                raise SplitBuildError(
                    f"Image leakage between {names[i]} and {names[j]}: {example}"
                )


def make_report(
    variant: str,
    records: list[dict[str, Any]],
    image_split: dict[str, str],
    attempt_id: int,
) -> dict[str, Any]:
    record_counts = Counter(image_split[r["name"]] for r in records)
    image_counts = Counter(image_split.values())
    condition_counts = Counter(r["condition"] for r in records)
    subject_counts = Counter(str(r["subject"]) for r in records)
    split_subject_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }
    for r in records:
        split_subject_counts[image_split[r["name"]]][str(r["subject"])] += 1

    shared = shared_support_candidates(records, image_split)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "variant": variant,
        "attempt_id": attempt_id,
        "total_records": len(records),
        "total_unique_images": len(image_split),
        "condition_counts": dict(sorted(condition_counts.items())),
        "subject_record_counts": dict(sorted(subject_counts.items(), key=lambda kv: int(kv[0]))),
        "record_counts_by_split": dict(record_counts),
        "image_counts_by_split": dict(image_counts),
        "subject_record_counts_by_split": {
            split: dict(sorted(counts.items(), key=lambda kv: int(kv[0])))
            for split, counts in split_subject_counts.items()
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
    max_attempts: int,
    force: bool,
) -> dict[str, Any]:
    records = variant_records(source_records, variant)
    attempt_id, image_split = choose_split(records, variant, max_attempts)

    variant_dir = output_root / variant
    if variant_dir.exists() and any(variant_dir.iterdir()):
        if not force:
            raise SplitBuildError(
                f"Output directory is not empty: {variant_dir}. Use --force to regenerate."
            )
        shutil.rmtree(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)

    split_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for record in records:
        split = image_split[record["name"]]
        split_rows[split].append(materialized_record(record, variant, split))

    assert_exact_partition(records, split_rows)
    verify_image_disjointness(split_rows)

    validation_supports = build_validation_supports(records, image_split)
    support_draws = build_final_support_draws(records, image_split)

    split_paths = {
        split: variant_dir / f"{split}.json" for split in ("train", "validation", "test")
    }
    for split, path in split_paths.items():
        dump_json(path, split_rows[split], pretty=False)

    split_hashes = {split: sha256_file(path) for split, path in split_paths.items()}
    source_hash = sha256_file(source_path)

    seen_subjects = sorted(
        {int(r["subject"]) for r in records} - set(UNSEEN_SUBJECTS)
    )
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "COCO-Search18",
        "benchmark_kind": "SemGaze-curated",
        "variant": variant,
        "condition_filter": {
            "tp_only": 'condition == "present"',
            "ta_only": 'condition == "absent"',
            "all": "all curated records",
        }[variant],
        "curated_source_file": source_path.as_posix(),
        "curated_source_sha256": source_hash,
        "split_generation": {
            "algorithm": "SHA256 ranking over unique stimulus_id=name",
            "attempt_id": attempt_id,
            "image_ratio": [TRAIN_RATIO, VAL_RATIO, 1.0 - TRAIN_RATIO - VAL_RATIO],
            "split_unit": "stimulus_id=name",
            "variant_split_independently": True,
        },
        "split_files": {split: path.as_posix() for split, path in split_paths.items()},
        "split_file_sha256": split_hashes,
        "train_stimulus_ids": sorted(
            name for name, split in image_split.items() if split == "train"
        ),
        "validation_stimulus_ids": sorted(
            name for name, split in image_split.items() if split == "validation"
        ),
        "test_stimulus_ids": sorted(
            name for name, split in image_split.items() if split == "test"
        ),
        "seen_subject_ids": seen_subjects,
        "unseen_subject_ids": list(UNSEEN_SUBJECTS),
        "validation_supports": validation_supports,
        "support_sampling": {
            "unit": "trial_key=(task,image_name)",
            "same_trial_keys_for_unseen_subjects": True,
            "distinct_image_within_block": True,
            "exclusive_across_draws_within_k": True,
            "num_draws_per_k": NUM_DRAWS,
            "python_random_seed_by_k": {str(k): k for k in KS},
            "nested_k": False,
        },
        "support_draws": support_draws,
    }

    report = make_report(variant, records, image_split, attempt_id)
    report["split_file_sha256"] = split_hashes

    dump_json(variant_dir / "split_manifest.json", manifest, pretty=True)
    dump_json(variant_dir / "preprocess_report.json", report, pretty=True)

    return {
        "variant": variant,
        "attempt_id": attempt_id,
        "records": len(records),
        "images": len(image_split),
        "image_counts": dict(Counter(image_split.values())),
        "record_counts": dict(Counter(image_split[r["name"]] for r in records)),
        "shared_support_distinct_images": report["shared_train_support_distinct_images"],
        "output_dir": variant_dir.as_posix(),
        "manifest_sha256": sha256_file(variant_dir / "split_manifest.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic SemGaze COCO-Search18 81/9/10 image-level splits "
            "for tp_only, ta_only, and all variants."
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

    summaries = []
    for variant in ("tp_only", "ta_only", "all"):
        summary = build_variant(
            source_records=source_records,
            source_path=args.input,
            output_root=args.output_root,
            variant=variant,
            max_attempts=args.max_attempts,
            force=args.force,
        )
        summaries.append(summary)
        print(
            f"[{variant}] attempt={summary['attempt_id']} "
            f"records={summary['records']} images={summary['images']} "
            f"image_counts={summary['image_counts']} "
            f"shared_support_images={summary['shared_support_distinct_images']}"
        )

    index = {
        "protocol_version": PROTOCOL_VERSION,
        "source_file": args.input.as_posix(),
        "source_sha256": sha256_file(args.input),
        "unseen_subject_ids": list(UNSEEN_SUBJECTS),
        "k_shot": list(KS),
        "num_exclusive_draws_per_k": NUM_DRAWS,
        "variants": summaries,
    }
    dump_json(args.output_root / "split_index.json", index, pretty=True)
    print(f"Wrote split index: {args.output_root / 'split_index.json'}")


if __name__ == "__main__":
    try:
        main()
    except SplitBuildError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
