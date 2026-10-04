#!/usr/bin/env python3
"""SemGaze flat baseline smoke harness.

Two levels:

  --contract-only  Pure contract/golden checks. No model download/GPU required.
  --with-model     One differentiable K=1 vertical slice. Requires implementation
                   modules, the HF InternVL base, released DeepGaze adapter, and GPU.

This file contains orchestration/assertions rather than model logic.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_FIXTURE = REPO_ROOT / "tests/fixtures/flat_episode.json"
DEFAULT_EXPECTED_WHERE = REPO_ROOT / "tests/fixtures/expected_where.txt"
DEFAULT_EXPECTED_FLAT = REPO_ROOT / "tests/fixtures/expected_flat_target.txt"
DEFAULT_CONFIG = REPO_ROOT / "configs/flat_single.yaml"


@dataclass(frozen=True)
class SymbolSpec:
    module: str
    name: str
    purpose: str


CONTRACT_SYMBOLS = (
    SymbolSpec(
        "semgaze.data.schema",
        "normalized_episode_from_dict",
        "convert normalized fixture mapping into typed FlatEpisode/NormalizedRecord objects",
    ),
    SymbolSpec(
        "semgaze.where.serialization",
        "serialize_xyd_record",
        "canonical XYD + <END_FIX> serializer",
    ),
    SymbolSpec(
        "semgaze.semantic.flat.target",
        "build_flat_target",
        "canonical flat WHAT/WHY/HOW target builder",
    ),
    SymbolSpec(
        "semgaze.semantic.flat.parser",
        "parse_flat_output",
        "deterministic strict flat-output parser",
    ),
)

MODEL_SYMBOLS = (
    SymbolSpec(
        "semgaze.model.build",
        "build_flat_model_bundle",
        "HF InternVL + released trainable DeepGaze adapter + END_FIX + P_E loader",
    ),
    SymbolSpec(
        "semgaze.training.flat_step",
        "run_flat_training_step",
        "one differentiable WHERE -> states -> flat semantic joint step",
    ),
)


def _import_symbol(spec: SymbolSpec) -> Callable[..., Any]:
    module = importlib.import_module(spec.module)
    fn = getattr(module, spec.name)
    if not callable(fn):
        raise TypeError(f"{spec.module}.{spec.name} exists but is not callable")
    return fn


def _load_symbols(specs: tuple[SymbolSpec, ...]) -> dict[str, Callable[..., Any]]:
    loaded: dict[str, Callable[..., Any]] = {}
    missing: list[str] = []
    for spec in specs:
        try:
            loaded[spec.name] = _import_symbol(spec)
        except Exception as exc:  # deliberate diagnostic boundary for a skeleton
            missing.append(
                f"- {spec.module}.{spec.name}: {spec.purpose}\n"
                f"  import error: {type(exc).__name__}: {exc}"
            )
    if missing:
        raise RuntimeError(
            "Required implementation symbols are not ready:\n" + "\n".join(missing)
        )
    return loaded


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_expected_where(path: Path) -> tuple[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise AssertionError(f"Malformed golden WHERE line: {raw!r}")
        values[key] = value
    if set(values) != {"SUPPORT", "QUERY"}:
        raise AssertionError(f"Expected SUPPORT and QUERY golden lines, got {sorted(values)}")
    return values["SUPPORT"], values["QUERY"]


def _basic_fixture_invariants(fixture: dict[str, Any]) -> None:
    supports = fixture["supports"]
    query = fixture["query"]
    assert len(supports) == 1, "This smoke fixture is intentionally K=1"
    assert supports[0]["subject"] == query["subject"]
    assert supports[0]["stimulus_id"] != query["stimulus_id"]

    for record in [*supports, query]:
        n = len(record["x_px"])
        assert n > 0
        assert len(record["y_px"]) == n
        assert len(record["duration_ms"]) == n
        semantic = record["semantic"]
        assert len(semantic["what"]) == n

        groups = semantic["why_groups"]
        members = [idx for group in groups for idx in group["members"]]
        assert sorted(members) == list(range(1, n + 1))
        assert len(members) == len(set(members))
        mins = [min(group["members"]) for group in groups]
        assert mins == sorted(mins)
        assert semantic["how"].strip()

    # Deliberate coverage conditions for the fixture.
    assert query["semantic"]["why_groups"][0]["members"] == [1, 3]
    assert query["semantic"]["why_groups"][1]["members"] == [2, 4]
    assert any(float(d) > 999 for d in query["duration_ms"])
    assert "\n" in query["semantic"]["how"]


def run_contract_only(args: argparse.Namespace) -> None:
    fixture = _load_json(args.fixture)
    _basic_fixture_invariants(fixture)

    symbols = _load_symbols(CONTRACT_SYMBOLS)
    normalized_episode_from_dict = symbols["normalized_episode_from_dict"]
    serialize_xyd_record = symbols["serialize_xyd_record"]
    build_flat_target = symbols["build_flat_target"]
    parse_flat_output = symbols["parse_flat_output"]

    expected_support, expected_query = _load_expected_where(args.expected_where)
    expected_flat = args.expected_flat.read_text(encoding="utf-8").rstrip("\n")

    episode = normalized_episode_from_dict(fixture)
    supports = episode["supports"] if isinstance(episode, dict) else episode.supports
    query = episode["query"] if isinstance(episode, dict) else episode.query
    query_semantic = query["semantic"] if isinstance(query, dict) else query.semantic

    got_support = serialize_xyd_record(supports[0])
    got_query = serialize_xyd_record(query)
    got_flat = build_flat_target(query_semantic)

    assert got_support == expected_support, (
        "Support WHERE serialization mismatch\n"
        f"EXPECTED: {expected_support}\nGOT:      {got_support}"
    )
    assert got_query == expected_query, (
        "Query WHERE serialization mismatch\n"
        f"EXPECTED: {expected_query}\nGOT:      {got_query}"
    )
    assert got_flat == expected_flat, (
        "Flat target mismatch\n"
        f"EXPECTED:\n{expected_flat}\n\nGOT:\n{got_flat}"
    )

    n = len(fixture["query"]["x_px"])
    m = len(fixture["query"]["semantic"]["why_groups"])
    parsed = parse_flat_output(got_flat, n=n, m=m)

    valid = parsed["flat_format_valid"] if isinstance(parsed, dict) else parsed.flat_format_valid
    assert valid is True

    print("[OK] fixture structural invariants")
    print("[OK] support XYD serialization golden")
    print("[OK] query XYD serialization golden")
    print("[OK] flat target golden")
    print("[OK] flat parser round-trip")


def _finite_scalar(value: Any, name: str) -> float:
    if hasattr(value, "detach"):
        value = value.detach().float().item()
    value = float(value)
    assert math.isfinite(value), f"{name} is not finite: {value}"
    return value


def run_with_model(args: argparse.Namespace) -> None:
    # Contract checks first: model smoke must never hide a broken pure codec.
    run_contract_only(args)

    fixture = _load_json(args.fixture)
    symbols = _load_symbols(MODEL_SYMBOLS)
    build_flat_model_bundle = symbols["build_flat_model_bundle"]
    run_flat_training_step = symbols["run_flat_training_step"]

    bundle = build_flat_model_bundle(config_path=args.config)

    # Expected bundle diagnostics. The implementation may use a dataclass or dict.
    diag = bundle["diagnostics"] if isinstance(bundle, dict) else bundle.diagnostics
    assert diag["adapter_source"].endswith("DeepGaze-VL/model/visual_search_adapter")
    assert diag["adapter_trainable"] is True
    assert diag["end_fix_token_count"] == 1
    assert diag["lora_trainable_parameter_count"] > 0
    assert diag["projector_trainable_parameter_count"] > 0
    assert diag["vision_tower_frozen"] is True
    assert diag["native_multimodal_projector_frozen"] is True

    result = run_flat_training_step(
        model_bundle=bundle,
        episode=fixture,
        config_path=args.config,
        optimizer_step=args.step,
    )

    get = result.get if isinstance(result, dict) else lambda k: getattr(result, k)

    loss_where = _finite_scalar(get("loss_where"), "L_WHERE")
    loss_flat = _finite_scalar(get("loss_flat"), "L_FLAT")
    loss_total = _finite_scalar(get("loss_total"), "L_total")

    n = len(fixture["query"]["x_px"])
    assert int(get("query_fixation_count")) == n
    assert int(get("query_end_fix_state_count")) == n
    assert int(get("supervised_query_end_fix_count")) == n
    assert int(get("supervised_support_end_fix_count")) == 0
    assert tuple(get("projector_output_shape"))[0] == n
    assert int(get("inserted_state_count")) == n

    assert bool(get("lora_grad_finite"))
    assert bool(get("projector_grad_finite"))
    assert bool(get("projector_grad_nonzero"))
    assert bool(get("end_fix_input_grad_finite"))

    if bool(get("embeddings_untied")):
        assert bool(get("end_fix_output_grad_finite"))

    print("[OK] released DeepGaze visual_search_adapter loaded trainably")
    print("[OK] END_FIX atomic and trainable contract")
    print(f"[OK] query fixation/state count = {n}")
    print(f"[OK] P_E output shape = {tuple(get('projector_output_shape'))}; {n} aligned state insertions")
    print(f"[OK] L_WHERE = {loss_where:.6f}")
    print(f"[OK] L_FLAT = {loss_flat:.6f}")
    print(f"[OK] L_total = {loss_total:.6f}")
    print("[OK] LoRA gradient finite")
    print("[OK] P_E gradient finite and nonzero")
    print("[OK] END_FIX input gradient finite")
    if bool(get("embeddings_untied")):
        print("[OK] END_FIX output gradient finite")
    else:
        print("[OK] tied vocabulary weights; shared END_FIX row checked")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--contract-only", action="store_true", help="run pure fixture/golden checks")
    mode.add_argument("--with-model", action="store_true", help="run one differentiable model vertical slice")
    parser.add_argument("--step", action="store_true", help="allow optimizer step in --with-model mode")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--expected-where", type=Path, default=DEFAULT_EXPECTED_WHERE)
    parser.add_argument("--expected-flat", type=Path, default=DEFAULT_EXPECTED_FLAT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.contract_only and not args.with_model:
        args.contract_only = True

    try:
        if args.with_model:
            run_with_model(args)
        else:
            run_contract_only(args)
    except Exception as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
