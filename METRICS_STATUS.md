# Evaluation metrics status

Gate B1 and Gate B2 are resolved from the checked-in reference trees and
executable fixtures under `tests/fixtures/isp_coordinate_pairs.json`.

* **SM — PRODUCTION_READY.** The runtime uses the vendored ISP ScanMatch
  reference with 512x320, 16x12, zero offset, 50 ms duration bins, and 3.5
  threshold. It records both variants and applies the SciPy harmonic mean.
* **MM — IMPLEMENTED_BUT_GATED.** The runtime preserves all five audited
  `multimatch-gaze==0.1.3` components, converts durations to seconds, pads
  short paths with `(1.0, 1.0, 0.001)`, and has deterministic zero/count
  handling for non-finite output. The package is declared in `pyproject.toml`
  but is unavailable in this restricted environment, so an installed-package
  parity run remains the blocker.
* **SED — PRODUCTION_READY.** The runtime calls the vendored ISP VAME
  implementation on a 512x320 frame with a 5x5 grid and reports raw edit
  distance without sequence-length normalization.
* **LL — IMPLEMENTED_BUT_GATED.** Outcome B performs an exact SemGaze-context
  teacher-forced probability pass and reduces digit-normalized transitions to
  scalars. It is disabled by default and covered by configuration guards.
* **IG — BLOCKED.** Canonical per-image COCO-Search18 center-bias assets are
  still absent; the evaluation path rejects missing assets and never uses a
  synthetic prior.
* **BERTScore-F1 — IMPLEMENTED_BUT_GATED.** Frozen settings, missing-unit
  accounting, unique semantic IDs, and batch-size-independent reduction are
  implemented; a real model-runtime parity run remains required.
* **CIDEr-R — IMPLEMENTED_BUT_GATED.** The authors' scorer and PTB wrapper
  are vendored with provenance; the authors-runtime parity check remains.

The active canonical configurations enable SM and SED; MM remains disabled
until its installed-package parity run is available. IG remains disabled and unchanged. Metrics are produced from the existing prediction
records; frozen rescoring does not load the VLM or generate additional text.

Focused result: 93 repository tests passed, plus 4 new frozen-runtime/failure-policy
tests passed. Two COCO protocol tests could not start because the checked-out
dataset split is absent; 16 tests were skipped by optional model dependencies.
Full model-generation and
installed-MultiMatch regression tests were skipped or blocked by this
environment (`peft` and `multimatch-gaze==0.1.3` are unavailable). Generation
call regression, real frozen artifact replay, and batch-size invariance are
therefore not claimed as passed here.

* Tests: 93 passed, 2 failed to start because the COCO split is absent, and
  16 skipped (optional model/dependency coverage); the added frozen-runtime
  suite is 4/4 passed.
* Generation-call regression: not run (no `peft` runtime).
* Frozen rescoring: passed with the checked-in synthetic frozen records; real
  package replay is blocked by missing MultiMatch.
* Batch-size invariance: not run in the model runtime.
* Config: SM and SED are enabled in `configs/defaults.yaml` and
  `configs/flat_single.yaml`; MM is explicitly disabled pending parity. IG and
  unrelated semantic/probability metrics are unchanged.
* Remaining blocker: install `multimatch-gaze==0.1.3` and run the real model
  generation-call, reference-parity, and batch-invariance regressions.
The coordinate adapter is now explicitly identified as
`deepgaze_vl_predict_scanpath_round` in both YAML configurations.

Authoritative local evidence:

* ISP preprocessing: `isp-senet/ISP/COCO_Search18/GazeformerISP/src/preprocess/preprocess_fixations.py` subtracts one from subjects and writes the 512x320 TP file.
* ISP evaluation: `isp-senet/ISP/COCO_Search18/GazeformerISP/src/utils/evaluation.py` uses raw `X/Y`, ScanMatch `16x12`, `TempBin=50`, `Threshold=3.5`, and `[512,320]` MultiMatch screensize.
* Gazeformer-ISP’s COCO dataset path independently confirms direct `X/Y`, duration conversion from milliseconds to seconds, and subject `-1` at evaluation loading.
  Its neighboring preprocessing script is an OSIE-style MAT-file splitter,
  so it is hash-pinned as traced evidence but is not used as the COCO source.
* DeepGaze inverse: `DeepGaze-VL/predict_scanpath.py:238-244` uses `int(round(x/100.0*W))` and `int(round(y/100.0*H))`.

No training, loss, support draws, split membership, generation path, or
semantic conditioning was changed by this audit.
