# Evaluation metrics status

Gate B1 and Gate B2 are resolved from the checked-in reference trees and
executable fixtures under `tests/fixtures/isp_coordinate_pairs.json`.

* **SM/MM/SED — IMPLEMENTED_BUT_GATED.** The coordinate protocol is proven:
  raw COCO-Search18 coordinates use `x*512/1680` and `y*320/1050`, subjects
  map from SemGaze `1..10` to ISP `0..9`, and prediction bins use the
  DeepGaze-VL representative `int(round(bin/100*frame_size))`. The local
  TP reference has 13,167 full-length matches and 833 shorter valid prefixes
  across 14,000 present-condition key matches; this is a source-population
  difference, not a coordinate mismatch. Runtime MultiMatch remains an
  optional audited dependency and full evaluation orchestration is still
  separately gated.
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

The canonical metrics block remains opt-in (`evaluation.metrics.enabled:
false`) because IG and independent runtime/preflight gates are unresolved.
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
