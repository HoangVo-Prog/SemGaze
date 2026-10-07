# Evaluation metrics status

The metric primitives are implemented in `semgaze/evaluation/metrics_scanpath.py`,
`metrics_probability.py`, `metrics_semantic.py`, and `records.py`.  They are
additive utilities over frozen prediction records and never call the VLM.

Current per-metric state:

* **SM/MM/SED — BLOCKED.**  The wrappers remain fail-closed until the raw-GT
  parity and predicted-bin representative gates are resolved.
* **LL — IMPLEMENTED_BUT_GATED.**  Outcome B performs an exact SemGaze-context
  teacher-forced probability pass and reduces digit-normalized transitions to
  scalars.  It is disabled by default and is covered by configuration guards.
* **IG — BLOCKED.**  The evaluation path rejects missing canonical per-image
  center-bias assets and never substitutes a synthetic prior.
* **BERTScore-F1 — IMPLEMENTED_BUT_GATED.**  Frozen settings, missing-unit
  accounting, unique semantic IDs, and batch-size independent reduction are
  implemented; a real model-runtime parity run is still required.
* **CIDEr-R — IMPLEMENTED_BUT_GATED.**  The authors' scorer and PTB wrapper are
  vendored with provenance.  Java/CoreNLP availability and an authors-runtime
  parity fixture remain required.

The canonical metric block is present in both YAML configurations but remains
disabled (`evaluation.metrics.enabled: false`) because the required preflight
stop conditions are not all resolved on this checkout:

* The official ISP-SENet ScanMatch and VAME sources are vendored with hashes in
  `third_party/isp_reference/README_PROVENANCE.md`.  Their source preprocessing
  uses a 512x320 reference file, but the persisted SemGaze records do not have
  full-sequence parity with those records.  The raw-GT adapter therefore cannot
  be enabled as a proven parity contract.
* The historical DeepGaze prediction inverse is recorded as
  `round(bin / 100 * frame_size)`, but it does not by itself establish raw-GT
  parity.  `CoordinateAdapter` therefore requires that mapping to be supplied
  explicitly and `score_scanpath_pair` remains fail-closed.
* The repository contains MIT center-bias priors, not canonical per-image
  COCO-Search18 priors.  IG consequently fails closed instead of using a
  synthetic Gaussian.
* The authors' CIDEr-R scorer and PTB wrapper are vendored in
  `third_party/cider_r/README_PROVENANCE.md`; the Stanford CoreNLP runtime and
  parity fixture are still required before enabling semantic CIDEr-R.

No training, loss, support sampling, split membership, or generation path is
changed by this gated implementation.
