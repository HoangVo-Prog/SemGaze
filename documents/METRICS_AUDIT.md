# Scanpath metrics continuation audit — 2026-10-07

The earlier report treated the coordinate gate as blocked because it did not
find an expected path. This audit traced the actual local COCO reference
trees and built an executable fixture from the persisted records.

## Gate B1 — raw GT to ISP frame: PASS

`isp-senet/ISP/COCO_Search18/GazeformerISP/src/preprocess/preprocess_fixations.py`
loads the 512x320 COCO fixation source, subtracts one from each subject ID,
separates `present` and `absent`, sorts records, and writes `TP_fixations.json`
and `TA_fixations.json` (under its configured TPTA output directory). The
checked-in active test path consumes `src/data/TP_fixations.json`, which has
the same preprocessed 512x320 schema. The active test/evaluation path passes width `512`,
height `320`, and consumes the persisted `X/Y` values directly. The sibling
`gazeformer-isp/COCO_Search18/GazeformerISP/src/dataset/dataset.py` confirms
the same direct evaluation path and performs the subject subtraction while
loading its COCO records.

The preprocessing script's relative PHAT input is not checked in under this
checkout, but its generated `isp-senet/.../src/data/TP_fixations.json` output
is present and hash-pinned. The executable audit uses that persisted output
and the present-condition SemGaze records, so the absent upstream input path
does not make the coordinate gate indeterminate.

The neighboring Gazeformer-ISP `preprocess_fixations.py` is an OSIE-style
MAT-file splitter and does not define the COCO-Search18 coordinate protocol.
Its active COCO dataset and test paths are therefore the relevant
Gazeformer-ISP evidence here; the preprocessing file is still hash-pinned in
the fixture so that this distinction remains auditable.

SemGaze’s persisted COCO records use the original `1680x1050` annotation
frame. Matching `(subject, task, image, condition)` after the subject remap
produces 14,000 present-condition key matches. The direct transform is:

```text
x_isp = x_raw * 512 / 1680
y_isp = y_raw * 320 / 1050
subject_isp = subject_semgaze - 1
```

The exhaustive local audit records 13,167 records with the same source and
reference sequence length and 833 shorter valid prefixes. Twenty-seven source
records contain 41 out-of-frame raw points; after the source validity rule
`0 <= x < 1680` and `0 <= y < 1050`, all 14,000 matched reference prefixes
have zero coordinate mismatches. This is a sequence-population caveat, not
an unresolved transform. Durations remain source milliseconds; the local
reference retains them while its evaluator converts them to seconds only for
MultiMatch and back to milliseconds for ScanMatch.

The fixture stores representative full and prefix records, boundary probes,
source hashes, and the exhaustive counts:
`tests/fixtures/isp_coordinate_pairs.json` and
`tests/test_isp_coordinate_parity.py`.

## Gate B2 — prediction bins to ISP frame: PASS

`DeepGaze-VL/predict_scanpath.py:238-244` is authoritative local prediction
code:

```python
int(round(x / 100.0 * W)), int(round(y / 100.0 * H))
```

The parser in `evaluate_vllm_unified.py` clips parsed coordinates to integer
bins `0..99`. Therefore the verified SemGaze inverse is
`int(round(bin_x*512/100))`, `int(round(bin_y*320/100))`; for example
`(0,0)->(0,0)`, `(50,50)->(256,160)`, and `(99,99)->(507,317)`. The adapter
and boundary cases are tested in `tests/test_scanpath_parity.py`.

## Frozen provenance

| Source | SHA-256 |
| --- | --- |
| `isp-senet/.../preprocess/preprocess_fixations.py` | `81aa0754346a382f1f818270645c934af275b6f5e96f6f44b97049108e44512f` |
| `isp-senet/.../utils/evaluation.py` | `d68c1c0554c8195785084f34e4b7f74b06ca085dc3deff7c729517eee79cdb2a` |
| `isp-senet/.../dataset/dataset.py` | `c3430adfa56768189036c56edbb3e4f6b1aec0da2060241d6e0dab862f2f932f` |
| `isp-senet/.../test.py` | `f5de2882ba32388c2f0478a731cdbc990aa5a75e3204f2516859e302bad29ec0` |
| `gazeformer-isp/.../preprocess/preprocess_fixations.py` | `0d4c326bf0fcf56e9ccf4f1ac4f88078a1da711fddc341da67ff8b03ef9984a6` |
| `gazeformer-isp/.../dataset/dataset.py` | `f871983b028ba596c2421584127a396cdf5a9a4526947befec0603b65b35d5cb` |
| `gazeformer-isp/.../utils/evaluation.py` | `6c7713ca02e0f2a0383ec7f837437ef3fad573b49411e62013d2abfd6cdbabf1` |
| `gazeformer-isp/.../test.py` | `7dcb9a9b38b65a576c5fa3a06e297ca8dfa091aec6687fdc5c45d99e482144be` |
| `DeepGaze-VL/predict_scanpath.py` | `4f059a30e829ff7414924d1841f20909c7b776d5227767b5abeda863e872108f` |
| `DeepGaze-VL/evaluate_vllm_unified.py` | `159af99b5919ebb1088cc9c68ea58f2aae6e29cc0c56f7f9bbbb785f9290f19b` |
| SemGaze COCO records | `bb80d858c390ee8ce03039fcc97da2d5430a5c8809c29551e99a0f4624f997d6` |
| ISP TP records | `c9471fb7fca6ceccdbe5a930f5d81d618ae1148ae27386b2718b1e9a6f8732a7` |

The vendored official ScanMatch/VAME sources and hashes remain documented in
`third_party/isp_reference/README_PROVENANCE.md`. The scanpath adapter uses
the verified inverse label `deepgaze_vl_predict_scanpath_round` and retains a
read-compatible alias for the earlier gated label.
