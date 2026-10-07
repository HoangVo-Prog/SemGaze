# ISP-SENet reference provenance

These two files are copied from the official CVPR 2025 repository
`cvlab-stonybrook/few-shot-scanpath` (remote HEAD `1bdc5c4887754c14dc252d08ef34d988251639b0`),
under `ISP/COCO_Search18/GazeformerISP/src/utils/evaltools/`:

| file | SHA256 |
| --- | --- |
| `COCO_Search18/scanmatch.py` | `32be3d37b3435dc511cde5d702e5432ef7605a1449ed32f340dddfbb218451b0` |
| `COCO_Search18/visual_attention_metrics.py` | `ce93cbf9955b1cc496720fbc8f5f0c0a13070d8e21654576fdd1a89e05fcdcd0` |

The checked-in `isp-senet` copies are LF-normalized byte-identical to the
official checkout.  SemGaze uses the COCO-Search18 `ScanMatch` configuration
and the VAME `string_edit_distance` implementation only through the explicit
wrappers in `semgaze/evaluation/metrics_scanpath.py`.

The local COCO reference audit traced the active preprocessing and evaluation
files rather than relying on an expected checkout layout:

| local path | SHA256 |
| --- | --- |
| `isp-senet/ISP/COCO_Search18/GazeformerISP/src/preprocess/preprocess_fixations.py` | `81aa0754346a382f1f818270645c934af275b6f5e96f6f44b97049108e44512f` |
| `isp-senet/ISP/COCO_Search18/GazeformerISP/src/utils/evaluation.py` | `d68c1c0554c8195785084f34e4b7f74b06ca085dc3deff7c729517eee79cdb2a` |
| `isp-senet/ISP/COCO_Search18/GazeformerISP/src/dataset/dataset.py` | `c3430adfa56768189036c56edbb3e4f6b1aec0da2060241d6e0dab862f2f932f` |
| `isp-senet/ISP/COCO_Search18/GazeformerISP/src/test.py` | `f5de2882ba32388c2f0478a731cdbc990aa5a75e3204f2516859e302bad29ec0` |
| `gazeformer-isp/COCO_Search18/GazeformerISP/src/preprocess/preprocess_fixations.py` | `0d4c326bf0fcf56e9ccf4f1ac4f88078a1da711fddc341da67ff8b03ef9984a6` |
| `gazeformer-isp/COCO_Search18/GazeformerISP/src/dataset/dataset.py` | `f871983b028ba596c2421584127a396cdf5a9a4526947befec0603b65b35d5cb` |
| `gazeformer-isp/COCO_Search18/GazeformerISP/src/utils/evaluation.py` | `6c7713ca02e0f2a0383ec7f837437ef3fad573b49411e62013d2abfd6cdbabf1` |
| `gazeformer-isp/COCO_Search18/GazeformerISP/src/test.py` | `7dcb9a9b38b65a576c5fa3a06e297ca8dfa091aec6687fdc5c45d99e482144be` |
| `DeepGaze-VL/predict_scanpath.py` | `4f059a30e829ff7414924d1841f20909c7b776d5227767b5abeda863e872108f` |
| `DeepGaze-VL/evaluate_vllm_unified.py` | `159af99b5919ebb1088cc9c68ea58f2aae6e29cc0c56f7f9bbbb785f9290f19b` |

The ISP path subtracts one from COCO subject IDs, writes target-present
records in the 512x320 frame, and the active evaluator consumes `X/Y`
directly. Its relative PHAT input is not present in this checkout, while the
generated `src/data/TP_fixations.json` output is present and hash-pinned; the
executable fixture audits the persisted output directly. The neighboring Gazeformer-ISP preprocessing
file is an OSIE-style MAT-file splitter; its COCO dataset/evaluation paths are
the relevant evidence for that tree. The fixture records matched source data
and the measured sequence-population caveat.
