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
