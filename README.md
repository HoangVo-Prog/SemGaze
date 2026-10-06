# SemGaze

COCO-Search18 uses the persisted 95/5 train/test split under
`data/COCO_Search18/split_95_5/`. Each training epoch visits all 18,508 eligible
seen-subject train records exactly once in shuffled order. Every query draws K
uniformly from 1..10 and distinct same-subject train supports. Overflow retries
retain the query and K. Same-K physical batching preserves these episode draws,
including the final partial optimizer window.

After each complete epoch, test evaluation uses unseen subjects 7/8/9, K=1/5/10,
and ten frozen support draws per K. There is no validation split.

See [training and evaluation usage](README_FLAT.md) and
[migration details](documents/COCO_955_MIGRATION.md). GPU throughput must be measured
again under this sampling protocol; earlier benchmark reports are historical.
