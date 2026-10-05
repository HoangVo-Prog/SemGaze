# SemGaze

Training samples one K from the configured distribution per physical episode
batch. All episodes in that batch share K; subject, query, support selection and
support order retain their existing conditional sampling. WHERE context-overflow
retries keep that K. `per_device_train_batch_size` is physical B and
`gradient_accumulation_steps` is the number of physical batches per optimizer
update. Length sorting stays within a same-K batch.

See [the A100 benchmark guide](documents/A100_BENCHMARK.md) for the B=1 baseline,
B=2 same-K and optional B=4 same-K comparison, server command, padding statistics,
and JSON/CSV outputs. Target-hardware results must be measured on the server.
