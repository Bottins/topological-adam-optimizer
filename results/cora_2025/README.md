# Verified Cora run — 10 October 2026

All seven configurations completed 200 epochs with seed 2025. The updated
repository implementation reproduces every accuracy and loss value from the
run used for the revised manuscript, with exact equality of the stored numbers
in the same environment. The optimizer regression suite also passed (7 tests).

The recorded command was:

```bash
python benchmark_cora.py --seeds 2025 --data ../data --output results/cora_2025
```

`../data` was an existing Planetoid Cora cache. For a fresh checkout, use
`python benchmark_cora.py --seeds 2025`; it downloads the same public dataset
into `data/` and writes to `results/cora_run/` instead of replacing these files.

- `results.csv`: final test metrics, one row per configuration.
- `protocol.json`: model/training settings, TRF, PyTorch/PyG versions, source hashes.
- `environment.json`: Python and installed versions of the main dependencies.

The experiment uses a reconstructed protocol, not a certified reproduction of
the unavailable earlier Cora experiment. Results are preliminary and provide
no multi-seed uncertainty estimate. The homophily adapter improves accuracy by
0.20 percentage points over Adam in this run; the remaining variants do not.

The graph is disconnected, so the original repository TRF code skips algebraic
connectivity. Its computed TRF is 16.408224656052. The repository multiplier and
shape-based local weighting differ from the manuscript equations, as explained
in the root README. The OnlyTRF/Combined output-equality assertion passed.
