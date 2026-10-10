# TAdam: Topologically-informed Adam

Experimental PyTorch implementation accompanying **TAdam: Topologically-informed Adaptive Optimization for Graph Neural Networks**, by Alessandro Bottino, Salvatore Cuomo, and Vincenzo Vocca.

The synthetic study motivates topology-informed optimization as a promising research direction. The limited Cora experiment gives mixed results. Systematic validation across real-world datasets, seeds, and architectures remains future work; these experiments do not demonstrate mitigation of over-squashing or over-smoothing.

Repository: [Bottins/topological-adam-optimizer](https://github.com/Bottins/topological-adam-optimizer). A link to the published proceeding will be added when its DOI becomes available.

## What this code implements

The public class is `TAdam`, imported from `tadam.py`. It extends PyTorch Adam with:

- A global graph-dependent multiplier, computed from connectivity, efficiency, and path-length statistics.
- Optional local weights based on degree, embedding homophily, degree-based Forman-like curvature, or their combination.
- Periodic topology refresh and an explicit `update_graph(...)` method.

This repository retains the experimental update used in the Cora run. It is **not a complete implementation of the manuscript equations**:

| Component | Repository behavior |
|---|---|
| Global learning rate | `lr * (0.01 + 0.99 ** (-(TRF - 1)))`; differs from the manuscript's `lr / TRF`. Even TRF=1 gives 1.01 after refresh. |
| TRF | Uses the code's capped/normalized terms; skips algebraic connectivity on disconnected graphs. |
| Degree weights | Inverse `degree + 1`, min-max scaling, then mean normalization. |
| Homophily weights | Neighbor cosine similarities, z-score, sigmoid, then mean normalization. |
| Curvature weights | Mean incident `4 - degree(u) - degree(v)`, z-score, inverted sigmoid, then mean normalization. |
| Combined weights | Product by default; `sum` and `weighted_sum` are also available. |
| Local gradient scaling | Applied only when a gradient's first dimension equals the number of nodes. This shape check is not a semantic node-to-parameter mapping. |

Shared GCN weights normally do not have a node dimension. Consequently, unadapted `OnlyTRF` and `Combined` produce identical outputs in the recorded Cora experiment. The explicit loss adapters in `benchmark_cora.py` apply local weights to training-node losses instead. Homophily is computed each epoch from detached hidden embeddings before dropout. These adapters also differ from the manuscript formulation.

The default `local_scaling='none'` means **global scaling only when graph data is supplied**. Use `graph_data=None` for ordinary Adam behavior. With the default update frequency, the first topology refresh is at step 10; on Cora the adaptive interval then becomes 30 steps. The static TRF is cached in the benchmark without changing that schedule.

## Installation

Use Python 3.10 or newer. The recorded Cora run was executed with Python 3.12, PyTorch 2.14.0+cu130 and PyG 2.8.0.post1, on CPU. Exact installed package versions are recorded alongside the results; other versions have not been verified for identical metrics.

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r requirements-cora.txt
```

For optimizer use and the legacy synthetic demo without Cora, `requirements.txt` is sufficient. SciPy and NetworkX support graph metrics. PyG is required for Cora. The first Cora run downloads the public Planetoid data and needs network access.

## Minimal optimizer example

```python
import torch
from tadam import TAdam

# Undirected graph: include both directions for each edge.
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
node_embeddings = torch.nn.Parameter(torch.randn(3, 8))
optimizer = TAdam(
    [node_embeddings], lr=0.01,
    graph_data=(edge_index, 3),
    local_scaling='degree', topology_update_freq=1,
)
optimizer.zero_grad()
loss = node_embeddings.square().mean()
loss.backward()
optimizer.step()
```

This example deliberately uses node-indexed parameters. For shared GCN weights, use the explicit benchmark loss adapters rather than assuming that optimizer weights automatically map to nodes.

Supported local modes are `none`, `Only_TRFscale`, `degree`, `homophily`, `curvature`, and `combined`. For homophily, provide `node_features` in the graph dictionary or assign current detached embeddings to `optimizer.node_features`. Node features are not updated automatically by the optimizer.

## Cora: 200 epochs

From the repository root:

```bash
python benchmark_cora.py --seeds 2025
```

The script evaluates all seven methods using the public split (140 training, 500 validation, 1,000 test nodes), normalized input features, a two-layer GCN with 16 hidden units, ReLU and dropout 0.5, learning rate 0.01, and weight decay 0.0005 on all parameters. Each method starts from the same seed and initialization. Test metrics are measured in evaluation mode after the 200th update, without early stopping or checkpoint selection. Weighted losses are used only for training; reported test loss is unweighted cross-entropy.

The protocol was reconstructed because the earlier Cora script and its original seed were unavailable. These are the new measured results used in the revised manuscript:

| Method | Test accuracy (%) | Test loss |
|---|---:|---:|
| Adam | 80.20 | 0.79314 |
| TAdam-OnlyTRF | 80.10 | 0.76971 |
| TAdam-Combined | 80.10 | 0.76971 |
| Adapted-repository-degree | 79.40 | 0.79308 |
| Adapted-repository-homophily | 80.40 | 0.76504 |
| Adapted-repository-curvature | 79.80 | 0.76620 |
| Adapted-repository-combined | 79.40 | 0.79147 |

The homophily adapter exceeds Adam by 0.20 percentage points in this single run. Other variants have lower accuracy; a single seed does not establish statistical significance or general superiority.

See [recorded results](results/cora_2025/results.csv), [protocol and source hashes](results/cora_2025/protocol.json), and [verification notes](results/cora_2025/README.md). The script checks exact equality of the unadapted OnlyTRF and Combined outputs.

New runs default to `results/cora_run/`. Existing results are not overwritten; choose another directory for another run:

```bash
python benchmark_cora.py --seeds 2025 2026 2027 --output results/cora_multiseed
```

This command is an example of future evaluation, not a claim that these additional runs have been performed. Use `--data PATH` to reuse an existing Planetoid cache. The benchmark writes one CSV row per method and seed, plus its configuration, versions, and source hashes.

## Synthetic experiments and curvature

The revised chapter discusses complete, star, path, barbell, and two Erdos-Renyi graph families. These controlled, small-graph experiments are proof-of-concept evidence, not validation of deployment performance.

`test_tadam.py` is a **legacy synthetic demonstration**, not an assertion-based test suite or an exact reproduction of the manuscript figures. It uses its own mean-aggregation `SimpleGNN`, learning rate 0.01, 100 epochs and two trials in the main routine. The manuscript reports a different nominal learning rate; the original synthetic experiment provenance still needs reconciliation. Paired trial seeding has now been fixed, so rerunning this demo need not recreate old figures. Importing the module no longer launches experiments.

```bash
python test_tadam.py
```

The demo displays plots and saves them together in `figs/`, including the combined comparison and the individual graph plots. Its bands are standard deviations, not confidence intervals. The images currently included in `figs/` are historical synthetic figures, not evidence for the new Cora run; running the demo will replace them.

For the barbell with two 25-node cliques, bridge endpoints have degree 25 and regular clique nodes degree 24. Average incident curvatures are -45.04 and -44 - 1/24 respectively. These fixed, degree-derived values are distinct but do not establish general bottleneck detection. The manuscript's OnlyTRF result supports a contribution from global scaling without establishing dominance over local mechanisms.

## Verification

```bash
python -m unittest discover -s tests -v
```

Regression checks cover Adam equivalence without graph data, the default global-only mode, scalar gradients, edgeless homophily, graph refresh, invalid options, and safe import/basic execution of the legacy demo. The full Cora run separately verifies all seven 200-epoch configurations.

## Scope and future work

Priorities are matched implementation ablations, multiple seeds, validation-selected checkpoints, more datasets and architectures, and runtime/memory measurements. Topological metric computation can be expensive on larger graphs. The current code retains experimental normalization and scaling choices; exact equivalence to Adam on complete graphs is not claimed for this implementation. Direct diagnostics are needed before claiming effects on over-squashing or over-smoothing.

`Adam_v/` contains historical optimizer reference files and is not used by the Cora benchmark. All included images are collected in `figs/`.

No license file was supplied in this snapshot; this README does not introduce or change licensing terms.

Maintainer: [Bottins](https://github.com/Bottins) (Alessandro Bottino). Last revised: October 2026.
