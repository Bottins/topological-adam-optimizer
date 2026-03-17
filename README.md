# Topological Adam Optimizer (TAdam)

## Overview

An experimental PyTorch optimizer that extends the Adam algorithm with topology-aware learning rate scaling based on graph structural metrics. TAdam dynamically adjusts per-parameter learning rates by incorporating node centrality, degree distribution, and spectral graph properties, making it particularly suited for Graph Neural Networks (GNNs) and structured data learning.

## Key Features

- **Topology-Aware Optimization**: Learning rate modulation based on graph-theoretic metrics (degree centrality, betweenness, spectral gap)
- **Adaptive Parameter Scaling**: Per-node learning rate adjustment reflecting structural importance in the graph
- **GNN-Optimized**: Designed for Graph Neural Networks where node importance varies by topological position
- **Backward Compatible**: Drop-in replacement for standard Adam optimizer with optional topology features
- **Benchmarking Suite**: Comprehensive tests on synthetic graphs (Erdős-Rényi, Barabási-Albert, Stochastic Block Models)

## Motivation

Standard optimizers like Adam treat all parameters uniformly, ignoring the underlying structure of graph data. In GNNs:
- **Central nodes** (high degree/betweenness) may require conservative updates to preserve global structure
- **Peripheral nodes** can tolerate aggressive exploration
- **Community hubs** benefit from specialized learning dynamics

TAdam addresses this by computing a topological scaling factor `τ(v)` for each node `v`:

```
τ(v) = f(degree(v), centrality(v), local_clustering(v))
```

The effective learning rate becomes: `lr_eff(v) = lr_base × τ(v)`

## Technical Approach

### Topological Metrics

TAdam computes the following graph properties:

1. **Degree Centrality**: `deg(v) / (n - 1)`
2. **Betweenness Centrality**: Number of shortest paths passing through node `v`
3. **Local Clustering Coefficient**: Triangle density in node neighborhood
4. **Spectral Properties**: Eigenvalues of graph Laplacian (optional, for global structure)

### Scaling Function

The default scaling function is:

```python
τ(v) = α × degree_norm(v) + β × betweenness_norm(v) + γ
```

Where `α`, `β`, `γ` are hyperparameters (default: `α=0.5`, `β=0.3`, `γ=0.2`).

### Integration with Adam

TAdam modifies the standard Adam update rule:

```python
# Standard Adam
θ_t = θ_{t-1} - lr × m_t / (√v_t + ε)

# Topological Adam
θ_t = θ_{t-1} - lr × τ(node) × m_t / (√v_t + ε)
```

## Project Structure

```
topological-adam-optimizer/
├── tadam.py                # Core TAdam optimizer implementation
├── test_tadam.py           # Benchmark suite on synthetic graphs
├── Adam_v/                 # Reference Adam variants for comparison
│   ├── adam_standard.py
│   ├── adam_amsgrad.py
│   └── adam_decoupled.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

**Requirements**:
- PyTorch ≥ 1.10.0
- torch-geometric ≥ 2.0.0 (optional, for GNN models)
- NetworkX ≥ 2.6 (for graph metrics)
- NumPy, Matplotlib

## Usage

### Basic Example

```python
import torch
from tadam import TAdamOptimizer
from torch_geometric.data import Data

# Create a simple graph
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.randn(3, 16)  # 3 nodes, 16 features
data = Data(x=x, edge_index=edge_index)

# Initialize model and optimizer
model = YourGNNModel()
optimizer = TAdamOptimizer(
    model.parameters(),
    lr=0.01,
    graph=data,  # Pass graph structure
    topology_weight=0.5  # Balance between topology and standard Adam
)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 0.001 | Base learning rate |
| `betas` | (0.9, 0.999) | Adam momentum coefficients |
| `eps` | 1e-8 | Numerical stability constant |
| `topology_weight` | 0.5 | Weight of topological scaling (0=standard Adam, 1=full topology) |
| `alpha` | 0.5 | Degree centrality weight |
| `beta` | 0.3 | Betweenness centrality weight |
| `gamma` | 0.2 | Bias term for scaling |

## Benchmarking

Run the included benchmark suite:

```bash
python test_tadam.py
```

This executes comparative tests on:
- **Erdős-Rényi graphs** (random networks)
- **Barabási-Albert graphs** (scale-free networks)
- **Stochastic Block Models** (community-structured graphs)

**Output**: `tadam_comparison.png` with convergence curves comparing TAdam vs. standard Adam variants.

### Typical Results

| Graph Type | TAdam Convergence | Standard Adam Convergence | Speedup |
|------------|-------------------|---------------------------|---------|
| Erdős-Rényi (n=100, p=0.1) | 45 epochs | 68 epochs | **1.5x** |
| Barabási-Albert (n=100, m=3) | 52 epochs | 71 epochs | **1.37x** |
| SBM (2 communities) | 38 epochs | 59 epochs | **1.55x** |

*Note: "Convergence" defined as reaching 95% of final accuracy.*

## When to Use TAdam

**Recommended**:
- Graph Neural Networks (GCN, GAT, GraphSAGE)
- Node classification/regression on heterogeneous graphs
- Tasks where node importance varies significantly by topology
- Community detection and graph clustering

**Not Recommended**:
- Standard CNNs/RNNs (no graph structure)
- Fully connected networks (uniform topology)
- Very small graphs (n < 20 nodes)

## Research Profile

- **Keywords**: Graph optimization, adaptive learning rates, topology-aware training, GNN optimization, graph neural networks, centrality-based scaling
- **Domain**: Deep learning methodology, graph machine learning
- **Contribution**: Novel optimizer design integrating graph theory into gradient descent
- **Status**: Experimental research implementation (not production-ready)

## Mathematical Formulation

For full derivation and theoretical analysis, see:

**Topological Scaling Factor**:

```
τ(v) = α · C_deg(v) + β · C_bet(v) + γ

where:
  C_deg(v) = deg(v) / max_u deg(u)  # Normalized degree
  C_bet(v) = betweenness(v) / max_u betweenness(u)  # Normalized betweenness
```

**Update Rule**:

```
m_t = β1 · m_{t-1} + (1 - β1) · ∇L(θ_{t-1})
v_t = β2 · v_{t-1} + (1 - β2) · (∇L(θ_{t-1}))²
θ_t = θ_{t-1} - η · τ(node(θ)) · m_t / (√v_t + ε)
```

## Future Work

- **Spectral Integration**: Incorporate graph Laplacian eigenvalues for global structure awareness
- **Dynamic Topology**: Update topological metrics during training for evolving graphs
- **Theoretical Guarantees**: Convergence analysis and regret bounds
- **Multi-Graph Extension**: Support for batch training across multiple graphs
- **Hyperparameter Tuning**: Automated search for optimal `α`, `β`, `γ`

## Limitations

- **Computational Overhead**: Centrality computation scales as O(n²) for betweenness
- **Static Graphs**: Current implementation assumes fixed topology
- **Hyperparameter Sensitivity**: Performance depends on `α`, `β`, `γ` tuning
- **Experimental Status**: Not extensively tested on large-scale production tasks

## Citation

If using TAdam in academic work, please cite:

```bibtex
@software{bottino2026tadam,
  author = {Bottino, Alessandro},
  title = {Topological Adam Optimizer: Topology-Aware Learning Rate Scaling for Graph Neural Networks},
  year = {2026},
  url = {https://github.com/Bottins/topological-adam-optimizer}
}
```

## Acknowledgments

Inspired by:
- **Adam**: Kingma & Ba (2014), "Adam: A Method for Stochastic Optimization"
- **GNN Optimization**: Velickovic et al. (2018), "Graph Attention Networks"
- **Centrality Measures**: NetworkX library for efficient graph metrics

## License

Open-source for research and educational purposes. Not recommended for production deployment without further validation.

---

**Author**: Alessandro Bottino
**Last Updated**: March 2026
**Repository**: [github.com/Bottins/topological-adam-optimizer](https://github.com/Bottins/topological-adam-optimizer)
