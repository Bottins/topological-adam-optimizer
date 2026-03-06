"""
Test suite and examples for TAdam optimizer.

This script demonstrates:
1. Basic usage with different scaling strategies
2. Performance comparison with standard Adam
3. Visualization of topology-aware learning
"""

import random
import numpy as np
import torch

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Per avere determinismo completo
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
import warnings


from tadam import TAdam

# Try to import optional dependencies
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    warnings.warn("PyTorch Geometric not installed. Using simple GNN implementation.")

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    warnings.warn("NetworkX not installed. Some visualizations will be disabled.")


# Simple GNN implementation if PyG not available
class SimpleGNN(nn.Module):
    """Simple Graph Neural Network for testing."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Simple message passing forward pass."""
        for i, layer in enumerate(self.layers[:-1]):
            # Linear transformation
            x = layer(x)
            
            # Simple aggregation (mean of neighbors)
            row, col = edge_index
            out = torch.zeros_like(x)
            out.index_add_(0, row, x[col])
            degree = torch.bincount(row, minlength=x.size(0)).float().clamp(min=1)
            x = out / degree.unsqueeze(1)
            
            # Activation
            x = F.relu(x)
        
        # Output layer
        x = self.layers[-1](x)
        return x


def create_test_graphs() -> List[Dict]:
    """Create various test graphs with different topological properties."""
    graphs = []
    
    # 1. Complete graph (high connectivity)
    n = 50
    edge_list = []
    for i in range(n):
        for j in range(i+1, n):
            edge_list.append([i, j])
            edge_list.append([j, i])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    graphs.append({
        'name': 'Complete Graph',
        'edge_index': edge_index,
        'num_nodes': n,
        'features': torch.randn(n, 16),
        'labels': torch.randint(0, 4, (n,))
    })
    
    # 2. Star graph (hub structure)
    n = 50
    edge_list = []
    for i in range(1, n):
        edge_list.append([0, i])
        edge_list.append([i, 0])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    graphs.append({
        'name': 'Star Graph',
        'edge_index': edge_index,
        'num_nodes': n,
        'features': torch.randn(n, 16),
        'labels': torch.randint(0, 4, (n,))
    })
    
    # 3. Path graph (low connectivity)
    n = 50
    edge_list = []
    for i in range(n-1):
        edge_list.append([i, i+1])
        edge_list.append([i+1, i])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    graphs.append({
        'name': 'Path Graph',
        'edge_index': edge_index,
        'num_nodes': n,
        'features': torch.randn(n, 16),
        'labels': torch.randint(0, 4, (n,))
    })
    
    # 4. Barbell graph (two clusters connected by bridge)
    n1, n2 = 25,25
    edge_list = []
    # First cluster (complete)
    for i in range(n1):
        for j in range(i+1, n1):
            edge_list.append([i, j])
            edge_list.append([j, i])
    # Second cluster (complete)
    for i in range(n1, n1+n2):
        for j in range(i+1, n1+n2):
            edge_list.append([i, j])
            edge_list.append([j, i])
    # Bridge
    edge_list.append([n1-1, n1])
    edge_list.append([n1, n1-1])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    graphs.append({
        'name': 'Barbell Graph',
        'edge_index': edge_index,
        'num_nodes': n1 + n2,
        'features': torch.randn(n1 + n2, 16),
        'labels': torch.cat([torch.zeros(n1, dtype=torch.long), 
                           torch.ones(n2, dtype=torch.long)])
    })
    
    # 5. Random graph (Erdos-Renyi)
    n = 50
    p = 0.15  # Edge probability
    edge_list = []
    for i in range(n):
        for j in range(i+1, n):
            if torch.rand(1).item() < p:
                edge_list.append([i, j])
                edge_list.append([j, i])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t() if edge_list else torch.zeros((2, 0), dtype=torch.long)
    
    graphs.append({
        'name': 'Random Graph (p=0.15)',
        'edge_index': edge_index,
        'num_nodes': n,
        'features': torch.randn(n, 16),
        'labels': torch.randint(0, 4, (n,))
    })
    
    # 5. Random graph (Erdos-Renyi)
    n = 50
    p = 0.4  # Edge probability
    edge_list = []
    for i in range(n):
        for j in range(i+1, n):
            if torch.rand(1).item() < p:
                edge_list.append([i, j])
                edge_list.append([j, i])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t() if edge_list else torch.zeros((2, 0), dtype=torch.long)
    
    graphs.append({
        'name': 'Random Graph (p=0.4)',
        'edge_index': edge_index,
        'num_nodes': n,
        'features': torch.randn(n, 16),
        'labels': torch.randint(0, 4, (n,))
    })
    
    return graphs


def train_with_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    graph_data: Dict,
    num_epochs: int = 100,
    verbose: bool = True
) -> List[float]:
    """Train a model with given optimizer and return loss history."""
    
    edge_index = graph_data['edge_index']
    features = graph_data['features']
    labels = graph_data['labels']
    
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(features, edge_index)
        loss = F.cross_entropy(out, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if verbose and (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    return losses


def compare_optimizers(graph_data: Dict, num_trials: int = 3) -> Dict:
    """Compare different optimizer configurations on a graph."""
    
    input_dim = graph_data['features'].shape[1]
    hidden_dim = 32
    output_dim = graph_data['labels'].max().item() + 1
    num_epochs = 100
    lr = 0.01
    
    results = {}
    
    # Test configurations
    configs = [
        ('Adam', None),
        ('TAdam-Only_TRFscale', 'Only_TRFscale'),
        ('TAdam-Degree', 'degree'),
        ('TAdam-Homophily', 'homophily'),
        ('TAdam-Curvature', 'curvature'),
        ('TAdam-Combined', 'combined'),
    ]
    set_global_seed(2025)
    for config_name, local_scaling in configs:
        print(f"\nTesting {config_name}...")
        trial_losses = []
        trial_times = []
        
        for trial in range(num_trials):
                       
            model = SimpleGNN(input_dim, hidden_dim, output_dim)
            
            # Create optimizer
            if config_name == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                optimizer = TAdam(
                    model.parameters(),
                    lr=lr,
                    graph_data=(graph_data['edge_index'], graph_data['num_nodes']),
                    local_scaling=local_scaling,
                    topology_update_freq=10
                )
                # Set node features for homophily computation
                if local_scaling in ['homophily', 'combined']:
                    optimizer.node_features = graph_data['features']
            
            # Train
            start_time = time.time()
            losses = train_with_optimizer(
                model, optimizer, graph_data, 
                num_epochs=num_epochs, verbose=False
            )
            train_time = time.time() - start_time
            
            trial_losses.append(losses)
            trial_times.append(train_time)
        
        # Average over trials
        avg_losses = np.mean(trial_losses, axis=0)
        std_losses = np.std(trial_losses, axis=0)
        avg_time = np.mean(trial_times)
        
        results[config_name] = {
            'losses': avg_losses,
            'std': std_losses,
            'time': avg_time,
            'final_loss': avg_losses[-1],
            'convergence_rate': (avg_losses[0] - avg_losses[-1]) / num_epochs
        }
        
        print(f"  Final loss: {avg_losses[-1]:.4f} ± {std_losses[-1]:.4f}")
        print(f"  Training time: {avg_time:.2f}s")
        print(f"  Convergence rate: {results[config_name]['convergence_rate']:.4f}")
    
    return results


def visualize_results(all_results: Dict[str, Dict], save_path: Optional[str] = None):
    """Visualize comparison results across different graphs."""
    
    num_graphs = len(all_results)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    colors = {
        'Adam': 'black',
        'TAdam-Only_TRFscale': 'blue',
        'TAdam-Degree': 'green',
        'TAdam-Homophily': 'orange',
        'TAdam-Curvature': 'red',
        'TAdam-Combined': 'purple'
    }
    
    for idx, (graph_name, results) in enumerate(all_results.items()):
        if idx >= 6:
            break
            
        ax = axes[idx]
        
        for optimizer_name, metrics in results.items():
            losses = metrics['losses']
            std = metrics['std']
            epochs = np.arange(len(losses))
            
            ax.plot(epochs, losses, label=optimizer_name, 
                   color=colors[optimizer_name], linewidth=2)
            ax.fill_between(epochs, losses - std, losses + std,
                           alpha=0.2, color=colors[optimizer_name])
        
        ax.set_title(f'{graph_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
        # ---- Salvataggio singolo subplot ----
        plt.figure()
        for optimizer_name, metrics in results.items():
            losses = metrics['losses']
            std = metrics['std']
            epochs = np.arange(len(losses))
            plt.plot(epochs, losses,label=optimizer_name, color=colors[optimizer_name], linewidth=2)
            plt.fill_between(epochs, losses - std, losses + std,
                             alpha=0.2, color=colors[optimizer_name])
        plt.yscale('log')
        plt.title(graph_name)
        plt.legend(loc='upper right', fontsize=6)
        plt.savefig(f"tadam_fig{idx+1}.png", dpi=150, bbox_inches='tight')
        plt.close()
    # -------------------------------------
    
    plt.suptitle('TAdam vs Adam: Performance Comparison on Different Graph Topologies', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def test_dynamic_graph():
    """Test TAdam with dynamically changing graphs."""
    print("\n" + "="*50)
    print("Testing Dynamic Graph Updates")
    print("="*50)
    
    # Create initial graph
    initial_graph = create_test_graphs()[0]  # Complete graph
    
    # Create model
    model = SimpleGNN(
        input_dim=initial_graph['features'].shape[1],
        hidden_dim=32,
        output_dim=initial_graph['labels'].max().item() + 1
    )
    
    # Create TAdam optimizer
    optimizer = TAdam(
        model.parameters(),
        lr=0.01,
        graph_data=(initial_graph['edge_index'], initial_graph['num_nodes']),
        local_scaling='combined',
        topology_update_freq=5
    )
    
    print("\nTraining on initial graph (Complete)...")
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(initial_graph['features'], initial_graph['edge_index'])
        loss = F.cross_entropy(out, initial_graph['labels'])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Switch to different graph
    new_graph = create_test_graphs()[1]  # Star graph
    optimizer.update_graph((new_graph['edge_index'], new_graph['num_nodes']))
    
    print("\nSwitched to new graph (Star)...")
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(new_graph['features'], new_graph['edge_index'])
        loss = F.cross_entropy(out, new_graph['labels'])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    print("\nDynamic graph update test completed successfully!")


def analyze_trf_values():
    """Analyze TRF values for different graph topologies."""
    print("\n" + "="*50)
    print("TRF Analysis for Different Graph Topologies")
    print("="*50)
    
    graphs = create_test_graphs()
    
    from tadam import TopologyMetrics
    metrics = TopologyMetrics()
    
    trf_weights = {
        'wA': 1.0,
        'wNc': 1.0,
        'wEc': 1.0,
        'wE': 1.0,
        'wL': 1.0,
        'wmu': 1.0
    }
    
    results = []
    
    for graph in graphs:
        edge_index = graph['edge_index']
        num_nodes = graph['num_nodes']
        num_edges = edge_index.shape[1] // 2
        
        trf = metrics.compute_trf(edge_index, num_nodes, num_edges, trf_weights)
        
        results.append({
            'Graph': graph['name'],
            'Nodes': num_nodes,
            'Edges': num_edges,
            'TRF': trf,
            'LR Scale': 1.0 / trf
        })
        
        print(f"\n{graph['name']}:")
        print(f"  Nodes: {num_nodes}, Edges: {num_edges}")
        print(f"  TRF: {trf:.4f}")
        print(f"  Effective LR multiplier: {1.0/trf:.4f}")
    
    return results



"""Main test function."""
print("="*60)
print("TAdam Optimizer Test Suite")
print("="*60)

# 1. Test TRF computation
print("\n1. Analyzing TRF values...")
trf_results = analyze_trf_values()

# 2. Test basic functionality
print("\n2. Testing basic functionality...")
graphs = create_test_graphs()
test_graph = graphs[0]  # Use complete graph

model = SimpleGNN(
    input_dim=test_graph['features'].shape[1],
    hidden_dim=32,
    output_dim=test_graph['labels'].max().item() + 1
)

optimizer = TAdam(
    model.parameters(),
    lr=0.01,
    graph_data=(test_graph['edge_index'], test_graph['num_nodes']),
    local_scaling='degree'
)

print("  TAdam initialized successfully!")
print(f"  Number of parameter groups: {len(optimizer.param_groups)}")

# 3. Compare optimizers on different graphs
print("\n3. Comparing optimizers on different graph topologies...")
all_results = {}

for graph in graphs[:6]:  # Test on first 3 graphs to save time
    print(f"\n{'='*40}")
    print(f"Graph: {graph['name']}")
    print(f"{'='*40}")
    
    results = compare_optimizers(graph, num_trials=2)
    all_results[graph['name']] = results

# 4. Visualize results
print("\n4. Visualizing results...")
visualize_results(all_results, save_path='tadam_comparison.png')

# 5. Test dynamic graphs
test_dynamic_graph()

# 6. Summary
print("\n" + "="*60)
print("Test Summary")
print("="*60)

print("\nBest performing configurations by graph:")
for graph_name, results in all_results.items():
    best_config = min(results.items(), key=lambda x: x[1]['final_loss'])
    print(f"\n{graph_name}:")
    print(f"  Best: {best_config[0]}")
    print(f"  Final loss: {best_config[1]['final_loss']:.4f}")
    # print(f"  Speedup vs Adam: {results['Adam']['time']/best_config[1]['time']:.2f}x")

print("\n✅ All tests completed successfully!")


