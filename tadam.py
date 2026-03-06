# mypy: allow-untyped-defs
"""
TAdam: Topologically-informed Adam Optimizer for Graph Neural Networks

This optimizer extends Adam with topology-aware learning rate scaling:
- Global scaling: Adjusts lr based on Topological Relevance Function (TRF)
- Local scaling: Node-wise gradient weighting based on topological properties
"""

import warnings
from typing import Optional, Union, Dict, Tuple, Any, Callable
import math

import torch
from torch import Tensor
from torch.optim.adam import Adam, adam
from torch.optim.optimizer import ParamsT

try:
    # import torch_geometric
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    
try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False


__all__ = ["TAdam", "tadam"]


class TopologyMetrics:
    """Compute and cache topological metrics for graphs."""
    
    def __init__(self, update_frequency: int = 10, adaptive_frequency: bool = True):
        self.update_frequency = update_frequency
        self.adaptive_frequency = adaptive_frequency
        self.base_frequency = update_frequency
        self._cache = {}
        self._step_count = 0
        
    def should_update(self, num_nodes: int) -> bool:
        """Determine if metrics should be updated based on graph size and step count."""
        if self._step_count % self.update_frequency == 0:
            # Adaptive frequency: increase update interval for larger graphs
            if self.adaptive_frequency and num_nodes > 1000:
                self.update_frequency = min(
                    self.base_frequency * (1 + num_nodes // 1000),
                    100  # Cap at 100 steps
                )
            return True
        return False
    
    def increment_step(self):
        """Increment the internal step counter."""
        self._step_count += 1
        
    def compute_trf(self, 
                   edge_index: Tensor,
                   num_nodes: int,
                   num_edges: int,
                   weights: Dict[str, float]) -> float:
        """
        Compute Topological Relevance Function (TRF).
        
        TRF(G) = 1 + μG × [wA×(n-A)/A + wNc×(n-1-Nc)/Nc + wEc×(n-1-Ec)/Ec + wE×(1-E)/E + wL×(L-1)]
        where μG = wμ × m/n
        """
        # Default weights
        w = {
            'wA': weights.get('wA', 1.0),
            'wNc': weights.get('wNc', 1.0),
            'wEc': weights.get('wEc', 1.0),
            'wE': weights.get('wE', 1.0),
            'wL': weights.get('wL', 1.0),
            'wmu': weights.get('wmu', 1.0)
        }
        
        # Compute μG
        mu_G = w['wmu'] * (num_edges / num_nodes)
        
        # Initialize TRF components
        trf_value = 1.0
        components = []
        
        # Convert to NetworkX for complex metrics
        if HAS_NX:
            G = self._to_networkx(edge_index, num_nodes)
            
            # Algebraic connectivity (Fiedler value)
            try:
                if nx.is_connected(G):
                    A = nx.algebraic_connectivity(G)
                    if A<0.1:
                        A=0.1
                    A_normalized = min((num_nodes - A) / A, 10) / 10
                    if A > 0:
                        components.append(w['wA'] * A_normalized)
                else:
                    warnings.warn("Graph is disconnected, skipping algebraic connectivity", RuntimeWarning)
            except Exception as e:
                warnings.warn(f"Could not compute algebraic connectivity: {e}")
            
            # Node connectivity
            try:
                Nc = nx.node_connectivity(G)
                if Nc<0.1:
                    Nc=0.1
                Nc_normalized = min((num_nodes - Nc) / Nc, 10) / 10
                if Nc > 0:
                    components.append(w['wNc'] * Nc_normalized)
            except Exception as e:
                warnings.warn(f"Could not compute node connectivity: {e}", RuntimeWarning)
            
            # Edge connectivity
            try:
                Ec = nx.edge_connectivity(G)
                if Ec<0.1:
                    Ec=0.1
                Ec_normalized = min((num_nodes - Ec) / Ec, 10) / 10
                if Ec > 0:
                    components.append(w['wEc'] * Ec_normalized)
            except Exception as e:
                warnings.warn(f"Could not compute edge connectivity: {e}", RuntimeWarning)
            
            # Global efficiency
            try:
                E = nx.global_efficiency(G)
                if E<0.1:
                    E=0.1
                E_normalized = min((num_nodes - E) / E, 10) / 10
                if E > 0 and E < 1:
                    components.append(w['wE'] * E_normalized)
            except Exception as e:
                warnings.warn(f"Could not compute global efficiency: {e}", RuntimeWarning)
            
            # Average path length
            try:
                if nx.is_connected(G):
                    L = nx.average_shortest_path_length(G)
                    if L > 1:
                        components.append(w['wL'] * (L - 1))
                else:
                    # Use average over connected components
                    total_L = 0
                    total_nodes = 0
                    for component in nx.connected_components(G):
                        subG = G.subgraph(component)
                        if len(component) > 1:
                            L_comp = nx.average_shortest_path_length(subG)
                            total_L += L_comp * len(component)
                            total_nodes += len(component)
                    if total_nodes > 0:
                        L = total_L / total_nodes
                        if L > 1:
                            components.append(w['wL'] * (L - 1))
            except Exception as e:
                warnings.warn(f"Could not compute average path length: {e}", RuntimeWarning)
        else:
            warnings.warn("NetworkX not available, using simplified TRF", RuntimeWarning)
            # Simplified TRF using only edge density
            density = num_edges / (num_nodes * (num_nodes - 1) / 2)
            components.append(2 * (1 - density))
        
        # Combine components
        if components:
            trf_value = 1.0 + mu_G * sum(components)
        
        return max(trf_value, 1.0)  # Ensure TRF is at least 1
    
    def _to_networkx(self, edge_index: Tensor, num_nodes: int) -> Any:
        """Convert edge_index tensor to NetworkX graph."""
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index.t().cpu().numpy()
        G.add_edges_from(edges)
        return G
    
    def compute_degree_weights(self, edge_index: Tensor, num_nodes: int) -> Tensor:
        """
        Anti-Hub Dominance weighting (più discriminativo):
            - nodo con grado alto => peso basso
            - normalizzazione globale per avere media ≈ 1
        """
        device = edge_index.device
    
        # Considero edge_index come bidirezionale (come nei tuoi test)
        src, _ = edge_index
        degrees = torch.bincount(src, minlength=num_nodes).float().to(device)
    
        # Anti-hub semplice: ϕ_v ~ 1/(deg(v)+1)
        raw = 1.0 / (degrees + 1.0)  # già molto più sensibile delle log + e
    
        # Se tutti i gradi sono uguali, non possiamo distinguerli in nessun modo con degree-only
        if torch.allclose(raw, raw[0].expand_as(raw)):
            # Ritorna vettore costante ma normalizzato a 1
            return torch.ones_like(raw)
    
        # Normalizzazione min-max in [0.5, 1.5] per evitare estremi
        r_min, r_max = raw.min(), raw.max()
        norm = (raw - r_min) / (r_max - r_min + 1e-8)  # [0,1]
        weights = 0.5 + norm  # [0.5,1.5]
    
        # Opzionale: normalizza la media a 1
        weights = weights / (weights.mean() + 1e-8)
    
        return weights

    
    def compute_homophily_weights(self, 
                              edge_index: Tensor, 
                              node_features: Optional[Tensor],
                              num_nodes: int) -> Tensor:

        device = edge_index.device
    
        if node_features is None:
            return torch.ones(num_nodes, device=device)
    
        homophily_scores = torch.zeros(num_nodes, device=device)
        counts = torch.zeros(num_nodes, device=device)
    
        src, dst = edge_index
    
        src_features = node_features[src]
        dst_features = node_features[dst]
    
        # Normalizza features (cosine similarity)
        src_norm = torch.nn.functional.normalize(src_features, p=2, dim=-1)
        dst_norm = torch.nn.functional.normalize(dst_features, p=2, dim=-1)
    
        # Cosine similarity per edge
        similarities = (src_norm * dst_norm).sum(dim=-1)
    
        # Aggrega sulle estremità
        homophily_scores.scatter_add_(0, src, similarities)
        homophily_scores.scatter_add_(0, dst, similarities)
        counts.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
        counts.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
    
        # Media locale
        mask = counts > 0
        homophily_scores[mask] = homophily_scores[mask] / counts[mask]
    
        # Se tutti identici → niente informazione utile
        if torch.allclose(homophily_scores[mask], homophily_scores[mask][0].expand_as(homophily_scores[mask])):
            return torch.ones(num_nodes, device=device)
    
        # Z-score sul grafo (solo dove counts>0)
        mean = homophily_scores[mask].mean()
        std = homophily_scores[mask].std() + 1e-8
        z = torch.zeros_like(homophily_scores)
        z[mask] = (homophily_scores[mask] - mean) / std
    
        # Mappa con sigmoide: nodi più omofili → peso più alto
        raw = torch.sigmoid(z)  # [0,1] circa
    
        # Espandi un po’ il range: [0.5, 1.5]
        weights = 0.5 + raw
    
        # Normalizza a media ≈ 1
        weights = weights / (weights.mean() + 1e-8)
    
        return weights

    
    def compute_curvature_weights(self, edge_index: Tensor, num_nodes: int) -> Tensor:
        """
        Forman-like curvature-based weighting:
            - Calcola una curvatura locale per nodo come media delle curvature degli archi incidenti
            - ϕ_v derivato da curvatura normalizzata (z-score + sigmoide)
        """
        device = edge_index.device
    
        if not HAS_NX:
            warnings.warn("NetworkX not available for curvature computation", RuntimeWarning)
            return torch.ones(num_nodes, device=device)
    
        try:
            G = self._to_networkx(edge_index, num_nodes)
    
            # Pre-calcolo dei gradi
            deg = dict(G.degree())
    
            curvatures = torch.zeros(num_nodes, device=device)
            counts = torch.zeros(num_nodes, device=device)
    
            # Forman-like edge curvature: F(e) = 4 - (deg(u) + deg(v))
            for u, v in G.edges():
                f_e = 4.0 - float(deg[u] + deg[v])
    
                curvatures[u] += f_e
                curvatures[v] += f_e
                counts[u] += 1.0
                counts[v] += 1.0
    
            mask = counts > 0
            curvatures[mask] = curvatures[mask] / counts[mask]
    
            if torch.any(mask):
                # Se tutte uguali -> niente informazione dalla curvatura
                if torch.allclose(curvatures[mask], curvatures[mask][0].expand_as(curvatures[mask])):
                    return torch.ones(num_nodes, device=device)
    
                # Z-score
                mean = curvatures[mask].mean()
                std = curvatures[mask].std() + 1e-8
                z = torch.zeros_like(curvatures)
                z[mask] = (curvatures[mask] - mean) / std
    
                # Nodi con curvatura "alta" (più positivi) → peso un po' più basso (anti-hub/bridge)
                raw = torch.sigmoid(-z)  # invertito: curvatura alta => peso piccolo
    
                # Range [0.5, 1.5]
                weights = 0.5 + raw
    
                # Normalizza a media ≈ 1
                weights = weights / (weights.mean() + 1e-8)
            else:
                weights = torch.ones(num_nodes, device=device)
    
            return weights
    
        except Exception as e:
            warnings.warn(f"Error computing curvature weights: {e}", RuntimeWarning)
            return torch.ones(num_nodes, device=device)

    
    def compute_combined_weights(self,
                                edge_index: Tensor,
                                node_features: Optional[Tensor],
                                num_nodes: int,
                                combination: str = 'product') -> Tensor:
        """
        Combine all three weighting strategies.
        
        Args:
            combination: 'product', 'sum', or 'weighted_sum'
        """
        degree_w = self.compute_degree_weights(edge_index, num_nodes)
        homophily_w = self.compute_homophily_weights(edge_index, node_features, num_nodes)
        curvature_w = self.compute_curvature_weights(edge_index, num_nodes)
        
        if combination == 'product':
            weights = degree_w * homophily_w * curvature_w
        elif combination == 'sum':
            weights = (degree_w + homophily_w + curvature_w) / 3
        elif combination == 'weighted_sum':
            # Can be customized with different weights
            weights = 0.4 * degree_w + 0.4 * homophily_w + 0.2 * curvature_w
        else:
            raise ValueError(f"Unknown combination method: {combination}")
        
        return weights


class TAdam(Adam):
    """
    Topologically-informed Adam optimizer for Graph Neural Networks.
    
    Extends Adam with:
    - Global learning rate scaling based on graph topology (TRF)
    - Local gradient weighting based on node properties
    """
    
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        *,
        graph_data: Optional[Union[Tuple[Tensor, int], Dict[str, Any]]] = None,
        trf_weights: Optional[Dict[str, float]] = None,
        local_scaling: str = 'none',  # 'none', 'degree', 'homophily', 'curvature', 'combined'
        topology_update_freq: int = 10,
        adaptive_frequency: bool = True,
        combination_method: str = 'product',
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        """
        Initialize TAdam optimizer.
        
        Args:
            params: iterable of parameters to optimize or dicts defining parameter groups
            lr: learning rate (default: 1e-3)
            betas: coefficients for computing running averages (default: (0.9, 0.999))
            eps: term added to denominator for numerical stability (default: 1e-8)
            weight_decay: weight decay coefficient (default: 0)
            amsgrad: whether to use AMSGrad variant (default: False)
            graph_data: Either tuple (edge_index, num_nodes) or dict with keys:
                - 'edge_index': edge connectivity tensor [2, num_edges]
                - 'num_nodes': number of nodes
                - 'node_features': optional node features for homophily computation
            trf_weights: weights for TRF calculation (default: all 1.0)
            local_scaling: type of local scaling to apply
            topology_update_freq: how often to update topology metrics
            adaptive_frequency: whether to adapt update frequency based on graph size
            combination_method: how to combine multiple local scaling methods
            maximize: maximize objective function (default: False)
            foreach: use foreach implementation (default: None)
            capturable: use capturable implementation (default: False)
            differentiable: enable differentiable implementation (default: False)
            fused: use fused implementation (default: None)
        """
        super().__init__(
            params, lr, betas, eps, weight_decay, amsgrad,
            maximize=maximize, foreach=foreach, capturable=capturable,
            differentiable=differentiable, fused=fused
        )
        
        # Initialize topology components
        self.metrics = TopologyMetrics(topology_update_freq, adaptive_frequency)
        self.graph_data = graph_data
        self.trf_weights = trf_weights or {}
        self.local_scaling = local_scaling
        self.combination_method = combination_method
        
        # Cache for topology metrics
        self._trf_scale = 1.0
        self._node_weights = None
        
        # Validate and process graph data
        if graph_data is not None:
            self._process_graph_data()
    
    def _process_graph_data(self):
        """Process and validate graph data."""
        if isinstance(self.graph_data, tuple):
            edge_index, num_nodes = self.graph_data
            self.edge_index = edge_index
            self.num_nodes = num_nodes
            self.num_edges = edge_index.shape[1] // 2  # Assuming undirected
            self.node_features = None
        elif isinstance(self.graph_data, dict):
            self.edge_index = self.graph_data['edge_index']
            self.num_nodes = self.graph_data['num_nodes']
            self.num_edges = self.graph_data.get('num_edges', self.edge_index.shape[1] // 2)
            self.node_features = self.graph_data.get('node_features', None)
        else:
            raise ValueError("graph_data must be tuple (edge_index, num_nodes) or dict")
    
    def update_graph(self, graph_data: Union[Tuple[Tensor, int], Dict[str, Any]]):
        """Update graph data for dynamic graphs."""
        self.graph_data = graph_data
        self._process_graph_data()
        # Force metric update on next step
        self.metrics._step_count = 0
    
    def _update_topology_metrics(self):
        """Update cached topology metrics if needed."""
        if self.graph_data is None:
            return
        
        if self.metrics.should_update(self.num_nodes):
            # Update global TRF scale
            TRF=self.metrics.compute_trf(
                self.edge_index, self.num_nodes, self.num_edges, self.trf_weights
            )
            # print(f"TRF: {TRF}")
            self._trf_scale = 0.01+0.99**(-1*(TRF-1))
            
            # Update local node weights
            if self.local_scaling == 'degree':
                self._node_weights = self.metrics.compute_degree_weights(
                    self.edge_index, self.num_nodes
                )
            elif self.local_scaling == 'homophily':
                self._node_weights = self.metrics.compute_homophily_weights(
                    self.edge_index, self.node_features, self.num_nodes
                )
            elif self.local_scaling == 'curvature':
                self._node_weights = self.metrics.compute_curvature_weights(
                    self.edge_index, self.num_nodes
                )
            elif self.local_scaling == 'combined':
                self._node_weights = self.metrics.compute_combined_weights(
                    self.edge_index, self.node_features, self.num_nodes,
                    self.combination_method
                )
            elif self.local_scaling == 'Only_TRFscale':
                self._node_weights = None
            else:
                raise ValueError(f"Unknown local_scaling method: {self.local_scaling}")
    
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Perform a single optimization step with topology-aware scaling.
        """
        # Update metrics
        self.metrics.increment_step()
        self._update_topology_metrics()
        
        # Apply global TRF scaling to learning rate
        if self._trf_scale != 1.0:
            # Temporarily scale learning rates
            original_lrs = []
            for group in self.param_groups:
                original_lrs.append(group['lr'])
                # print(f"old LR: {group['lr']}")
                group['lr'] = group['lr'] * self._trf_scale
                # print(f"factor scaled LR: {self._trf_scale}")
        
        # Apply local node weights to gradients if available
        if self._node_weights is not None:
            original_grads = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        original_grads.append(p.grad.data.clone())
                        # Assuming gradient shape matches node weights for node embeddings
                        if p.grad.shape[0] == self._node_weights.shape[0]:
                            # Apply node-wise scaling
                            if p.grad.dim() == 1:
                                p.grad.data = p.grad.data * self._node_weights
                            elif p.grad.dim() == 2:
                                p.grad.data = p.grad.data * self._node_weights.unsqueeze(1)
                            else:
                                # For higher dimensional tensors, scale first dimension
                                scaling_shape = [self._node_weights.shape[0]] + [1] * (p.grad.dim() - 1)
                                p.grad.data = p.grad.data * self._node_weights.view(scaling_shape)
                    else:
                        original_grads.append(None)
        
        # Perform standard Adam step
        loss = super().step(closure)
        
        # Restore original learning rates
        if self._trf_scale != 1.0:
            for group, orig_lr in zip(self.param_groups, original_lrs):
                group['lr'] = orig_lr
        
        # Restore original gradients
        if self._node_weights is not None:
            grad_idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    if original_grads[grad_idx] is not None:
                        p.grad.data = original_grads[grad_idx]
                    grad_idx += 1
        
        return loss
    
    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the optimizer as a dict."""
        state = super().state_dict()
        state['tadam_state'] = {
            'trf_scale': self._trf_scale,
            'metrics_step_count': self.metrics._step_count,
            'update_frequency': self.metrics.update_frequency,
        }
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load the optimizer state."""
        tadam_state = state_dict.pop('tadam_state', {})
        super().load_state_dict(state_dict)
        
        # Restore TAdam-specific state
        self._trf_scale = tadam_state.get('trf_scale', 1.0)
        self.metrics._step_count = tadam_state.get('metrics_step_count', 0)
        self.metrics.update_frequency = tadam_state.get('update_frequency', 
                                                       self.metrics.base_frequency)


def tadam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    # Additional TAdam parameters
    trf_scale: float = 1.0,
    node_weights: Optional[Tensor] = None,
    # Standard Adam parameters
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
):
    """
    Functional API for TAdam algorithm.
    
    This is a wrapper around the standard Adam functional API with
    topology-aware scaling applied.
    """
    # Apply TRF scaling to learning rate
    scaled_lr = lr * trf_scale if isinstance(lr, float) else lr * trf_scale
    
    # Apply node weights to gradients if provided
    if node_weights is not None:
        scaled_grads = []
        for grad in grads:
            if grad is not None and grad.shape[0] == node_weights.shape[0]:
                if grad.dim() == 1:
                    scaled_grads.append(grad * node_weights)
                elif grad.dim() == 2:
                    scaled_grads.append(grad * node_weights.unsqueeze(1))
                else:
                    scaling_shape = [node_weights.shape[0]] + [1] * (grad.dim() - 1)
                    scaled_grads.append(grad * node_weights.view(scaling_shape))
            else:
                scaled_grads.append(grad)
    else:
        scaled_grads = grads
    
    # Call standard Adam with scaled parameters
    adam(
        params,
        scaled_grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        foreach=foreach,
        capturable=capturable,
        differentiable=differentiable,
        fused=fused,
        grad_scale=grad_scale,
        found_inf=found_inf,
        has_complex=has_complex,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=scaled_lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
    )


# Documentation
TAdam.__doc__ = r"""
Implements TAdam (Topologically-informed Adam) algorithm for Graph Neural Networks.

TAdam extends Adam with topology-aware learning rate scaling:

1. **Global Scaling (TRF)**: Adjusts learning rate based on graph complexity
   
   .. math::
      lr_{effective} = \frac{lr_{base}}{TRF(G)}
   
   where TRF (Topological Relevance Function) measures graph difficulty:
   
   .. math::
      TRF(G) = 1 + \mu_G \times \sum_i w_i \cdot metric_i
   
2. **Local Scaling**: Node-wise gradient weighting based on:
   - **Degree-based** (Anti-Hub): :math:`\phi_v = \frac{1}{\log(\deg(v) + e)}`
   - **Homophily-based**: :math:`\phi_v = 1 + \tanh(H_v)`
   - **Curvature-based**: :math:`\phi_v = e^{-\kappa_v}`

Args:
    params: iterable of parameters to optimize
    lr (float, optional): base learning rate (default: 1e-3)
    betas (Tuple[float, float], optional): coefficients for running averages (default: (0.9, 0.999))
    eps (float, optional): numerical stability term (default: 1e-8)
    weight_decay (float, optional): weight decay coefficient (default: 0)
    amsgrad (bool, optional): use AMSGrad variant (default: False)
    graph_data (tuple or dict, optional): graph structure information
    trf_weights (dict, optional): weights for TRF components
    local_scaling (str, optional): type of node-wise scaling
    topology_update_freq (int, optional): steps between metric updates
    adaptive_frequency (bool, optional): adapt frequency to graph size
    
Example:
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> 
    >>> # Create graph data
    >>> edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    >>> num_nodes = 3
    >>> 
    >>> # Initialize model and optimizer
    >>> model = GNN(in_channels=16, out_channels=10)
    >>> optimizer = TAdam(
    ...     model.parameters(),
    ...     lr=0.01,
    ...     graph_data=(edge_index, num_nodes),
    ...     local_scaling='degree',
    ...     topology_update_freq=10
    ... )
    >>> 
    >>> # Training loop
    >>> for epoch in range(100):
    ...     optimizer.zero_grad()
    ...     loss = compute_loss(model, data)
    ...     loss.backward()
    ...     optimizer.step()

References:
    - Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)
    - Graph Neural Networks (overview): https://arxiv.org/abs/1901.00596
    - Ollivier-Ricci Curvature: https://arxiv.org/abs/1712.02943
"""
