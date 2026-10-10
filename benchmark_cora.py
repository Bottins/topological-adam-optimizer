"""Reconstructed Cora protocol, NOT an exact reproduction of the missing original.

Run: python benchmark_cora.py --seeds 2025
Requires torch, torch-geometric, numpy, scipy, networkx and Cora download access.
Defaults are new explicit assumptions; do not attribute them to the proceeding.
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures

from tadam import TAdam, TopologyMetrics


METHODS = [
    'Adam', 'TAdam-OnlyTRF', 'TAdam-Combined',
    'Adapted-repository-degree', 'Adapted-repository-homophily',
    'Adapted-repository-curvature', 'Adapted-repository-combined',
]


class GCN(torch.nn.Module):
    def __init__(self, inputs, hidden, outputs):
        super().__init__()
        self.conv1 = GCNConv(inputs, hidden, cached=True)
        self.conv2 = GCNConv(hidden, outputs, cached=True)

    def forward(self, data):
        embedding = F.relu(self.conv1(data.x, data.edge_index))
        logits = self.conv2(F.dropout(embedding, p=0.5, training=self.training),
                            data.edge_index)
        return logits, embedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2025])
    parser.add_argument('--hidden', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--data', default='data')
    parser.add_argument('--output', default='results/cora_run')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dataset = Planetoid(args.data, 'Cora', split='public', transform=NormalizeFeatures())
    data = dataset[0]
    assert [int(mask.sum()) for mask in
            (data.train_mask, data.val_mask, data.test_mask)] == [140, 500, 1000]
    metrics = TopologyMetrics()
    print('Computing static graph TRF...', flush=True)
    trf = metrics.compute_trf(data.edge_index, data.num_nodes,
                              data.edge_index.shape[1] // 2, {})
    degree = metrics.compute_degree_weights(data.edge_index, data.num_nodes)
    curvature = metrics.compute_curvature_weights(data.edge_index, data.num_nodes)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'results.csv').exists() or (output / 'protocol.json').exists():
        raise FileExistsError(f'Choose a new --output directory; results already exist in {output}')
    manifest = dict(vars(args), epochs=200, device='cpu', torch=torch.__version__,
                    pyg=torch_geometric.__version__, trf=trf,
                    status='reconstructed protocol; original seed and adapters unavailable',
                    protocol='GCN-ReLU-dropout(0.5)-GCN; normalized features; '
                    'weight decay on all parameters; final epoch; no checkpoint selection; '
                    'adapted loss = mean(node_weight * node_loss) on training nodes; '
                    'homophily from detached pre-dropout hidden embeddings each epoch; '
                    'static TRF cached, repository update schedule preserved',
                    source_hash_format='sha256 of UTF-8 source with LF line endings',
                    source_hashes={name: hashlib.sha256(Path(__file__).with_name(name).read_text(encoding='utf-8').encode('utf-8')).hexdigest()
                                   for name in ('tadam.py', 'benchmark_cora.py')})
    (output / 'protocol.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    fields = ['seed', 'method', 'accuracy', 'loss']
    with (output / 'results.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for seed in args.seeds:
            original_logits = None
            for name in METHODS:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                model = GCN(dataset.num_features, args.hidden, dataset.num_classes)
                common = dict(lr=args.lr, weight_decay=args.weight_decay)
                if name == 'Adam':
                    optimizer = torch.optim.Adam(model.parameters(), **common)
                else:
                    optimizer = TAdam(model.parameters(), **common,
                        graph_data=dict(edge_index=data.edge_index, num_nodes=data.num_nodes,
                                        node_features=data.x),
                        local_scaling='combined' if name == 'TAdam-Combined' else 'Only_TRFscale')
                    # Exact static value; avoid repeating expensive NetworkX calculations.
                    optimizer.metrics.compute_trf = lambda *a, **kw: trf
                for _ in range(200):
                    model.train()
                    optimizer.zero_grad()
                    logits, embedding = model(data)
                    losses = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask],
                                             reduction='none')
                    if name.startswith('Adapted-repository-'):
                        variant = name.removeprefix('Adapted-repository-')
                        with torch.no_grad():
                            if variant in ('homophily', 'combined'):
                                homophily = metrics.compute_homophily_weights(
                                    data.edge_index, embedding.detach(), data.num_nodes)
                            weights = degree if variant == 'degree' else curvature
                            if variant == 'homophily':
                                weights = homophily
                            elif variant == 'combined':
                                weights = degree * homophily * curvature
                        losses = losses * weights[data.train_mask]
                    losses.mean().backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    logits, _ = model(data)
                    accuracy = 100 * float((logits[data.test_mask].argmax(1) ==
                                            data.y[data.test_mask]).double().mean())
                    loss = float(F.cross_entropy(logits[data.test_mask], data.y[data.test_mask]))
                if name == 'TAdam-OnlyTRF':
                    original_logits = logits.clone()
                elif name == 'TAdam-Combined':
                    assert torch.equal(original_logits, logits), 'Unexpected local-scaling effect'
                row = dict(zip(fields, [seed, name, accuracy, loss]))
                writer.writerow(row)
                handle.flush()
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
