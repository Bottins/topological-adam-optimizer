"""Regression checks for the public optimizer API and revision fixes."""
import contextlib
import importlib
import io
import unittest

import torch

from tadam import TAdam, TopologyMetrics


def graph(nodes):
    edges = [(i, i + 1) for i in range(nodes - 1)]
    return torch.tensor(edges + [(v, u) for u, v in edges]).t().contiguous()


class OptimizerTests(unittest.TestCase):
    def test_without_graph_matches_adam(self):
        first = torch.nn.Parameter(torch.tensor([1., -2.]))
        second = torch.nn.Parameter(first.detach().clone())
        adam = torch.optim.Adam([first], lr=0.01, weight_decay=0.001)
        tadam = TAdam([second], lr=0.01, weight_decay=0.001)
        for _ in range(12):
            for param, optimizer in ((first, adam), (second, tadam)):
                optimizer.zero_grad()
                param.square().sum().backward()
                optimizer.step()
            self.assertTrue(torch.equal(first, second))

    def test_default_local_mode_matches_global_only(self):
        params = [torch.nn.Parameter(torch.tensor([1., -2.])) for _ in range(2)]
        opts = [TAdam([param], graph_data=(graph(3), 3), local_scaling=mode)
                for param, mode in zip(params, ('none', 'Only_TRFscale'))]
        for opt in opts:
            opt.metrics.compute_trf = lambda *args: 2.0
        for _ in range(12):
            for param, opt in zip(params, opts):
                opt.zero_grad()
                param.square().sum().backward()
                opt.step()
        self.assertTrue(torch.equal(*params))
        self.assertGreater(opts[0]._trf_scale, 1.)

    def test_scalar_parameter_with_local_weights(self):
        scalar = torch.nn.Parameter(torch.tensor(2.))
        opt = TAdam([scalar], graph_data=(graph(3), 3), local_scaling='degree',
                    topology_update_freq=1)
        opt.metrics.compute_trf = lambda *args: 2.0
        scalar.square().backward()
        opt.step()
        self.assertTrue(torch.isfinite(scalar))
        self.assertLess(float(scalar.detach()), 2.)

    def test_empty_graph_homophily_is_neutral(self):
        weights = TopologyMetrics().compute_homophily_weights(
            torch.empty((2, 0), dtype=torch.long), torch.ones(3, 4), 3)
        self.assertTrue(torch.equal(weights, torch.ones(3)))

    def test_graph_update_refreshes_on_next_step(self):
        param = torch.nn.Parameter(torch.ones(3))
        opt = TAdam([param], graph_data=(graph(3), 3), local_scaling='degree')
        opt.metrics.compute_trf = lambda *args: 2.0
        opt._node_weights = torch.ones(3)
        opt.update_graph((graph(4), 4))
        param.sum().backward()
        opt.step()
        self.assertEqual(opt._node_weights.shape, (4,))

    def test_invalid_options_fail_early(self):
        param = torch.nn.Parameter(torch.ones(1))
        for kwargs in ({'local_scaling': 'typo'}, {'topology_update_freq': 0},
                       {'combination_method': 'typo'}):
            with self.assertRaises(ValueError):
                TAdam([param], **kwargs)

    def test_legacy_demo_is_safe_to_import(self):
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            demo = importlib.import_module('test_tadam')
        self.assertEqual(capture.getvalue(), '')
        demo.set_global_seed(2025)
        graphs = demo.create_test_graphs()
        self.assertEqual(len(graphs), 6)
        model = demo.SimpleGNN(16, 8, 4)
        losses = demo.train_with_optimizer(model, torch.optim.Adam(model.parameters()),
                                           graphs[0], num_epochs=2, verbose=False)
        self.assertEqual(len(losses), 2)
        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))


if __name__ == '__main__':
    unittest.main()
