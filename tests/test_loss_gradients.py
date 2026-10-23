import unittest

import torch
from torch import nn

from semgaze.training.loss_gradients import LossGradientProbe


class DummyBundle:
    def __init__(self):
        self.model = nn.Module()
        self.model.register_parameter('lora_shared', nn.Parameter(torch.tensor([1., -1.])))
        self.input_row = nn.Parameter(torch.tensor([2.]))
        self.output_row = self.input_row  # tied: must not count twice
        self.model.register_parameter('end_fix_row', self.input_row)
        self.projector = nn.Linear(2, 1, bias=False)

    def trainable_parameters(self):
        return list(self.model.parameters()) + list(self.projector.parameters())


class ProbeTest(unittest.TestCase):
    def test_aggregated_gradients_equal_joint_backward_and_detect_conflict(self):
        bundle = DummyBundle()
        probe = LossGradientProbe(bundle)
        w = bundle.model.lora_shared
        p = bundle.projector.weight
        r = bundle.input_row
        for scale in (0.25, 0.75):
            where = (w[0] * 3 + w[1] * 2 + r * 4).sum()
            semantic = (-w[0] * 5 + w[1] * 2 + p.sum() * 3 + r * 2).sum()
            old_grads = [param.grad.clone() if param.grad is not None else None
                         for param in bundle.trainable_parameters()]
            probe.record(where_loss=where, semantic_loss=semantic,
                         lambda_where=2.0, lambda_sem=0.5, micro_weight=scale)
            self.assertTrue(all((old is None and param.grad is None) or
                                (old is not None and torch.equal(old, param.grad))
                                for old, param in zip(old_grads, bundle.trainable_parameters())))
            (scale * (2 * where + 0.5 * semantic)).backward()
        stats = probe.summary()
        self.assertEqual(stats['microbatches'], 2)
        self.assertLess(stats['groups']['shared_lora']['cosine'], 0)
        self.assertEqual(stats['groups']['projector']['where_norm'], 0.0)
        self.assertEqual(len(probe.groups['end_fix_rows']), 1)
        for i, param in enumerate(probe.params):
            total = sum((a[i] for a in probe.gradients.values() if a[i] is not None),
                        torch.zeros_like(param, dtype=torch.float32))
            torch.testing.assert_close(total, param.grad.float(), atol=1e-6, rtol=1e-6)

    def test_one_objective_has_undefined_cosine(self):
        bundle = DummyBundle()
        probe = LossGradientProbe(bundle)
        probe.record(where_loss=bundle.model.lora_shared.sum(),
                     semantic_loss=None, lambda_sem=0)
        stats = probe.summary()['groups']['shared_lora']
        self.assertIsNone(stats['cosine'])
        self.assertEqual(stats['semantic_norm'], 0.0)


if __name__ == '__main__':
    unittest.main()
