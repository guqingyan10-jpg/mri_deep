import unittest

import torch

from models.resunet_edge import ResUNetEdge, ResUpEdge


class GatedEdgeIntegrationTests(unittest.TestCase):
    def test_gated_concat_starts_as_exact_concat(self):
        torch.manual_seed(17)
        concat = ResUpEdge(16, 8, 8, fusion="concat")
        torch.manual_seed(17)
        gated = ResUpEdge(16, 8, 8, fusion="gated_concat")

        self.assertEqual(torch.count_nonzero(gated.edge_gate.weight).item(), 0)
        self.assertEqual(torch.count_nonzero(gated.edge_gate.bias).item(), 0)

        x1 = torch.randn(1, 8, 2, 2, 2)
        x2 = torch.randn(1, 8, 4, 4, 4)
        edge = torch.randn(1, 8, 4, 4, 4)

        concat.eval()
        gated.eval()
        with torch.no_grad():
            expected = concat(x1, x2, edge)
            actual = gated(x1, x2, edge)

        self.assertTrue(torch.equal(expected, actual))

    def test_gates_do_not_change_shared_model_initialization(self):
        torch.manual_seed(123)
        concat = ResUNetEdge(
            in_channels=4,
            n_classes=3,
            n_channels=8,
            fusion="concat",
            edge_type="laplacian",
        )
        torch.manual_seed(123)
        gated = ResUNetEdge(
            in_channels=4,
            n_classes=3,
            n_channels=8,
            fusion="gated_concat",
            edge_type="laplacian",
        )

        concat_state = concat.state_dict()
        gated_state = {
            key: value
            for key, value in gated.state_dict().items()
            if ".edge_gate." not in key
        }

        self.assertEqual(concat_state.keys(), gated_state.keys())
        for key, expected in concat_state.items():
            self.assertTrue(
                torch.equal(expected, gated_state[key]),
                f"shared initialization differs at {key}",
            )


if __name__ == "__main__":
    unittest.main()
