import unittest
from pathlib import Path

import torch

from models.resunet_edge import ResUNetEdge, ResUpEdge
from models.resunet_hf_concat_boundary import ResUNetHFConcatBoundary


ROOT = Path(__file__).resolve().parents[1]


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

    def test_boundary_model_supports_gated_concat_and_keeps_two_heads(self):
        model = ResUNetHFConcatBoundary(
            in_channels=4,
            n_classes=3,
            n_channels=8,
            fusion="gated_concat",
        )

        model.eval()
        with torch.no_grad():
            seg, boundary = model(torch.randn(1, 4, 16, 16, 16))

        self.assertEqual(seg.shape, (1, 3, 16, 16, 16))
        self.assertEqual(boundary.shape, seg.shape)

    def test_boundary_gate_does_not_change_shared_initialization(self):
        torch.manual_seed(123)
        concat = ResUNetHFConcatBoundary(n_channels=8, fusion="concat")
        torch.manual_seed(123)
        gated = ResUNetHFConcatBoundary(n_channels=8, fusion="gated_concat")

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
                f"boundary shared initialization differs at {key}",
            )

    def test_training_entrypoints_expose_gated_concat(self):
        edge_source = (ROOT / "scripts" / "train_v2_edge.py").read_text(
            encoding="utf-8"
        )
        boundary_source = (
            ROOT / "scripts" / "train_hf_concat_boundary.py"
        ).read_text(encoding="utf-8")

        self.assertIn("gated_concat", edge_source)
        self.assertIn('add_argument("--fusion"', boundary_source)
        self.assertIn("gated_concat", boundary_source)
        self.assertIn("fusion=args.fusion", boundary_source)


if __name__ == "__main__":
    unittest.main()
