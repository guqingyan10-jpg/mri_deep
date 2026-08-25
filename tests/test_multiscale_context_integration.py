import ast
import importlib.util
import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _class_node(relative_path, class_name):
    tree = ast.parse(_source(relative_path))
    return next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )


def _load_runner():
    path = ROOT / "scripts" / "run_multiscale_seed_screen.py"
    spec = importlib.util.spec_from_file_location("run_multiscale_seed_screen", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_runner_v2():
    path = ROOT / "scripts" / "run_multiscale_v2_seed_screen.py"
    spec = importlib.util.spec_from_file_location("run_multiscale_v2_seed_screen", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MultiScaleContextIntegrationTests(unittest.TestCase):
    def test_context_module_has_expected_four_branch_contract(self):
        source = _source("models/resunet_edge.py")
        module = _class_node("models/resunet_edge.py", "MultiScaleContext3d")

        self.assertIn("dilation=1", source)
        self.assertIn("dilation=2", source)
        self.assertIn("dilation=3", source)
        self.assertIn("kernel_size=1", source)
        self.assertTrue(
            any(
                isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
                for node in ast.walk(module)
            ),
            "context output must retain a residual addition",
        )

    def test_model_flags_default_to_disabled(self):
        for path, class_name in (
            ("models/resunet_edge.py", "ResUNetEdge"),
            ("models/resunet_hf_concat_boundary.py", "ResUNetHFConcatBoundary"),
        ):
            node = _class_node(path, class_name)
            init = next(
                item for item in node.body
                if isinstance(item, ast.FunctionDef) and item.name == "__init__"
            )
            defaults = dict(zip(
                [arg.arg for arg in init.args.args[-len(init.args.defaults):]],
                init.args.defaults,
            ))
            self.assertIn("multiscale_context", defaults)
            self.assertIs(defaults["multiscale_context"].value, False)

    def test_training_entrypoint_forwards_multiscale_flag(self):
        source = _source("scripts/train_hf_concat_boundary.py")
        self.assertIn('"--multiscale_context"', source)
        self.assertIn("multiscale_context=args.multiscale_context", source)

    def test_runner_builds_fair_seed_42_and_123_jobs(self):
        runner = _load_runner()
        output_root = ROOT / "stability-output"
        jobs = runner.build_jobs([42, 123], output_root, epochs=200, lr=5e-4)

        self.assertEqual([job.seed for job in jobs], [42, 123])
        for job in jobs:
            baseline = output_root / f"seed{job.seed}" / "baseline" / "best_model_9.pth"
            command = job.command(str(baseline))
            self.assertEqual(
                job.checkpoint_dir,
                output_root / f"seed{job.seed}" / "hf_concat_boundary_w0.1_multiscale",
            )
            self.assertEqual(command[command.index("--seed") + 1], str(job.seed))
            self.assertEqual(command[command.index("--epochs") + 1], "200")
            self.assertEqual(command[command.index("--lr") + 1], "0.0005")
            self.assertEqual(command[command.index("--fusion") + 1], "concat")
            self.assertEqual(command[command.index("--boundary_weight") + 1], "0.1")
            self.assertEqual(
                command[command.index("--baseline_checkpoint") + 1], str(baseline)
            )
            self.assertIn("--multiscale_context", command)

    def test_seed_evaluation_registers_multiscale_model(self):
        source = _source("scripts/eval_key_comparison.py")
        self.assertIn("hf_concat_boundary_w0.1_multiscale", source)
        self.assertIn("'multiscale_context': True", source)

    def test_v2_uses_zero_initialized_residual_scale(self):
        source = _source("models/resunet_edge.py")
        self.assertIn("identity_start=False", source)
        self.assertIn("self.alpha = nn.Parameter(torch.zeros(1))", source)
        self.assertIn("x + self.alpha * context", source)

    def test_v2_training_entrypoint_and_runner_are_seed_42_specific(self):
        train_source = _source("scripts/train_hf_concat_boundary.py")
        runner_source = _source("scripts/run_multiscale_v2_seed_screen.py")
        self.assertIn('"--multiscale_context_v2"', train_source)
        self.assertIn("multiscale_context_v2=args.multiscale_context_v2", train_source)
        self.assertIn("hf_concat_boundary_w0.1_multiscale_v2", runner_source)
        self.assertIn('default=[42]', runner_source)

        runner = _load_runner_v2()
        jobs = runner.build_jobs([42], ROOT / "stability-output", 200, 5e-4)
        command = jobs[0].command(
            str(ROOT / "stability-output/seed42/baseline/best_model_9.pth")
        )
        self.assertIn("--multiscale_context_v2", command)
        self.assertEqual(
            str(jobs[0].checkpoint_dir),
            str(ROOT / "stability-output/seed42/hf_concat_boundary_w0.1_multiscale_v2"),
        )

    def test_seed_evaluation_registers_v2_model(self):
        source = _source("scripts/eval_key_comparison.py")
        self.assertIn("hf_concat_boundary_w0.1_multiscale_v2", source)
        self.assertIn("'multiscale_context_v2': True", source)

    @unittest.skipIf(torch is None, "PyTorch is only available in the AutoDL environment")
    def test_enabled_model_preserves_shape_and_shared_initialization(self):
        from models.resunet_hf_concat_boundary import ResUNetHFConcatBoundary

        torch.manual_seed(42)
        control = ResUNetHFConcatBoundary(n_channels=8)
        torch.manual_seed(42)
        multiscale = ResUNetHFConcatBoundary(
            n_channels=8,
            multiscale_context=True,
        )

        multiscale_shared = {
            key: value
            for key, value in multiscale.state_dict().items()
            if not key.startswith("multiscale_context.")
        }
        self.assertEqual(control.state_dict().keys(), multiscale_shared.keys())
        for key, expected in control.state_dict().items():
            self.assertTrue(torch.equal(expected, multiscale_shared[key]), key)

        multiscale.eval()
        with torch.no_grad():
            seg, boundary = multiscale(torch.randn(1, 4, 16, 16, 16))
        self.assertEqual(seg.shape, (1, 3, 16, 16, 16))
        self.assertEqual(boundary.shape, seg.shape)


if __name__ == "__main__":
    unittest.main()
