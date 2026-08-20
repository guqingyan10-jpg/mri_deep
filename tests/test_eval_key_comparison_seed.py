import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_key_comparison.py"


def test_key_comparison_help_exposes_seed_option():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    argument_values = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }

    assert "--seed" in argument_values
    assert "--stability-root" in argument_values


def test_seed_experiment_builder_defines_the_four_paired_directories():
    source = SCRIPT.read_text(encoding="utf-8")

    for directory in (
        "baseline",
        "edge_laplacian_concat",
        "hf_concat_boundary_w0.1",
        "hf_concat_boundary_w0.05",
    ):
        assert directory in source
