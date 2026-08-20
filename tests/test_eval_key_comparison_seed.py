import ast
import os
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_key_comparison.py"


def _load_seed_builder():
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_seed_experiments"
    )
    namespace = {
        "os": os,
        "eval_all": SimpleNamespace(
            ResUNet3d=type("ResUNet3d", (), {}),
            ResUNetEdge=type("ResUNetEdge", (), {}),
            ResUNetHFConcatBoundary=type("ResUNetHFConcatBoundary", (), {}),
        ),
    }
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(SCRIPT), "exec"), namespace)
    return namespace["build_seed_experiments"]


def _load_primary_contract():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    assignment = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "PRIMARY"
            for target in node.targets
        )
    )
    return ast.literal_eval(assignment.value)


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


def test_seed_experiment_builder_defines_existing_and_gated_directories():
    source = SCRIPT.read_text(encoding="utf-8")

    for directory in (
        "baseline",
        "edge_laplacian_concat",
        "hf_concat_boundary_w0.1",
        "hf_concat_boundary_w0.05",
        "edge_laplacian_gated_concat",
        "hf_gated_concat_boundary_w0.1",
    ):
        assert directory in source


def test_seed_experiment_builder_configures_two_gated_models():
    build_seed_experiments = _load_seed_builder()
    experiments = build_seed_experiments(123, "/root/autodl-tmp/stability")

    assert len(experiments) == 6
    gated = [
        spec for spec in experiments
        if spec["model_kwargs"].get("fusion") == "gated_concat"
    ]
    assert len(gated) == 2
    assert gated[0]["model_class"].__name__ == "ResUNetEdge"
    assert gated[0]["model_kwargs"]["edge_type"] == "laplacian"
    assert gated[1]["model_class"].__name__ == "ResUNetHFConcatBoundary"


def test_primary_indicator_contract_remains_exactly_four_metrics():
    assert _load_primary_contract() == [
        ("Macro_Dice_mean", "Macro Dice", "high"),
        ("ET_Dice_mean", "ET Dice", "high"),
        ("Small_case_ET_Dice_mean", "Small-case Dice", "high"),
        ("ET_HD95_mean", "ET HD95", "low"),
    ]
