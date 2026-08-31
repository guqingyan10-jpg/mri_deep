import ast
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_wt_lesion_stratified.py"


def _registry_literal():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "MODEL_SPECS"
                for target in node.targets)
    )
    return ast.literal_eval(assignment.value)


def test_wt_evaluation_registry_contains_requested_five_models():
    registry = _registry_literal()
    labels = [spec["label"] for spec in registry]
    assert labels == [
        "Baseline (BCEDice)",
        "Edge (Laplacian, concat)",
        "HF Concat Boundary(w=0.1)",
        "HF Concat Boundary +Multi-scale V2 (w=0.1)",
        "HF Concat Boundary(w=0.05)",
    ]

    checkpoint_dirs = [spec["checkpoint_dir"] for spec in registry]
    assert checkpoint_dirs == [
        "/root/autodl-tmp/ResUNet_model",
        "/root/autodl-tmp/ResUNet_Edge_concat_laplacian_model",
        "/root/autodl-tmp/ResUNet_HFConcatBoundary_w0.1_model",
        "/root/autodl-tmp/ResUNet_HFConcatBoundary_w0.1_multiscale_v2_model",
        "/root/autodl-tmp/ResUNet_HFConcatBoundary_w0.05_model",
    ]

    assert registry[1]["model_kwargs"]["edge_type"] == "laplacian"
    assert registry[3]["model_kwargs"]["multiscale_context_v2"] is True
