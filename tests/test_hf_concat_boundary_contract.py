import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_ast(relative_path):
    path = ROOT / relative_path
    return ast.parse(path.read_text(encoding="utf-8"))


def _find_class(tree, name):
    return next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == name)


def test_combined_model_exposes_segmentation_and_boundary_outputs():
    tree = _read_ast("models/resunet_hf_concat_boundary.py")
    model = _find_class(tree, "ResUNetHFConcatBoundary")

    forward = next(node for node in model.body
                   if isinstance(node, ast.FunctionDef) and node.name == "forward")
    return_nodes = [node for node in ast.walk(forward) if isinstance(node, ast.Return)]

    assert return_nodes, "combined model must define a forward return"
    returned = return_nodes[-1].value
    assert isinstance(returned, ast.Tuple)
    assert len(returned.elts) == 2


def test_combined_model_reuses_multiscale_concat_edge_backbone_and_boundary_head():
    source = (ROOT / "models/resunet_hf_concat_boundary.py").read_text(encoding="utf-8")
    assert "ResUNetEdge" in source
    assert "fusion='concat'" in source or 'fusion="concat"' in source
    assert "boundary_head" in source
    assert "edge_type='laplacian'" in source or 'edge_type="laplacian"' in source


def test_training_entrypoint_keeps_baseline_fairness_contract():
    source = (ROOT / "scripts/train_hf_concat_boundary.py").read_text(encoding="utf-8")
    required_fragments = [
        "BCEDiceWithBoundaryLoss",
        "boundary_weight=0.3",
        "lr=args.lr",
        "accumulation_steps=4",
        "early_stopping_patience=25",
        "min_delta=1e-4",
        "check_exist(config.ResUNet_checkpoint_dir)",
        'startswith("best_model_")',
        "check_exist_last(CHECKPOINT_DIR)",
        "raise FileNotFoundError",
    ]
    for fragment in required_fragments:
        assert fragment in source, f"missing fairness contract: {fragment}"


def test_combined_model_is_registered_in_both_evaluation_entrypoints():
    for relative_path in (
        "scripts/eval_all_experiments.py",
        "scripts/eval_comprehensive.py",
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "ResUNetHFConcatBoundary" in source
        assert "/root/autodl-tmp/ResUNet_HFConcatBoundary_model" in source
        assert "HF Concat Boundary" in source
