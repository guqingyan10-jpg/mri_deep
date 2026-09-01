import ast
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_wt_lesion_stratified.py"
ET_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_et_lesion_stratified.py"
STRATA_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "derive_train_lesion_strata.py"


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
        "ResUNet (BCE–Dice)",
        "ResUNet + LHFC",
        "ResUNet + LHFC + ABS (λ_b = 0.1)",
        "AFBMS-ResUNet",
        "ResUNet + LHFC + ABS (λ_b = 0.05)",
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


def test_wt_comparison_plot_has_a_color_for_each_registered_model():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    colors_assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "colors"
                for target in node.targets)
    )
    colors = ast.literal_eval(colors_assignment.value)
    assert len(colors) >= len(_registry_literal())


def test_et_entry_point_reuses_shared_five_model_evaluation():
    source = ET_SCRIPT.read_text(encoding="utf-8")
    assert "from eval_wt_lesion_stratified import main" in source
    assert 'main(default_region="ET")' in source


def test_evaluation_fits_strata_on_train_and_applies_to_selected_phase():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'phase="train"' in source
    assert 'choices=("valid", "test", "valid_test")' in source
    assert 'default="test"' in source
    assert '{"WT": 0, "ET": 2}' in source
    assert "derive_size_strata(training_sizes" in source
    assert '"--strata-json"' in source
    assert 'strata_metadata.get("fit_split") != "train"' in source


def test_pooled_validation_test_mode_is_auditable_and_does_not_overwrite_test():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'split_names = ("valid", "test") if phase == "valid_test"' in source
    assert "ConcatDataset" in source
    assert 'f"{region.lower()}_lesion_stratified_valid_test_results"' in source
    assert 'f"{prefix}_evaluated_cases.csv"' in source
    assert 'strata_metadata["evaluation_splits"]' in source


def test_training_distribution_supports_wt_and_et_regions():
    source = STRATA_SCRIPT.read_text(encoding="utf-8")
    assert '"WT": (1, 2, 4)' in source
    assert '"ET": (4,)' in source
    assert 'default="BOTH"' in source
    assert "train_test_split(" in source
    assert "random_state=10" in source
