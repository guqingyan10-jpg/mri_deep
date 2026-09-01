import ast
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "eval_alpha_sensitivity.py"
)


def _source():
    return SCRIPT.read_text(encoding="utf-8")


def test_alpha_sensitivity_uses_fixed_validation_and_three_gate_modes():
    source = _source()
    assert 'phase="valid"' in source
    assert 'ALPHA_MODES = ("zero", "learned", "one")' in source
    assert "model.multiscale_context.alpha.fill_(evaluation_alpha)" in source
    assert '"evaluation_split": "valid"' in source


def test_small_metric_reuses_training_defined_et_lesion_strata():
    source = _source()
    assert '"--et-strata-json"' in source
    assert "et_training_lesion_strata.json" in source
    assert 'metadata.get("fit_split") != "train"' in source
    assert 'metadata.get("region") != "ET"' in source
    assert "match_lesion_components(" in source
    assert "summarize_stratified_cases(case_lesion_results, strata)" in source
    assert '"small_lesion_gt_anchored_dice"' in source
    assert '"small_lesion_matched_dice"' in source
    assert '"small_lesion_recall"' in source
    assert '"small_lesion_miss_rate"' in source
    assert "small_case_definition" not in source


def test_seed55_uses_the_distinct_main_experiment_directory():
    tree = ast.parse(_source())
    defaults = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    strings = [node.value for node in defaults]
    assert "main_experiment" in strings
    assert "stability_runner" in strings
    assert any(
        "ResUNet_HFConcatBoundary_w0.1_multiscale_v2_model" in value
        for value in strings
    )


def test_sensitivity_does_not_claim_to_reconstruct_training_history():
    source = _source()
    assert "last_epoch_model" not in source
    assert "checkpoint-history reconstruction is performed" in source


def test_outputs_include_auditable_cases_alpha_summary_and_figures():
    source = _source()
    for filename in (
        "alpha_sensitivity_per_seed.csv",
        "alpha_sensitivity_per_case.csv",
        "alpha_sensitivity_per_lesion.csv",
        "alpha_checkpoint_values.csv",
        "alpha_checkpoint_summary.json",
        "small_et_validation_lesions.csv",
        "et_lesion_strata_applied.json",
        "alpha_sensitivity_metrics.png",
    ):
        assert filename in source


def test_cross_seed_summary_is_sample_standard_deviation():
    source = _source()
    assert ".std(ddof=1)" in source
    assert "mixed_training_protocols" in source


def test_checkpoint_inspection_can_run_without_validation_inference():
    source = _source()
    assert '"--inspect-only"' in source
    assert "if args.inspect_only:" in source
    assert "checkpoint alpha inspection" in source
