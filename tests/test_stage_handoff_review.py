import importlib.util
from pathlib import Path
import sys
from unittest.mock import MagicMock


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "stage_handoff_review.py"
)
SPEC = importlib.util.spec_from_file_location("stage_handoff_review", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_selected_model_scope_and_standard_names():
    artifacts = MODULE.model_artifacts()
    ids = {item.model_id for item in artifacts}
    assert {
        "baseline_unet3d",
        "baseline_resunet3d",
        "baseline_attention_unet3d",
        "baseline_nnunet3d",
        "resunet_lhfc",
        "resunet_lhfc_abs_lb0p05",
        "resunet_lhfc_abs_lb0p10",
        "resunet_lhfc_abs_msc",
        "afbms_resunet",
        "afbms_resunet_alpha_trace_supplement",
    } == ids


def test_seed55_uses_main_directories_and_records_missing_nonadaptive_msc():
    seed55 = [item for item in MODULE.model_artifacts() if item.seed == 55]
    mapping = {item.model_id: item.source_dir for item in seed55}
    assert mapping["baseline_resunet3d"] == "ResUNet_model"
    assert mapping["resunet_lhfc"] == "ResUNet_Edge_concat_laplacian_model"
    assert mapping["afbms_resunet"] == (
        "ResUNet_HFConcatBoundary_w0.1_multiscale_v2_model"
    )
    assert mapping["resunet_lhfc_abs_msc"] is None


def test_stability_directory_names_match_training_registry():
    artifacts = MODULE.model_artifacts()
    seed42 = {
        item.model_id: item.source_dir for item in artifacts if item.seed == 42
    }
    assert seed42["resunet_lhfc_abs_msc"].endswith(
        "hf_concat_boundary_w0.1_multiscale"
    )
    assert seed42["afbms_resunet"].endswith(
        "hf_concat_boundary_w0.1_multiscale_v2"
    )


def test_full_method_naming_has_no_gated_variant():
    artifacts = MODULE.model_artifacts()
    assert not any("gated" in item.model_id for item in artifacts)
    full = [item for item in artifacts if item.model_id == "afbms_resunet"]
    assert full
    assert all("AR-MSC (learnable alpha)" in item.components for item in full)
    assert MODULE.METHOD_NAMING[-1][0] == "AFBMS-ResUNet"
    assert MODULE.METHOD_NAMING[-1][1] == (
        "Adaptive Frequency-Boundary Multi-Scale ResUNet"
    )


def test_only_latest_numbered_best_checkpoint_is_resolved():
    directory = MagicMock()
    directory.glob.return_value = [
        Path("best_model_10.pth"), Path("best_model_33.pth")
    ]
    assert MODULE.best_checkpoint(directory).name == "best_model_33.pth"


def test_result_categories_are_separated():
    assert "alpha_sensitivity_test_results" in MODULE.FORMAL_RESULT_DIRS
    assert "et_lesion_stratified_valid_test_results" in (
        MODULE.EXPLORATORY_RESULT_DIRS
    )
    assert "comprehensive_results" in MODULE.LEGACY_RESULT_DIRS
    assert ".png" not in MODULE.TABLE_SUFFIXES
