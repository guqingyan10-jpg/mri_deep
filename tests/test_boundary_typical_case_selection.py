import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "select_boundary_typical_cases.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The selector itself is executed on AutoDL with PyTorch.  Its deterministic
# ranking logic and boundary metrics can still be tested in the lightweight
# local review environment by stubbing only the training/inference imports.
sys.modules.setdefault("torch", types.ModuleType("torch"))

evaluation_package = types.ModuleType("evaluation")
evaluation_package.__path__ = [str(ROOT / "evaluation")]
sys.modules["evaluation"] = evaluation_package
sys.modules["evaluation.advanced_metrics"] = _load_module(
    "boundary_advanced_metrics", ROOT / "evaluation" / "advanced_metrics.py"
)
sys.modules["evaluation.wt_lesion_stratified"] = _load_module(
    "boundary_wt_lesion_stratified",
    ROOT / "evaluation" / "wt_lesion_stratified.py",
)

dataset_module = types.ModuleType("data.dataset")
dataset_module.BratsDataset = object
dataset_module.get_dataloader = lambda *args, **kwargs: None
sys.modules["data.dataset"] = dataset_module

for module_name, class_name in (
    ("models.resunet3d", "ResUNet3d"),
    ("models.resunet_edge", "ResUNetEdge"),
    ("models.resunet_hf_concat_boundary", "ResUNetHFConcatBoundary"),
):
    model_module = types.ModuleType(module_name)
    setattr(model_module, class_name, type(class_name, (), {}))
    sys.modules[module_name] = model_module

SPEC = importlib.util.spec_from_file_location("boundary_case_selector", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _case_rows(case_id, values, small_gt=0):
    rows = []
    for model_key in MODULE.MODEL_KEYS:
        model = values[model_key]
        rows.append(
            {
                "case_id": case_id,
                "model_key": model_key,
                "model": model_key,
                "gt_et_voxels": 100,
                "pred_et_voxels": 100,
                "et_dice": model.get("et", 0.7),
                "hd95_mm": model["hd"],
                "boundary_dice": model["bd"],
                "boundary_valid": True,
                "small_gt_lesions": small_gt,
                "small_detected_lesions": model.get("det", 0),
                "small_missed_lesions": small_gt - model.get("det", 0),
                "small_gt_anchored_dice": model.get("small", np.nan),
            }
        )
    return rows


def _small_lesion_rows(case_id, gt_index, gt_id, gt_size, dice_by_model):
    rows = []
    for model_key in MODULE.MODEL_KEYS:
        dice = float(dice_by_model[model_key])
        rows.append(
            {
                "case_id": case_id,
                "gt_index": gt_index,
                "gt_id": gt_id,
                "gt_size": gt_size,
                "model_key": model_key,
                "detected": dice > 0,
                "pred_index": gt_index if dice > 0 else -1,
                "pred_size": gt_size if dice > 0 else 0,
                "lesion_dice": dice,
            }
        )
    return rows


def test_identical_and_one_voxel_shifted_boundaries():
    gt = np.zeros((10, 10, 10), dtype=bool)
    gt[3:7, 3:7, 3:7] = True
    assert MODULE.hd95_single(gt, gt) == pytest.approx(0.0)
    assert MODULE.nsd_single(gt, gt, tau=1.0) == pytest.approx(1.0)

    shifted = np.zeros_like(gt)
    shifted[4:8, 3:7, 3:7] = True
    assert MODULE.hd95_single(shifted, gt) == pytest.approx(1.0)
    assert MODULE.nsd_single(shifted, gt, tau=1.0) == pytest.approx(1.0)


def test_focus_uses_complete_matched_component_not_unrelated_prediction():
    gt = np.zeros((12, 20, 20), dtype=bool)
    gt[4:8, 6:11, 6:11] = True
    baseline = np.zeros_like(gt)
    baseline[4:8, 6:10, 6:10] = True
    baseline[2:4, 16:18, 16:18] = True  # unrelated false-positive component
    full = np.zeros_like(gt)
    full[4:8, 6:11, 6:11] = True
    predictions = {
        "baseline": baseline,
        "lhfc": full.copy(),
        "full": full,
    }

    focus_gt, focus_predictions = MODULE._focus_masks_for_case(
        gt,
        predictions,
        "small_lesion_improvement",
        min_component_size=1,
        selected_gt_index=0,
    )

    assert np.array_equal(focus_gt, gt)
    assert int(focus_predictions["baseline"].sum()) == 4 * 4 * 4
    assert not focus_predictions["baseline"][2:4, 16:18, 16:18].any()
    assert np.array_equal(focus_predictions["full"], full)


def test_small_lesion_overlay_separates_tp_fn_and_fp_regions():
    gt = np.array([[1, 1], [0, 0]], dtype=bool)
    prediction = np.array([[1, 0], [1, 0]], dtype=bool)
    overlay = MODULE._region_error_overlay(gt, prediction)

    assert np.allclose(overlay[0, 0], MODULE.REGION_COLORS["tp"])
    assert np.allclose(overlay[0, 1], MODULE.REGION_COLORS["fn"])
    assert np.allclose(overlay[1, 0], MODULE.REGION_COLORS["fp"])
    assert np.allclose(overlay[1, 1], 0.0)


def test_case_selection_returns_four_unique_predefined_roles():
    rows = []
    rows += _case_rows(
        "small",
        {
            "baseline": {"hd": 12, "bd": 0.50, "det": 0, "small": 0.0},
            "lhfc": {"hd": 10, "bd": 0.60, "det": 1, "small": 0.6},
            "full": {"hd": 8, "bd": 0.75, "det": 1, "small": 0.8},
        },
        small_gt=1,
    )
    rows += _case_rows(
        "hd_best",
        {
            "baseline": {"hd": 24, "bd": 0.50},
            "lhfc": {"hd": 15, "bd": 0.60},
            "full": {"hd": 5, "bd": 0.72},
        },
    )
    rows += _case_rows(
        "bd_best",
        {
            "baseline": {"hd": 12, "bd": 0.25},
            "lhfc": {"hd": 10, "bd": 0.65},
            "full": {"hd": 9, "bd": 0.90},
        },
    )
    rows += _case_rows(
        "regression",
        {
            "baseline": {"hd": 5, "bd": 0.90},
            "lhfc": {"hd": 8, "bd": 0.70},
            "full": {"hd": 18, "bd": 0.40},
        },
    )
    rows += _case_rows(
        "neutral",
        {
            "baseline": {"hd": 10, "bd": 0.60},
            "lhfc": {"hd": 9, "bd": 0.62},
            "full": {"hd": 8, "bd": 0.65},
        },
    )
    lesion_rows = []
    lesion_rows += _small_lesion_rows(
        "small", 0, 1, 18,
        {"baseline": 0.60, "lhfc": 0.65, "full": 0.70},
    )
    lesion_rows += _small_lesion_rows(
        "small", 1, 2, 14,
        {"baseline": 0.10, "lhfc": 0.55, "full": 0.80},
    )
    # A Baseline miss is a detection improvement, not a matched-Dice
    # comparison, so it must not win even with a numerically larger gain.
    lesion_rows += _small_lesion_rows(
        "missed_small", 0, 1, 12,
        {"baseline": 0.0, "lhfc": 0.75, "full": 0.95},
    )
    comparison = MODULE.build_case_comparison(pd.DataFrame(rows))
    small_lesions = MODULE.build_small_lesion_comparison(
        pd.DataFrame(lesion_rows)
    )
    selected = MODULE.select_typical_cases(comparison, small_lesions)
    chosen = dict(zip(selected["selection_role"].astype(str), selected["case_id"]))

    assert chosen == {
        "small_lesion_improvement": "small",
        "hd95_improvement": "hd_best",
        "boundary_dice_improvement": "bd_best",
        "regression": "regression",
    }
    assert selected["case_id"].nunique() == 4
    small_selected = selected[
        selected["selection_role"].astype(str) == "small_lesion_improvement"
    ].iloc[0]
    assert small_selected["gt_id"] == 2
    assert small_selected["baseline_lesion_dice"] == pytest.approx(0.10)
    assert small_selected["full_lesion_dice"] == pytest.approx(0.80)
    assert small_selected["small_lesion_dice_gain"] == pytest.approx(0.70)


def test_script_fixes_test_cohort_and_best_checkpoints():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'phase="test"' in source
    assert 'default=37' in source
    assert 'requires best_model_*.pth' in source
    assert "last_epoch_model" not in source
    assert 'linestyle="dashed"' not in source
    assert "Small-lesion Dice" in source
    assert "Small lesion: TP overlap" in source
    assert "small_lesion_comparison.csv" in source
    assert '"Baseline"' in source
    assert '"LHFC"' in source
    assert '"Full"' in source
