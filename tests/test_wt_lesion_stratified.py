import numpy as np
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "wt_lesion_stratified_for_test",
    Path(__file__).resolve().parents[1] / "evaluation" / "wt_lesion_stratified.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

derive_size_strata = _MODULE.derive_size_strata
classify_wt_size = _MODULE.classify_wt_size
match_wt_components = _MODULE.match_wt_components
summarize_stratified_cases = _MODULE.summarize_stratified_cases


def test_size_strata_use_inclusive_integer_boundaries():
    strata = derive_size_strata([10, 20, 30, 40, 50, 60])
    assert classify_wt_size(10, strata) == "small"
    assert classify_wt_size(20, strata) == "small"
    assert classify_wt_size(21, strata) == "medium"
    assert classify_wt_size(40, strata) == "medium"
    assert classify_wt_size(41, strata) == "large"


def test_strata_are_fitted_from_supplied_training_components_only():
    train_strata = derive_size_strata([10, 20, 30, 40, 50, 60])
    # Validation/test sizes are classified by the frozen training bounds and
    # are not passed back into threshold fitting.
    assert train_strata["small"] == (10, 20)
    assert train_strata["medium"] == (21, 40)
    assert train_strata["large"] == (41, None)
    assert classify_wt_size(10000, train_strata) == "large"


def test_cut_points_optimize_lesion_counts_without_splitting_ties():
    sizes = [10] * 5 + [20] * 4 + [30] + [40] * 5
    strata = derive_size_strata(sizes)
    assert strata == {
        "small": (10, 10),
        "medium": (11, 30),
        "large": (31, None),
    }


def test_one_to_one_matching_and_matched_dice():
    gt = np.zeros((1, 1, 9), dtype=np.uint8)
    gt[0, 0, 1:3] = 1
    gt[0, 0, 6:8] = 1
    pred = np.zeros_like(gt)
    pred[0, 0, 1:8] = 1  # one prediction overlaps both GT lesions

    result = match_wt_components(pred, gt, min_component_size=1)
    assert result["gt_lesions"] == 2
    assert result["pred_lesions"] == 1
    assert result["tp"] == 1
    assert result["fn"] == 1
    assert result["fp"] == 0
    assert len(result["matches"]) == 1
    assert result["matches"][0]["dice"] == 4 / 9


def test_no_gt_case_is_not_scored_as_perfect_recall():
    empty = np.zeros((1, 1, 5), dtype=np.uint8)
    result = match_wt_components(empty, empty, min_component_size=1)
    strata = derive_size_strata([10, 20, 30])
    summary = summarize_stratified_cases([result], strata=strata)
    assert summary["small"]["gt_lesions"] == 0
    assert np.isnan(summary["small"]["lesion_recall"])
    assert np.isnan(summary["small"]["miss_rate"])
    assert summary["all"]["n_true_negative_cases"] == 1


def test_prediction_without_gt_counts_as_false_positive():
    gt = np.zeros((1, 1, 5), dtype=np.uint8)
    pred = np.zeros_like(gt)
    pred[0, 0, 2] = 1
    result = match_wt_components(pred, gt, min_component_size=1)
    strata = derive_size_strata([10, 20, 30])
    summary = summarize_stratified_cases([result], strata=strata)
    assert summary["all"]["fp"] == 1


def test_summary_is_pooled_by_lesion_not_averaged_by_case():
    strata = {
        "small": (1, 10),
        "medium": (11, 20),
        "large": (21, None),
    }
    nine_detected = {
        "gt_components": [{"size": 1}] * 9,
        "pred_components": [{}] * 9,
        "matches": [
            {"gt_index": index, "dice": 0.8} for index in range(9)
        ],
        "gt_lesions": 9,
        "pred_lesions": 9,
        "fp": 0,
    }
    one_missed = {
        "gt_components": [{"size": 1}],
        "pred_components": [],
        "matches": [],
        "gt_lesions": 1,
        "pred_lesions": 0,
        "fp": 0,
    }

    summary = summarize_stratified_cases(
        [nine_detected, one_missed], strata=strata
    )["small"]

    # A case-wise mean would be (1 + 0) / 2 = 0.5.  Lesion-level pooling is
    # 9 detected lesions / 10 total GT lesions = 0.9.
    assert summary["gt_lesions"] == 10
    assert summary["detected"] == 9
    assert summary["missed"] == 1
    assert summary["lesion_recall"] == 0.9
    assert summary["miss_rate"] == 0.1
    assert summary["matched_lesion_dice"] == 0.8
