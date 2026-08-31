import numpy as np
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "wt_lesion_stratified_for_test",
    Path(__file__).resolve().parents[1] / "evaluation" / "wt_lesion_stratified.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

STRATA = _MODULE.STRATA
classify_wt_size = _MODULE.classify_wt_size
match_wt_components = _MODULE.match_wt_components
summarize_stratified_cases = _MODULE.summarize_stratified_cases


def test_size_strata_use_inclusive_integer_boundaries():
    assert classify_wt_size(10) == "small"
    assert classify_wt_size(74) == "small"
    assert classify_wt_size(75) == "medium"
    assert classify_wt_size(64999) == "medium"
    assert classify_wt_size(65000) == "large"


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
    summary = summarize_stratified_cases([result], strata=STRATA)
    assert summary["small"]["gt_lesions"] == 0
    assert np.isnan(summary["small"]["lesion_recall"])
    assert np.isnan(summary["small"]["miss_rate"])
    assert summary["small"]["true_negative_cases"] == 1


def test_prediction_without_gt_counts_as_false_positive():
    gt = np.zeros((1, 1, 5), dtype=np.uint8)
    pred = np.zeros_like(gt)
    pred[0, 0, 2] = 1
    result = match_wt_components(pred, gt, min_component_size=1)
    summary = summarize_stratified_cases([result], strata=STRATA)
    assert summary["all"]["fp"] == 1
