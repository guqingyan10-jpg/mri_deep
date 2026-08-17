import importlib.util
import numpy as np
from pathlib import Path
import pytest
import sys
import types


ROOT = Path(__file__).resolve().parents[1]


def _load_advanced_metrics():
    if "torch" not in sys.modules:
        sys.modules["torch"] = types.ModuleType("torch")
    spec = importlib.util.spec_from_file_location(
        "advanced_metrics_for_test", ROOT / "evaluation" / "advanced_metrics.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


advanced_metrics = _load_advanced_metrics()
lesion_wise_detection = advanced_metrics.lesion_wise_detection


def test_one_prediction_cannot_match_multiple_gt_lesions():
    gt = np.zeros((1, 1, 7), dtype=np.uint8)
    gt[0, 0, 1] = 1
    gt[0, 0, 5] = 1

    pred = np.zeros_like(gt)
    pred[0, 0, 1:6] = 1

    result = lesion_wise_detection(pred, gt, min_size=1)

    assert result["gt_lesions"] == 2
    assert result["pred_lesions"] == 1
    assert result["detected"] == 1
    assert result["tp_lesions"] == 1
    assert result["fp_lesions"] == 0
    assert result["fn_lesions"] == 1
    assert result["lesion_precision"] == pytest.approx(1.0)
    assert result["lesion_recall"] == pytest.approx(0.5)
    assert result["lesion_f1"] == pytest.approx(2.0 / 3.0)


def test_multiple_predictions_cannot_match_one_gt_lesion():
    gt = np.zeros((1, 1, 7), dtype=np.uint8)
    gt[0, 0, 2:5] = 1

    pred = np.zeros_like(gt)
    pred[0, 0, 2] = 1
    pred[0, 0, 4] = 1

    result = lesion_wise_detection(pred, gt, min_size=1)

    assert result["tp_lesions"] == 1
    assert result["fp_lesions"] == 1
    assert result["fn_lesions"] == 0
    assert result["lesion_precision"] == pytest.approx(0.5)
    assert result["lesion_recall"] == pytest.approx(1.0)
    assert result["lesion_f1"] == pytest.approx(2.0 / 3.0)


def test_empty_prediction_counts_gt_lesions_as_false_negatives():
    gt = np.zeros((1, 1, 5), dtype=np.uint8)
    gt[0, 0, 2] = 1
    pred = np.zeros_like(gt)

    result = lesion_wise_detection(pred, gt, min_size=1)

    assert result["tp_lesions"] == 0
    assert result["fp_lesions"] == 0
    assert result["fn_lesions"] == 1
    assert result["lesion_precision"] == pytest.approx(0.0)
    assert result["lesion_recall"] == pytest.approx(0.0)
    assert result["lesion_f1"] == pytest.approx(0.0)


def test_predictions_without_gt_are_false_positives():
    gt = np.zeros((1, 1, 7), dtype=np.uint8)
    pred = np.zeros_like(gt)
    pred[0, 0, 1] = 1
    pred[0, 0, 5] = 1

    result = lesion_wise_detection(pred, gt, min_size=1)

    assert result["gt_lesions"] == 0
    assert result["pred_lesions"] == 2
    assert result["tp_lesions"] == 0
    assert result["fp_lesions"] == 2
    assert result["fn_lesions"] == 0
    assert result["lesion_precision"] == pytest.approx(0.0)
    assert np.isnan(result["lesion_recall"])
    assert result["lesion_f1"] == pytest.approx(0.0)


def test_empty_gt_and_prediction_have_undefined_lesion_scores():
    gt = np.zeros((1, 1, 5), dtype=np.uint8)
    pred = np.zeros_like(gt)

    result = lesion_wise_detection(pred, gt, min_size=1)

    assert result["gt_lesions"] == 0
    assert result["pred_lesions"] == 0
    assert result["tp_lesions"] == 0
    assert result["fp_lesions"] == 0
    assert result["fn_lesions"] == 0
    assert np.isnan(result["lesion_precision"])
    assert np.isnan(result["lesion_recall"])
    assert np.isnan(result["lesion_f1"])


def test_min_size_filters_components_before_matching():
    gt = np.zeros((1, 1, 8), dtype=np.uint8)
    gt[0, 0, 1] = 1
    gt[0, 0, 4:6] = 1
    pred = gt.copy()

    result = lesion_wise_detection(pred, gt, min_size=2)

    assert result["gt_lesions"] == 1
    assert result["pred_lesions"] == 1
    assert result["tp_lesions"] == 1
    assert result["fp_lesions"] == 0
    assert result["fn_lesions"] == 0


def test_overlap_threshold_preserves_gt_coverage_semantics():
    gt = np.zeros((1, 1, 6), dtype=np.uint8)
    gt[0, 0, 1:5] = 1
    pred = np.zeros_like(gt)
    pred[0, 0, 1] = 1

    rejected = lesion_wise_detection(
        pred, gt, min_size=1, overlap_thresh=0.25
    )
    accepted = lesion_wise_detection(
        pred, gt, min_size=1, overlap_thresh=0.20
    )

    assert rejected["tp_lesions"] == 0
    assert accepted["tp_lesions"] == 1


def test_summary_uses_tp_fp_fn_and_includes_false_positive_only_cases():
    results = [
        {
            "gt_lesions": 2,
            "pred_lesions": 1,
            "tp_lesions": 1,
            "fp_lesions": 0,
            "fn_lesions": 1,
            "lesion_precision": 1.0,
            "lesion_recall": 0.5,
            "lesion_f1": 2.0 / 3.0,
        },
        {
            "gt_lesions": 0,
            "pred_lesions": 2,
            "tp_lesions": 0,
            "fp_lesions": 2,
            "fn_lesions": 0,
            "lesion_precision": 0.0,
            "lesion_recall": np.nan,
            "lesion_f1": 0.0,
        },
    ]

    summary = advanced_metrics.summarize_lesion_results(results)

    assert summary["total_tp_lesions"] == 1
    assert summary["total_fp_lesions"] == 2
    assert summary["total_fn_lesions"] == 1
    assert summary["overall_lesion_precision"] == pytest.approx(1.0 / 3.0)
    assert summary["overall_lesion_recall"] == pytest.approx(0.5)
    assert summary["overall_lesion_f1"] == pytest.approx(0.4)
    assert summary["mean_lesion_precision"] == pytest.approx(0.5)
    assert summary["mean_lesion_recall"] == pytest.approx(0.5)
    assert summary["mean_lesion_f1"] == pytest.approx(1.0 / 3.0)


def test_evaluation_entrypoints_export_corrected_lesion_metrics():
    advanced_source = (ROOT / "evaluation" / "advanced_metrics.py").read_text(
        encoding="utf-8"
    )
    for key in (
        "Lesion_F1_mean",
        "Total_TP_lesions",
        "Total_FP_lesions",
        "Total_FN_lesions",
        "Overall_lesion_precision",
        "Overall_lesion_recall",
        "Overall_lesion_f1",
    ):
        assert key in advanced_source

    for relative_path in (
        "scripts/eval_all_experiments.py",
        "scripts/eval_comprehensive.py",
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "Lesion_F1_mean" in source
        assert "Overall_lesion_precision" in source
        assert "Overall_lesion_f1" in source
