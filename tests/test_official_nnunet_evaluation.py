"""Prediction adapters must preserve the study's frozen evaluation meaning."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from scripts.eval_official_nnunet import (
    evaluate_experiment,
    fit_training_strata,
    label_regions,
    load_experiment,
    load_prediction,
    load_target,
    surface_metrics,
    validate_prediction_inventory,
)


def make_experiment(tmp_path, dataset="brats"):
    shape_xyz = (16, 14, 12)
    affine = np.diag([1.0, 1.5, 2.0, 1.0])
    cases = []
    split = {"train": [], "valid": [], "test": []}
    for number, phase in enumerate(["train", "train", "train", "valid", "test"]):
        case_id = f"case{number:03d}"
        target = np.zeros(shape_xyz, dtype=np.uint8)
        # Three distinct training lesion sizes fit both voxel and volume tertiles.
        target[2:4, 3:8, 2:2 + (number + 1 if number < 3 else 1)] = 3 if dataset == "brats" else 1
        target_path = tmp_path / f"{case_id}_gt.nii.gz"
        nib.save(nib.Nifti1Image(target, affine), target_path)
        cases.append({"case_id": case_id, "split": phase, "patient_group": f"patient{number}",
                      "label_file": str(target_path), "image_files": [],
                      "spacing_dhw": [2.0, 1.5, 1.0], "shape_dhw": list(shape_xyz[::-1])})
        split[phase].append(case_id)
    experiment = {"schema_version": 1, "dataset": dataset,
                  "regions": ["WT", "TC", "ET"] if dataset == "brats" else ["BM"],
                  "threshold": 0.33, "split": split, "cases": cases}
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(experiment), encoding="utf-8")
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    test_case = cases[-1]
    reference = nib.load(test_case["label_file"])
    nib.save(reference, predictions / f"{test_case['case_id']}.nii.gz")
    probability = label_regions(np.asarray(reference.dataobj), dataset).astype(np.float32) * 0.9
    np.savez_compressed(predictions / f"{test_case['case_id']}.npz", probabilities=probability)
    return path, predictions, experiment


def test_region_mapping_preserves_original_brats_tc_semantics():
    labels = np.asarray([0, 1, 2, 3], dtype=np.uint8).reshape(4, 1, 1)
    regions = label_regions(labels, "brats")
    assert regions.shape == (3, 1, 1, 4)
    assert regions[0, 0, 0].tolist() == [False, True, True, True]
    assert regions[1, 0, 0].tolist() == [False, True, False, True]
    assert regions[2, 0, 0].tolist() == [False, False, False, True]
    with pytest.raises(ValueError, match="4->3"):
        label_regions(np.asarray([4]).reshape(1, 1, 1), "brats")


def test_surface_metrics_identical_shifted_and_empty_masks():
    target = np.zeros((12, 13, 14), dtype=bool)
    target[3:7, 3:7, 3:7] = True
    assert surface_metrics(target, target, (1, 1, 1)) == (0.0, 1.0)
    shifted = np.roll(target, 1, axis=0)
    hd, boundary = surface_metrics(shifted, target, (1, 1, 1), tolerance_mm=1.0)
    assert hd == 1.0 and boundary == 1.0
    _, narrow_boundary = surface_metrics(shifted, target, (1, 1, 1), tolerance_mm=0.1)
    assert 0 < narrow_boundary < 1
    empty = np.zeros_like(target)
    assert surface_metrics(empty, empty, (2, 1, 1)) == (0.0, 1.0)
    hd, boundary = surface_metrics(empty, target, (2, 1, 1))
    assert hd == pytest.approx(np.linalg.norm([24, 13, 14]))
    assert boundary == 0.0
    legacy_hd, legacy_boundary = surface_metrics(empty, target, (2, 1, 1), profile="legacy-ucsf")
    assert np.isnan(legacy_hd) and legacy_boundary == 0


def test_probabilities_are_not_sigmoided_again(tmp_path):
    path, predictions, _ = make_experiment(tmp_path)
    experiment = load_experiment(path, allow_nonstandard_cohort=True)
    case = experiment["_case_map"]["case004"]
    target, reference, _ = load_target(case, "brats")
    result = load_prediction(predictions, case, reference, "brats", "probabilities", 0.33)
    assert np.array_equal(result, target)
    assert not result[:, 0, 0, 0].any()  # sigmoid(0) would wrongly mark this foreground.


@pytest.mark.parametrize("error", ["shape", "logit", "nan", "affine"])
def test_malformed_probability_or_geometry_is_rejected(tmp_path, error):
    path, predictions, _ = make_experiment(tmp_path)
    experiment = load_experiment(path, allow_nonstandard_cohort=True)
    case = experiment["_case_map"]["case004"]
    _, reference, _ = load_target(case, "brats")
    archive = predictions / "case004.npz"
    with np.load(archive) as saved:
        probability = saved["probabilities"].copy()
    if error == "shape":
        probability = probability[..., :-1]
    elif error == "logit":
        probability[0, 0, 0, 0] = -0.1
    elif error == "nan":
        probability[0, 0, 0, 0] = np.nan
    else:
        affine = reference.affine.copy()
        affine[0, 3] += 1
        nib.save(nib.Nifti1Image(np.asarray(reference.dataobj), affine), predictions / "case004.nii.gz")
    np.savez_compressed(archive, probabilities=probability)
    with pytest.raises(ValueError):
        load_prediction(predictions, case, reference, "brats", "probabilities", 0.33)


def test_prediction_inventory_cannot_silently_drop_or_add_cases(tmp_path):
    path, predictions, experiment = make_experiment(tmp_path)
    expected = experiment["split"]["test"]
    validate_prediction_inventory(predictions, expected, "probabilities")
    with pytest.raises(ValueError, match="missing"):
        validate_prediction_inventory(predictions, [*expected, "missing_case"], "probabilities")
    np.savez_compressed(predictions / "extra.npz", probabilities=np.zeros((3, 12, 14, 16)))
    with pytest.raises(ValueError, match="extra"):
        validate_prediction_inventory(predictions, expected, "probabilities")


def test_training_strata_do_not_read_test_or_validation_labels(tmp_path):
    path, predictions, _ = make_experiment(tmp_path)
    experiment = load_experiment(path, allow_nonstandard_cohort=True)
    strata, rows = fit_training_strata(experiment, 10)
    assert dict(strata) == {"small": (10, 10), "medium": (11, 20), "large": (21, None)}
    for phase in ("valid", "test"):
        for case_id in experiment["split"][phase]:
            Path(experiment["_case_map"][case_id]["label_file"]).unlink()
    again, again_rows = fit_training_strata(experiment, 10)
    assert strata == again and rows == again_rows


def test_split_and_patient_leakage_are_rejected(tmp_path):
    path, _, experiment = make_experiment(tmp_path)
    experiment["cases"][-1]["patient_group"] = experiment["cases"][0]["patient_group"]
    path.write_text(json.dumps(experiment), encoding="utf-8")
    with pytest.raises(ValueError, match="Patient leakage"):
        load_experiment(path, allow_nonstandard_cohort=True)


@pytest.mark.parametrize("dataset", ["brats", "ucsf"])
def test_synthetic_end_to_end_has_correct_pooled_lesion_counts(tmp_path, dataset):
    path, predictions, _ = make_experiment(tmp_path, dataset)
    result = evaluate_experiment(path, predictions, tmp_path / "evaluation", allow_nonstandard_cohort=True)
    assert result["dice_mean"] == 1
    assert result["hd95_mm_mean"] == 0
    assert result["boundary_dice_mean"] == 1
    assert result["tp_lesions"] == 1
    assert result["fn_lesions"] == result["fp_lesions"] == 0
    assert result["lesion_f1"] == result["lesion_recall"] == 1
    protocol = json.loads((tmp_path / "evaluation/evaluation_protocol.json").read_text())
    assert protocol["probability_transform"].startswith("none")
    assert protocol["std_definition"].startswith("population SD between test cases")
    assert len(protocol["ground_truth_sha256"]) == 5


def test_native_mask_is_separate_and_cannot_overwrite_probability_analysis(tmp_path):
    path, predictions, _ = make_experiment(tmp_path)
    out = tmp_path / "evaluation"
    evaluate_experiment(path, predictions, out, allow_nonstandard_cohort=True)
    with pytest.raises(ValueError, match="different evaluation protocol"):
        evaluate_experiment(path, predictions, out, decision="native-mask", allow_nonstandard_cohort=True)
    result = evaluate_experiment(path, predictions, tmp_path / "native", decision="native-mask", allow_nonstandard_cohort=True)
    assert result["decision"] == "native-mask" and result["dice_mean"] == 1
    protocol = json.loads((tmp_path / "native/evaluation_protocol.json").read_text())
    assert protocol["threshold"] is None


def test_test_threshold_change_is_rejected(tmp_path):
    path, predictions, _ = make_experiment(tmp_path)
    with pytest.raises(ValueError, match="frozen experiment"):
        evaluate_experiment(path, predictions, tmp_path / "out", threshold=0.5, allow_nonstandard_cohort=True)


def test_one_prediction_cannot_count_two_gt_lesions(tmp_path):
    path, predictions, experiment = make_experiment(tmp_path)
    case = experiment["cases"][-1]
    reference = nib.load(case["label_file"])
    target = np.zeros(reference.shape, dtype=np.uint8)
    target[2:4, 2:7, 2:3] = 3
    target[2:4, 2:7, 7:8] = 3
    nib.save(nib.Nifti1Image(target, reference.affine), case["label_file"])
    bridge = target.copy()
    bridge[2:4, 2:7, 3:7] = 3
    nib.save(nib.Nifti1Image(bridge, reference.affine), predictions / "case004.nii.gz")
    probability = label_regions(bridge, "brats").astype(np.float32) * 0.9
    np.savez_compressed(predictions / "case004.npz", probabilities=probability)
    result = evaluate_experiment(path, predictions, tmp_path / "evaluation", allow_nonstandard_cohort=True)
    assert result["tp_lesions"] == 1 and result["fn_lesions"] == 1
    assert result["lesion_recall"] == 0.5


def test_official_ucsf_uses_softmax_foreground_channel_one(tmp_path):
    path, predictions, _ = make_experiment(tmp_path, "ucsf")
    archive = predictions / "case004.npz"
    with np.load(archive) as saved:
        foreground = saved["probabilities"][0].copy()
    np.savez_compressed(archive, probabilities=np.stack([1 - foreground, foreground]))
    result = evaluate_experiment(path, predictions, tmp_path / "official", probability_layout="official",
                                 allow_nonstandard_cohort=True)
    assert result["dice_mean"] == result["lesion_recall"] == 1
    protocol = json.loads((tmp_path / "official/evaluation_protocol.json").read_text())
    assert protocol["probability_layout"]["selected"] == "official binary softmax"
    assert protocol["probability_layout"]["foreground_channel"] == 1
    with pytest.raises(ValueError, match="incompatible"):
        evaluate_experiment(path, predictions, tmp_path / "wrong", probability_layout="study",
                            allow_nonstandard_cohort=True)


def test_official_ucsf_rejects_single_channel_study_layout(tmp_path):
    path, predictions, _ = make_experiment(tmp_path, "ucsf")
    with pytest.raises(ValueError, match="incompatible"):
        evaluate_experiment(path, predictions, tmp_path / "wrong", probability_layout="official",
                            allow_nonstandard_cohort=True)


def test_invalid_binary_softmax_sum_is_rejected(tmp_path):
    path, predictions, _ = make_experiment(tmp_path, "ucsf")
    np.savez_compressed(predictions / "case004.npz", probabilities=np.zeros((2, 12, 14, 16), dtype=np.float32))
    with pytest.raises(ValueError, match="sum to one"):
        evaluate_experiment(path, predictions, tmp_path / "wrong", probability_layout="official",
                            allow_nonstandard_cohort=True)
