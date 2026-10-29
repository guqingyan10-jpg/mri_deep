"""Evaluate official nnU-Net predictions on the frozen study test cases.

The default reads exported *probabilities*, never logits, and applies the
study's prespecified 0.33 threshold. The native integer-mask decision is a
separate, explicitly labelled analysis. Existing model/evaluation entry
points are not changed by this adapter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path.insert(0, str(REPO_ROOT))
# Avoid importing evaluation/__init__.py, which also imports visualization.
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
from wt_lesion_stratified import (  # noqa: E402
    classify_wt_size,
    derive_size_strata,
    match_wt_components,
    summarize_stratified_cases,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_experiment(path: Path, *, allow_nonstandard_cohort: bool = False) -> dict:
    experiment = json.loads(path.read_text(encoding="utf-8"))
    if experiment.get("schema_version") != 1:
        raise ValueError("Unsupported experiment schema_version")
    dataset = experiment.get("dataset")
    expected_regions = {"brats": ["WT", "TC", "ET"], "ucsf": ["BM"]}
    if dataset not in expected_regions or experiment.get("regions") != expected_regions[dataset]:
        raise ValueError("Unexpected dataset or region/channel order")
    splits = experiment.get("split", {})
    if set(splits) != {"train", "valid", "test"}:
        raise ValueError("Expected frozen train/valid/test case lists")
    case_map = {}
    groups = {}
    for case in experiment.get("cases", []):
        case_id = str(case["case_id"])
        if case_id in case_map or not case_id or "/" in case_id or "\\" in case_id:
            raise ValueError(f"Invalid or duplicate case ID: {case_id}")
        split = case["split"]
        if split not in splits:
            raise ValueError(f"Invalid split for {case_id}")
        group = str(case.get("patient_group", case_id))
        if group in groups and groups[group] != split:
            raise ValueError(f"Patient leakage across splits: {group}")
        groups[group] = split
        for key in ("label_file", "native_label_file", "prediction_reference_file"):
            if key not in case:
                continue
            item = Path(case[key])
            case[key] = str(item if item.is_absolute() else (path.parent / item).resolve())
        case_map[case_id] = case
    all_ids = []
    for split, ids in splits.items():
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Empty or duplicate IDs in {split}")
        for case_id in ids:
            if case_id not in case_map or case_map[case_id]["split"] != split:
                raise ValueError(f"Case list/manifest mismatch: {case_id}")
        all_ids.extend(ids)
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(case_map):
        raise ValueError("Frozen splits overlap or do not cover the manifest exactly")
    if not allow_nonstandard_cohort:
        counts = {key: len(ids) for key, ids in splits.items()}
        expected = {"brats": {"train": 257, "valid": 74, "test": 37},
                    "ucsf": {"train": 321, "valid": 94, "test": 46}}[dataset]
        if counts != expected:
            raise ValueError(f"Study cohort counts differ: {counts}; expected {expected}")
    experiment["_case_map"] = case_map
    return experiment


def label_regions(label_xyz: np.ndarray, dataset: str) -> np.ndarray:
    """Convert exported integer labels to region masks in C,Z,Y,X order."""
    label_xyz = np.asarray(label_xyz)
    if label_xyz.ndim != 3 or not np.isfinite(label_xyz).all():
        raise ValueError("Expected a finite three-dimensional integer label map")
    if not np.equal(label_xyz, np.rint(label_xyz)).all():
        raise ValueError("Fractional values found in integer labels")
    allowed = {0, 1, 2, 3} if dataset == "brats" else {0, 1}
    if not set(np.unique(label_xyz).tolist()).issubset(allowed):
        raise ValueError("Unexpected label values; BraTS must already use the exported 4->3 mapping")
    if dataset == "brats":
        regions = [np.isin(label_xyz, [1, 2, 3]), np.isin(label_xyz, [1, 3]), label_xyz == 3]
    else:
        regions = [label_xyz == 1]
    return np.stack([region.transpose(2, 1, 0) for region in regions])


def load_target(case: dict, dataset: str):
    path = Path(case["label_file"])
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D GT: {path}")
    target = label_regions(np.asanyarray(image.dataobj), dataset)
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3][::-1])
    if any(not math.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError(f"Invalid physical spacing: {path}")
    if tuple(case["shape_dhw"]) != target.shape[1:]:
        raise ValueError(f"Manifest/GT shape mismatch for {case['case_id']}")
    if not np.allclose(case["spacing_dhw"], spacing, rtol=1e-5, atol=1e-5):
        raise ValueError(f"Manifest/GT spacing mismatch for {case['case_id']}")
    return target, image, spacing


def validate_prediction_inventory(directory: Path, expected_ids: list[str], decision: str):
    nifti = {path.name[:-7] for path in directory.glob("*.nii.gz")}
    expected = set(expected_ids)
    if nifti != expected:
        raise ValueError(f"Prediction NIfTI inventory differs; missing={sorted(expected-nifti)}, extra={sorted(nifti-expected)}")
    if decision == "probabilities":
        archives = {path.stem for path in directory.glob("*.npz")}
        if archives != expected:
            raise ValueError(f"Probability inventory differs; missing={sorted(expected-archives)}, extra={sorted(archives-expected)}")


def align_to_study_grid(array: np.ndarray, case: dict, alignment: str,
                        *, integer_labels: bool = False) -> np.ndarray:
    """Restore the exact legacy scoring grid, without changing ground truth."""
    if alignment == "identity":
        result = array
    elif alignment == "brats-study-crop":
        result = array[:, 20:120, 40:210, 40:210]
    elif alignment == "ucsf-canonical-study-resize":
        import torch
        import torch.nn.functional as functional
        tensor = torch.from_numpy(np.ascontiguousarray(array)).float()[None]
        kwargs = {"size": tuple(case["shape_dhw"]),
                  "mode": "nearest" if integer_labels else "trilinear"}
        if not integer_labels:
            kwargs["align_corners"] = False
        result = functional.interpolate(tensor, **kwargs)[0].numpy()
    else:
        raise ValueError(f"Unknown prediction alignment: {alignment}")
    if result.shape[1:] != tuple(case["shape_dhw"]):
        raise ValueError(f"Aligned prediction shape differs from study GT: {case['case_id']}")
    return result


def load_prediction(directory: Path, case: dict, reference, dataset: str,
                    decision: str, threshold: float, probability_layout: str = "auto") -> np.ndarray:
    case_id = case["case_id"]
    image = nib.load(str(directory / f"{case_id}.nii.gz"))
    alignment = case.get("prediction_alignment", "identity") if probability_layout != "study" else "identity"
    prediction_reference = (nib.load(case["prediction_reference_file"])
                            if alignment != "identity" else reference)
    if image.shape != prediction_reference.shape or not np.allclose(image.affine, prediction_reference.affine, rtol=1e-5, atol=1e-4):
        raise ValueError(f"Prediction/GT geometry mismatch for {case_id}")
    if decision == "native-mask":
        labels = np.asanyarray(image.dataobj)
        # Validate original integer semantics before nearest-neighbour resize.
        label_regions(labels, dataset)
        aligned = align_to_study_grid(labels.transpose(2, 1, 0)[None],case,alignment,integer_labels=True)
        return label_regions(aligned[0].transpose(2, 1, 0), dataset)
    with np.load(directory / f"{case_id}.npz", allow_pickle=False) as archive:
        if "probabilities" not in archive:
            raise ValueError(f"Missing official 'probabilities' array for {case_id}")
        probability = np.asarray(archive["probabilities"], dtype=np.float32)
    expected_channels = (3,) if dataset == "brats" else ((2,) if probability_layout == "official" else
                                                        (1,) if probability_layout == "study" else (1, 2))
    expected_shape = tuple(case.get("native_shape_dhw", case["shape_dhw"])) if alignment != "identity" else tuple(case["shape_dhw"])
    if (probability.ndim != 4 or probability.shape[0] not in expected_channels or
            probability.shape[1:] != expected_shape):
        raise ValueError(f"Probability shape mismatch for {case_id}: {probability.shape}; "
                         f"expected channels {expected_channels} and DHW {expected_shape}")
    if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
        raise ValueError(f"Expected probabilities in [0,1], not logits: {case_id}")
    if dataset == "ucsf" and probability.shape[0] == 2:
        if not np.allclose(probability.sum(axis=0), 1.0, rtol=1e-4, atol=1e-4):
            raise ValueError(f"Official binary softmax probabilities must sum to one: {case_id}")
        probability = probability[1:2]  # Official binary nnU-Net: background, metastasis.
    # Intentionally no sigmoid: these are nnU-Net's exported probabilities.
    return align_to_study_grid(probability,case,alignment) >= threshold


def resolve_probability_layout(directory: Path, experiment: dict, decision: str, requested: str) -> dict:
    if requested not in {"auto", "official", "study"}:
        raise ValueError("Unknown probability layout")
    if decision == "native-mask":
        return {"requested": requested, "selected": "native integer labels", "channels": None,
                "foreground_channel": None}
    channels = set()
    for case_id in experiment["split"]["test"]:
        with np.load(directory / f"{case_id}.npz", allow_pickle=False) as archive:
            if "probabilities" not in archive:
                raise ValueError(f"Missing official 'probabilities' array for {case_id}")
            shape = archive["probabilities"].shape
            channels.add(shape[0] if len(shape) == 4 else -1)
    if len(channels) != 1:
        raise ValueError("Probability channel layout must be consistent across all test cases")
    channel_count = next(iter(channels))
    if experiment["dataset"] == "brats":
        if channel_count != 3:
            raise ValueError("BraTS requires three WT/TC/ET sigmoid region probabilities")
        selected, foreground_channel = "WT/TC/ET sigmoid regions", None
    elif channel_count == 2 and requested != "study":
        selected, foreground_channel = "official binary softmax", 1
    elif channel_count == 1 and requested != "official":
        selected, foreground_channel = "study single-channel sigmoid", 0
    else:
        raise ValueError(f"Probability shape/channel layout is incompatible with {requested}: {channel_count}")
    return {"requested": requested, "selected": selected, "channels": channel_count,
            "foreground_channel": foreground_channel}


def binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    return 2.0 * int(np.logical_and(prediction, target).sum()) / denominator if denominator else 1.0


def surface_metrics(prediction: np.ndarray, target: np.ndarray, spacing,
                    tolerance_mm: float = 1.0, profile: str = "manuscript") -> tuple[float, float]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Surface metric masks must share a 3D shape")
    if not prediction.any() or not target.any():
        both_empty = not prediction.any() and not target.any()
        boundary = 1.0 if both_empty else 0.0
        if profile == "legacy-ucsf":
            return float("nan"), boundary
        diagonal = float(np.linalg.norm(np.asarray(target.shape, dtype=float) * np.asarray(spacing)))
        return (0.0 if both_empty else diagonal), boundary
    structure = ndimage.generate_binary_structure(3, 1)
    pred_surface = prediction & ~ndimage.binary_erosion(prediction, structure=structure, border_value=0)
    gt_surface = target & ~ndimage.binary_erosion(target, structure=structure, border_value=0)
    to_gt = ndimage.distance_transform_edt(~gt_surface, sampling=spacing)[pred_surface]
    to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)[gt_surface]
    boundary = float((np.count_nonzero(to_gt <= tolerance_mm) + np.count_nonzero(to_pred <= tolerance_mm)) /
                     (len(to_gt) + len(to_pred)))
    if profile == "legacy-ucsf":
        from advanced_metrics import hd95_single
        return float(hd95_single(prediction, target, voxel_spacing=spacing)), boundary
    return float(np.percentile(np.concatenate([to_gt, to_pred]), 95)), boundary


def fit_training_strata(experiment: dict, min_size: int):
    rows = []
    dataset = experiment["dataset"]
    for case_id in experiment["split"]["train"]:
        case = experiment["_case_map"][case_id]
        regions, _, spacing = load_target(case, dataset)
        target = regions[-1]
        matching = match_wt_components(np.zeros_like(target), target, min_component_size=min_size)
        for index, component in enumerate(matching["gt_components"]):
            rows.append({"case_id": case_id, "gt_component_index": index,
                         "gt_voxels": component["size"],
                         "gt_volume_mm3": component["size"] * float(np.prod(spacing))})
    if dataset == "brats":
        strata = derive_size_strata([row["gt_voxels"] for row in rows], min_size)
        for row in rows:
            row["stratum"] = classify_wt_size(row["gt_voxels"], strata)
    else:
        # Reuse the study's existing physical-volume thresholds unchanged.
        from scripts.eval_ucsf_baseline_full import classify_volume, derive_volume_strata
        strata = derive_volume_strata([row["gt_volume_mm3"] for row in rows])
        for row in rows:
            row["stratum"] = classify_volume(row["gt_volume_mm3"], strata)
    return strata, rows


def _mean_std(values):
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    return ((float(finite.mean()), float(finite.std(ddof=0))) if finite.size else (float("nan"), float("nan")))


def write_csv(path: Path, rows: list[dict], fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_lesions(case_rows, lesion_rows) -> list[dict]:
    output = []
    for stratum in ("small", "medium", "large", "all"):
        subset = [row for row in lesion_rows if stratum == "all" or row["stratum"] == stratum]
        detected = [row for row in subset if row["detected"]]
        tp = len(detected)
        fn = len(subset) - tp
        fp = sum(row["fp_lesions"] for row in case_rows) if stratum == "all" else None
        output.append({"stratum": stratum, "gt_lesions": len(subset), "detected_lesions": tp,
                       "missed_lesions": fn, "lesion_recall": tp / len(subset) if subset else float("nan"),
                       "matched_lesion_dice": _mean_std([row["matched_dice"] for row in detected])[0],
                       "gt_anchored_lesion_dice": _mean_std([row["gt_anchored_dice"] for row in subset])[0],
                       "fp_lesions": fp,
                       "lesion_precision": tp / (tp + fp) if fp is not None and tp + fp else float("nan"),
                       "lesion_f1": 2 * tp / (2 * tp + fp + fn) if fp is not None and 2 * tp + fp + fn else float("nan")})
    return output


def evaluate_experiment(experiment_path: Path, predictions_dir: Path, output_dir: Path,
                        *, threshold: float = 0.33, decision: str = "probabilities",
                        metric_profile: str = "manuscript", min_component_size: int = 10,
                        boundary_tolerance_mm: float = 1.0, allow_nonstandard_cohort: bool = False,
                        model_name: str = "nnU-Net v2 official pipeline", probability_layout: str = "auto") -> dict:
    if not 0 < threshold < 1 or min_component_size < 1 or boundary_tolerance_mm <= 0:
        raise ValueError("Invalid threshold, component size, or surface tolerance")
    if decision not in {"probabilities", "native-mask"} or metric_profile not in {"manuscript", "legacy-ucsf"}:
        raise ValueError("Invalid decision or metric profile")
    experiment = load_experiment(experiment_path, allow_nonstandard_cohort=allow_nonstandard_cohort)
    if metric_profile == "legacy-ucsf" and experiment["dataset"] != "ucsf":
        raise ValueError("legacy-ucsf applies only to historical UCSF results")
    frozen_threshold = float(experiment.get("threshold", 0.33))
    if decision == "probabilities" and threshold != frozen_threshold:
        raise ValueError(f"Threshold differs from frozen experiment value {frozen_threshold}; test-set tuning is not allowed")
    validate_prediction_inventory(predictions_dir, experiment["split"]["test"], decision)
    layout = resolve_probability_layout(predictions_dir, experiment, decision, probability_layout)
    hashes = {case["case_id"]: file_sha256(Path(case["label_file"])) for case in experiment["cases"]}
    strata, training_rows = fit_training_strata(experiment, min_component_size)
    if experiment["dataset"] == "brats" and not allow_nonstandard_cohort:
        expected = {"small": (10, 44), "medium": (45, 4678), "large": (4679, None)}
        if dict(strata) != expected:
            raise ValueError(f"Training-derived ET strata differ from frozen study: {dict(strata)}")
    protocol = {"schema_version": 1, "model": model_name, "dataset": experiment["dataset"],
                "experiment_sha256": file_sha256(experiment_path),
                "split_sha256": canonical_hash(experiment["split"]),
                "ground_truth_sha256": hashes, "decision": decision,
                "threshold": threshold if decision == "probabilities" else None,
                "native_mask_note": ("Official BraTS region decisions use threshold 0.5 and ordered label overwrite; "
                                     "official binary UCSF uses softmax/argmax; neither is the study 0.33 decision") if decision == "native-mask" else None,
                "metric_profile": metric_profile, "min_component_size": min_component_size,
                "allow_nonstandard_cohort": allow_nonstandard_cohort,
                "boundary_tolerance_mm": boundary_tolerance_mm,
                "lesion_region": experiment["regions"][-1], "lesion_size_strata": strata,
                "strata_units": "voxels" if experiment["dataset"] == "brats" else "mm3",
                "std_definition": "population SD between test cases, not SD between random seeds",
                "empty_hd95": "both empty 0; one empty norm(shape_dhw * spacing_dhw)" if metric_profile == "manuscript" else "NaN if either empty; finite cases averaged only",
                "hd95_definition": "95th percentile of pooled bidirectional surface distances" if metric_profile == "manuscript" else "maximum of two directional 95th percentiles",
                "probability_transform": "none; official exported probabilities are thresholded directly",
                "probability_layout": layout,
                "threshold_selection": "frozen before evaluating the test split",
                "predictions_dir": str(predictions_dir.resolve())}
    protocol["training_input_space"] = experiment.get("input_space", "study")
    protocol["evaluation_space"] = experiment.get("evaluation_space", "exported study GT grid")
    protocol["alignment"] = ("identity" if probability_layout == "study" else
                             sorted({case.get("prediction_alignment", "identity") for case in experiment["cases"]}))
    protocol["prediction_sha256"] = {case_id: {
        suffix: file_sha256(predictions_dir/f"{case_id}{suffix}")
        for suffix in ([".nii.gz",".npz"] if decision == "probabilities" else [".nii.gz"])}
        for case_id in experiment["split"]["test"]}
    protocol_file = output_dir / "evaluation_protocol.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not protocol_file.exists():
            raise ValueError("Refusing to write into an unrelated/nonempty output directory")
        prior = json.loads(protocol_file.read_text(encoding="utf-8"))
        if prior != json_safe(protocol):
            raise ValueError("Output directory belongs to a different evaluation protocol; choose another directory")
    case_rows, lesion_rows = [], []
    for case_id in experiment["split"]["test"]:
        case = experiment["_case_map"][case_id]
        target, reference, spacing = load_target(case, experiment["dataset"])
        prediction = load_prediction(predictions_dir, case, reference, experiment["dataset"], decision, threshold, probability_layout)
        values = [binary_dice(p, t) for p, t in zip(prediction, target)]
        hd95, boundary = surface_metrics(prediction[-1], target[-1], spacing, boundary_tolerance_mm, metric_profile)
        matching = match_wt_components(prediction[-1], target[-1], min_component_size=min_component_size)
        row = {"model": model_name, "case_id": case_id, "subject_id": case_id,
               "patient_group": case.get("patient_group", case_id), "decision": decision,
               "threshold": threshold if decision == "probabilities" else None,
               **{f"{region}_Dice": value for region, value in zip(experiment["regions"], values)},
               "dice": float(np.mean(values)), "hd95_mm": hd95, "boundary_dice": boundary,
               "gt_empty": not bool(target[-1].any()), "pred_empty": not bool(prediction[-1].any()),
               "gt_lesions": matching["gt_lesions"], "pred_lesions": matching["pred_lesions"],
               "tp_lesions": matching["tp"], "fp_lesions": matching["fp"], "fn_lesions": matching["fn"]}
        if experiment["dataset"] == "brats":
            row["Macro_Dice"] = row["dice"]
        case_rows.append(row)
        matches = {pair["gt_index"]: pair for pair in matching["matches"]}
        if experiment["dataset"] == "brats":
            # Invoke the existing aggregation as a per-case consistency check.
            reused_summary = summarize_stratified_cases([matching], strata)["all"]
            if reused_summary["detected"] != matching["tp"]:
                raise RuntimeError("Lesion matching/summary inconsistency")
        for index, component in enumerate(matching["gt_components"]):
            matched = matches.get(index)
            volume = component["size"] * float(np.prod(spacing))
            if experiment["dataset"] == "brats":
                stratum = classify_wt_size(component["size"], strata)
            else:
                from scripts.eval_ucsf_baseline_full import classify_volume
                stratum = classify_volume(volume, strata)
            lesion_rows.append({"model": model_name, "case_id": case_id, "subject_id": case_id,
                                "gt_component_index": index, "gt_voxels": component["size"],
                                "gt_volume_mm3": volume, "stratum": stratum, "volume_stratum": stratum,
                                "detected": matched is not None,
                                "matched_dice": float(matched["dice"]) if matched else float("nan"),
                                "gt_anchored_dice": float(matched["dice"]) if matched else 0.0})
        print(f"Evaluated {len(case_rows)}/{len(experiment['split']['test'])}: {case_id}", flush=True)
    lesion_summary = summarize_lesions(case_rows, lesion_rows)
    if experiment["dataset"] == "brats" and not allow_nonstandard_cohort:
        if lesion_summary[0]["gt_lesions"] != 31 or lesion_summary[1]["gt_lesions"] != 36:
            raise ValueError("Test ET lesion identities/counts differ from the frozen 31 small/36 medium study lesions")
    overall = lesion_summary[-1]
    summary = {"model": model_name, "dataset": experiment["dataset"], "decision": decision,
               "metric_profile": metric_profile, "n_test_scans": len(case_rows),
               "dice_mean": _mean_std([row["dice"] for row in case_rows])[0],
               "dice_std": _mean_std([row["dice"] for row in case_rows])[1],
               "hd95_mm_mean": _mean_std([row["hd95_mm"] for row in case_rows])[0],
               "hd95_mm_std": _mean_std([row["hd95_mm"] for row in case_rows])[1],
               "hd95_finite_cases": sum(math.isfinite(row["hd95_mm"]) for row in case_rows),
               "boundary_dice_mean": _mean_std([row["boundary_dice"] for row in case_rows])[0],
               "boundary_dice_std": _mean_std([row["boundary_dice"] for row in case_rows])[1],
               "gt_empty_cases": sum(row["gt_empty"] for row in case_rows),
               "pred_empty_cases": sum(row["pred_empty"] for row in case_rows),
               "tp_lesions": overall["detected_lesions"], "fp_lesions": overall["fp_lesions"],
               "fn_lesions": overall["missed_lesions"], "lesion_recall": overall["lesion_recall"],
               "lesion_precision": overall["lesion_precision"], "lesion_f1": overall["lesion_f1"],
               "lesion_gt_anchored_dice": overall["gt_anchored_lesion_dice"],
               "small_lesion_gt_anchored_dice": lesion_summary[0]["gt_anchored_lesion_dice"]}
    if experiment["dataset"] == "brats":
        summary["Macro_Dice_mean"] = summary["dice_mean"]
        for region in experiment["regions"]:
            mean, sd = _mean_std([row[f"{region}_Dice"] for row in case_rows])
            summary[f"{region}_Dice_mean"], summary[f"{region}_Dice_std"] = mean, sd
        summary["ET_HD95_mean"] = summary["hd95_mm_mean"]
        summary["Boundary_Dice_mean"] = summary["boundary_dice_mean"]
        summary["Lesion_F1"] = summary["lesion_f1"]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_case.csv", case_rows)
    write_csv(output_dir / "per_gt_lesion.csv", lesion_rows, list(lesion_rows[0]) if lesion_rows else
              ["model", "case_id", "subject_id", "gt_component_index", "gt_voxels", "gt_volume_mm3", "stratum", "volume_stratum", "detected", "matched_dice", "gt_anchored_dice"])
    write_csv(output_dir / "summary_by_lesion_size.csv", lesion_summary)
    write_csv(output_dir / "training_lesion_size_distribution.csv", training_rows)
    write_csv(output_dir / "summary.csv", [summary])
    (output_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    protocol_file.write_text(json.dumps(json_safe(protocol), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(json_safe(summary), indent=2, allow_nan=False))
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.33)
    parser.add_argument("--decision", choices=("probabilities", "native-mask"), default="probabilities")
    parser.add_argument("--metric-profile", choices=("manuscript", "legacy-ucsf"), default="manuscript")
    parser.add_argument("--min-component-size", type=int, default=10)
    parser.add_argument("--boundary-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--allow-nonstandard-cohort", action="store_true", help="Explicitly permit synthetic/other cohorts; not the manuscript comparison")
    parser.add_argument("--model-name", default="nnU-Net v2 official pipeline")
    parser.add_argument("--probability-layout", choices=("auto", "official", "study"), default="auto",
                        help="Official UCSF has background/foreground softmax channels; study models have one sigmoid channel")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    evaluate_experiment(args.experiment, args.predictions_dir, args.output_dir,
                        threshold=args.threshold, decision=args.decision, metric_profile=args.metric_profile,
                        min_component_size=args.min_component_size, boundary_tolerance_mm=args.boundary_tolerance_mm,
                        allow_nonstandard_cohort=args.allow_nonstandard_cohort, model_name=args.model_name,
                        probability_layout=args.probability_layout)


if __name__ == "__main__":
    main()
