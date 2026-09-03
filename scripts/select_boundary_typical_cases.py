"""Automatically select and plot four representative ET boundary cases.

The script evaluates the fixed 37-case test split with the main-experiment
Baseline, LHFC, and Full AFBMS checkpoints.  It selects the individual small
GT lesion, matched by both models, with the largest Full-vs-Baseline
matched-lesion Dice gain, two boundary improvements, and one regression using
deterministic metric rules.  The small-lesion row uses filled TP/FN/FP regions
because its endpoint is Dice; the boundary rows use solid contour comparisons.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy import ndimage
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import BratsDataset, get_dataloader
from evaluation.advanced_metrics import hd95_single, nsd_single
from evaluation.wt_lesion_stratified import (
    classify_lesion_size,
    match_lesion_components,
    summarize_stratified_cases,
)
from models.resunet3d import ResUNet3d
from models.resunet_edge import ResUNetEdge
from models.resunet_hf_concat_boundary import ResUNetHFConcatBoundary


MODEL_SPECS = (
    {
        "key": "baseline",
        "label": "Baseline",
        "checkpoint_dir": "/root/autodl-tmp/ResUNet_model",
        "model_class": ResUNet3d,
        "model_kwargs": {
            "in_channels": 4,
            "n_classes": 3,
            "n_channels": 24,
        },
        "key_remap": None,
    },
    {
        "key": "lhfc",
        "label": "LHFC",
        "checkpoint_dir": (
            "/root/autodl-tmp/ResUNet_Edge_concat_laplacian_model"
        ),
        "model_class": ResUNetEdge,
        "model_kwargs": {
            "in_channels": 4,
            "n_classes": 3,
            "n_channels": 24,
            "fusion": "concat",
            "edge_type": "laplacian",
        },
        "key_remap": "edge",
    },
    {
        "key": "full",
        "label": "Full",
        "checkpoint_dir": (
            "/root/autodl-tmp/"
            "ResUNet_HFConcatBoundary_w0.1_multiscale_v2_model"
        ),
        "model_class": ResUNetHFConcatBoundary,
        "model_kwargs": {
            "in_channels": 4,
            "n_classes": 3,
            "n_channels": 24,
            "fusion": "concat",
            "multiscale_context_v2": True,
        },
        "key_remap": None,
    },
)
MODEL_KEYS = tuple(spec["key"] for spec in MODEL_SPECS)
ROLE_ORDER = (
    "small_lesion_improvement",
    "hd95_improvement",
    "boundary_dice_improvement",
    "regression",
)
ROLE_LABELS = {
    "small_lesion_improvement": "Small-lesion Dice improvement",
    "hd95_improvement": "HD95 improvement",
    "boundary_dice_improvement": "Boundary Dice improvement",
    "regression": "Regression",
}
STRUCTURE_26 = ndimage.generate_binary_structure(3, 3)


def load_et_strata(path: Path, min_component_size: int):
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run scripts/derive_train_lesion_strata.py "
            "--region BOTH first"
        )
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("region") != "ET":
        raise ValueError("strata JSON must describe ET")
    if metadata.get("fit_split") != "train":
        raise ValueError("ET strata must be fitted on the training split")
    if metadata.get("connectivity") != 26:
        raise ValueError("ET strata must use 26-connectivity")
    if metadata.get("min_component_size") != min_component_size:
        raise ValueError("ET strata minimum component size does not match CLI")
    strata = OrderedDict(
        (name, tuple(metadata["strata"][name]))
        for name in ("small", "medium", "large")
    )
    return strata, metadata


def _checkpoint_epoch(path: Path) -> int:
    match = re.search(r"_(\d+)\.pth$", path.name)
    return int(match.group(1)) if match else -1


def resolve_best_checkpoint(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_file():
        if not path.name.startswith("best_model_"):
            raise ValueError(f"formal evaluation requires best_model_*.pth: {path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint path does not exist: {path}")
    candidates = list(path.glob("best_model_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"no best_model_*.pth found in {path}")
    return max(candidates, key=_checkpoint_epoch)


def _torch_load(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def load_model(spec, checkpoint: Path, device):
    model = spec["model_class"](**spec["model_kwargs"]).to(device)
    state = _torch_load(checkpoint)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint does not contain a state dictionary: {checkpoint}")
    cleaned = {}
    for key, value in state.items():
        key = key[7:] if key.startswith("module.") else key
        key = key.replace("out.conv.0.", "out.conv.")
        if spec.get("key_remap") == "edge" and key.startswith("sobel."):
            key = key.replace("sobel.", "edge_extractor.", 1)
        cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print(
            f"[WARN] {spec['label']} checkpoint compatibility: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    model.eval()
    return model


def _case_id(data) -> str:
    value = data.get("Id", "unknown")
    if isinstance(value, (list, tuple)):
        return str(value[0])
    return str(value)


def _predict_et(model, images, threshold: float) -> np.ndarray:
    logits = model(images)
    if isinstance(logits, tuple):
        logits = logits[0]
    return (
        (torch.sigmoid(logits)[0, 2] >= threshold)
        .detach()
        .cpu()
        .numpy()
        .astype(bool)
    )


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    intersection = int(np.logical_and(prediction, target).sum())
    denominator = int(prediction.sum()) + int(target.sum())
    return 2.0 * intersection / denominator if denominator else 1.0


def _small_metrics(match_result, strata):
    summary = summarize_stratified_cases([match_result], strata)["small"]
    return {
        "small_gt_lesions": int(summary["gt_lesions"]),
        "small_detected_lesions": int(summary["detected"]),
        "small_missed_lesions": int(summary["missed"]),
        "small_gt_anchored_dice": float(summary["gt_anchored_lesion_dice"]),
    }


def evaluate_test_cases(
    models,
    dataloader,
    device,
    threshold,
    min_component_size,
    strata,
    boundary_tolerance_mm,
    voxel_spacing,
):
    rows = []
    lesion_rows = []
    with torch.inference_mode():
        for data in tqdm(dataloader, desc="Screen ET boundary cases"):
            case_id = _case_id(data)
            images = data["image"].to(device)
            gt_et = (data["mask"][0, 2].cpu().numpy() > 0)
            for model_key in MODEL_KEYS:
                pred_et = _predict_et(models[model_key], images, threshold)
                match_result = match_lesion_components(
                    pred_et,
                    gt_et,
                    min_component_size=min_component_size,
                )
                matches_by_gt = {
                    match["gt_index"]: match
                    for match in match_result["matches"]
                }
                for gt_index, component in enumerate(
                    match_result["gt_components"]
                ):
                    if classify_lesion_size(component["size"], strata) != "small":
                        continue
                    match = matches_by_gt.get(gt_index)
                    lesion_rows.append(
                        {
                            "case_id": case_id,
                            "gt_index": int(gt_index),
                            "gt_id": int(component["id"]),
                            "gt_size": int(component["size"]),
                            "model_key": model_key,
                            "detected": match is not None,
                            "pred_index": (
                                int(match["pred_index"])
                                if match is not None
                                else -1
                            ),
                            "pred_size": (
                                int(match["pred_size"])
                                if match is not None
                                else 0
                            ),
                            # A missed GT lesion receives zero.  This makes
                            # the per-lesion comparison GT-anchored and avoids
                            # selecting only among lesions already detected.
                            "lesion_dice": (
                                float(match["dice"])
                                if match is not None
                                else 0.0
                            ),
                        }
                    )
                small = _small_metrics(match_result, strata)
                boundary_valid = bool(gt_et.any() and pred_et.any())
                hd95 = (
                    hd95_single(pred_et, gt_et, voxel_spacing=voxel_spacing)
                    if boundary_valid
                    else float("nan")
                )
                boundary_dice = (
                    nsd_single(
                        pred_et,
                        gt_et,
                        tau=boundary_tolerance_mm,
                        voxel_spacing=voxel_spacing,
                    )
                    if boundary_valid
                    else float("nan")
                )
                rows.append(
                    {
                        "case_id": case_id,
                        "model_key": model_key,
                        "model": next(
                            spec["label"]
                            for spec in MODEL_SPECS
                            if spec["key"] == model_key
                        ),
                        "gt_et_voxels": int(gt_et.sum()),
                        "pred_et_voxels": int(pred_et.sum()),
                        "et_dice": _dice(pred_et, gt_et),
                        "hd95_mm": float(hd95),
                        "boundary_dice": float(boundary_dice),
                        "boundary_valid": boundary_valid,
                        **small,
                    }
                )
            del images
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("test dataloader produced no cases")
    return frame, pd.DataFrame(lesion_rows)


def build_small_lesion_comparison(per_lesion: pd.DataFrame) -> pd.DataFrame:
    """Place each individual small GT lesion's three model results in one row."""
    if per_lesion.empty:
        raise ValueError("the test split contains no eligible small ET lesions")
    identity = ["case_id", "gt_index", "gt_id", "gt_size"]
    counts = per_lesion.groupby(identity)["model_key"].nunique()
    if not (counts == len(MODEL_KEYS)).all():
        raise ValueError("every small GT lesion must have all three model results")

    output = per_lesion[identity].drop_duplicates().set_index(identity)
    for model_key in MODEL_KEYS:
        model_rows = per_lesion[per_lesion["model_key"] == model_key].set_index(
            identity
        )
        for column in ("detected", "pred_index", "pred_size", "lesion_dice"):
            output[f"{model_key}_{column}"] = model_rows[column]
    output = output.reset_index()
    output["small_lesion_dice_gain"] = (
        output["full_lesion_dice"] - output["baseline_lesion_dice"]
    )
    output["small_lesion_dice_gain_vs_lhfc"] = (
        output["full_lesion_dice"] - output["lhfc_lesion_dice"]
    )
    return output


def build_case_comparison(per_case: pd.DataFrame) -> pd.DataFrame:
    required_models = set(MODEL_KEYS)
    observed = set(per_case["model_key"].unique())
    if observed != required_models:
        raise ValueError(f"expected models {required_models}, got {observed}")
    counts = per_case.groupby("case_id")["model_key"].nunique()
    if not (counts == len(MODEL_KEYS)).all():
        raise ValueError("every test case must have all three model results")

    metadata = (
        per_case.sort_values("model_key")
        .groupby("case_id", as_index=False)
        .first()[["case_id", "gt_et_voxels", "small_gt_lesions"]]
    )
    value_columns = (
        "pred_et_voxels",
        "et_dice",
        "hd95_mm",
        "boundary_dice",
        "boundary_valid",
        "small_detected_lesions",
        "small_missed_lesions",
        "small_gt_anchored_dice",
    )
    output = metadata.set_index("case_id")
    for model_key in MODEL_KEYS:
        model_rows = per_case[per_case["model_key"] == model_key].set_index(
            "case_id"
        )
        for column in value_columns:
            output[f"{model_key}_{column}"] = model_rows[column]
    output = output.reset_index()

    output["small_detection_gain"] = (
        output["full_small_detected_lesions"]
        - output["baseline_small_detected_lesions"]
    )
    output["small_dice_gain"] = (
        output["full_small_gt_anchored_dice"]
        - output["baseline_small_gt_anchored_dice"]
    )
    output["et_dice_gain"] = output["full_et_dice"] - output["baseline_et_dice"]
    output["hd95_improvement"] = (
        output["baseline_hd95_mm"] - output["full_hd95_mm"]
    )
    output["boundary_dice_improvement"] = (
        output["full_boundary_dice"] - output["baseline_boundary_dice"]
    )
    boundary_columns = [
        f"{key}_{metric}"
        for key in MODEL_KEYS
        for metric in ("hd95_mm", "boundary_dice")
    ]
    output["all_models_boundary_valid"] = np.isfinite(
        output[boundary_columns].to_numpy(dtype=float)
    ).all(axis=1)

    valid = output["all_models_boundary_valid"]
    hd_worsening = -output["hd95_improvement"]
    bd_worsening = -output["boundary_dice_improvement"]

    def standardize(series):
        values = series[valid].astype(float)
        scale = float(values.std(ddof=0))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        center = float(values.mean()) if len(values) else 0.0
        return (series.astype(float) - center) / scale

    output["regression_score"] = standardize(hd_worsening) + standardize(
        bd_worsening
    )
    return output


def _choose_row(pool, sort_columns, ascending):
    if pool.empty:
        raise ValueError("not enough eligible cases for deterministic selection")
    return pool.sort_values(
        list(sort_columns) + ["case_id"],
        ascending=list(ascending) + [True],
        kind="mergesort",
    ).iloc[0]


def select_typical_cases(
    comparison: pd.DataFrame,
    small_lesions: pd.DataFrame,
) -> pd.DataFrame:
    """Select four unique cases with deterministic, auditable rules."""
    selected = []
    used = set()

    small_pool = small_lesions[
        small_lesions["baseline_detected"]
        & small_lesions["full_detected"]
        & (small_lesions["small_lesion_dice_gain"] > 0)
    ]
    if small_pool.empty:
        raise ValueError(
            "no small ET lesion matched by both Baseline and Full has a "
            "higher Full matched-lesion Dice"
        )
    small = _choose_row(
        small_pool,
        ("small_lesion_dice_gain", "full_lesion_dice", "gt_size"),
        (False, False, True),
    )
    small_case = comparison[comparison["case_id"] == small["case_id"]].iloc[0]
    selected.append(
        {
            **small_case.to_dict(),
            **small.to_dict(),
            "selection_role": "small_lesion_improvement",
            "selection_strict": True,
            "selection_reason": (
                "largest matched small-lesion Dice gain over Baseline "
                "among lesions detected by both models"
            ),
        }
    )
    used.add(small["case_id"])

    boundary_all = comparison[
        comparison["all_models_boundary_valid"]
        & ~comparison["case_id"].isin(used)
    ]
    boundary_strict = boundary_all[
        (boundary_all["hd95_improvement"] > 0)
        & (boundary_all["boundary_dice_improvement"] > 0)
    ]

    hd_target = boundary_strict
    if hd_target.empty:
        hd_target = boundary_all[boundary_all["hd95_improvement"] > 0]
    if hd_target.empty:
        hd_target = boundary_all
    hd_case = _choose_row(
        hd_target,
        ("hd95_improvement", "boundary_dice_improvement"),
        (False, False),
    )
    selected.append(
        {
            **hd_case.to_dict(),
            "selection_role": "hd95_improvement",
            "selection_strict": (
                hd_case["hd95_improvement"] > 0
                and hd_case["boundary_dice_improvement"] > 0
            ),
            "selection_reason": "largest Full-vs-Baseline HD95 reduction",
        }
    )
    used.add(hd_case["case_id"])

    bd_all = comparison[
        comparison["all_models_boundary_valid"]
        & ~comparison["case_id"].isin(used)
    ]
    bd_strict = bd_all[
        (bd_all["hd95_improvement"] > 0)
        & (bd_all["boundary_dice_improvement"] > 0)
    ]
    bd_target = bd_strict
    if bd_target.empty:
        bd_target = bd_all[bd_all["boundary_dice_improvement"] > 0]
    if bd_target.empty:
        bd_target = bd_all
    bd_case = _choose_row(
        bd_target,
        ("boundary_dice_improvement", "hd95_improvement"),
        (False, False),
    )
    selected.append(
        {
            **bd_case.to_dict(),
            "selection_role": "boundary_dice_improvement",
            "selection_strict": (
                bd_case["hd95_improvement"] > 0
                and bd_case["boundary_dice_improvement"] > 0
            ),
            "selection_reason": (
                "largest Full-vs-Baseline Boundary Dice increase"
            ),
        }
    )
    used.add(bd_case["case_id"])

    regression_all = comparison[
        comparison["all_models_boundary_valid"]
        & ~comparison["case_id"].isin(used)
    ]
    regression_strict = regression_all[
        (regression_all["hd95_improvement"] < 0)
        & (regression_all["boundary_dice_improvement"] < 0)
    ]
    regression_pool = (
        regression_strict if not regression_strict.empty else regression_all
    )
    regression = _choose_row(
        regression_pool,
        ("regression_score",),
        (False,),
    )
    selected.append(
        {
            **regression.to_dict(),
            "selection_role": "regression",
            "selection_strict": not regression_strict.empty,
            "selection_reason": (
                "largest standardized HD95/Boundary Dice deterioration"
            ),
        }
    )

    result = pd.DataFrame(selected)
    if result["case_id"].duplicated().any() or len(result) != 4:
        raise AssertionError("case selection must return four unique cases")
    result["selection_role"] = pd.Categorical(
        result["selection_role"], categories=ROLE_ORDER, ordered=True
    )
    return result.sort_values("selection_role").reset_index(drop=True)


def _focus_masks_for_case(
    gt_et,
    predictions,
    role,
    min_component_size,
    selected_gt_index=None,
):
    """Return one GT lesion and each model's complete matched component."""
    match_results = {
        key: match_lesion_components(
            predictions[key], gt_et, min_component_size=min_component_size
        )
        for key in MODEL_KEYS
    }
    gt_components = match_results["full"]["gt_components"]
    if not gt_components:
        empty = np.zeros_like(gt_et, dtype=bool)
        return gt_et.astype(bool), {key: empty.copy() for key in MODEL_KEYS}

    if role == "small_lesion_improvement":
        if selected_gt_index is None:
            raise ValueError("selected small lesion is missing its GT index")
        gt_index = int(selected_gt_index)
        if gt_index < 0 or gt_index >= len(gt_components):
            raise IndexError(f"selected GT lesion index out of range: {gt_index}")
    else:
        gt_index = max(
            range(len(gt_components)),
            key=lambda index: gt_components[index]["size"],
        )

    focus_gt = gt_components[gt_index]["mask"]
    focus_predictions = {}
    for key, result in match_results.items():
        match = next(
            (
                item
                for item in result["matches"]
                if item["gt_index"] == gt_index
            ),
            None,
        )
        focus_predictions[key] = (
            result["pred_components"][match["pred_index"]]["mask"]
            if match is not None
            else np.zeros_like(gt_et, dtype=bool)
        )
    return focus_gt, focus_predictions


def _crop_bounds(mask_2d, margin=16, min_size=64):
    height, width = mask_2d.shape
    coords = np.argwhere(mask_2d)
    if not len(coords):
        return 0, height, 0, width
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    y0, y1 = max(0, y0 - margin), min(height, y1 + margin)
    x0, x1 = max(0, x0 - margin), min(width, x1 + margin)

    def expand(start, end, limit):
        needed = max(0, min_size - (end - start))
        start -= needed // 2
        end += needed - needed // 2
        if start < 0:
            end = min(limit, end - start)
            start = 0
        if end > limit:
            start = max(0, start - (end - limit))
            end = limit
        return int(start), int(end)

    y0, y1 = expand(y0, y1, height)
    x0, x1 = expand(x0, x1, width)
    return y0, y1, x0, x1


def collect_selected_visuals(
    dataloader,
    models,
    selected,
    device,
    threshold,
    min_component_size,
    strata,
    crop_margin,
    minimum_crop_size,
):
    selected_by_case = selected.set_index("case_id")
    role_by_case = dict(zip(selected["case_id"], selected["selection_role"].astype(str)))
    wanted = set(role_by_case)
    visuals = {}
    with torch.inference_mode():
        for data in tqdm(dataloader, desc="Render selected cases"):
            case_id = _case_id(data)
            if case_id not in wanted:
                continue
            images = data["image"].to(device)
            gt_et = data["mask"][0, 2].cpu().numpy() > 0
            predictions = {
                key: _predict_et(models[key], images, threshold)
                for key in MODEL_KEYS
            }
            role = role_by_case[case_id]
            selected_gt_index = (
                selected_by_case.loc[case_id].get("gt_index")
                if role == "small_lesion_improvement"
                else None
            )
            focus, focus_predictions = _focus_masks_for_case(
                gt_et,
                predictions,
                role,
                min_component_size,
                selected_gt_index,
            )
            slice_sums = focus.reshape(focus.shape[0], -1).sum(axis=1)
            z_index = (
                int(np.argmax(slice_sums))
                if slice_sums.max() > 0
                else int(np.argmax(gt_et.reshape(gt_et.shape[0], -1).sum(axis=1)))
            )
            # Include every matched prediction in the crop so the relevant
            # red contour is never cut into an apparently open curve.
            crop_support = focus[z_index].copy()
            for prediction in focus_predictions.values():
                crop_support |= prediction[z_index]
            crop = _crop_bounds(
                crop_support, margin=crop_margin, min_size=minimum_crop_size
            )
            visuals[case_id] = {
                "t1ce": images[0, 2].detach().cpu().numpy(),
                "gt_et": gt_et,
                "predictions": predictions,
                "focus_predictions": focus_predictions,
                "focus": focus,
                "z_index": z_index,
                "crop": crop,
            }
            del images
            if len(visuals) == len(wanted):
                break
    missing = wanted - set(visuals)
    if missing:
        raise ValueError(f"selected cases missing from test loader: {sorted(missing)}")
    return visuals


def _normalize_mri(slice_2d):
    values = slice_2d[slice_2d > 0]
    if not len(values):
        return np.zeros_like(slice_2d, dtype=float)
    lower, upper = np.percentile(values, (1, 99))
    if upper <= lower:
        upper = lower + 1e-6
    return np.clip((slice_2d - lower) / (upper - lower), 0, 1)


REGION_COLORS = {
    "tp": (0.12, 0.70, 0.36, 0.68),
    "fn": (0.12, 0.47, 0.71, 0.68),
    "fp": (0.89, 0.10, 0.11, 0.68),
}


def _region_error_overlay(gt, prediction):
    """Create a filled error map for a whole matched lesion cross-section."""
    gt = np.asarray(gt, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    overlay = np.zeros((*gt.shape, 4), dtype=float)
    overlay[np.logical_and(gt, prediction)] = REGION_COLORS["tp"]
    overlay[np.logical_and(gt, ~prediction)] = REGION_COLORS["fn"]
    overlay[np.logical_and(~gt, prediction)] = REGION_COLORS["fp"]
    return overlay


def _draw_contour(ax, mask, color, linewidth=1.5, linestyle="solid"):
    if np.asarray(mask).any():
        ax.contour(
            np.asarray(mask, dtype=float),
            levels=[0.5],
            colors=[color],
            linewidths=linewidth,
            linestyles=linestyle,
        )


def _metric_text(row, model_key, selected_row, role):
    if role == "small_lesion_improvement":
        if not bool(selected_row[f"{model_key}_detected"]):
            return "Small lesion missed"
        value = float(selected_row[f"{model_key}_lesion_dice"])
        return f"Matched lesion Dice {value:.3f}"
    hd95 = row[f"{model_key}_hd95_mm"]
    boundary_dice = row[f"{model_key}_boundary_dice"]
    if not np.isfinite(hd95) or not np.isfinite(boundary_dice):
        return "ET prediction empty"
    return f"HD95 {hd95:.2f} mm  |  BD {boundary_dice:.3f}"


def plot_case_grid(selected, comparison, visuals, output_stem: Path, zoomed: bool):
    comparison_by_case = comparison.set_index("case_id")
    selected_by_role = selected.set_index("selection_role")
    columns = ("MRI (T1ce)", "Ground truth", "Baseline", "LHFC", "Full")
    fig, axes = plt.subplots(4, 5, figsize=(16.2, 12.8))

    for row_index, role in enumerate(ROLE_ORDER):
        selected_row = selected_by_role.loc[role]
        case_id = selected_row["case_id"]
        metrics = comparison_by_case.loc[case_id]
        visual = visuals[case_id]
        z_index = visual["z_index"]
        y0, y1, x0, x1 = visual["crop"]
        crop = (slice(y0, y1), slice(x0, x1)) if zoomed else (slice(None), slice(None))
        mri = _normalize_mri(visual["t1ce"][z_index])[crop]
        gt_source = (
            visual["focus"]
            if role == "small_lesion_improvement"
            else visual["gt_et"]
        )
        gt = gt_source[z_index][crop]
        focus = visual["focus"][z_index][crop]
        prediction_source = (
            visual["focus_predictions"]
            if role == "small_lesion_improvement"
            else visual["predictions"]
        )
        prediction_slices = {
            key: prediction_source[key][z_index][crop] for key in MODEL_KEYS
        }

        for column_index, title in enumerate(columns):
            ax = axes[row_index, column_index]
            ax.imshow(mri, cmap="gray", interpolation="nearest")
            if row_index == 0:
                ax.set_title(title, fontsize=11, fontweight="semibold", pad=8)
            ax.axis("off")

        if gt.any():
            overlay = np.zeros((*gt.shape, 4), dtype=float)
            overlay[..., 1] = 0.85
            overlay[..., 3] = gt.astype(float) * 0.25
            axes[row_index, 1].imshow(overlay)
            _draw_contour(axes[row_index, 1], gt, "#00A878", linewidth=1.8)

        for column_index, model_key in enumerate(MODEL_KEYS, start=2):
            ax = axes[row_index, column_index]
            if role == "small_lesion_improvement":
                # Dice measures the complete region, so expose overlap and
                # both error types instead of reducing this row to contours.
                ax.imshow(
                    _region_error_overlay(gt, prediction_slices[model_key]),
                    interpolation="nearest",
                )
            else:
                # Boundary examples compare only the two complete contours.
                # GT is wider underneath so exact overlap remains visible.
                _draw_contour(
                    ax, gt, "#00A878", linewidth=2.6, linestyle="solid"
                )
                _draw_contour(
                    ax,
                    prediction_slices[model_key],
                    "#E84A5F",
                    linewidth=1.4,
                    linestyle="solid",
                )
            ax.text(
                0.5,
                -0.035,
                _metric_text(metrics, model_key, selected_row, role),
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=7.7,
            )

        if not zoomed:
            rect = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="#FFC857",
                linewidth=1.5,
            )
            axes[row_index, 0].add_patch(rect)

        if role == "small_lesion_improvement":
            row_label = (
                f"{ROLE_LABELS[role]}\n{case_id}, lesion {int(selected_row['gt_id'])}\n"
                f"{int(selected_row['gt_size'])} vox, "
                f"Dice gain={float(selected_row['small_lesion_dice_gain']):+.3f}"
            )
        else:
            row_label = (
                f"{ROLE_LABELS[role]}\n{case_id}\n"
                f"GT ET={int(metrics['gt_et_voxels'])} vox, z={z_index}"
            )
        axes[row_index, 0].set_ylabel(
            row_label,
            fontsize=9,
            rotation=0,
            ha="right",
            va="center",
            labelpad=95,
        )

    legend = [
        Patch(
            facecolor=REGION_COLORS["tp"],
            edgecolor="none",
            label="Small lesion: TP overlap",
        ),
        Patch(
            facecolor=REGION_COLORS["fn"],
            edgecolor="none",
            label="Small lesion: FN (GT only)",
        ),
        Patch(
            facecolor=REGION_COLORS["fp"],
            edgecolor="none",
            label="Small lesion: FP (prediction only)",
        ),
        Line2D([0], [0], color="#00A878", lw=2, label="GT ET boundary"),
        Line2D(
            [0],
            [0],
            color="#E84A5F",
            lw=2,
            linestyle="-",
            label="Predicted ET boundary",
        ),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False)
    view_name = "zoomed ROIs" if zoomed else "full-slice context"
    fig.suptitle(
        f"Automatically selected ET cases — {view_name}",
        fontsize=15,
        fontweight="semibold",
        y=0.995,
    )
    fig.subplots_adjust(left=0.18, right=0.99, top=0.955, bottom=0.065, wspace=0.08, hspace=0.25)
    suffix = "zoom" if zoomed else "context"
    png_path = output_stem.with_name(f"{output_stem.name}_{suffix}.png")
    pdf_path = output_stem.with_name(f"{output_stem.name}_{suffix}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="tumourCSV.csv")
    parser.add_argument(
        "--et-strata-json",
        type=Path,
        default=Path(
            "training_lesion_distributions/et_training_lesion_strata.json"
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.33)
    parser.add_argument("--boundary-tolerance-mm", type=float, default=1.0)
    parser.add_argument(
        "--voxel-spacing",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("DZ", "DY", "DX"),
    )
    parser.add_argument("--min-component-size", type=int, default=10)
    parser.add_argument("--expected-test-cases", type=int, default=37)
    parser.add_argument("--crop-margin", type=int, default=16)
    parser.add_argument("--minimum-crop-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--baseline-checkpoint",
        default=MODEL_SPECS[0]["checkpoint_dir"],
    )
    parser.add_argument(
        "--lhfc-checkpoint",
        default=MODEL_SPECS[1]["checkpoint_dir"],
    )
    parser.add_argument(
        "--full-checkpoint",
        default=MODEL_SPECS[2]["checkpoint_dir"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("boundary_typical_case_results"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    strata, strata_metadata = load_et_strata(
        args.et_strata_json, args.min_component_size
    )
    test_loader = get_dataloader(
        BratsDataset, args.csv, phase="test", batch_size=1
    )
    if len(test_loader.dataset) != args.expected_test_cases:
        raise ValueError(
            f"expected {args.expected_test_cases} fixed test cases, "
            f"got {len(test_loader.dataset)}"
        )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    checkpoint_inputs = {
        "baseline": args.baseline_checkpoint,
        "lhfc": args.lhfc_checkpoint,
        "full": args.full_checkpoint,
    }
    checkpoints = {
        key: resolve_best_checkpoint(path)
        for key, path in checkpoint_inputs.items()
    }
    models = {}
    for spec in MODEL_SPECS:
        checkpoint = checkpoints[spec["key"]]
        print(f"{spec['label']}: {checkpoint}")
        models[spec["key"]] = load_model(spec, checkpoint, device)

    per_case, per_lesion = evaluate_test_cases(
        models,
        test_loader,
        device,
        args.threshold,
        args.min_component_size,
        strata,
        args.boundary_tolerance_mm,
        tuple(args.voxel_spacing),
    )
    comparison = build_case_comparison(per_case)
    small_lesions = build_small_lesion_comparison(per_lesion)
    selected = select_typical_cases(comparison, small_lesions)

    per_case.to_csv(args.output_dir / "boundary_metrics_per_case.csv", index=False)
    per_lesion.to_csv(
        args.output_dir / "small_lesion_metrics_long.csv", index=False
    )
    small_lesions.to_csv(
        args.output_dir / "small_lesion_comparison.csv", index=False
    )
    comparison.to_csv(args.output_dir / "case_selection_ranking.csv", index=False)
    selected.to_csv(args.output_dir / "selected_typical_cases.csv", index=False)

    visuals = collect_selected_visuals(
        test_loader,
        models,
        selected,
        device,
        args.threshold,
        args.min_component_size,
        strata,
        args.crop_margin,
        args.minimum_crop_size,
    )
    output_stem = args.output_dir / "typical_cases_4x5"
    plot_case_grid(selected, comparison, visuals, output_stem, zoomed=True)
    plot_case_grid(selected, comparison, visuals, output_stem, zoomed=False)

    audit = {
        "evaluation_split": "test",
        "expected_test_cases": args.expected_test_cases,
        "threshold": args.threshold,
        "boundary_metric": "Boundary Dice (NSD)",
        "boundary_tolerance_mm": args.boundary_tolerance_mm,
        "voxel_spacing": list(args.voxel_spacing),
        "strata_source": str(args.et_strata_json),
        "strata": strata_metadata,
        "checkpoints": {key: str(value) for key, value in checkpoints.items()},
        "selected_cases": selected.to_dict(orient="records"),
    }
    with (args.output_dir / "selected_cases.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(_json_safe(audit), handle, indent=2, ensure_ascii=False)

    print("\nSelected cases:")
    print(
        selected[
            [
                "selection_role",
                "case_id",
                "hd95_improvement",
                "boundary_dice_improvement",
                "small_lesion_dice_gain",
                "baseline_lesion_dice",
                "full_lesion_dice",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved results to {args.output_dir}")

    models.clear()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
