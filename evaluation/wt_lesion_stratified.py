"""WT lesion-level matching and size-stratified metrics.

The functions in this module deliberately operate on individual 3-D WT
connected components, not on whole cases.  Ground-truth component size
determines the stratum; predictions are only used for one-to-one matching.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


# Empirical WT tertiles from the 729-component distribution.  Rounded bounds
# give 245/242/242 components while remaining easy to report in a paper.
STRATA = OrderedDict(
    (
        ("small", (10, 74)),
        ("medium", (75, 64999)),
        ("large", (65000, None)),
    )
)


def classify_wt_size(size: int, strata=STRATA) -> str | None:
    """Return the stratum for a component size, or ``None`` if out of range."""
    size = int(size)
    for name, (lower, upper) in strata.items():
        if size >= lower and (upper is None or size <= upper):
            return name
    return None


def _components(mask: np.ndarray, structure, min_component_size: int):
    """Extract component masks and sizes after minimum-size filtering."""
    labeled, count = ndimage.label(np.asarray(mask) > 0, structure=structure)
    components = []
    for component_id in range(1, count + 1):
        component_mask = labeled == component_id
        size = int(component_mask.sum())
        if size >= min_component_size:
            components.append(
                {
                    "id": len(components) + 1,
                    "mask": component_mask,
                    "size": size,
                }
            )
    return components


def _dice(intersection: int, size_a: int, size_b: int) -> float:
    denominator = size_a + size_b
    return 2.0 * intersection / denominator if denominator else 0.0


def match_wt_components(
    pred_wt_mask: np.ndarray,
    gt_wt_mask: np.ndarray,
    *,
    structure=None,
    min_component_size: int = 10,
):
    """Match predicted and GT WT lesions one-to-one.

    Pair scores are component Dice values.  Hungarian assignment maximizes
    the total score, and an assignment is accepted only when the components
    share at least one voxel.  Thus one prediction cannot detect two GT
    lesions, and one GT lesion cannot be detected by two predictions.
    """
    if structure is None:
        structure = ndimage.generate_binary_structure(3, 3)

    gt_components = _components(gt_wt_mask, structure, min_component_size)
    pred_components = _components(pred_wt_mask, structure, min_component_size)

    scores = np.zeros((len(gt_components), len(pred_components)), dtype=float)
    intersections = np.zeros_like(scores, dtype=np.int64)
    for gi, gt_component in enumerate(gt_components):
        for pi, pred_component in enumerate(pred_components):
            overlap = int(np.logical_and(gt_component["mask"], pred_component["mask"]).sum())
            intersections[gi, pi] = overlap
            scores[gi, pi] = _dice(
                overlap, gt_component["size"], pred_component["size"]
            )

    matches = []
    matched_gt = set()
    matched_pred = set()
    if scores.size:
        gt_indices, pred_indices = linear_sum_assignment(scores, maximize=True)
        for gi, pi in zip(gt_indices.tolist(), pred_indices.tolist()):
            overlap = int(intersections[gi, pi])
            if overlap <= 0:
                continue
            matched_gt.add(gi)
            matched_pred.add(pi)
            gt_component = gt_components[gi]
            pred_component = pred_components[pi]
            matches.append(
                {
                    "gt_index": gi,
                    "pred_index": pi,
                    "gt_id": gt_component["id"],
                    "pred_id": pred_component["id"],
                    "gt_size": gt_component["size"],
                    "pred_size": pred_component["size"],
                    "intersection": overlap,
                    "dice": float(scores[gi, pi]),
                }
            )

    return {
        "gt_components": gt_components,
        "pred_components": pred_components,
        "matches": matches,
        "gt_lesions": len(gt_components),
        "pred_lesions": len(pred_components),
        "tp": len(matches),
        "fn": len(gt_components) - len(matches),
        "fp": len(pred_components) - len(matches),
        "matched_gt_indices": matched_gt,
        "matched_pred_indices": matched_pred,
    }


def _empty_summary(name: str):
    return {
        "stratum": name,
        "gt_lesions": 0,
        "detected": 0,
        "missed": 0,
        "matched_pairs": 0,
        "pred_lesions": 0,
        "fp": 0,
        "lesion_recall": float("nan"),
        "miss_rate": float("nan"),
        "matched_lesion_dice": float("nan"),
        "gt_anchored_lesion_dice": float("nan"),
        "matched_dice_values": [],
        "n_cases_with_gt": 0,
        "n_cases_evaluated": 0,
        "n_gt_empty_cases": 0,
        "n_fp_only_cases": 0,
        "n_true_negative_cases": 0,
    }


def summarize_stratified_cases(case_results, strata=STRATA):
    """Aggregate lesion-level metrics across cases for each WT size stratum.

    Recall and miss rate use all GT lesions in a stratum.  ``matched_lesion_dice``
    is conditional on a valid match; ``gt_anchored_lesion_dice`` assigns Dice=0
    to every missed GT lesion and is the less optimistic companion metric.
    Cases with no GT lesion are never scored as perfect recall.  They are
    counted separately as true-negative or false-positive-only cases.
    """
    summaries = OrderedDict((name, _empty_summary(name)) for name in strata)
    summaries["all"] = _empty_summary("all")

    for result in case_results:
        gt_components = result["gt_components"]
        matches_by_gt = {m["gt_index"]: m for m in result["matches"]}
        gt_strata_present = set()
        for key in summaries:
            summaries[key]["n_cases_evaluated"] += 1
        for gi, component in enumerate(gt_components):
            stratum = classify_wt_size(component["size"], strata)
            if stratum is None:
                continue
            gt_strata_present.add(stratum)
            for key in (stratum, "all"):
                summary = summaries[key]
                summary["gt_lesions"] += 1
                match = matches_by_gt.get(gi)
                if match is None:
                    summary["missed"] += 1
                else:
                    summary["detected"] += 1
                    summary["matched_pairs"] += 1
                    summary["matched_dice_values"].append(match["dice"])

        # Predicted components are reported globally.  They cannot be assigned
        # to a GT size stratum when they are false positives.
        summaries["all"]["pred_lesions"] += result["pred_lesions"]
        summaries["all"]["fp"] += result["fp"]

        for key in strata:
            if key not in gt_strata_present:
                summaries[key]["n_gt_empty_cases"] += 1
        if result["pred_lesions"] == 0 and result["gt_lesions"] == 0:
            summaries["all"]["n_gt_empty_cases"] += 1
            summaries["all"]["n_true_negative_cases"] += 1
        elif result["gt_lesions"] == 0 and result["pred_lesions"] > 0:
            summaries["all"]["n_gt_empty_cases"] += 1
            summaries["all"]["n_fp_only_cases"] += 1
        for key in gt_strata_present:
            summaries[key]["n_cases_with_gt"] += 1
        if result["gt_lesions"] > 0:
            summaries["all"]["n_cases_with_gt"] += 1

    for summary in summaries.values():
        gt_count = summary["gt_lesions"]
        if gt_count:
            summary["lesion_recall"] = summary["detected"] / gt_count
            summary["miss_rate"] = summary["missed"] / gt_count
            anchored = list(summary["matched_dice_values"])
            anchored.extend([0.0] * summary["missed"])
            summary["gt_anchored_lesion_dice"] = float(np.mean(anchored))
        if summary["matched_dice_values"]:
            summary["matched_lesion_dice"] = float(
                np.mean(summary["matched_dice_values"])
            )

    return summaries
