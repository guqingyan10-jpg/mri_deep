"""Evaluate paired UCSF-BMSR Baseline and Full checkpoints on the fixed test set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from data.ucsf_bmsr import UCSFPreparedDataset, get_ucsf_dataloader
from advanced_metrics import hd95_single
from wt_lesion_stratified import match_wt_components
from models.resunet3d import ResUNet3d
from models.resunet_hf_concat_boundary import ResUNetHFConcatBoundary


def checkpoint_epoch(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def find_best(directory: Path) -> Path:
    checkpoints = list(directory.glob("best_model_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No best_model_*.pth in {directory}")
    return max(checkpoints, key=checkpoint_epoch)


def build_model(model_name: str):
    if model_name == "baseline":
        return ResUNet3d(in_channels=4, n_classes=1, n_channels=24)
    if model_name == "full":
        return ResUNetHFConcatBoundary(
            in_channels=4,
            n_classes=1,
            n_channels=24,
            fusion="concat",
            multiscale_context_v2=True,
        )
    raise ValueError(model_name)


def binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * float(np.logical_and(prediction, target).sum()) / denominator


def safe_mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def safe_std(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.std(finite)) if finite else float("nan")


def derive_volume_strata(volumes_mm3: list[float]) -> dict[str, tuple[float, float | None]]:
    """Fit approximately balanced lesion-volume strata without test data."""
    values, counts = np.unique(np.asarray(volumes_mm3, dtype=np.float64), return_counts=True)
    if len(values) < 3:
        raise ValueError("At least three distinct training lesion volumes are required")
    cumulative = np.cumsum(counts)
    first = int(np.argmin(np.abs(cumulative[:-2] - cumulative[-1] / 3.0)))
    candidates = np.arange(first + 1, len(values) - 1)
    second = int(candidates[np.argmin(np.abs(cumulative[candidates] - 2.0 * cumulative[-1] / 3.0))])
    return {
        "small": (0.0, float(values[first])),
        "medium": (float(values[first]), float(values[second])),
        "large": (float(values[second]), None),
    }


def classify_volume(
    volume_mm3: float, strata: dict[str, tuple[float, float | None]],
) -> str:
    if volume_mm3 <= strata["small"][1]:
        return "small"
    if volume_mm3 <= strata["medium"][1]:
        return "medium"
    return "large"


def collect_training_lesions(dataloader, min_component_size: int) -> list[dict]:
    """Measure GT components from the fixed training split in physical units."""
    rows = []
    for batch in dataloader:
        subject_id = str(batch["Id"][0])
        target = batch["mask"][0, 0].numpy() > 0.5
        spacing = tuple(float(value) for value in batch["spacing_dhw"][0].numpy())
        matching = match_wt_components(
            np.zeros_like(target), target, min_component_size=min_component_size,
        )
        voxel_volume = float(np.prod(spacing))
        for component_index, component in enumerate(matching["gt_components"]):
            rows.append({
                "subject_id": subject_id,
                "gt_component_index": component_index,
                "gt_voxels": component["size"],
                "gt_volume_mm3": component["size"] * voxel_volume,
            })
    if not rows:
        raise ValueError("No training lesions survived component filtering")
    return rows


def evaluate_model(
    model_name: str,
    checkpoint: Path,
    dataloader,
    device: str,
    threshold: float,
    min_component_size: int,
    volume_strata: dict[str, tuple[float, float | None]],
):
    model = build_model(model_name).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(state, dict):
        state = state.state_dict()
    model.load_state_dict(state, strict=True)
    model.eval()
    case_rows = []
    lesion_rows = []
    with torch.no_grad():
        for batch in dataloader:
            subject_id = str(batch["Id"][0])
            image = batch["image"].to(device)
            target = batch["mask"][0, 0].numpy() > 0.5
            spacing = tuple(float(value) for value in batch["spacing_dhw"][0].numpy())
            output = model(image)
            if isinstance(output, tuple):
                output = output[0]
            probability = torch.sigmoid(output)[0, 0].cpu().numpy()
            prediction = probability >= threshold
            matching = match_wt_components(
                prediction,
                target,
                min_component_size=min_component_size,
            )
            hd95 = hd95_single(prediction, target, voxel_spacing=spacing)
            case_rows.append({
                "model": model_name,
                "subject_id": subject_id,
                "threshold": threshold,
                "dice": binary_dice(prediction, target),
                "hd95_mm": hd95,
                "gt_lesions": matching["gt_lesions"],
                "pred_lesions": matching["pred_lesions"],
                "tp_lesions": matching["tp"],
                "fp_lesions": matching["fp"],
                "fn_lesions": matching["fn"],
            })
            matches = {item["gt_index"]: item for item in matching["matches"]}
            voxel_volume = float(np.prod(spacing))
            for gt_index, component in enumerate(matching["gt_components"]):
                match = matches.get(gt_index)
                volume_mm3 = component["size"] * voxel_volume
                lesion_rows.append({
                    "model": model_name,
                    "subject_id": subject_id,
                    "gt_component_index": gt_index,
                    "gt_voxels": component["size"],
                    "gt_volume_mm3": volume_mm3,
                    "volume_stratum": classify_volume(volume_mm3, volume_strata),
                    "detected": match is not None,
                    "matched_dice": float(match["dice"]) if match is not None else float("nan"),
                    "gt_anchored_dice": float(match["dice"]) if match is not None else 0.0,
                })
    return case_rows, lesion_rows


def summarize_lesion_strata(lesion_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, stratum), subset in lesion_frame.groupby(
        ["model", "volume_stratum"], sort=False,
    ):
        detected = subset["detected"].astype(bool)
        rows.append({
            "model": model_name,
            "volume_stratum": stratum,
            "gt_lesions": len(subset),
            "detected_lesions": int(detected.sum()),
            "missed_lesions": int((~detected).sum()),
            "lesion_recall": float(detected.mean()),
            "matched_lesion_dice": safe_mean(subset.loc[detected, "matched_dice"].tolist()),
            "gt_anchored_lesion_dice": safe_mean(subset["gt_anchored_dice"].tolist()),
        })
    order = {"small": 0, "medium": 1, "large": 2}
    result = pd.DataFrame(rows)
    result["_order"] = result["volume_stratum"].map(order)
    return result.sort_values(["model", "_order"]).drop(columns="_order")


def summarize_core(
    model_name: str, checkpoint: Path, cases: list[dict], lesions: list[dict],
) -> dict:
    """Return the compact UCSF endpoints corresponding to the final BraTS analysis."""
    tp = sum(row["tp_lesions"] for row in cases)
    fp = sum(row["fp_lesions"] for row in cases)
    fn = sum(row["fn_lesions"] for row in cases)
    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    lesion_f1 = (
        2 * recall * precision / (recall + precision)
        if recall + precision else 0.0
    )
    hd95 = [row["hd95_mm"] for row in cases]
    dice = [row["dice"] for row in cases]
    small = [row for row in lesions if row["volume_stratum"] == "small"]
    return {
        "model": model_name,
        "checkpoint": str(checkpoint.resolve()),
        "n_test_scans": len(cases),
        "dice_mean": safe_mean(dice),
        "dice_std": safe_std(dice),
        "hd95_mm_mean": safe_mean(hd95),
        "hd95_mm_std": safe_std(hd95),
        "lesion_gt_anchored_dice": safe_mean(
            [row["gt_anchored_dice"] for row in lesions]
        ),
        "small_lesion_gt_anchored_dice": safe_mean(
            [row["gt_anchored_dice"] for row in small]
        ),
        "lesion_f1": lesion_f1,
        "small_gt_lesions": len(small),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("/root/autodl-tmp/ucsf_bmsr_prepared/manifest.csv"),
    )
    parser.add_argument(
        "--training-root", type=Path,
        default=Path("/root/autodl-tmp/ucsf_baseline_full"),
    )
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--threshold", type=float, default=0.33)
    parser.add_argument("--min-component-size", type=int, default=10)
    parser.add_argument(
        "--boundary-tolerance-mm", type=float, default=1.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.training_root / f"seed{args.seed}" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    training_loader = get_ucsf_dataloader(
        UCSFPreparedDataset,
        str(args.manifest),
        "train",
        batch_size=1,
        num_workers=0,
    )
    training_lesions = collect_training_lesions(
        training_loader, args.min_component_size,
    )
    volume_strata = derive_volume_strata(
        [row["gt_volume_mm3"] for row in training_lesions]
    )
    for row in training_lesions:
        row["volume_stratum"] = classify_volume(row["gt_volume_mm3"], volume_strata)
    pd.DataFrame(training_lesions).to_csv(
        output_dir / "training_lesion_size_distribution.csv", index=False,
    )
    dataloader = get_ucsf_dataloader(
        UCSFPreparedDataset,
        str(args.manifest),
        "test",
        batch_size=1,
        num_workers=0,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_cases = []
    all_lesions = []
    summaries = []
    for model_name in ("baseline", "full"):
        checkpoint = find_best(
            args.training_root / f"seed{args.seed}" / model_name
        )
        cases, lesions = evaluate_model(
            model_name,
            checkpoint,
            dataloader,
            device,
            args.threshold,
            args.min_component_size,
            volume_strata,
        )
        all_cases.extend(cases)
        all_lesions.extend(lesions)
        summaries.append(summarize_core(model_name, checkpoint, cases, lesions))

    case_frame = pd.DataFrame(all_cases)
    lesion_frame = pd.DataFrame(all_lesions)
    summary_frame = pd.DataFrame(summaries)
    lesion_strata_frame = summarize_lesion_strata(lesion_frame)
    case_frame.to_csv(output_dir / "per_case.csv", index=False)
    lesion_frame.to_csv(output_dir / "per_gt_lesion.csv", index=False)
    summary_frame.to_csv(output_dir / "summary.csv", index=False)
    lesion_strata_frame.to_csv(output_dir / "summary_by_lesion_size.csv", index=False)
    paired = case_frame.pivot(index="subject_id", columns="model", values="dice")
    paired["full_minus_baseline_dice"] = paired["full"] - paired["baseline"]
    paired.reset_index().to_csv(output_dir / "paired_case_dice.csv", index=False)
    metadata = {
        "manifest": str(args.manifest.resolve()),
        "seed": args.seed,
        "threshold": args.threshold,
        "min_component_size": args.min_component_size,
        "lesion_size_definition": "individual GT connected-component physical volume; thresholds fitted on training split only",
        "lesion_volume_strata_mm3": volume_strata,
        "notes": [
            "The fixed test split is evaluated once; no five-fold averaging.",
            "A GT lesion is detected when one retained predicted component overlaps it after one-to-one Dice matching.",
            "Missed GT lesions receive zero GT-anchored Dice.",
            "Small/medium/large thresholds are never fitted on validation or test data.",
            "The primary table is restricted to Dice, HD95, overall GT-anchored lesion Dice, small-lesion GT-anchored Dice, and lesion F1.",
        ],
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary_frame.to_string(index=False))
    print(f"Results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
