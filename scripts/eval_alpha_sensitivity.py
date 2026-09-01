"""Run the minimal V2 alpha sensitivity experiment on the fixed test set.

For each seed-specific V2 best checkpoint, the script fixes the model weights,
test cases, preprocessing, and binarization threshold.  It evaluates the
same checkpoint three times with alpha=0, the learned checkpoint alpha, and
alpha=1.  Macro/ET Dice keep the existing case-level definitions; the third
primary metric is training-stratified small-lesion ET GT-anchored Dice, using
the repository's existing one-to-one lesion matcher.  No training or
checkpoint-history reconstruction is performed.

Seed 55 intentionally uses the main-experiment directory.  Seeds 42 and 123
use the stability-runner directories, so cross-seed summaries are descriptive
and the training protocol remains explicit in every output.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import BratsDataset, get_dataloader
from evaluation.wt_lesion_stratified import (
    classify_lesion_size,
    match_lesion_components,
    summarize_stratified_cases,
)
from models.resunet_hf_concat_boundary import ResUNetHFConcatBoundary


ALPHA_KEY = "multiscale_context.alpha"
ALPHA_MODES = ("zero", "learned", "one")
METRIC_COLUMNS = (
    "macro_dice",
    "et_dice",
    "small_lesion_gt_anchored_dice",
)
SEED_COLORS = {42: "#4E79A7", 55: "#E15759", 123: "#59A14F"}


@dataclass(frozen=True)
class SeedCheckpointSpec:
    seed: int
    checkpoint_dir: Path
    training_protocol: str


def load_et_strata(path: Path, min_component_size: int):
    """Load ET lesion-size bounds fitted on the fixed training split."""
    with path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("region") != "ET":
        raise ValueError(f"strata JSON region must be ET, got {metadata.get('region')!r}")
    if metadata.get("fit_split") != "train":
        raise ValueError("ET strata must be fitted on the training split")
    if metadata.get("connectivity") != 26:
        raise ValueError("ET strata must use 26-connectivity")
    if metadata.get("min_component_size") != min_component_size:
        raise ValueError(
            "ET strata minimum component size does not match "
            "--min-component-size"
        )
    strata = OrderedDict(
        (name, tuple(metadata["strata"][name]))
        for name in ("small", "medium", "large")
    )
    return strata, metadata


def default_seed_specs(stability_root: Path, seed55_dir: Path):
    """Return the repository's three V2 checkpoint locations."""
    return {
        42: SeedCheckpointSpec(
            seed=42,
            checkpoint_dir=(
                stability_root
                / "seed42"
                / "hf_concat_boundary_w0.1_multiscale_v2"
            ),
            training_protocol="stability_runner",
        ),
        55: SeedCheckpointSpec(
            seed=55,
            checkpoint_dir=seed55_dir,
            training_protocol="main_experiment",
        ),
        123: SeedCheckpointSpec(
            seed=123,
            checkpoint_dir=(
                stability_root
                / "seed123"
                / "hf_concat_boundary_w0.1_multiscale_v2"
            ),
            training_protocol="stability_runner",
        ),
    }


def parse_checkpoint_overrides(values):
    """Parse repeated ``SEED=PATH`` command-line overrides."""
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"checkpoint override must be SEED=PATH, got {value!r}"
            )
        seed_text, path_text = value.split("=", 1)
        overrides[int(seed_text)] = Path(path_text)
    return overrides


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"_(\d+)\.pth$", path.name)
    if not match:
        raise ValueError(f"checkpoint filename has no epoch number: {path}")
    return int(match.group(1))


def find_best_checkpoint(directory: Path) -> Path:
    candidates = list(directory.glob("best_model_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"no best_model_*.pth found in {directory}")
    return max(candidates, key=checkpoint_epoch)


def _torch_load(path: Path):
    """Load a trusted local state dictionary across PyTorch versions."""
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def clean_state_dict(checkpoint):
    """Unwrap common containers and normalize historical state keys."""
    state = checkpoint
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain a state dictionary")
    cleaned = {}
    for key, value in state.items():
        key = key[7:] if key.startswith("module.") else key
        key = key.replace("out.conv.0.", "out.conv.")
        cleaned[key] = value
    return cleaned


def extract_scalar_alpha(state, source: Path) -> float:
    """Read the unconstrained V2 scalar alpha and reject V3/missing states."""
    if ALPHA_KEY not in state:
        candidates = [key for key in state if key.endswith(".alpha")]
        raise KeyError(
            f"{source} has no {ALPHA_KEY!r}; alpha-like keys={candidates}"
        )
    alpha = state[ALPHA_KEY].detach().cpu().reshape(-1)
    if alpha.numel() != 1:
        raise ValueError(
            f"{source} contains {alpha.numel()} alpha values; expected V2 scalar"
        )
    return float(alpha.item())


def load_best_state(spec: SeedCheckpointSpec):
    checkpoint = find_best_checkpoint(spec.checkpoint_dir)
    state = clean_state_dict(_torch_load(checkpoint))
    alpha = extract_scalar_alpha(state, checkpoint)
    return checkpoint, state, alpha


def build_v2_model(state, device):
    model = ResUNetHFConcatBoundary(
        in_channels=4,
        n_classes=3,
        n_channels=24,
        fusion="concat",
        multiscale_context_v2=True,
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "V2 checkpoint is not architecture-compatible: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model


def _batch_case_ids(data, batch_size):
    ids = data.get("Id", ["unknown"] * batch_size)
    if isinstance(ids, str):
        return [ids]
    if isinstance(ids, (list, tuple)):
        return [str(value) for value in ids]
    return [str(ids)]


def _binary_dice(prediction, target):
    intersection = int(np.logical_and(prediction, target).sum())
    denominator = int(prediction.sum()) + int(target.sum())
    return 2.0 * intersection / denominator if denominator else 1.0


def evaluate_case_dice_and_et_lesions(
    model,
    dataloader,
    device,
    threshold,
    min_component_size,
    strata,
    description,
):
    """Return case Dice plus one-to-one matched ET lesion results."""
    case_rows = []
    case_lesion_results = []
    lesion_rows = []
    with torch.no_grad():
        for data in tqdm(dataloader, desc=description):
            images = data["image"].to(device)
            targets = data["mask"].cpu().numpy() > 0
            logits = model(images)
            if isinstance(logits, tuple):
                logits = logits[0]
            predictions = (
                torch.sigmoid(logits) >= threshold
            ).cpu().numpy()
            case_ids = _batch_case_ids(data, len(images))
            for index, case_id in enumerate(case_ids):
                dice = [
                    _binary_dice(
                        predictions[index, channel], targets[index, channel]
                    )
                    for channel in range(3)
                ]
                case_rows.append(
                    {
                        "case_id": case_id,
                        "wt_dice": dice[0],
                        "tc_dice": dice[1],
                        "et_dice": dice[2],
                        "gt_et_voxels": int(targets[index, 2].sum()),
                    }
                )

                result = match_lesion_components(
                    predictions[index, 2],
                    targets[index, 2],
                    min_component_size=min_component_size,
                )
                result["case_id"] = case_id
                case_lesion_results.append(result)
                matched_by_gt = {
                    match["gt_index"]: match for match in result["matches"]
                }
                for gt_index, component in enumerate(result["gt_components"]):
                    match = matched_by_gt.get(gt_index)
                    lesion_rows.append(
                        {
                            "case_id": case_id,
                            "gt_lesion_id": component["id"],
                            "stratum": classify_lesion_size(
                                component["size"], strata
                            ),
                            "gt_voxels": component["size"],
                            "pred_lesion_id": match["pred_id"] if match else "",
                            "pred_voxels": match["pred_size"] if match else "",
                            "intersection_voxels": (
                                match["intersection"] if match else 0
                            ),
                            "detected": bool(match),
                            "matched_dice": match["dice"] if match else np.nan,
                            "gt_anchored_dice": match["dice"] if match else 0.0,
                        }
                    )
    case_frame = pd.DataFrame(case_rows)
    lesion_frame = pd.DataFrame(lesion_rows)
    if case_frame.empty:
        raise ValueError("test dataloader produced no cases")
    if case_frame["case_id"].duplicated().any():
        raise ValueError("test dataloader produced duplicate case IDs")
    return case_frame, case_lesion_results, lesion_frame


def assert_same_ground_truth(reference, candidate):
    left = reference[["case_id", "gt_et_voxels"]].sort_values("case_id")
    right = candidate[["case_id", "gt_et_voxels"]].sort_values("case_id")
    if not left.reset_index(drop=True).equals(right.reset_index(drop=True)):
        raise ValueError("test cases or GT ET volumes changed between runs")


def assert_same_gt_lesions(reference, candidate):
    columns = ["case_id", "gt_lesion_id", "stratum", "gt_voxels"]
    left = reference[columns].sort_values(["case_id", "gt_lesion_id"])
    right = candidate[columns].sort_values(["case_id", "gt_lesion_id"])
    if not left.reset_index(drop=True).equals(right.reset_index(drop=True)):
        raise ValueError("test GT ET lesions changed between alpha runs")


def summarize_run(per_case, case_lesion_results, strata):
    class_means = [
        float(per_case[column].mean())
        for column in ("wt_dice", "tc_dice", "et_dice")
    ]
    small = summarize_stratified_cases(case_lesion_results, strata)["small"]
    if not small["gt_lesions"]:
        raise ValueError("test set has no small ET lesions")
    return {
        "macro_dice": float(np.mean(class_means)),
        "et_dice": class_means[2],
        "small_lesion_gt_anchored_dice": small[
            "gt_anchored_lesion_dice"
        ],
        "small_lesion_matched_dice": small["matched_lesion_dice"],
        "small_lesion_recall": small["lesion_recall"],
        "small_lesion_miss_rate": small["miss_rate"],
        "small_gt_lesions": small["gt_lesions"],
        "small_detected_lesions": small["detected"],
        "small_missed_lesions": small["missed"],
    }


def alpha_mode_value(mode, learned_alpha):
    return {"zero": 0.0, "learned": learned_alpha, "one": 1.0}[mode]


def evaluate_seed(
    spec,
    dataloader,
    device,
    threshold,
    min_component_size,
    strata,
    reference_gt,
    reference_gt_lesions,
):
    checkpoint, state, learned_alpha = load_best_state(spec)
    model = build_v2_model(state, device)
    summary_rows = []
    case_detail_frames = []
    lesion_detail_frames = []

    for mode in ALPHA_MODES:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"could not restore fixed seed {spec.seed} checkpoint"
            )
        evaluation_alpha = alpha_mode_value(mode, learned_alpha)
        with torch.no_grad():
            model.multiscale_context.alpha.fill_(evaluation_alpha)
        per_case, case_lesion_results, per_lesion = (
            evaluate_case_dice_and_et_lesions(
                model,
                dataloader,
                device,
                threshold,
                min_component_size,
                strata,
                f"Seed {spec.seed}, alpha={mode}",
            )
        )
        if reference_gt is None:
            reference_gt = per_case.copy()
            reference_gt_lesions = per_lesion.copy()
        else:
            assert_same_ground_truth(reference_gt, per_case)
            assert_same_gt_lesions(reference_gt_lesions, per_lesion)

        metrics = summarize_run(per_case, case_lesion_results, strata)
        summary_rows.append(
            {
                "seed": spec.seed,
                "training_protocol": spec.training_protocol,
                "checkpoint": str(checkpoint),
                "best_epoch": checkpoint_epoch(checkpoint),
                "checkpoint_alpha": learned_alpha,
                "alpha_mode": mode,
                "evaluation_alpha": evaluation_alpha,
                "evaluation_split": "test",
                "threshold": threshold,
                "n_cases": len(per_case),
                **metrics,
            }
        )
        per_case.insert(0, "alpha_mode", mode)
        per_case.insert(0, "evaluation_alpha", evaluation_alpha)
        per_case.insert(0, "checkpoint_alpha", learned_alpha)
        per_case.insert(0, "training_protocol", spec.training_protocol)
        per_case.insert(0, "seed", spec.seed)
        case_detail_frames.append(per_case)
        per_lesion.insert(0, "alpha_mode", mode)
        per_lesion.insert(0, "evaluation_alpha", evaluation_alpha)
        per_lesion.insert(0, "checkpoint_alpha", learned_alpha)
        per_lesion.insert(0, "training_protocol", spec.training_protocol)
        per_lesion.insert(0, "seed", spec.seed)
        lesion_detail_frames.append(per_lesion)

    del model, state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (
        summary_rows,
        case_detail_frames,
        lesion_detail_frames,
        reference_gt,
        reference_gt_lesions,
    )


def aggregate_sensitivity(summary: pd.DataFrame):
    rows = []
    learned_by_seed = summary[summary["alpha_mode"] == "learned"].set_index(
        "seed"
    )
    for mode in ALPHA_MODES:
        group = summary[summary["alpha_mode"] == mode]
        row = {"alpha_mode": mode, "n_seeds": len(group)}
        for metric in METRIC_COLUMNS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
            deltas = (
                group.set_index("seed")[metric]
                - learned_by_seed[metric]
            )
            row[f"{metric}_delta_vs_learned_mean"] = float(deltas.mean())
            row[f"{metric}_delta_vs_learned_std"] = float(deltas.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def checkpoint_alpha_summary(best_metadata):
    table = pd.DataFrame(
        [
            {
                "seed": spec.seed,
                "training_protocol": spec.training_protocol,
                "checkpoint": str(checkpoint),
                "best_epoch": checkpoint_epoch(checkpoint),
                "checkpoint_alpha": alpha,
            }
            for spec, checkpoint, alpha in best_metadata
        ]
    ).sort_values("seed").reset_index(drop=True)
    values = table["checkpoint_alpha"].to_numpy(dtype=float)
    aggregate = {
        "definition": "scalar alpha read from each evaluated best checkpoint",
        "seeds": table["seed"].astype(int).tolist(),
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "range": [float(values.min()), float(values.max())],
        "mixed_training_protocols": table["training_protocol"].nunique() > 1,
    }
    return table, aggregate


def plot_sensitivity(summary, output):
    labels = {
        "zero": "alpha=0",
        "learned": "learned alpha",
        "one": "alpha=1",
    }
    titles = {
        "macro_dice": "Macro Dice",
        "et_dice": "ET Dice",
        "small_lesion_gt_anchored_dice": (
            "Small-lesion ET GT-anchored Dice"
        ),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    x = np.arange(len(ALPHA_MODES))
    for ax, metric in zip(axes, METRIC_COLUMNS):
        metric_values = []
        for seed in sorted(summary["seed"].unique()):
            group = summary[summary["seed"] == seed].set_index("alpha_mode")
            values = [float(group.loc[mode, metric]) for mode in ALPHA_MODES]
            metric_values.extend(values)
            protocol = group.iloc[0]["training_protocol"]
            suffix = "main" if protocol == "main_experiment" else "stability"
            ax.plot(
                x,
                values,
                marker="o",
                linewidth=2,
                color=SEED_COLORS.get(seed),
                label=f"Seed {seed} ({suffix})",
            )
            for xpos, value in zip(x, values):
                ax.annotate(
                    f"{value:.4f}",
                    (xpos, value),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.5,
                )
        lower = max(0.0, min(metric_values) - 0.025)
        upper = min(1.0, max(metric_values) + 0.025)
        if upper - lower < 0.06:
            midpoint = (upper + lower) / 2
            lower = max(0.0, midpoint - 0.03)
            upper = min(1.0, midpoint + 0.03)
        ax.set_ylim(lower, upper)
        ax.set_xticks(x, [labels[mode] for mode in ALPHA_MODES])
        ax.set_title(titles[metric], fontweight="semibold")
        ax.set_ylabel("Dice")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Test-set sensitivity to the V2 residual gate",
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.01,
        "All weights and test cases are fixed within seed; only alpha is overwritten. "
        "Axes use local ranges and exact values are annotated.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    fig.savefig(output, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(summary, aggregate, alpha_table, alpha_aggregate, output):
    lines = [
        "# V2 Alpha Sensitivity (Test Set)",
        "",
        "Seed 55 is the main experiment; seeds 42 and 123 use the stability runner. "
        "Cross-seed summaries are descriptive because the training protocols differ.",
        "",
        "## Learned alpha from evaluated best checkpoints",
        "",
        "| Seed | Protocol | Best epoch | Learned alpha |",
        "|---:|---|---:|---:|",
    ]
    for row in alpha_table.itertuples(index=False):
        lines.append(
            f"| {row.seed} | {row.training_protocol} | {row.best_epoch} | "
            f"{row.checkpoint_alpha:.8g} |"
        )
    lines.extend(
        [
            "",
            f"Best-checkpoint alpha mean: **{alpha_aggregate['mean']:.8g}**; "
            f"range: **[{alpha_aggregate['minimum']:.8g}, "
            f"{alpha_aggregate['maximum']:.8g}]**.",
            "",
            "## Per-seed sensitivity",
            "",
            "| Seed | Protocol | Alpha mode | Eval alpha | Macro Dice | ET Dice | Small-lesion GT-anchored Dice |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.seed} | {row.training_protocol} | {row.alpha_mode} | "
            f"{row.evaluation_alpha:.8g} | {row.macro_dice:.5f} | "
            f"{row.et_dice:.5f} | "
            f"{row.small_lesion_gt_anchored_dice:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Small-lesion ET diagnostics",
            "",
            "| Seed | Alpha mode | GT lesions | Detected | Missed | Recall | Miss rate | Matched Dice | GT-anchored Dice |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.seed} | {row.alpha_mode} | {row.small_gt_lesions} | "
            f"{row.small_detected_lesions} | {row.small_missed_lesions} | "
            f"{row.small_lesion_recall:.5f} | "
            f"{row.small_lesion_miss_rate:.5f} | "
            f"{row.small_lesion_matched_dice:.5f} | "
            f"{row.small_lesion_gt_anchored_dice:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Descriptive cross-seed summary",
            "",
            "| Alpha mode | Macro Dice | ET Dice | Small-lesion GT-anchored Dice |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.alpha_mode} | {row.macro_dice_mean:.5f} +/- "
            f"{row.macro_dice_std:.5f} | {row.et_dice_mean:.5f} +/- "
            f"{row.et_dice_std:.5f} | "
            f"{row.small_lesion_gt_anchored_dice_mean:.5f} +/- "
            f"{row.small_lesion_gt_anchored_dice_std:.5f} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="tumourCSV.csv")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 55, 123])
    parser.add_argument(
        "--stability-root",
        type=Path,
        default=Path("/root/autodl-tmp/stability"),
    )
    parser.add_argument(
        "--seed55-dir",
        type=Path,
        default=Path(
            "/root/autodl-tmp/"
            "ResUNet_HFConcatBoundary_w0.1_multiscale_v2_model"
        ),
        help="Main-experiment V2 directory; intentionally not under stability/seed55",
    )
    parser.add_argument(
        "--checkpoint-dir",
        action="append",
        default=[],
        metavar="SEED=PATH",
        help="Override a seed checkpoint directory; may be repeated",
    )
    parser.add_argument("--threshold", type=float, default=0.33)
    parser.add_argument(
        "--et-strata-json",
        type=Path,
        default=Path(
            "training_lesion_distributions/et_training_lesion_strata.json"
        ),
        help="Frozen ET lesion-size strata fitted on the training split",
    )
    parser.add_argument("--min-component-size", type=int, default=10)
    parser.add_argument(
        "--expected-test-cases",
        type=int,
        default=37,
        help="Fail unless the fixed test split contains this many cases",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("alpha_sensitivity_test_results"),
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only resolve best checkpoints and report their learned alpha values",
    )
    args = parser.parse_args()

    specs = default_seed_specs(args.stability_root, args.seed55_dir)
    overrides = parse_checkpoint_overrides(args.checkpoint_dir)
    for seed, directory in overrides.items():
        protocol = specs.get(
            seed,
            SeedCheckpointSpec(seed, directory, "custom"),
        ).training_protocol
        specs[seed] = SeedCheckpointSpec(seed, directory, protocol)
    unknown = [seed for seed in args.seeds if seed not in specs]
    if unknown:
        raise ValueError(
            f"no default checkpoint mapping for seeds {unknown}; use --checkpoint-dir"
        )
    selected = [specs[seed] for seed in args.seeds]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_metadata = []
    for spec in selected:
        checkpoint, state, alpha = load_best_state(spec)
        best_metadata.append((spec, checkpoint, alpha))
        del state
    alpha_table, alpha_aggregate = checkpoint_alpha_summary(best_metadata)
    alpha_table.to_csv(
        args.output_dir / "alpha_checkpoint_values.csv", index=False
    )
    with open(
        args.output_dir / "alpha_checkpoint_summary.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(alpha_aggregate, handle, indent=2)
    print(alpha_table.to_string(index=False))
    print(
        "Best-checkpoint alpha: "
        f"mean={alpha_aggregate['mean']:.8g}, "
        f"range=[{alpha_aggregate['minimum']:.8g}, "
        f"{alpha_aggregate['maximum']:.8g}]"
    )
    if args.inspect_only:
        print(f"Saved checkpoint alpha inspection to {args.output_dir}")
        return

    strata, strata_metadata = load_et_strata(
        args.et_strata_json, args.min_component_size
    )
    print(
        "Training-defined ET strata:",
        ", ".join(
            f"{name}={lower}-{upper if upper is not None else 'inf'}"
            for name, (lower, upper) in strata.items()
        ),
    )

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    test_loader = get_dataloader(
        BratsDataset, args.csv, phase="test", batch_size=1
    )
    actual_test_cases = len(test_loader.dataset)
    if actual_test_cases != args.expected_test_cases:
        raise ValueError(
            f"expected {args.expected_test_cases} fixed test cases, "
            f"but {args.csv} produced {actual_test_cases}"
        )
    print(f"Device: {device}; fixed test cases: {actual_test_cases}")

    summary_rows = []
    case_detail_frames = []
    lesion_detail_frames = []
    reference_gt = None
    reference_gt_lesions = None
    for spec in selected:
        result = evaluate_seed(
            spec,
            test_loader,
            device,
            args.threshold,
            args.min_component_size,
            strata,
            reference_gt,
            reference_gt_lesions,
        )
        (
            rows,
            case_details,
            lesion_details,
            reference_gt,
            reference_gt_lesions,
        ) = result
        summary_rows.extend(rows)
        case_detail_frames.extend(case_details)
        lesion_detail_frames.extend(lesion_details)

    summary = pd.DataFrame(summary_rows)
    case_details = pd.concat(case_detail_frames, ignore_index=True)
    lesion_details = pd.concat(lesion_detail_frames, ignore_index=True)
    learned_metrics = summary[summary["alpha_mode"] == "learned"].set_index(
        "seed"
    )
    for metric in METRIC_COLUMNS:
        summary[f"{metric}_delta_vs_learned"] = summary.apply(
            lambda row: row[metric] - learned_metrics.loc[row["seed"], metric],
            axis=1,
        )
    aggregate = aggregate_sensitivity(summary)

    summary.to_csv(
        args.output_dir / "alpha_sensitivity_per_seed.csv", index=False
    )
    aggregate.to_csv(
        args.output_dir / "alpha_sensitivity_summary.csv", index=False
    )
    case_details.to_csv(
        args.output_dir / "alpha_sensitivity_per_case.csv", index=False
    )
    lesion_details.to_csv(
        args.output_dir / "alpha_sensitivity_per_lesion.csv", index=False
    )
    reference_gt_lesions[
        reference_gt_lesions["stratum"] == "small"
    ][["case_id", "gt_lesion_id", "gt_voxels", "stratum"]].to_csv(
        args.output_dir / "small_et_test_lesions.csv", index=False
    )
    strata_output = dict(strata_metadata)
    strata_output["source_json"] = str(args.et_strata_json)
    strata_output["apply_split"] = "test"
    with open(
        args.output_dir / "et_lesion_strata_applied.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(strata_output, handle, indent=2)

    plot_sensitivity(
        summary, args.output_dir / "alpha_sensitivity_metrics.png"
    )
    write_report(
        summary,
        aggregate,
        alpha_table,
        alpha_aggregate,
        args.output_dir / "alpha_sensitivity_report.md",
    )
    print(f"Saved alpha sensitivity results to {args.output_dir}")


if __name__ == "__main__":
    main()
