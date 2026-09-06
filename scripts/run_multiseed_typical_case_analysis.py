"""Run ET small-lesion and boundary-case visualization for all three seeds.

Each seed is evaluated independently on the same fixed 37-case test split.
The script then aggregates the same GT lesion/case across seeds, so qualitative
examples can be chosen by consistency instead of by the best-looking seed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SELECTOR = REPO_ROOT / "scripts" / "select_boundary_typical_cases.py"
SEEDS = (42, 55, 123)
COMPLETE_SEED_OUTPUTS = (
    "small_lesion_comparison.csv",
    "case_selection_ranking.csv",
    "selected_typical_cases.csv",
    "small_lesion_region_dice.png",
    "typical_cases_4x5_zoom.png",
    "typical_cases_4x5_context.png",
)


@dataclass(frozen=True)
class SeedModelTriplet:
    seed: int
    baseline: Path
    lhfc: Path
    full: Path
    protocol: str

    def selector_command(
        self,
        output_dir: Path,
        strata_json: Path,
        csv_path: Path,
    ) -> list[str]:
        return [
            sys.executable,
            str(SELECTOR),
            "--csv",
            str(csv_path),
            "--et-strata-json",
            str(strata_json),
            "--baseline-checkpoint",
            str(self.baseline),
            "--lhfc-checkpoint",
            str(self.lhfc),
            "--full-checkpoint",
            str(self.full),
            "--output-dir",
            str(output_dir / f"seed{self.seed}"),
        ]


def default_seed_triplets(
    stability_root: Path,
    seed55_baseline: Path,
    seed55_lhfc: Path,
    seed55_full: Path,
    seed123_full_override: Path | None = None,
) -> list[SeedModelTriplet]:
    def stability_triplet(seed: int, full_override=None):
        root = stability_root / f"seed{seed}"
        return SeedModelTriplet(
            seed=seed,
            baseline=root / "baseline",
            lhfc=root / "edge_laplacian_concat",
            full=(
                Path(full_override)
                if full_override is not None
                else root / "hf_concat_boundary_w0.1_multiscale_v2"
            ),
            protocol="stability_runner",
        )

    return [
        stability_triplet(42),
        SeedModelTriplet(
            seed=55,
            baseline=seed55_baseline,
            lhfc=seed55_lhfc,
            full=seed55_full,
            protocol="main_experiment",
        ),
        stability_triplet(123, seed123_full_override),
    ]


def _flatten_seed_columns(frame, index_columns, value_columns):
    output = frame[index_columns].drop_duplicates().set_index(index_columns)
    for seed in SEEDS:
        seed_rows = frame[frame["seed"] == seed].set_index(index_columns)
        for column in value_columns:
            output[f"seed{seed}_{column}"] = seed_rows[column]
    return output.reset_index()


def seed_output_complete(seed_dir: Path) -> bool:
    return all((seed_dir / name).is_file() for name in COMPLETE_SEED_OUTPUTS)


def aggregate_small_lesions(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the same small GT component across three seed predictions."""
    identity = ["case_id", "gt_index", "gt_id", "gt_size"]
    required = set(SEEDS)
    observed = frame.groupby(identity)["seed"].agg(lambda values: set(values))
    if not observed.map(lambda values: values == required).all():
        raise ValueError("every small GT lesion must be present for all three seeds")

    frame = frame.copy()
    frame["comparable"] = frame["baseline_detected"] & frame["full_detected"]
    frame["improved"] = frame["comparable"] & (
        frame["small_lesion_dice_gain"] > 0
    )
    frame["comparable_gain"] = frame["small_lesion_dice_gain"].where(
        frame["comparable"]
    )
    grouped = frame.groupby(identity, as_index=False).agg(
        seeds_evaluated=("seed", "nunique"),
        comparable_seeds=("comparable", "sum"),
        improved_seeds=("improved", "sum"),
        mean_dice_gain=("comparable_gain", "mean"),
        std_dice_gain=("comparable_gain", "std"),
        mean_baseline_dice=("baseline_lesion_dice", "mean"),
        mean_full_dice=("full_lesion_dice", "mean"),
    )
    details = _flatten_seed_columns(
        frame,
        identity,
        (
            "baseline_detected",
            "full_detected",
            "baseline_lesion_dice",
            "full_lesion_dice",
            "small_lesion_dice_gain",
        ),
    )
    output = grouped.merge(details, on=identity, how="left")
    output["all_seeds_improved"] = output["improved_seeds"] == len(SEEDS)
    return output.sort_values(
        [
            "all_seeds_improved",
            "improved_seeds",
            "comparable_seeds",
            "mean_dice_gain",
            "mean_full_dice",
            "case_id",
            "gt_index",
        ],
        ascending=[False, False, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def aggregate_boundary_cases(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate case-level HD95 and Boundary Dice changes across seeds."""
    if not (
        frame.groupby("case_id")["seed"].agg(lambda values: set(values))
        == set(SEEDS)
    ).all():
        raise ValueError("every boundary case must be present for all three seeds")
    frame = frame.copy()
    frame["comparable"] = frame["all_models_boundary_valid"].astype(bool)
    frame["hd95_improved"] = frame["comparable"] & (
        frame["hd95_improvement"] > 0
    )
    frame["boundary_dice_improved"] = frame["comparable"] & (
        frame["boundary_dice_improvement"] > 0
    )
    frame["both_improved"] = (
        frame["hd95_improved"] & frame["boundary_dice_improved"]
    )
    frame["comparable_hd95_gain"] = frame["hd95_improvement"].where(
        frame["comparable"]
    )
    frame["comparable_boundary_dice_gain"] = frame[
        "boundary_dice_improvement"
    ].where(frame["comparable"])

    grouped = frame.groupby("case_id", as_index=False).agg(
        seeds_evaluated=("seed", "nunique"),
        comparable_seeds=("comparable", "sum"),
        both_improved_seeds=("both_improved", "sum"),
        hd95_improved_seeds=("hd95_improved", "sum"),
        boundary_dice_improved_seeds=("boundary_dice_improved", "sum"),
        mean_hd95_reduction_mm=("comparable_hd95_gain", "mean"),
        std_hd95_reduction_mm=("comparable_hd95_gain", "std"),
        mean_boundary_dice_gain=("comparable_boundary_dice_gain", "mean"),
        std_boundary_dice_gain=("comparable_boundary_dice_gain", "std"),
    )
    details = _flatten_seed_columns(
        frame,
        ["case_id"],
        (
            "hd95_improvement",
            "boundary_dice_improvement",
            "all_models_boundary_valid",
        ),
    )
    output = grouped.merge(details, on="case_id", how="left")
    output["all_seeds_both_improved"] = (
        output["both_improved_seeds"] == len(SEEDS)
    )
    return output.sort_values(
        [
            "all_seeds_both_improved",
            "both_improved_seeds",
            "comparable_seeds",
            "mean_boundary_dice_gain",
            "mean_hd95_reduction_mm",
            "case_id",
        ],
        ascending=[False, False, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def load_and_aggregate(output_dir: Path, triplets):
    small_frames = []
    boundary_frames = []
    for triplet in triplets:
        seed_dir = output_dir / f"seed{triplet.seed}"
        small_path = seed_dir / "small_lesion_comparison.csv"
        boundary_path = seed_dir / "case_selection_ranking.csv"
        if not small_path.is_file() or not boundary_path.is_file():
            raise FileNotFoundError(
                f"seed{triplet.seed} outputs are incomplete under {seed_dir}"
            )
        small = pd.read_csv(small_path)
        small.insert(0, "protocol", triplet.protocol)
        small.insert(0, "seed", triplet.seed)
        small_frames.append(small)
        boundary = pd.read_csv(boundary_path)
        boundary.insert(0, "protocol", triplet.protocol)
        boundary.insert(0, "seed", triplet.seed)
        boundary_frames.append(boundary)

    small_all = pd.concat(small_frames, ignore_index=True)
    boundary_all = pd.concat(boundary_frames, ignore_index=True)
    small_summary = aggregate_small_lesions(small_all)
    boundary_summary = aggregate_boundary_cases(boundary_all)

    small_all.to_csv(output_dir / "all_seed_small_lesions_long.csv", index=False)
    boundary_all.to_csv(output_dir / "all_seed_boundary_cases_long.csv", index=False)
    small_summary.to_csv(
        output_dir / "cross_seed_small_lesion_ranking.csv", index=False
    )
    boundary_summary.to_csv(
        output_dir / "cross_seed_boundary_case_ranking.csv", index=False
    )
    return small_summary, boundary_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("tumourCSV.csv"))
    parser.add_argument(
        "--et-strata-json",
        type=Path,
        default=Path(
            "training_lesion_distributions/et_training_lesion_strata.json"
        ),
    )
    parser.add_argument(
        "--stability-root",
        type=Path,
        default=Path("/root/autodl-tmp/stability"),
    )
    parser.add_argument(
        "--seed55-baseline",
        type=Path,
        default=Path("/root/autodl-tmp/ResUNet_model"),
    )
    parser.add_argument(
        "--seed55-lhfc",
        type=Path,
        default=Path("/root/autodl-tmp/ResUNet_Edge_concat_laplacian_model"),
    )
    parser.add_argument(
        "--seed55-full",
        type=Path,
        default=Path(
            "/root/autodl-tmp/"
            "ResUNet_HFConcatBoundary_w0.1_multiscale_v2_model"
        ),
    )
    parser.add_argument(
        "--seed123-full-override",
        type=Path,
        default=None,
        help="Optional supplemental checkpoint; formal default keeps original seed123",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("boundary_typical_case_multiseed_results"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Recompute seeds whose metrics and figures are already complete",
    )
    args = parser.parse_args()

    triplets = default_seed_triplets(
        args.stability_root,
        args.seed55_baseline,
        args.seed55_lhfc,
        args.seed55_full,
        args.seed123_full_override,
    )
    for triplet in triplets:
        command = triplet.selector_command(
            args.output_dir, args.et_strata_json, args.csv
        )
        print(f"\n[seed{triplet.seed} | {triplet.protocol}]")
        print(subprocess.list2cmdline(command))
        if not args.dry_run and not args.aggregate_only:
            seed_dir = args.output_dir / f"seed{triplet.seed}"
            if seed_output_complete(seed_dir) and not args.rerun_existing:
                print(f"Already complete; skipping {seed_dir}")
            else:
                subprocess.run(command, cwd=REPO_ROOT, check=True)

    if args.dry_run:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    small_summary, boundary_summary = load_and_aggregate(
        args.output_dir, triplets
    )
    print("\nTop cross-seed small-lesion candidate:")
    print(small_summary.head(1).to_string(index=False))
    print("\nTop cross-seed boundary candidate:")
    print(boundary_summary.head(1).to_string(index=False))
    print(f"\nSaved multi-seed results to {args.output_dir}")


if __name__ == "__main__":
    main()
