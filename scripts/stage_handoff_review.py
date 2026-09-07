"""Stage AutoDL artifacts for handoff review under one Git branch.

The staged directory contains selected best checkpoints, result tables,
training records, split manifests, and an auditable model-name mapping.  It
never copies rendered figures, plotting scripts, datasets, last-epoch
checkpoints, notebook caches, or trash-directory contents.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


TABLE_SUFFIXES = {".csv", ".json", ".md", ".txt", ".tsv"}
TRAINING_RECORD_NAMES = {
    "train_log.csv",
    "training_complete.json",
    "alpha_history.csv",
    "multiscale_gates.txt",
}
DISALLOWED_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".ipynb",
    ".nii", ".gz", ".zip", ".tar",
}

FORMAL_RESULT_DIRS = (
    "alpha_sensitivity_test_results",
    "boundary_typical_case_multiseed_results",
    "et_lesion_stratified_results",
    "wt_lesion_stratified_results",
    "training_lesion_distributions",
)
EXPLORATORY_RESULT_DIRS = (
    "et_lesion_stratified_valid_test_results",
    "wt_lesion_stratified_valid_test_results",
)
LEGACY_RESULT_DIRS = ("comprehensive_results",)
LEGACY_ROOT_FILES = (
    "all_experiments_results.csv",
    "all_experiments_results.json",
    "key_comparison_cache.json",
    "key_comparison_results.json",
    "key_comparison_table.md",
    "lambda_results.csv",
    "paper_table.md",
    "v2_edge_results.csv",
    "et_components_detail.csv",
    "et_lesion_size_distribution.csv",
    "et_statistics.csv",
    "wt_components_detail.csv",
    "wt_lesion_size_distribution.csv",
    "wt_statistics.csv",
)
SPLIT_FILES = ("train_df.csv", "val_df.csv", "test_df.csv", "tumourCSV.csv")
METHOD_NAMING = (
    (
        "LHFC", "Laplacian High-Frequency Feature Concatenation",
        "Laplacian high-frequency features concatenated into the decoder.",
    ),
    (
        "ABS", "Auxiliary Boundary Supervision",
        "Independent boundary supervision weighted by lambda_b.",
    ),
    (
        "AR-MSC", "Adaptive Residual Multi-Scale Context",
        "Residual multi-scale context controlled by one learnable scalar alpha.",
    ),
    (
        "AFBMS-ResUNet", "Adaptive Frequency-Boundary Multi-Scale ResUNet",
        "Complete method: ResUNet + LHFC + ABS + AR-MSC.",
    ),
)


@dataclass(frozen=True)
class ModelArtifact:
    model_id: str
    display_name: str
    seed: int
    protocol: str
    source_dir: str | None
    architecture: str
    components: str
    notes: str = ""


def model_artifacts() -> tuple[ModelArtifact, ...]:
    models = [
        ModelArtifact(
            "baseline_unet3d", "3D U-Net", 55, "main_experiment",
            "Unet", "UNet3d", "baseline", "Original directory uses Unet",
        ),
        ModelArtifact(
            "baseline_resunet3d", "3D ResUNet", 55, "main_experiment",
            "ResUNet_model", "ResUNet3d", "baseline",
        ),
        ModelArtifact(
            "baseline_attention_unet3d", "3D Attention U-Net", 55,
            "main_experiment", ".autodl/attention-unet", "AttUNet3d",
            "baseline", "Actual directory differs from legacy config",
        ),
        ModelArtifact(
            "baseline_nnunet3d", "3D nnU-Net", 55, "main_experiment",
            "nnUnet", "nnUNet3d", "baseline",
            "Actual directory differs from legacy config",
        ),
    ]

    for seed in (42, 123):
        models.append(ModelArtifact(
            "baseline_resunet3d", "3D ResUNet", seed,
            "stability_runner", f"stability/seed{seed}/baseline",
            "ResUNet3d", "baseline",
            "Seed-matched reference for stability experiments",
        ))

    improved = (
        (
            "resunet_lhfc", "ResUNet + LHFC", "edge_laplacian_concat",
            "ResUNetEdge", "LHFC",
        ),
        (
            "resunet_lhfc_abs_lb0p05", "ResUNet + LHFC + ABS (lambda_b=0.05)",
            "hf_concat_boundary_w0.05", "ResUNetHFConcatBoundary",
            "LHFC + ABS(lambda_b=0.05)",
        ),
        (
            "resunet_lhfc_abs_lb0p10", "ResUNet + LHFC + ABS (lambda_b=0.10)",
            "hf_concat_boundary_w0.1", "ResUNetHFConcatBoundary",
            "LHFC + ABS(lambda_b=0.10)",
        ),
        (
            "resunet_lhfc_abs_msc", "ResUNet + LHFC + ABS + MSC",
            "hf_concat_boundary_w0.1_multiscale", "ResUNetHFConcatBoundary",
            "LHFC + ABS(lambda_b=0.10) + non-adaptive MSC",
        ),
        (
            "afbms_resunet", "AFBMS-ResUNet",
            "hf_concat_boundary_w0.1_multiscale_v2",
            "ResUNetHFConcatBoundary",
            "LHFC + ABS(lambda_b=0.10) + AR-MSC (learnable alpha)",
        ),
    )
    main_dirs = {
        "edge_laplacian_concat": "ResUNet_Edge_concat_laplacian_model",
        "hf_concat_boundary_w0.05": "ResUNet_HFConcatBoundary_w0.05_model",
        "hf_concat_boundary_w0.1": "ResUNet_HFConcatBoundary_w0.1_model",
        "hf_concat_boundary_w0.1_multiscale_v2": (
            "ResUNet_HFConcatBoundary_w0.1_multiscale_v2_model"
        ),
    }
    for model_id, display, run_name, architecture, components in improved:
        source_dir = main_dirs.get(run_name)
        note = ""
        if source_dir is None:
            note = "No seed55 main-experiment checkpoint found in AutoDL inventory"
        models.append(ModelArtifact(
            model_id, display, 55, "main_experiment", source_dir,
            architecture, components, note,
        ))
        for seed in (42, 123):
            models.append(ModelArtifact(
                model_id, display, seed, "stability_runner",
                f"stability/seed{seed}/{run_name}", architecture, components,
            ))

    models.append(ModelArtifact(
        "afbms_resunet_alpha_trace_supplement",
        "AFBMS-ResUNet alpha-trace supplement", 123,
        "supplemental_retraining",
        "stability/seed123/hf_concat_boundary_w0.1_multiscale_v2_alpha_trace",
        "ResUNetHFConcatBoundary",
        "LHFC + ABS(lambda_b=0.10) + AR-MSC (learnable alpha)",
        "Supplement only; must not replace the formal seed123 checkpoint",
    ))
    return tuple(models)


def best_checkpoint(directory: Path) -> Path:
    candidates = list(directory.glob("best_model_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"no best_model_*.pth under {directory}")

    def epoch(path: Path) -> int:
        match = re.fullmatch(r"best_model_(\d+)\.pth", path.name)
        if match is None:
            raise ValueError(f"unexpected best checkpoint name: {path.name}")
        return int(match.group(1))

    return max(candidates, key=epoch)


def checkpoint_epoch(path: Path) -> int:
    return int(re.fullmatch(r"best_model_(\d+)\.pth", path.name).group(1))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_table_tree(source: Path, destination: Path) -> int:
    if not source.is_dir():
        return 0
    copied = 0
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TABLE_SUFFIXES:
            continue
        if ".ipynb_checkpoints" in path.parts:
            continue
        target = destination / source.name / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    return copied


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def stage(args) -> Path:
    autodl_root = args.autodl_root.resolve()
    repo_root = args.repo_root.resolve()
    output = (repo_root / args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing staging directory: {output}"
        )
    output.mkdir(parents=True)

    for name in FORMAL_RESULT_DIRS:
        copy_table_tree(repo_root / name, output / "results" / "formal")
    for name in EXPLORATORY_RESULT_DIRS:
        copy_table_tree(
            repo_root / name, output / "results" / "exploratory_valid_test"
        )
    for name in LEGACY_RESULT_DIRS:
        copy_table_tree(repo_root / name, output / "results" / "legacy")

    legacy_dir = output / "results" / "legacy" / "root_files"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    for name in LEGACY_ROOT_FILES:
        source = repo_root / name
        if source.is_file():
            shutil.copy2(source, legacy_dir / name)

    split_dir = output / "manifests" / "data_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_FILES:
        source = autodl_root / name
        if source.is_file():
            shutil.copy2(source, split_dir / name)
    overview = repo_root / "autodl_overview.txt"
    if overview.is_file():
        shutil.copy2(overview, output / "manifests" / overview.name)

    manifest_rows = []
    checksum_rows = []
    for artifact in model_artifacts():
        row = asdict(artifact)
        row.update({
            "availability": "not_trained" if artifact.source_dir is None else "pending",
            "source_checkpoint": "",
            "packaged_checkpoint": "",
            "best_epoch": "",
            "size_bytes": "",
            "sha256": "",
        })
        if artifact.source_dir is not None:
            source_dir = autodl_root / artifact.source_dir
            checkpoint = best_checkpoint(source_dir)
            epoch = checkpoint_epoch(checkpoint)
            target_dir = (
                output / "weights" / artifact.model_id / f"seed_{artifact.seed}"
            )
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"best_model_epoch_{epoch:03d}.pth"
            if not args.without_weights:
                shutil.copy2(checkpoint, target)
                checksum = sha256(target)
                checksum_rows.append({
                    "sha256": checksum,
                    "path": str(target.relative_to(output)),
                })
            else:
                checksum = sha256(checkpoint)
            record_dir = (
                output / "training_records" / artifact.model_id
                / f"seed_{artifact.seed}"
            )
            for name in TRAINING_RECORD_NAMES:
                source_record = source_dir / name
                if source_record.is_file():
                    record_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_record, record_dir / name)
            row.update({
                "availability": "available",
                "source_checkpoint": str(checkpoint),
                "packaged_checkpoint": (
                    str(target.relative_to(output))
                    if not args.without_weights else "not copied in review-only mode"
                ),
                "best_epoch": epoch,
                "size_bytes": checkpoint.stat().st_size,
                "sha256": checksum,
            })
        manifest_rows.append(row)

    write_csv(output / "manifests" / "model_manifest.csv", manifest_rows)
    write_csv(output / "manifests" / "checkpoint_sha256.csv", checksum_rows)

    readme = "# AutoDL handoff review staging\n\n"
    readme += "This directory is machine-generated for review. It contains no "
    readme += "rendered figures, plotting scripts, raw BraTS volumes, last-epoch "
    readme += "checkpoints, notebook caches, or trash contents.\n\n"
    readme += "Formal results, exploratory valid+test results, and legacy results "
    readme += "are intentionally separated. See `manifests/model_manifest.csv` "
    readme += "for standardized names and missing seed/model combinations.\n"
    (output / "README_REVIEW.md").write_text(readme, encoding="utf-8")

    naming_lines = [
        "# Method naming convention",
        "",
        "| Abbreviation | English name | Definition |",
        "|---|---|---|",
    ]
    naming_lines.extend(
        f"| {short} | {english} | {definition} |"
        for short, english, definition in METHOD_NAMING
    )
    naming_lines.extend([
        "",
        "AFBMS-ResUNet uses plain LHFC concatenation (`fusion=concat`). ",
        "It does not use gated concatenation. The learnable scalar alpha belongs ",
        "only to AR-MSC in the V2 model and is not a feature-fusion gate.",
        "",
    ])
    (output / "manifests" / "METHOD_NAMING.md").write_text(
        "\n".join(naming_lines), encoding="utf-8"
    )

    unexpected = [
        path for path in output.rglob("*")
        if path.is_file()
        and path.suffix.lower() in DISALLOWED_SUFFIXES
        and path.suffix.lower() != ".pth"
    ]
    if unexpected:
        raise AssertionError(f"disallowed files staged: {unexpected}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--autodl-root", type=Path, default=Path("/root/autodl-tmp")
    )
    parser.add_argument(
        "--repo-root", type=Path,
        default=Path("/root/autodl-tmp/mri_deep"),
    )
    parser.add_argument("--output-dir", default="handoff_review_data")
    parser.add_argument(
        "--without-weights", action="store_true",
        help="Create a small metadata-only staging directory",
    )
    args = parser.parse_args()
    output = stage(args)
    total_bytes = sum(
        path.stat().st_size for path in output.rglob("*") if path.is_file()
    )
    print(f"Staged handoff review data: {output}")
    print(f"Total size: {total_bytes / (1024 ** 2):.1f} MiB")
    print(f"Model rows: {len(model_artifacts())}")
    print("Missing combinations are recorded as availability=not_trained")


if __name__ == "__main__":
    main()
