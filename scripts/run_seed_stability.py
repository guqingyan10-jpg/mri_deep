"""Run paired stability-training experiments for multiple random seeds.

For each seed, the BCEDice baseline is trained/resumed first.  The same
baseline's best checkpoint is then used to warm-start Edge and both HF
boundary-weight variants.  Training runs sequentially to avoid GPU contention.
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrainingJob:
    name: str
    seed: int
    checkpoint_dir: Path
    script_name: str
    arguments: tuple[str, ...]
    needs_baseline: bool = False

    def command(self, baseline_checkpoint: str | None = None) -> list[str]:
        command = [sys.executable, str(REPO_ROOT / "scripts" / self.script_name)]
        command.extend(self.arguments)
        if self.needs_baseline:
            if not baseline_checkpoint:
                raise ValueError(f"{self.name} requires a baseline checkpoint")
            command.extend(["--baseline_checkpoint", baseline_checkpoint])
        return command


def build_jobs(
    seeds: Sequence[int], output_root: Path, epochs: int, lr: float,
) -> list[TrainingJob]:
    """Build baseline and derived-model jobs with isolated seed directories."""
    jobs: list[TrainingJob] = []
    for seed in seeds:
        seed_dir = output_root / f"seed{seed}"
        common = ("--seed", str(seed), "--epochs", str(epochs), "--lr", str(lr))
        jobs.extend([
            TrainingJob(
                name="baseline",
                seed=seed,
                checkpoint_dir=seed_dir / "baseline",
                script_name="train_baseline_bcedice.py",
                arguments=common + ("--checkpoint_dir", str(seed_dir / "baseline")),
            ),
            TrainingJob(
                name="edge_laplacian_concat",
                seed=seed,
                checkpoint_dir=seed_dir / "edge_laplacian_concat",
                script_name="train_v2_edge.py",
                arguments=common + (
                    "--fusion", "concat", "--edge_type", "laplacian",
                    "--checkpoint_dir", str(seed_dir / "edge_laplacian_concat"),
                ),
                needs_baseline=True,
            ),
            TrainingJob(
                name="hf_concat_boundary_w0.1",
                seed=seed,
                checkpoint_dir=seed_dir / "hf_concat_boundary_w0.1",
                script_name="train_hf_concat_boundary.py",
                arguments=common + (
                    "--boundary_weight", "0.1",
                    "--checkpoint_dir", str(seed_dir / "hf_concat_boundary_w0.1"),
                ),
                needs_baseline=True,
            ),
            TrainingJob(
                name="hf_concat_boundary_w0.05",
                seed=seed,
                checkpoint_dir=seed_dir / "hf_concat_boundary_w0.05",
                script_name="train_hf_concat_boundary.py",
                arguments=common + (
                    "--boundary_weight", "0.05",
                    "--checkpoint_dir", str(seed_dir / "hf_concat_boundary_w0.05"),
                ),
                needs_baseline=True,
            ),
        ])
    return jobs


def find_baseline_best(checkpoint_dir: Path) -> Path:
    checkpoints = list(checkpoint_dir.glob("best_model_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No baseline best_model_*.pth found in {checkpoint_dir}. "
            "The paired Edge/HF experiments will not be started."
        )
    return max(checkpoints, key=lambda path: int(path.stem.rsplit("_", 1)[1]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run seed-paired Baseline, Edge, and HF stability training",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 55, 123])
    parser.add_argument("--output_root", type=Path,
                        default=Path("/root/autodl-tmp/stability"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without starting training")
    args = parser.parse_args()

    jobs = build_jobs(args.seeds, args.output_root, args.epochs, args.lr)
    baseline_paths: dict[int, Path] = {}
    for job in jobs:
        if job.name == "baseline":
            command = job.command()
        else:
            baseline = baseline_paths.get(job.seed)
            if baseline is None:
                raise RuntimeError(f"Baseline was not prepared for seed {job.seed}")
            command = job.command(str(baseline))

        print(f"\n[{job.name} | seed={job.seed}]")
        print(subprocess.list2cmdline(command))
        if args.dry_run:
            if job.name == "baseline":
                baseline_paths[job.seed] = (
                    job.checkpoint_dir / "best_model_<best_epoch>.pth"
                )
            continue

        subprocess.run(command, cwd=REPO_ROOT, check=True)
        if job.name == "baseline":
            baseline_paths[job.seed] = find_baseline_best(job.checkpoint_dir)
            print(f"Using paired baseline checkpoint: {baseline_paths[job.seed]}")


if __name__ == "__main__":
    main()
