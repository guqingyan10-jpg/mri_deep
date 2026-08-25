"""Run seed-paired multi-scale context experiments on AutoDL.

Each job reuses the exact completed ResUNet baseline checkpoint for its seed
and changes only the optional bottleneck context module.
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_seed_stability import find_completed_baseline


@dataclass(frozen=True)
class MultiScaleTrainingJob:
    seed: int
    checkpoint_dir: Path
    arguments: tuple[str, ...]

    def command(self, baseline_checkpoint: str) -> list[str]:
        if not baseline_checkpoint:
            raise ValueError(f"seed {self.seed} requires a baseline checkpoint")
        return [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_hf_concat_boundary.py"),
            *self.arguments,
            "--baseline_checkpoint",
            baseline_checkpoint,
        ]


def build_jobs(seeds: Sequence[int], output_root: Path,
               epochs: int, lr: float) -> list[MultiScaleTrainingJob]:
    jobs = []
    for seed in seeds:
        checkpoint_dir = (
            output_root / f"seed{seed}" /
            "hf_concat_boundary_w0.1_multiscale"
        )
        jobs.append(MultiScaleTrainingJob(
            seed=seed,
            checkpoint_dir=checkpoint_dir,
            arguments=(
                "--seed", str(seed),
                "--epochs", str(epochs),
                "--lr", str(lr),
                "--fusion", "concat",
                "--boundary_weight", "0.1",
                "--multiscale_context",
                "--checkpoint_dir", str(checkpoint_dir),
            ),
        ))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train paired multi-scale context models for selected seeds",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    parser.add_argument("--output_root", type=Path,
                        default=Path("/root/autodl-tmp/stability"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs(args.seeds, args.output_root, args.epochs, args.lr)
    for job in jobs:
        baseline_dir = args.output_root / f"seed{job.seed}" / "baseline"
        baseline = find_completed_baseline(baseline_dir)
        command = job.command(str(baseline))
        print(f"\n[hf_concat_boundary_w0.1_multiscale | seed={job.seed}]")
        print(f"Paired baseline checkpoint: {baseline}")
        print(subprocess.list2cmdline(command))
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
