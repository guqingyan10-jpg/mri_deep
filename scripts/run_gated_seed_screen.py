"""Run the two gated Edge screening experiments for one completed seed.

Both arms warm-start from the exact same seed-matched baseline best checkpoint.
Training remains sequential to avoid GPU contention.
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_seed_stability import find_completed_baseline


@dataclass(frozen=True)
class GatedTrainingJob:
    name: str
    seed: int
    checkpoint_dir: Path
    script_name: str
    arguments: tuple[str, ...]

    def command(self, baseline_checkpoint: str) -> list[str]:
        if not baseline_checkpoint:
            raise ValueError(f"{self.name} requires a baseline checkpoint")
        return [
            sys.executable,
            str(REPO_ROOT / "scripts" / self.script_name),
            *self.arguments,
            "--baseline_checkpoint",
            baseline_checkpoint,
        ]


def build_jobs(seed: int, output_root: Path, epochs: int,
               lr: float) -> list[GatedTrainingJob]:
    """Build the gated Edge and gated Edge+Boundary jobs for one seed."""
    seed_dir = output_root / f"seed{seed}"
    common = ("--seed", str(seed), "--epochs", str(epochs), "--lr", str(lr))
    return [
        GatedTrainingJob(
            name="edge_laplacian_gated_concat",
            seed=seed,
            checkpoint_dir=seed_dir / "edge_laplacian_gated_concat",
            script_name="train_v2_edge.py",
            arguments=common + (
                "--fusion", "gated_concat",
                "--edge_type", "laplacian",
                "--checkpoint_dir",
                str(seed_dir / "edge_laplacian_gated_concat"),
            ),
        ),
        GatedTrainingJob(
            name="hf_gated_concat_boundary_w0.1",
            seed=seed,
            checkpoint_dir=seed_dir / "hf_gated_concat_boundary_w0.1",
            script_name="train_hf_concat_boundary.py",
            arguments=common + (
                "--fusion", "gated_concat",
                "--boundary_weight", "0.1",
                "--checkpoint_dir",
                str(seed_dir / "hf_gated_concat_boundary_w0.1"),
            ),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train paired gated Edge variants from one seed baseline",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_root", type=Path,
                        default=Path("/root/autodl-tmp/stability"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    seed_dir = args.output_root / f"seed{args.seed}"
    baseline = find_completed_baseline(seed_dir / "baseline")
    jobs = build_jobs(args.seed, args.output_root, args.epochs, args.lr)

    print(f"Paired baseline checkpoint: {baseline}")
    for job in jobs:
        command = job.command(str(baseline))
        print(f"\n[{job.name} | seed={job.seed}]")
        print(subprocess.list2cmdline(command))
        if not args.dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
