"""Retrain the V2 Full model for seed123 while preserving the original run.

The original seed123 directory is never used as an output target.  This runner
uses the same paired seed123 baseline warm-start and writes checkpoints plus an
epoch-by-epoch alpha trace to a dedicated sibling directory.
"""

from __future__ import annotations

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


SEED = 123
ORIGINAL_RUN_NAME = "hf_concat_boundary_w0.1_multiscale_v2"
TRACE_RUN_NAME = "hf_concat_boundary_w0.1_multiscale_v2_alpha_trace"


@dataclass(frozen=True)
class AlphaTraceJob:
    checkpoint_dir: Path
    original_dir: Path
    baseline_checkpoint: Path
    epochs: int
    lr: float

    def command(self) -> list[str]:
        if self.checkpoint_dir.resolve() == self.original_dir.resolve():
            raise ValueError("alpha-trace output must not overwrite the original run")
        return [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_hf_concat_boundary.py"),
            "--seed",
            str(SEED),
            "--epochs",
            str(self.epochs),
            "--lr",
            str(self.lr),
            "--fusion",
            "concat",
            "--boundary_weight",
            "0.1",
            "--multiscale_context_v2",
            "--checkpoint_dir",
            str(self.checkpoint_dir),
            "--baseline_checkpoint",
            str(self.baseline_checkpoint),
        ]


def build_job(output_root: Path, epochs: int, lr: float) -> AlphaTraceJob:
    seed_root = output_root / f"seed{SEED}"
    baseline = find_completed_baseline(seed_root / "baseline")
    return AlphaTraceJob(
        checkpoint_dir=seed_root / TRACE_RUN_NAME,
        original_dir=seed_root / ORIGINAL_RUN_NAME,
        baseline_checkpoint=baseline,
        epochs=epochs,
        lr=lr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-tmp/stability"),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    job = build_job(args.output_root, args.epochs, args.lr)
    command = job.command()
    print(f"Original run preserved: {job.original_dir}")
    print(f"New alpha-trace run:    {job.checkpoint_dir}")
    print(f"Paired baseline:        {job.baseline_checkpoint}")
    print(subprocess.list2cmdline(command))
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
