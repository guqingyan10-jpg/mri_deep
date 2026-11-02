"""Keep random-initialized Full experiments separate from legacy warm starts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def guard_full_initialization(
    directory: Path | str, *, from_scratch: bool, dataset: str, seed: int,
    manifest: Path | str, epochs: int, lr: float, model_settings: dict,
) -> None:
    """Record scratch provenance and reject incompatible/uncertified weights.

    Existing legacy warm-start runs retain their previous behavior. A scratch
    run may reuse its own recorded directory, but never an unlabelled legacy
    checkpoint. Launch-level locking is documented separately.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "initialization_protocol.json"
    if not from_scratch and not marker.exists():
        return
    manifest = Path(manifest).resolve()
    protocol = {
        "schema_version": 1,
        "initialization": "from_scratch" if from_scratch else "baseline_warm_start",
        "dataset": dataset,
        "seed": int(seed),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "epochs_argument": int(epochs),
        "lr": float(lr),
        "optimizer": "Adam",
        "batch_size": 1,
        "accumulation_steps": 4,
        "early_stopping": {"monitor": "valid_loss", "patience": 25, "min_delta": 1e-4},
        "model_settings": model_settings,
    }
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous != protocol:
            raise ValueError(
                f"Initialization/training protocol differs in {directory}. "
                "Use a new experiment directory; do not mix warm-start and scratch runs."
            )
        return
    existing = list(directory.glob("*.pth"))
    existing += [directory / name for name in (
        "train_log.csv", "training_complete.json", "alpha_history.csv"
    ) if (directory / name).exists()]
    if existing:
        raise ValueError(
            f"Cannot certify from-scratch initialization in existing run {directory}. "
            "Choose a new directory; the old weights/results were not modified."
        )
    # Exclusive creation refuses a simultaneous first-time metadata writer.
    with marker.open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2, sort_keys=True)
        stream.write("\n")
