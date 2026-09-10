"""Prepared UCSF-BMSR dataset with a fixed, patient-grouped data split."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


REQUIRED_MANIFEST_COLUMNS = {
    "subject_id",
    "patient_group",
    "split",
    "prepared_path",
}


def patient_group(subject_id: str) -> str:
    """Map longitudinal IDs such as 100108A/B/C to one patient."""
    match = re.match(r"^(\d+)", str(subject_id))
    if match:
        return match.group(1)
    return re.sub(r"[A-Za-z]+$", "", str(subject_id)) or str(subject_id)


def assign_grouped_splits(
    subject_ids: list[str], seed: int = 10,
) -> dict[str, str]:
    """Reproduce the BraTS 70/20/10 logic while keeping patients intact.

    The original project permutes the cohort with random_state=10, assigns
    30% to a temporary set, then takes the first two thirds as validation and
    the last third as test. Here the same operation is performed on patient
    groups, because UCSF can contain several scans from the same patient.
    """
    groups = sorted({patient_group(subject_id) for subject_id in subject_ids})
    if len(groups) < 4:
        raise ValueError("At least four patient groups are required")
    permutation = np.random.RandomState(seed).permutation(len(groups))
    n_temporary = int(math.ceil(0.30 * len(groups)))
    temporary = [groups[index] for index in permutation[:n_temporary]]
    training = [groups[index] for index in permutation[n_temporary:]]
    valid_cut = len(temporary) * 2 // 3
    validation = temporary[:valid_cut]
    testing = temporary[valid_cut:]
    group_to_split = {
        **{group: "train" for group in training},
        **{group: "valid" for group in validation},
        **{group: "test" for group in testing},
    }
    return {
        subject_id: group_to_split[patient_group(subject_id)]
        for subject_id in subject_ids
    }


def read_manifest(path_to_csv: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path_to_csv)
    missing = REQUIRED_MANIFEST_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"UCSF manifest is missing columns: {sorted(missing)}")
    invalid = set(frame["split"]) - {"train", "valid", "test"}
    if invalid:
        raise ValueError(f"UCSF manifest contains invalid splits: {sorted(invalid)}")
    duplicated = frame["subject_id"].duplicated(keep=False)
    if duplicated.any():
        values = frame.loc[duplicated, "subject_id"].tolist()
        raise ValueError(f"Duplicate UCSF subject IDs: {values[:10]}")
    leakage = frame.groupby("patient_group")["split"].nunique()
    if (leakage > 1).any():
        groups = leakage[leakage > 1].index.tolist()
        raise ValueError(f"Patient leakage across UCSF splits: {groups[:10]}")
    return frame


class UCSFPreparedDataset(Dataset):
    """Read preprocessed four-channel images and one-channel binary masks."""

    def __init__(self, df: pd.DataFrame, phase: str = "test"):
        self.df = df.reset_index(drop=True)
        self.phase = phase

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        path = Path(row["prepared_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Prepared UCSF case is missing: {path}")
        with np.load(path, allow_pickle=False) as case:
            image = np.asarray(case["image"], dtype=np.float32)
            mask = np.asarray(case["mask"], dtype=np.float32)
            spacing = np.asarray(case["spacing_dhw"], dtype=np.float32)
        if image.ndim != 4 or image.shape[0] != 4:
            raise ValueError(f"Expected image (4,D,H,W), got {image.shape}: {path}")
        if mask.shape != (1, *image.shape[1:]):
            raise ValueError(f"Mask/image shape mismatch {mask.shape}/{image.shape}: {path}")
        return {
            "Id": str(row["subject_id"]),
            "patient_group": str(row["patient_group"]),
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "spacing_dhw": torch.from_numpy(spacing),
        }


def get_ucsf_dataloader(
    dataset: type[Dataset],
    path_to_csv: str,
    phase: str,
    fold: int = 0,
    batch_size: int = 1,
    num_workers: int = 0,
):
    """Build a loader from the frozen manifest without resplitting it."""
    del fold
    if phase not in {"train", "valid", "test"}:
        raise ValueError(f"Unsupported UCSF phase: {phase}")
    frame = read_manifest(path_to_csv)
    selected = frame.loc[frame["split"] == phase].reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"No UCSF cases assigned to split {phase!r}")
    return DataLoader(
        dataset(selected, phase),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )
