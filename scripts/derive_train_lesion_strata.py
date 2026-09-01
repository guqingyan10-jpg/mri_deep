"""Derive fixed lesion-size strata from the training split only.

The split exactly mirrors :func:`data.dataset.get_dataloader`.  Connected
components are extracted after the same spatial crop used by ``BratsDataset``
so the resulting thresholds can be frozen and applied unchanged to validation
and test predictions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "lesion_stratified_for_thresholds",
    PROJECT_ROOT / "evaluation" / "wt_lesion_stratified.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
derive_size_strata = _MODULE.derive_size_strata


REGION_LABELS = {
    "WT": (1, 2, 4),
    "ET": (4,),
}


def training_rows(csv_path: str | Path) -> pd.DataFrame:
    """Return the deterministic 70% training partition used by the project."""
    frame = pd.read_csv(csv_path)
    train, _ = train_test_split(
        frame, test_size=0.3, random_state=10, shuffle=True
    )
    return train.reset_index(drop=True)


def component_sizes(
    rows: pd.DataFrame,
    region: str,
    data_dir: str | Path | None,
    min_component_size: int,
) -> list[int]:
    structure = ndimage.generate_binary_structure(3, 3)
    sizes: list[int] = []
    for row in rows.itertuples(index=False):
        case_id = row.Brats20ID
        case_dir = Path(data_dir) / case_id if data_dir else Path(row.path)
        mask_path = case_dir / f"{case_id}_seg.nii"
        image = nib.load(mask_path, mmap=False)
        mask = np.asarray(image.dataobj, dtype=np.int16)
        del image
        # Match BratsDataset.resize() before connected-component extraction.
        mask = mask[40:210, 40:210, 20:120]
        labeled, count = ndimage.label(
            np.isin(mask, REGION_LABELS[region]), structure=structure
        )
        counts = np.bincount(labeled.ravel())[1 : count + 1]
        sizes.extend(int(value) for value in counts if value >= min_component_size)
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="tumourCSV.csv")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--region", choices=("WT", "ET", "BOTH"), default="BOTH")
    parser.add_argument("--min-component-size", type=int, default=10)
    parser.add_argument("--output-dir", default="training_lesion_distributions")
    args = parser.parse_args()

    rows = training_rows(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    regions = REGION_LABELS if args.region == "BOTH" else (args.region,)
    for region in regions:
        sizes = component_sizes(
            rows, region, args.data_dir, args.min_component_size
        )
        strata = derive_size_strata(sizes, args.min_component_size)
        small_upper = strata["small"][1]
        medium_upper = strata["medium"][1]
        counts = {
            "small": sum(value <= small_upper for value in sizes),
            "medium": sum(small_upper < value <= medium_upper for value in sizes),
            "large": sum(value > medium_upper for value in sizes),
        }
        distribution = (
            pd.Series(sizes, name="lesion_voxels")
            .value_counts()
            .rename_axis("lesion_voxels")
            .rename("lesion_count")
            .sort_index()
            .reset_index()
        )
        distribution["cumulative_count"] = distribution["lesion_count"].cumsum()
        distribution["cumulative_fraction"] = (
            distribution["cumulative_count"] / len(sizes)
        )
        distribution["stratum"] = distribution["lesion_voxels"].map(
            lambda value: _MODULE.classify_lesion_size(value, strata)
        )
        prefix = region.lower()
        distribution.to_csv(
            output_dir / f"{prefix}_training_lesion_size_distribution.csv",
            index=False,
        )
        metadata = {
            "region": region,
            "fit_split": "train",
            "split_random_state": 10,
            "training_cases": len(rows),
            "connectivity": 26,
            "min_component_size": args.min_component_size,
            "training_component_count": len(sizes),
            "strata": strata,
            "stratum_counts": counts,
        }
        with open(output_dir / f"{prefix}_training_lesion_strata.json", "w",
                  encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        print(f"{region}: training_cases={len(rows)}, components={len(sizes)}")
        print(
            f"  small={args.min_component_size}-{small_upper} (n={counts['small']})\n"
            f"  medium={small_upper + 1}-{medium_upper} (n={counts['medium']})\n"
            f"  large={medium_upper + 1}-inf (n={counts['large']})"
        )


if __name__ == "__main__":
    main()
