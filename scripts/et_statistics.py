"""
=============================================================================
Per-Case Region and Lesion Statistics for BraTS2020
=============================================================================
Computes for each of the 369 training cases:
  1. ET voxel count        — number of voxels with label 4
  2. WT voxel count        — number of voxels with labels 1+2+4 (whole tumor)
  3. TC voxel count        — number of voxels with labels 1+4 (tumor core)
  4. ET / WT ratio         — enhancing tumor proportion in whole tumor
  5. ET / TC ratio         — enhancing tumor proportion in tumor core
  6. Selected-region connected components — number, sizes, multi-focal analysis
  7. Component-level details — size distribution, ratio of largest to total

Outputs:
  - et_statistics.csv       — one row per case
  - et_components_detail.csv — one row per individual ET lesion component
  - et_lesion_size_distribution.csv — exact lesion-size counts
  - et_lesion_size_distribution.png — lesion-size/count curve

Usage:
    python scripts/et_statistics.py --region WT

Requires: nibabel, numpy, scipy, pandas, tqdm
Data: BraTS2020 TrainingData at DATA_DIR

Author: Generated for ResUNet enhancement project
Date:   2026-08-01
=============================================================================
"""

import argparse
import os
import gc
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================

parser = argparse.ArgumentParser(
    description='Compute per-case and per-lesion statistics for ET, TC, or WT.'
)
parser.add_argument('--region', choices=('ET', 'TC', 'WT'), default='WT',
                    help='Region used for connected-component analysis (default: WT).')
parser.add_argument('--csv-path', default='tumourCSV.csv')
parser.add_argument('--data-dir', default=None,
                    help='Root directory containing one folder per case.')
parser.add_argument('--output-path', default=None,
                    help='Per-case CSV path (default: <region>_statistics.csv).')
parser.add_argument('--detail-output-path', default=None,
                    help='Component CSV path (default: <region>_components_detail.csv).')
parser.add_argument('--distribution-output-path', default=None,
                    help='Size distribution CSV path (default: <region>_lesion_size_distribution.csv).')
parser.add_argument('--plot-path', default=None,
                    help='Distribution plot path (default: <region>_lesion_size_distribution.png).')
parser.add_argument('--connectivity', type=int, choices=(6, 26), default=26,
                    help='3D connected-component definition (default: 26).')
parser.add_argument('--min-component-size', type=int, default=10,
                    help='Ignore components smaller than this many voxels.')
args = parser.parse_args()

CSV_PATH = args.csv_path
REGION = args.region
region_prefix = REGION.lower()
OUTPUT_PATH = args.output_path or f'{region_prefix}_statistics.csv'
OUTPUT_DETAIL_PATH = args.detail_output_path or f'{region_prefix}_components_detail.csv'
DISTRIBUTION_OUTPUT_PATH = (
    args.distribution_output_path
    or f'{region_prefix}_lesion_size_distribution.csv'
)
PLOT_PATH = args.plot_path or f'{region_prefix}_lesion_size_distribution.png'
MIN_COMPONENT_SIZE = args.min_component_size
CONNECTIVITY = args.connectivity
CONNECTIVITY_STRUCTURE = ndimage.generate_binary_structure(
    3, 1 if CONNECTIVITY == 6 else 3
)
REGION_LABELS = {
    'ET': (4,),
    'TC': (1, 4),
    'WT': (1, 2, 4),
}

# Data directory — ADAPT TO YOUR ENVIRONMENT:
#   Local Windows:
DATA_DIR = args.data_dir or r'D:\lunwen\base\enhance_resu\MICCAI_BraTS2020_TrainingData\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'
#   Server (autodl-tmp):
# DATA_DIR = r'/root/autodl-tmp/brats_project/MICCAI_BraTS2020_TrainingData/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'

GC_INTERVAL = 10

print("=" * 70)
print(f"BraTS2020 Per-Case {REGION} & Connected Component Statistics")
print(f"Connected-component region: {REGION}")
print(f"Connectivity: {CONNECTIVITY}-neighborhood")
print(f"Minimum component size: {MIN_COMPONENT_SIZE} voxels")
print("=" * 70)

# ============================================================
# Load patient list
# ============================================================

df = pd.read_csv(CSV_PATH)
print(f"\nTotal cases in tumourCSV.csv: {len(df)}")

# ============================================================
# Compute statistics for each case
# ============================================================

results = []
all_components = []  # detailed per-component records
errors = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing cases"):
    case_id = row['Brats20ID']
    seg_path = os.path.join(DATA_DIR, case_id, f"{case_id}_seg.nii")

    if not os.path.exists(seg_path):
        errors.append((case_id, "File missing"))
        continue

    try:
        # --- Load mask with explicit cleanup ---
        img = nib.load(seg_path, mmap=False)
        voxel_spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
        voxel_volume_mm3 = float(np.prod(voxel_spacing))
        mask_data = np.asarray(img.dataobj, dtype=np.int16)
        del img

        # BraTS label definitions:
        #  Label 0: Background
        #  Label 1: NCR/NET (necrotic core / non-enhancing tumor)
        #  Label 2: ED   (peritumoral edema)
        #  Label 4: ET   (enhancing tumor)

        et_mask = (mask_data == 4)
        wt_voxels = int(np.isin(mask_data, [1, 2, 4]).sum())
        tc_voxels = int(np.isin(mask_data, [1, 4]).sum())
        et_voxels = int(et_mask.sum())

        # --- Proportions ---
        et_wt_ratio = et_voxels / wt_voxels if wt_voxels > 0 else 0.0
        et_tc_ratio = et_voxels / tc_voxels if tc_voxels > 0 else 0.0

        # --- Connected component analysis on the selected region ---
        region_mask = np.isin(mask_data, REGION_LABELS[REGION])
        region_voxels = int(region_mask.sum())
        del mask_data
        component_sizes = []
        largest_comp = 0
        if region_voxels > 0:
            labeled, num_labels = ndimage.label(
                region_mask, structure=CONNECTIVITY_STRUCTURE
            )

            for comp_id in range(1, num_labels + 1):
                size = int((labeled == comp_id).sum())
                if size >= MIN_COMPONENT_SIZE:
                    component_sizes.append(size)

            del labeled
        del region_mask

        num_components = len(component_sizes)
        is_multifocal = (num_components >= 2)
        largest_comp = max(component_sizes) if component_sizes else 0

        # Ratio: largest component / total selected-region mass.
        if region_voxels > 0 and largest_comp > 0:
            largest_ratio = round(largest_comp / region_voxels, 4)
        else:
            largest_ratio = 0.0

        # How much of the selected region is in the 2nd+ components?
        secondary_mass = sum(sorted(component_sizes)[:-1]) if len(component_sizes) >= 2 else 0
        secondary_ratio = round(secondary_mass / region_voxels, 4) if region_voxels > 0 else 0.0

        results.append({
            'Brats20ID':           case_id,
            'ET_voxels':           et_voxels,
            'WT_voxels':           wt_voxels,
            'TC_voxels':           tc_voxels,
            'ET_WT_ratio':         round(et_wt_ratio, 6),
            'ET_TC_ratio':         round(et_tc_ratio, 6),
            # Legacy ET component fields are retained for ET runs only.
            'ET_components':       num_components if REGION == 'ET' else 0,
            'ET_largest_ratio':    largest_ratio if REGION == 'ET' else 0.0,
            'ET_secondary_ratio':  secondary_ratio if REGION == 'ET' else 0.0,
            'ET_comp_sizes':       str(component_sizes) if REGION == 'ET' else '[]',
            'is_multifocal':       is_multifocal,
            'region':              REGION,
            'region_voxels':       region_voxels,
            'region_components':  num_components,
            'region_largest_ratio': largest_ratio,
            'region_secondary_ratio': secondary_ratio,
            'region_comp_sizes':  str(component_sizes),
        })

        # --- Record each component individually ---
        for i, sz in enumerate(component_sizes):
            all_components.append({
                'Brats20ID':         case_id,
                'lesion_id':         i + 1,
                'lesion_voxels':     sz,
                'lesion_volume_mm3': round(sz * voxel_volume_mm3, 3),
                'region':            REGION,
            })

    except Exception as e:
        errors.append((case_id, str(e)))

    if (idx + 1) % GC_INTERVAL == 0:
        gc.collect()

gc.collect()

# ============================================================
# Report errors
# ============================================================
if errors:
    print(f"\n[WARNING] {len(errors)} case(s) had errors:")
    for case_id, err in errors[:10]:
        print(f"  {case_id}: {err[:120]}")
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more")

# ============================================================
# Build & save DataFrames
# ============================================================

stats_df = pd.DataFrame(results)
stats_df = stats_df.sort_values('ET_voxels', ascending=False).reset_index(drop=True)
stats_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved {len(stats_df)} cases to: {OUTPUT_PATH}")

comp_df = pd.DataFrame(all_components)
if len(comp_df) > 0:
    comp_df.to_csv(OUTPUT_DETAIL_PATH, index=False)
    print(f"Saved {len(comp_df)} individual components to: {OUTPUT_DETAIL_PATH}")

    # Exact count-by-size table and a log-scaled curve make threshold selection
    # readable despite the highly skewed lesion-size distribution.
    distribution_df = (
        comp_df.groupby('lesion_voxels').size()
        .rename('lesion_count')
        .reset_index()
        .sort_values('lesion_voxels')
    )
    distribution_df['cumulative_count'] = distribution_df['lesion_count'].cumsum()
    distribution_df['cumulative_fraction'] = (
        distribution_df['cumulative_count'] / len(comp_df)
    )
    distribution_df.to_csv(DISTRIBUTION_OUTPUT_PATH, index=False)
    print(f"Saved lesion-size distribution to: {DISTRIBUTION_OUTPUT_PATH}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
        ax.plot(
            distribution_df['lesion_voxels'],
            distribution_df['lesion_count'],
            color='#1f77b4', linewidth=1.2, marker='.', markersize=3,
        )
        ax.set_xscale('log')
        ax.set_xlabel(f'{REGION} lesion size (voxels, log scale)')
        ax.set_ylabel('Number of lesions')
        ax.set_title(
        f'{REGION} lesion-size distribution ({CONNECTIVITY}-connectivity, '
            f'minimum {MIN_COMPONENT_SIZE} voxels)'
        )
        ax.grid(True, which='both', alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOT_PATH, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved lesion-size curve to: {PLOT_PATH}")
    except ImportError:
        print('[WARNING] matplotlib is unavailable; skipped PNG curve.')

if len(stats_df) == 0:
    print("No results to summarize. Exiting.")
    exit(1)

# ============================================================
# SECTION A: OVERALL VOLUME STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("SECTION A: OVERALL VOLUME STATISTICS")
print("=" * 70)

headers = f"{'Metric':<38} {'Mean':>8} {'Std':>8} {'Min':>8} {'Q25':>8} {'Median':>8} {'Q75':>8} {'Max':>8}"
print(f"\n{headers}")
print("-" * len(headers))

for col, name in [
    ('ET_voxels',     'ET Voxels'),
    ('WT_voxels',     'WT Voxels'),
    ('TC_voxels',     'TC Voxels'),
    ('ET_WT_ratio',   'ET / WT Ratio'),
    ('ET_TC_ratio',   'ET / TC Ratio'),
]:
    vals = stats_df[col]
    q25, q50, q75 = vals.quantile([0.25, 0.5, 0.75])
    print(f"{name:<38} {vals.mean():>8.1f} {vals.std():>8.1f} {vals.min():>8.1f} "
          f"{q25:>8.1f} {q50:>8.1f} {q75:>8.1f} {vals.max():>8.1f}")

# ============================================================
# SECTION B: CONNECTED COMPONENT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print(f"SECTION B: CONNECTED COMPONENT ANALYSIS (Multi-Focal {REGION})")
print("=" * 70)

# B.1 — Per-case component count distribution
print(f"\n--- B.1: Cases by Number of {REGION} Components ---")
comp_counts = stats_df['region_components'].value_counts().sort_index()
for n_comp, count in comp_counts.items():
    pct = 100 * count / len(stats_df)
    bar = '█' * int(pct)
    label = "Single lesion" if n_comp == 1 else ("Multi-focal" if n_comp >= 2 else f"No {REGION}")
    print(f"  {n_comp:>2} component(s) [{label:<20}]: {count:>4} / {len(stats_df)} "
          f"({pct:>5.1f}%)  {bar}")

# B.2 — Summary counts
n_has_region   = (stats_df['region_voxels'] > 0).sum()
n_zero_region  = (stats_df['region_voxels'] == 0).sum()
n_single       = (stats_df['region_components'] == 1).sum()
n_multifocal   = stats_df['is_multifocal'].sum()
n_3plus        = (stats_df['region_components'] >= 3).sum()
n_5plus        = (stats_df['region_components'] >= 5).sum()

total_lesions  = stats_df['region_components'].sum()

print(f"\n--- B.2: Summary Counts ---")
print(f"  Total cases:                            {len(stats_df)}")
print(f"  Cases WITH {REGION} ({REGION}>0):                  {n_has_region}  ({100*n_has_region/len(stats_df):.1f}%)")
print(f"  Cases with ZERO {REGION}:                    {n_zero_region}  ({100*n_zero_region/len(stats_df):.1f}%)")
print(f"  Single-lesion cases (1 component):      {n_single}  ({100*n_single/len(stats_df):.1f}%)")
print(f"  Multi-focal cases (>=2 components):     {n_multifocal}  ({100*n_multifocal/len(stats_df):.1f}%)")
print(f"  Cases with >=3 components:              {n_3plus}  ({100*n_3plus/len(stats_df):.1f}%)")
print(f"  Cases with >=5 components:              {n_5plus}  ({100*n_5plus/len(stats_df):.1f}%)")
print(f"  TOTAL ET lesions across all cases:      {total_lesions}")

# B.3 — Component size statistics (individual lesions)
if len(comp_df) > 0:
    print(f"\n--- B.3: Individual Lesion Size Statistics (n={len(comp_df)}) ---")
    cs = comp_df['lesion_voxels']
    q25, q50, q75 = cs.quantile([0.25, 0.5, 0.75])
    print(f"  {'Lesion Size':<40} {'Value':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Mean component size:':<40} {cs.mean():>10.1f} voxels")
    print(f"  {'Std component size:':<40} {cs.std():>10.1f} voxels")
    print(f"  {'25th percentile:':<40} {q25:>10.1f} voxels")
    print(f"  {'Median component size:':<40} {q50:>10.1f} voxels")
    print(f"  {'75th percentile:':<40} {q75:>10.1f} voxels")
    print(f"  {'Total region volume (all lesions):':<40} {cs.sum():>10.0f} voxels")

    # Component size histogram bins
    print(f"\n--- B.4: Lesion Size Distribution ---")
    bins = [(0, 50), (50, 100), (100, 500), (500, 1000),
            (1000, 5000), (5000, 10000), (10000, 50000), (50000, 999999)]
    for lo, hi in bins:
        cnt = ((cs >= lo) & (cs < hi)).sum()
        pct = 100 * cnt / len(comp_df)
        bar = '|' * int(pct * 2)
        label = f">{hi-1}" if hi == 999999 else f"{lo}-{hi-1}"
        print(f"  {label:>10} voxels: {cnt:>6} lesions ({pct:>5.1f}%)  {bar}")

    # B.5 — Cases with most lesions
    print(f"\n--- B.5: Cases with Most {REGION} Lesions ---")
    top_multifocal = stats_df[stats_df['region_components'] >= 2].head(15)
    for _, r in top_multifocal.iterrows():
        print(f"  {r['Brats20ID']}: {r['region_components']} lesions, "
              f"{REGION}={r['region_voxels']}, sizes={r['region_comp_sizes']}")

    # B.6 — Fragmentation analysis: largest-to-total ratio
    print(f"\n--- B.6: Lesion Fragmentation (Largest/Total {REGION} ratio) ---")
    ratios = stats_df[stats_df['region_voxels'] > 0]['region_largest_ratio']
    print(f"  Ratio = 1.000 (single lesion):           {(ratios == 1.0).sum()} cases")
    print(f"  Ratio >= 0.900 (dominant main lesion):   {(ratios >= 0.9).sum()} cases")
    print(f"  Ratio 0.700-0.899 (moderate secondary):  {((ratios >= 0.7) & (ratios < 0.9)).sum()} cases")
    print(f"  Ratio 0.500-0.699 (large secondary):     {((ratios >= 0.5) & (ratios < 0.7)).sum()} cases")
    print(f"  Ratio < 0.500 (fragmented / scattered):  {(ratios < 0.5).sum()} cases")

# ============================================================
# SECTION C: ZERO ET & EXTREME CASES
# ============================================================

print(f"\n{'='*70}")
print("SECTION D: SPECIAL CASES")
print("=" * 70)

zero_region = stats_df[stats_df['region_voxels'] == 0]
if len(zero_region) > 0:
    print(f"\n--- D.1: Cases with ZERO {REGION} Voxels (n={len(zero_region)}) ---")
    for _, r in zero_region.iterrows():
        print(f"  {r['Brats20ID']}  (ET={r['ET_voxels']}, WT={r['WT_voxels']}, TC={r['TC_voxels']})")

print(f"\n--- D.2: Top 15 Cases by {REGION} Volume ---")
top_by_region = stats_df.sort_values('region_voxels', ascending=False).head(15)
for rank, (_, r) in enumerate(top_by_region.iterrows(), 1):
    print(f"  #{rank:<3} {r['Brats20ID']}: {REGION}={r['region_voxels']:>6}, WT={r['WT_voxels']:>6}, "
          f"ET/WT={r['ET_WT_ratio']:.4f}, Lesions={r['region_components']}, "
          f"Multifocal={'YES' if r['is_multifocal'] else 'no'}, "
          f"LargestRatio={r['region_largest_ratio']:.3f}")

print(f"\n--- D.3: Top 15 Cases by Number of {REGION} Lesions ---")
top_by_comp = stats_df.sort_values('region_components', ascending=False).head(15)
for rank, (_, r) in enumerate(top_by_comp.iterrows(), 1):
    print(f"  #{rank:<3} {r['Brats20ID']}: Lesions={r['region_components']}, {REGION}={r['region_voxels']:>6}, "
          f"Sizes={r['region_comp_sizes']}")

print("\n" + "=" * 70)
print("Done. Output files:")
print(f"  {OUTPUT_PATH}         — per-case statistics ({len(stats_df)} rows)")
if len(comp_df) > 0:
    print(f"  {OUTPUT_DETAIL_PATH} — per-lesion detail ({len(comp_df)} rows)")
    print(f"  {DISTRIBUTION_OUTPUT_PATH} — count by lesion size")
    print(f"  {PLOT_PATH} — lesion-size curve")
print("=" * 70)
