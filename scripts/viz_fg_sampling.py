"""
=============================================================================
Visualization: Verify Foreground-Aware Patch Sampling
=============================================================================
Samples 5 patches from each strategy and visualizes them with
ET mask overlay to confirm small lesions are being captured.

Strategy display:
  - random:        green border
  - foreground:    blue border
  - et_centered:   orange border
  - small_lesion:  red border

Output:
  - Each patch saves as a multi-slice PNG with ET overlay
  - Summary figure: ET centroid distribution + patch coverage

Usage:
    python scripts/viz_fg_sampling.py

Author: Generated for ResUNet enhancement project
Date:   2026-08-04
=============================================================================
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import BratsDataset, BratsDatasetWithFGSampling
from training.config import config


# ============================================================
# Config
# ============================================================

OUTPUT_DIR = 'viz_fg_sampling_output'
NUM_PATCHES_PER_STRATEGY = 5
strategies = ['random', 'foreground', 'et_centered', 'small_lesion']
strategy_colors = {
    'random':       '#2ecc71',  # green
    'foreground':   '#3498db',  # blue
    'et_centered':  '#e67e22',  # orange
    'small_lesion': '#e74c3c',  # red
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Load data
# ============================================================

print("Loading data...")
df = pd.read_csv(config.path_to_csv)
train_df, _ = train_test_split(df, test_size=0.3, random_state=10, shuffle=True)
train_df = train_df.reset_index(drop=True)

# Create FULL dataset (no patch sampling) to load complete volumes
full_dataset = BratsDataset(train_df, phase='train', is_resize=True)

# Create FG-aware dataset (for building index and sampling)
fg_dataset = BratsDatasetWithFGSampling(
    train_df, phase='train', is_resize=True,
    patch_size=(128, 128, 96),
    ratios={'random': 0.0, 'foreground': 0.0,
            'et_centered': 0.0, 'small_lesion': 1.0},
)

# Build foreground index
cache_path = os.path.join(OUTPUT_DIR, 'foreground_index_cache.pkl')
fg_dataset.build_foreground_index(cache_path=cache_path)

sampler = fg_dataset.sampler

# ============================================================
# Find cases suitable for visualization
# ============================================================

print("\nFinding cases with ET components...")
eligible_cases = []
for idx in range(len(full_dataset)):
    case_id = full_dataset.df.loc[idx, 'Brats20ID']
    if sampler.has_index(case_id):
        n_small = sampler.num_small_components(case_id)
        n_total = sampler.num_components(case_id)
        eligible_cases.append((idx, case_id, n_total, n_small))

eligible_cases.sort(key=lambda x: -x[3])  # sort by most small components
print(f"  {len(eligible_cases)} cases with ET, "
      f"{sum(1 for x in eligible_cases if x[3] > 0)} with small lesions")

if len(eligible_cases) == 0:
    print("ERROR: No cases with ET components found. Check your data.")
    sys.exit(1)

# Use the case with most small lesions for demonstration
demo_idx, demo_id, n_all, n_small = eligible_cases[0]
print(f"\nDemo case: {demo_id} ({n_all} ET components, {n_small} small)")


# ============================================================
# Load the demo case
# ============================================================

sample = full_dataset[demo_idx]
full_img = np.asarray(sample['image'])   # (4, D, H, W)
full_mask = np.asarray(sample['mask'])   # (3, D, H, W)
volume_shape = full_img.shape[1:]     # (D, H, W)

# Extract channels for visualization
t1ce = full_img[2]    # T1ce — best for tumor visualization
et_mask = full_mask[2]  # ET channel
wt_mask = full_mask[0]  # WT channel

print(f"  Volume shape: {volume_shape}")
print(f"  ET voxels: {et_mask.sum():.0f}")
print(f"  WT voxels: {wt_mask.sum():.0f}")


# ============================================================
# Sample patches and visualize
# ============================================================

def crop_middle_slice(patch_3d, axis=0):
    """Extract middle slice from a 3D patch along given axis."""
    mid = patch_3d.shape[axis] // 2
    if axis == 0:
        return patch_3d[mid, :, :]
    elif axis == 1:
        return patch_3d[:, mid, :]
    else:
        return patch_3d[:, :, mid]


def sample_one_patch(strategy_name, case_id, volume_shape, full_t1ce, full_et):
    """
    Force a specific strategy and sample one patch.
    Returns (patch_t1ce, patch_et, crop_bbox).
    """
    # Temporarily force the strategy
    old_probs = list(sampler.strategy_probs)
    strat_idx = sampler.strategy_names.index(strategy_name)

    new_probs = [0.0] * len(sampler.strategy_names)
    new_probs[strat_idx] = 1.0
    sampler.strategy_probs = new_probs

    z1, z2, y1, y2, x1, x2 = sampler.sample(case_id, volume_shape)

    # Restore
    sampler.strategy_probs = old_probs

    p_t1ce = full_t1ce[z1:z2, y1:y2, x1:x2]
    p_et   = full_et[z1:z2, y1:y2, x1:x2]

    return p_t1ce, p_et, (z1, z2, y1, y2, x1, x2)


# ============================================================
# Figure 1: 5-patch grid per strategy
# ============================================================

print("\nGenerating patch visualizations...")

fig, axes = plt.subplots(
    len(strategies), NUM_PATCHES_PER_STRATEGY,
    figsize=(NUM_PATCHES_PER_STRATEGY * 3, len(strategies) * 3.2)
)

for row, strat in enumerate(strategies):
    print(f"  Sampling {strat}...")

    for col in range(NUM_PATCHES_PER_STRATEGY):
        p_t1ce, p_et, bbox = sample_one_patch(
            strat, demo_id, volume_shape, t1ce, et_mask
        )

        # Middle slices (axial, coronal, sagittal → show axial for simplicity)
        slc_axial = crop_middle_slice(p_t1ce, axis=0)
        slc_et    = crop_middle_slice(p_et, axis=0)

        # Overlay ET mask on T1ce
        ax = axes[row, col]
        ax.imshow(slc_axial, cmap='gray', vmin=0, vmax=1)

        # ET overlay in red
        et_contour = np.ma.masked_where(slc_et < 0.5, slc_et)
        ax.imshow(et_contour, cmap='Reds', alpha=0.6, vmin=0, vmax=1)

        # Border color = strategy
        color = strategy_colors[strat]
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)

        # Title with ET content
        et_pct = p_et.sum() / p_et.size * 100
        ax.set_title(f"ET={p_et.sum():.0f}vox ({et_pct:.2f}%)", fontsize=8)
        ax.axis('off')

        if col == 0:
            ax.set_ylabel(strat.replace('_', '\n'), fontsize=11,
                         color=color, fontweight='bold', rotation=0,
                         labelpad=35)

fig.suptitle(f'Foreground-Aware Patch Sampling Verification\n'
             f'Case: {demo_id}  ({n_all} ET components, {n_small} small)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'patch_sampling_grid.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: patch_sampling_grid.png")


# ============================================================
# Figure 2: Multi-slice detail of one small_lesion patch
# ============================================================

print("\nGenerating multi-slice detail view...")

# Sample one small_lesion patch
p_t1ce, p_et, bbox = sample_one_patch(
    'small_lesion', demo_id, volume_shape, t1ce, et_mask
)

# Show every 8th slice along axial axis
n_slices = p_t1ce.shape[0]
slice_indices = list(range(4, n_slices, max(1, n_slices // 6)))[:6]

fig, axes = plt.subplots(2, len(slice_indices),
                          figsize=(len(slice_indices) * 2.2, 4.5))

for i, slc_idx in enumerate(slice_indices):
    # Top row: T1ce only
    axes[0, i].imshow(p_t1ce[slc_idx], cmap='gray', vmin=0, vmax=1)
    axes[0, i].set_title(f'slice {slc_idx}/{n_slices}', fontsize=8)
    axes[0, i].axis('off')

    # Bottom row: T1ce + ET overlay
    axes[1, i].imshow(p_t1ce[slc_idx], cmap='gray', vmin=0, vmax=1)
    et_slice = np.ma.masked_where(p_et[slc_idx] < 0.5, p_et[slc_idx])
    axes[1, i].imshow(et_slice, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('T1ce', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('T1ce + ET', fontsize=11, fontweight='bold')

fig.suptitle(f'Small-Lesion Patch: {p_et.sum():.0f} ET voxels'
             f' ({p_et.sum()/p_et.size*100:.2f}%)',
             fontsize=12, fontweight='bold',
             color=strategy_colors['small_lesion'])
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'small_lesion_multislice.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: small_lesion_multislice.png")


# ============================================================
# Figure 3: ET centroid map + sampled patch bboxes
# ============================================================

print("\nGenerating centroid coverage map...")

et_coords = np.argwhere(et_mask > 0.5)
if len(et_coords) == 0:
    print("  SKIP: no ET in this case")
else:
    # Sample many times from each strategy, record patch centers
    sample_centers = {s: [] for s in strategies}
    sample_bboxes = {s: [] for s in strategies}

    for strat in strategies:
        for _ in range(50):
            p_t1ce, p_et, bbox = sample_one_patch(
                strat, demo_id, volume_shape, t1ce, et_mask
            )
            z_center = (bbox[0] + bbox[1]) // 2
            y_center = (bbox[2] + bbox[3]) // 2
            x_center = (bbox[4] + bbox[5]) // 2
            sample_centers[strat].append((y_center, x_center))
            sample_bboxes[strat].append(bbox)

    # Plot ET heatmap + patch center scatter
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Axial projection
    et_proj_axial = et_mask.sum(axis=0)  # (H, W)
    axes[0].imshow(t1ce[t1ce.shape[0]//2], cmap='gray', vmin=0, vmax=1)
    axes[0].imshow(np.ma.masked_where(et_proj_axial < 1, et_proj_axial),
                   cmap='Reds', alpha=0.3)
    for strat in strategies:
        pts = np.array(sample_centers[strat])
        axes[0].scatter(pts[:, 1], pts[:, 0], c=strategy_colors[strat],
                       alpha=0.5, s=15, label=strat)
    axes[0].set_title('Axial (mid-slice)')
    axes[0].legend(fontsize=6, loc='upper right')

    # Coronal projection
    et_proj_cor = et_mask.sum(axis=1)
    axes[1].imshow(t1ce[:, t1ce.shape[1]//2, :], cmap='gray', vmin=0, vmax=1)
    axes[1].imshow(np.ma.masked_where(et_proj_cor < 1, et_proj_cor),
                   cmap='Reds', alpha=0.3)
    for strat in strategies:
        pts = np.array(sample_centers[strat])
        # z, x plane
        axes[1].scatter(pts[:, 1], [s[0] for s in sample_centers[strat]],
                       c=strategy_colors[strat], alpha=0.5, s=15)
    axes[1].set_title('Coronal (mid-slice)')

    # Sagittal projection
    et_proj_sag = et_mask.sum(axis=2)
    axes[2].imshow(t1ce[:, :, t1ce.shape[2]//2], cmap='gray', vmin=0, vmax=1)
    axes[2].imshow(np.ma.masked_where(et_proj_sag < 1, et_proj_sag),
                   cmap='Reds', alpha=0.3)
    for strat in strategies:
        pts = np.array(sample_centers[strat])
        axes[2].scatter(pts[:, 0], [s[0] for s in sample_centers[strat]],
                       c=strategy_colors[strat], alpha=0.5, s=15)
    axes[2].set_title('Sagittal (mid-slice)')

    fig.suptitle(f'Patch Center Distribution by Strategy\n'
                 f'Case: {demo_id}  ({n_all} ET components, {n_small} small)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'centroid_coverage.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: centroid_coverage.png")


# ============================================================
# Figure 4: ET content distribution (box plot per strategy)
# ============================================================

print("\nGenerating ET content statistics...")

strategy_et_counts = {s: [] for s in strategies}
n_samples = 100

for strat in strategies:
    for _ in range(n_samples):
        p_t1ce, p_et, bbox = sample_one_patch(
            strat, demo_id, volume_shape, t1ce, et_mask
        )
        strategy_et_counts[strat].append(int(p_et.sum()))

fig, ax = plt.subplots(figsize=(8, 5))

positions = list(range(len(strategies)))
bp = ax.boxplot([strategy_et_counts[s] for s in strategies],
                positions=positions, patch_artist=True,
                widths=0.5)

for i, strat in enumerate(strategies):
    bp['boxes'][i].set_facecolor(strategy_colors[strat])
    bp['boxes'][i].set_alpha(0.6)

    # Overlay individual points with jitter
    y = strategy_et_counts[strat]
    x = np.random.normal(i, 0.08, size=len(y))
    ax.scatter(x, y, c=strategy_colors[strat], alpha=0.3, s=12)

ax.set_xticks(positions)
ax.set_xticklabels([s.replace('_', '\n') for s in strategies])
ax.set_ylabel('ET Voxels in Patch')
ax.set_title(f'ET Content Distribution by Sampling Strategy\n'
             f'Case: {demo_id}  ({n_all} ET components, {n_small} small)')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'et_content_distribution.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: et_content_distribution.png")


# ============================================================
# Summary
# ============================================================

print(f"\n{'='*60}")
print("Visualization complete!")
print(f"Output directory: {OUTPUT_DIR}/")
print(f"\nFiles:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('.png'):
        size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"  {f:<35} ({size_kb:.0f} KB)")
print(f"\n{'='*60}")
print("\nHow to read the figures:")
print("  1. patch_sampling_grid.png")
print("     - 5 random samples per strategy")
print("     - Red = ET overlay on T1ce")
print("     - Border color = strategy")
print("     - ET=vox count in title")
print(f"     - EXPECT: et_centered/small_lesion consistently capture ET")
print(f"     - EXPECT: random sometimes misses ET entirely")
print(f"\n  2. small_lesion_multislice.png")
print(f"     - 6 slices through a small_lesion patch")
print(f"     - Top: T1ce only, Bottom: T1ce+ET overlay")
print(f"     - EXPECT: ET visible in most slices, not just edges")
print(f"\n  3. centroid_coverage.png")
print(f"     - Red heatmap = ET density")
print(f"     - Colored dots = patch centers from each strategy")
print(f"     - EXPECT: et_centered/small_lesion dots cluster on ET regions")
print(f"     - EXPECT: random dots spread across entire volume")
print(f"\n  4. et_content_distribution.png")
print(f"     - Box plot of ET voxels per patch")
print(f"     - EXPECT: et_centered/small_lesion has higher median ET count")
print(f"     - EXPECT: random has many zero-ET patches")
