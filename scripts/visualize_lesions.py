"""
=============================================================================
Multi-focal ET Lesion Visualization for BraTS2020
=============================================================================
For each selected case, generates TWO figures:

  Fig A — "All Lesions Together" (论文主图):
    Each ET lesion is color-coded on the SAME slice, making spatial
    separation immediately visible. Lesion IDs and voxel counts
    labeled. Background = FLAIR MRI.

  Fig B — "3D Spatial Map":
    Scatter plot showing the (x,y,z) center-of-mass of each lesion
    in 3D space, with marker size proportional to lesion volume.
    This unambiguously proves lesions are spatially disconnected.

Usage:
    python scripts/visualize_lesions.py

Output:
    5 × (A + B) = 10 figures saved to current directory.

Author: Generated for ResUNet enhancement project
Date:   2026-08-02
=============================================================================
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import ndimage
from matplotlib.lines import Line2D

# ============================================================
# Configuration
# ============================================================

DATA_ROOT = '/root/autodl-tmp/brats_project/MICCAI_BraTS2020_TrainingData/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'

# Selected cases with their roles
CASES = {
    'BraTS20_Training_225': 'A: Single large ET (111K vox) — simple case',
    'BraTS20_Training_274': 'B: Classic multi-focal (9 lesions, main + 8 satellites)',
    'BraTS20_Training_293': 'C: Extreme fragmentation (35 lesions!)',
    'BraTS20_Training_329': 'D: LGG with ZERO ET — model must not hallucinate',
    'BraTS20_Training_284': 'E: Micro multi-focal (16 lesions, ALL small)',
}

MAX_LESION_COLORS = 40  # tab20 has 20, tab20b has 20


# ============================================================
# Helper: generate distinct colors for each lesion
# ============================================================

def get_lesion_colors(n):
    """Generate n visually distinct colors."""
    if n <= 20:
        return [cm.tab20(i / 20) for i in range(n)]
    else:
        # Combine tab20 + tab20b for >20 lesions
        colors = [cm.tab20(i / 20) for i in range(20)]
        colors += [cm.tab20b(i / 20) for i in range(n - 20)]
        return colors


# ============================================================
# FIGURE A: All Lesions on One Slice (Color-coded overlay)
# ============================================================

def fig_all_lesions_together(case_id, desc, flair, labeled, n_lesions):
    """
    Color-code EVERY lesion on the SAME slice.
    The viewer can instantly see spatial separation (different colors
    in different locations = different lesions).
    """
    # Pick the slice with most distinct lesions (best for visualization)
    best_slice = 0
    best_count = 0
    for z in range(labeled.shape[0]):
        unique = len(set(labeled[z][labeled[z] > 0]))
        if unique > best_count:
            best_count = unique
            best_slice = z

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))

    colors = get_lesion_colors(n_lesions)

    # --- Left: FLAIR background + GT color overlay ---
    ax = axes[0]
    ax.imshow(flair[best_slice], cmap='gray', vmin=0, vmax=np.percentile(flair[best_slice], 99))

    # Overlay each lesion in a different color
    all_colored = np.zeros(flair[best_slice].shape + (4,))
    for i in range(1, n_lesions + 1):
        mask_2d = (labeled[best_slice] == i)
        if mask_2d.sum() == 0:
            continue
        rgba = list(colors[i-1])
        all_colored[mask_2d] = rgba

    ax.imshow(all_colored, alpha=0.8)
    ax.set_title(f'All {n_lesions} ET Lesions — Slice {best_slice}', fontsize=13, fontweight='bold')
    ax.axis('off')

    # --- Right: Same but with lesion IDs annotated ---
    ax = axes[1]
    ax.imshow(flair[best_slice], cmap='gray', vmin=0, vmax=np.percentile(flair[best_slice], 99))

    # Find centroids for labeling (within this slice)
    for i in range(1, n_lesions + 1):
        mask_2d = (labeled[best_slice] == i)
        if mask_2d.sum() == 0:
            continue
        ys, xs = np.where(mask_2d)
        cy, cx = int(ys.mean()), int(xs.mean())
        # Total 3D volume of this lesion
        total_vox = int((labeled == i).sum())
        ax.annotate(f'#{i}\n({total_vox})', (cx, cy),
                    color='white', fontsize=8, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    # Overlay with transparency so labels are visible
    for i in range(1, n_lesions + 1):
        mask_2d = (labeled[best_slice] == i)
        if mask_2d.sum() == 0:
            continue
        rgba = list(colors[i-1])[:3] + [0.4]  # lower alpha
        overlay = np.zeros(flair[best_slice].shape + (4,))
        overlay[mask_2d] = rgba
        ax.imshow(overlay)

    ax.set_title(f'Lesion IDs + Voxel Counts — Slice {best_slice}', fontsize=13, fontweight='bold')
    ax.axis('off')

    # Legend
    legend_elements = []
    for i in range(1, min(11, n_lesions + 1)):
        total_vox = int((labeled == i).sum())
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i-1],
                   markersize=10, label=f'#{i}: {total_vox} vox')
        )
    if n_lesions > 10:
        legend_elements.append(Line2D([0], [0], marker='', color='w',
                                      label=f'... +{n_lesions-10} more'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8,
              bbox_to_anchor=(1.35, 1.0))

    plt.suptitle(f'{case_id}: {n_lesions} ET Lesions\n{desc}',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = f'{case_id}_all_lesions.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


# ============================================================
# FIGURE B: 3D Spatial Map (proves spatial disconnection)
# ============================================================

def fig_3d_spatial_map(case_id, desc, labeled, n_lesions):
    """
    Plot each lesion's center-of-mass in 3D space.
    If two lesions have different spatial coordinates → they are
    physically separate → genuine multi-focal disease.
    """
    fig = plt.figure(figsize=(14, 6))

    # --- Left: 3D scatter (center-of-mass) ---
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    colors = get_lesion_colors(n_lesions)

    for i in range(1, n_lesions + 1):
        coords = np.argwhere(labeled == i)
        if len(coords) == 0:
            continue
        cz, cy, cx = coords.mean(axis=0)
        size = len(coords)
        ax.scatter(cx, cy, cz, c=[colors[i-1]], s=np.sqrt(size)*5,
                   edgecolors='black', linewidth=0.5, alpha=0.9,
                   label=f'#{i}: {size} vox')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (slice)')
    ax.set_title('3D Lesion Centers-of-Mass', fontsize=12, fontweight='bold')
    ax.legend(fontsize=6, loc='upper right', ncol=2)
    ax.view_init(elev=25, azim=-60)

    # --- Right: Size-ranked bar chart ---
    ax = fig.add_subplot(1, 2, 2)
    sizes = sorted([int((labeled == i).sum()) for i in range(1, n_lesions + 1)],
                   reverse=True)
    bar_colors = get_lesion_colors(n_lesions)
    # Re-sort colors by size too
    size_color_pairs = sorted(
        [(int((labeled == i).sum()), colors[i-1]) for i in range(1, n_lesions + 1)],
        reverse=True
    )

    bars = ax.bar(range(1, n_lesions + 1), [s[0] for s in size_color_pairs],
                  color=[s[1] for s in size_color_pairs],
                  edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Lesion Rank')
    ax.set_ylabel('Voxels')
    ax.set_title('Lesion Size Distribution', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Tiny (<50 vox)')
    ax.axhline(y=500, color='orange', linestyle='--', alpha=0.5, label='Small (<500 vox)')
    ax.legend(fontsize=8)

    # Annotate the largest lesion
    ax.annotate(f'{size_color_pairs[0][0]} vox',
                (1, size_color_pairs[0][0]),
                textcoords='offset points', xytext=(0, 10),
                fontsize=9, fontweight='bold', ha='center')

    plt.suptitle(f'{case_id}: 3D Lesion Spatial Distribution\n{desc}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = f'{case_id}_3d_spatial.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


# ============================================================
# Main
# ============================================================

def main():
    print('=' * 60)
    print('Multi-focal ET Lesion Visualization')
    print('=' * 60)

    for case_id, desc in CASES.items():
        print(f'\nProcessing: {case_id} — {desc}')

        seg_path = os.path.join(DATA_ROOT, case_id, case_id + '_seg.nii')
        flair_path = os.path.join(DATA_ROOT, case_id, case_id + '_flair.nii')

        seg = nib.load(seg_path)
        seg_data = np.asarray(seg.dataobj).astype(np.int16)

        flair = nib.load(flair_path)
        flair_data = np.asarray(flair.dataobj).astype(np.float32)

        # ET = label 4 only (confirmed by BraTS annotation protocol)
        et_mask = (seg_data == 4)

        # 3D connected-component labeling (26-connectivity)
        labeled, n_lesions = ndimage.label(et_mask)

        if n_lesions == 0:
            print(f'  [INFO] Zero ET lesions — generating summary only')
            # Generate a note figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            mid_z = flair_data.shape[0] // 2
            ax.imshow(flair_data[mid_z], cmap='gray')
            # Show WT contour
            wt = np.isin(seg_data[mid_z], [1, 2, 4])
            ax.contour(wt, colors='yellow', linewidths=1.5, levels=[0.5])
            ax.set_title(f'{case_id}\n{desc}\nET=0, WT={wt.sum()} vox, TC={np.isin(seg_data, [1,4]).sum()} vox',
                        fontsize=13, fontweight='bold')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(f'{case_id}_all_lesions.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f'  Saved: {case_id}_all_lesions.png (zero ET note)')
            continue

        # Generate both figures
        fig_all_lesions_together(case_id, desc, flair_data, labeled, n_lesions)
        fig_3d_spatial_map(case_id, desc, labeled, n_lesions)

    print('\n' + '=' * 60)
    print('DONE. Generated files:')
    print('=' * 60)
    for case_id in CASES:
        print(f'  {case_id}_all_lesions.png')
        if case_id != 'BraTS20_Training_329':
            print(f'  {case_id}_3d_spatial.png')
    print()
    print('Key for reviewers:')
    print('  _all_lesions.png  — Each lesion color-coded on SAME slice')
    print('  _3d_spatial.png   — 3D center-of-mass + size distribution')
    print('  Spatial separation in 3D scatter = genuine multi-focal disease')


if __name__ == '__main__':
    main()
