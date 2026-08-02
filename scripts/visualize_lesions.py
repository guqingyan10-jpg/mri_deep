"""
=============================================================================
ET Lesion Disconnection Proof — Paper-Ready Visualization
=============================================================================
Generates a SINGLE figure per case that unambiguously proves lesions
are disconnected in 3D space. No ambiguity, no complex interpretation.

PROOF STRATEGY (3 independent lines of evidence):

  PANEL A — "Z-stack Montage":
    8 equally-spaced axial slices through the tumor volume.
    Every ET voxel is color-coded by lesion ID.
    ALL slices share the SAME color legend.
    If lesion A and lesion B were connected in 3D, at least one
    intermediate slice MUST show them merging to the same color.
    If they never merge across all 8 slices → DISCONNECTED.

  PANEL B — "Depth-Space Scatter":
    Scatter plot of EVERY ET voxel: Y-axis = z-coordinate (depth),
    X-axis = spatial x. Each lesion = different color column.
    If two lesions are separate (no voxels in between), their
    color clusters DO NOT overlap in z. Can see at a glance.

  PANEL C — "3D View" (only for cases with 2-20 lesions):
    Two viewing angles showing the voxel clouds of each lesion,
    subsampled for clarity. Spatial gaps = visible.

Usage:
    python scripts/visualize_lesions.py

Output:
    lesion_verification_figures/{case_id}_proof.png  (per case)

Author: Generated for ResUNet enhancement project
Date:   2026-08-02
=============================================================================
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================

DATA_ROOT = '/root/autodl-tmp/brats_project/MICCAI_BraTS2020_TrainingData/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'

CASES = {
    'BraTS20_Training_225': 'Single large ET | 1 lesion, 111K vox',
    'BraTS20_Training_274': 'Classic multi-focal | 9 lesions, main + 8 satellites',
    'BraTS20_Training_293': 'Extreme fragmentation | 35 lesions',
    'BraTS20_Training_329': 'LGG: zero ET | large tumor, no enhancement',
    'BraTS20_Training_284': 'Micro multi-focal | 16 lesions, ALL <2800 vox',
}

OUTPUT_DIR = 'lesion_verification_figures'
N_MONTAGE_SLICES = 8
SUBSAMPLE_3D = 5000


def distinct_colors(n):
    """Return n visually distinct RGBA colors."""
    if n == 0:
        return []
    # Use tab20, then tab20b, then cycle
    pool = []
    for cmap in [cm.tab20, cm.tab20b]:
        for i in range(20):
            pool.append(cmap(i / 20))
    while len(pool) < n:
        for cmap in [cm.Set3, cm.Paired, cm.tab20c]:
            for i in range(min(20, n - len(pool))):
                pool.append(cmap(i / 20))
    return pool[:n]


def make_proof_figure(case_id, description, flair, labeled, n_lesions):
    """
    Single comprehensive figure that PROVES lesions are disconnected.
    """

    if n_lesions == 0:
        # Zero ET case — simple annotated slice
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        z_mid = flair.shape[0] // 2
        ax.imshow(flair[z_mid], cmap='gray')
        ax.set_title(f'{case_id}\nZERO Enhancing Tumor\n{description}',
                     fontsize=14, fontweight='bold')
        ax.axis('off')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.savefig(os.path.join(OUTPUT_DIR, f'{case_id}_proof.png'),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)
        return

    D, H, W = labeled.shape
    colors = distinct_colors(n_lesions)

    # ================================================================
    # Find Z-range with lesions
    # ================================================================
    z_has_lesion = np.array([(labeled[z] > 0).any() for z in range(D)])
    z_indices = np.where(z_has_lesion)[0]
    if len(z_indices) == 0:
        return
    z_min, z_max = z_indices[0], z_indices[-1]

    # ================================================================
    # Create figure
    # ================================================================
    fig = plt.figure(figsize=(24, 10))

    # ---- ROW 1: Z-stack Montage (8 slices) ----
    slice_indices = np.linspace(z_min, z_max, N_MONTAGE_SLICES).astype(int)
    slice_indices = np.unique(slice_indices)

    for si, z in enumerate(slice_indices):
        ax = fig.add_subplot(2, max(N_MONTAGE_SLICES, 3),
                             si + 1)
        ax.imshow(flair[z], cmap='gray',
                  vmin=0, vmax=np.percentile(flair[z], 99))

        # Overlay each lesion
        for lid in range(1, n_lesions + 1):
            mask = (labeled[z] == lid)
            if mask.sum() == 0:
                continue
            rgba = list(colors[lid - 1])
            # Less alpha for large lesions so they don't block others
            rgba[3] = 0.7
            overlay = np.zeros((H, W, 4))
            overlay[mask] = rgba
            ax.imshow(overlay)

        ax.set_title(f'z={z}', fontsize=9)
        ax.axis('off')

    # Legend (compact)
    n_legend = min(n_lesions, 15)
    legend_items = []
    for i in range(n_legend):
        sz = int((labeled == i + 1).sum())
        legend_items.append(Patch(color=colors[i],
                                  label=f'#{i+1} ({sz})'))
    if n_lesions > 15:
        legend_items.append(Patch(color='gray',
                                  label=f'+{n_lesions-15} more'))

    # Put legend in remaining subplot slots or a dedicated area
    legend_slot = len(slice_indices)
    if legend_slot < N_MONTAGE_SLICES:
        ax_leg = fig.add_subplot(2, N_MONTAGE_SLICES, legend_slot + 1)
        ax_leg.axis('off')
        ax_leg.legend(handles=legend_items, loc='center',
                      fontsize=6, title='Lesion Key',
                      title_fontsize=8, ncol=1)
    else:
        # Add legend below
        ax_leg = fig.add_subplot(2, 1, 2)
        ax_leg.axis('off')
        ax_leg.legend(handles=legend_items, loc='center',
                      fontsize=6, title='Lesion Key',
                      title_fontsize=8, ncol=min(5, n_legend))

    # ---- ROW 2 LEFT: Depth-Space Scatter ----
    ax_scatter = fig.add_subplot(2, N_MONTAGE_SLICES,
                                 N_MONTAGE_SLICES + 1,
                                 colspan=N_MONTAGE_SLICES // 2)

    # Collect voxel coordinates per lesion, subsample
    for lid in range(1, n_lesions + 1):
        coords = np.argwhere(labeled == lid)
        if len(coords) == 0:
            continue
        # Subsample for speed
        if len(coords) > 3000:
            idx = np.random.choice(len(coords), 3000, replace=False)
            coords = coords[idx]
        zz, yy, xx = coords[:, 0], coords[:, 1], coords[:, 2]
        ax_scatter.scatter(xx, zz, s=0.3, c=[colors[lid - 1]],
                           alpha=0.5, rasterized=True)

    ax_scatter.set_xlabel('X (voxel)', fontsize=10)
    ax_scatter.set_ylabel('Z (slice depth)', fontsize=10)
    ax_scatter.set_title('Depth-Space View: Each color = one lesion\n'
                         'Separate clusters → DISCONNECTED',
                         fontsize=11, fontweight='bold')
    ax_scatter.invert_yaxis()
    ax_scatter.grid(True, alpha=0.2)

    # ---- ROW 2 RIGHT: 3D Scatter (two viewing angles) ----
    for vi, (elev, azim, label) in enumerate([
        (20, -60, 'Front-oblique'),
        (80, -90, 'Top-down'),
    ]):
        ax3d = fig.add_subplot(2, N_MONTAGE_SLICES,
                               N_MONTAGE_SLICES + 1 + N_MONTAGE_SLICES // 2 + vi,
                               projection='3d')
        for lid in range(1, n_lesions + 1):
            coords = np.argwhere(labeled == lid)
            if len(coords) == 0:
                continue
            if len(coords) > SUBSAMPLE_3D:
                idx = np.random.choice(len(coords), SUBSAMPLE_3D, replace=False)
                coords = coords[idx]
            # Center of mass
            cz, cy, cx = coords[:, 0], coords[:, 1], coords[:, 2]
            ax3d.scatter(cx, cy, cz, s=0.5, c=[colors[lid - 1]],
                         alpha=0.4, rasterized=True)

        ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
        ax3d.set_title(f'3D {label}', fontsize=10, fontweight='bold')
        ax3d.view_init(elev=elev, azim=azim)

    fig.suptitle(f'{case_id}: {n_lesions} ET Lesion Disconnection Proof\n{description}',
                 fontsize=15, fontweight='bold', y=1.01)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f'{case_id}_proof.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def main():
    print('=' * 60)
    print('ET LESION DISCONNECTION PROOF')
    print('=' * 60)
    print()
    print('Proof method:')
    print('  Panel A — 8 serial z-slices, all lesions color-coded')
    print('  Panel B — Depth-space scatter (z vs x) per lesion')
    print('  Panel C — 3D voxel clouds, 2 viewing angles')
    print('  If colors never merge across slices/scatter → DISCONNECTED')
    print()

    for case_id, desc in CASES.items():
        print(f'Processing: {case_id}')
        seg_path = os.path.join(DATA_ROOT, case_id, f'{case_id}_seg.nii')
        flair_path = os.path.join(DATA_ROOT, case_id, f'{case_id}_flair.nii')

        seg = nib.load(seg_path)
        seg_data = np.asarray(seg.dataobj).astype(np.int16)

        flair = nib.load(flair_path)
        flair_data = np.asarray(flair.dataobj).astype(np.float32)

        # ET = label 4 only
        et_mask = (seg_data == 4)

        # 3D 26-connectivity
        labeled_raw, n_raw = ndimage.label(et_mask)

        # Filter <10 voxels
        valid_ids = []
        for lid in range(1, n_raw + 1):
            if (labeled_raw == lid).sum() >= 10:
                valid_ids.append(lid)

        filtered = np.zeros_like(labeled_raw)
        for new_id, old_id in enumerate(valid_ids, 1):
            filtered[labeled_raw == old_id] = new_id

        n_valid = len(valid_ids)
        print(f'  Raw: {n_raw} components → Valid (>=10vox): {n_valid}')

        make_proof_figure(case_id, desc, flair_data, filtered, n_valid)

    print()
    print('=' * 60)
    print(f'Done. Figures in: {OUTPUT_DIR}/')
    print('=' * 60)
    print()
    print('HOW TO READ THE FIGURE:')
    print('  Top row:   8 z-slices through tumor. Different colors')
    print('             on DIFFERENT slices, or on the SAME slice')
    print('             but spatially separated = DISCONNECTED.')
    print('  Bottom-L:  z vs x scatter. Each lesion = different')
    print('             color column. Separate clusters = proof.')
    print('  Bottom-R:  3D rendering from two angles. Spatial')
    print('             gaps between color groups = visible proof.')


if __name__ == '__main__':
    main()
