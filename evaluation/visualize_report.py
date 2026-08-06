"""
=============================================================================
Paper-Ready Visualization Generator for BraTS2020 Model Comparison
=============================================================================
Generates all quantitative and qualitative figures for the paper.

Quantitative:
  - ET HD95 bar chart (all models)
  - Lesion-wise Recall bar chart (all models)
  - Small-case ET Dice bar chart (all models)
  - Per-category Dice radar / grouped bar

Qualitative (per selected case):
  - MRI (FLAIR) axial slice
  - Ground Truth overlay
  - Baseline prediction overlay
  - Each experiment model prediction overlay
  - Small ET region zoom-in crop

Usage:
    from evaluation.visualize_report import generate_all_figures
    generate_all_figures(all_metrics, all_histories, case_data, output_dir='figures')

Author: ResUNet Enhancement Project
Date:   2026-08-06
=============================================================================
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from scipy import ndimage
from scipy.ndimage import center_of_mass


# ============================================================
# Color scheme — consistent across all figures
# ============================================================

CATEGORY_COLORS = {
    'Baseline':           '#2c3e50',
    'V1: Loss Function':  '#e74c3c',
    'V2: Architecture':   '#2ecc71',
    'SLA-FB: Data':       '#3498db',
    'SLA-FB: Loss':       '#9b59b6',
}

# For individual model color assignment
MODEL_COLORS = [
    '#2c3e50',   # Baseline — black/navy
    '#e74c3c',   # lb=0.1
    '#c0392b',   # lb=0.3
    '#922b21',   # lb=0.5
    '#27ae60',   # Edge Sobel concat
    '#2ecc71',   # Edge Sobel add
    '#1abc9c',   # Edge Laplacian
    '#16a085',   # FGFE
    '#3498db',   # FG Sampling
    '#2980b9',   # CC-Dice
    '#8e44ad',   # PM-Dice
]


def _get_color(i, model_name, category):
    """Get consistent color for a model."""
    if model_name.startswith('Baseline'):
        return '#2c3e50'
    return MODEL_COLORS[i % len(MODEL_COLORS)]


# ============================================================
# 1. Quantitative Bar Charts
# ============================================================

def plot_hd95_barchart(all_metrics, save_path='figures/hd95_barchart.png'):
    """
    Grouped bar chart: ET HD95 per model, sorted by value.
    Lower is better — bars are colored green→yellow→red by value.
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    models = [m['model_name'] for m in all_metrics]
    hd95_vals = [m.get('ET_HD95_mean', float('nan')) for m in all_metrics]

    # Sort by value
    sorted_idx = np.argsort(hd95_vals)
    sorted_models = [models[i] for i in sorted_idx]
    sorted_hd95 = [hd95_vals[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Color gradient: lower = greener, higher = redder
    min_v, max_v = min(sorted_hd95), max(sorted_hd95)
    colors = []
    for v in sorted_hd95:
        ratio = (v - min_v) / (max_v - min_v + 1e-9)  # 0=best, 1=worst
        # Green → Yellow → Red
        r = min(1.0, 2 * ratio)
        g = min(1.0, 2 * (1 - ratio))
        b = 0.15
        colors.append((r, g, b))

    bars = ax.barh(sorted_models, sorted_hd95, color=colors, edgecolor='white', linewidth=0.8)
    ax.set_xlabel('ET HD95 (mm)', fontsize=12)
    ax.set_title('ET Hausdorff Distance 95th Percentile (lower = better)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    # Annotate values
    for bar, val in zip(bars, sorted_hd95):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=10)

    # Baseline marker
    for m, v in zip(sorted_models, sorted_hd95):
        if 'Baseline' in m:
            ax.axvline(x=v, color='#2c3e50', linestyle='--', linewidth=1.5, alpha=0.7)
            ax.text(v + 0.2, ax.get_ylim()[0] + 0.5, f'Baseline {v:.2f}',
                    color='#2c3e50', fontsize=9, fontstyle='italic')
            break

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_lesion_recall_barchart(all_metrics, save_path='figures/lesion_recall_barchart.png'):
    """Grouped bar chart: Lesion-wise Recall per model (higher = better)."""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    models = [m['model_name'] for m in all_metrics]
    vals = [m.get('Lesion_Recall_mean', float('nan')) for m in all_metrics]
    sorted_idx = np.argsort(vals)[::-1]  # descending
    sorted_models = [models[i] for i in sorted_idx]
    sorted_vals = [vals[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = [_get_color(i, n, '') for i, n in enumerate(sorted_models)]
    bars = ax.barh(sorted_models, sorted_vals, color=colors, edgecolor='white', linewidth=0.8)
    ax.set_xlabel('Lesion-wise Recall', fontsize=12)
    ax.set_title('ET Lesion-wise Recall (higher = fewer missed small lesions)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    for bar, val in zip(bars, sorted_vals):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_small_case_dice_barchart(all_metrics, save_path='figures/small_case_dice_barchart.png'):
    """Grouped bar chart: Small-case ET Dice (bottom 25% ET volume)."""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    models = [m['model_name'] for m in all_metrics]
    vals = [m.get('Small_case_ET_Dice_mean', float('nan')) for m in all_metrics]
    sorted_idx = np.argsort(vals)[::-1]
    sorted_models = [models[i] for i in sorted_idx]
    sorted_vals = [vals[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = [_get_color(i, n, '') for i, n in enumerate(sorted_models)]
    bars = ax.barh(sorted_models, sorted_vals, color=colors, edgecolor='white', linewidth=0.8)
    ax.set_xlabel('Small-case ET Dice', fontsize=12)
    ax.set_title('Small-case ET Dice (bottom 25% ET volume — target population)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    for bar, val in zip(bars, sorted_vals):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_et_dice_barchart(all_metrics, save_path='figures/et_dice_barchart.png'):
    """Grouped bar chart: ET Dice per model, descending."""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    models = [m['model_name'] for m in all_metrics]
    vals = [m.get('ET_Dice_mean', float('nan')) for m in all_metrics]
    sorted_idx = np.argsort(vals)[::-1]
    sorted_models = [models[i] for i in sorted_idx]
    sorted_vals = [vals[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = [_get_color(i, n, '') for i, n in enumerate(sorted_models)]
    bars = ax.barh(sorted_models, sorted_vals, color=colors, edgecolor='white', linewidth=0.8)
    ax.set_xlabel('ET Dice', fontsize=12)
    ax.set_title('ET Dice Score (higher = better overall ET segmentation)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    for bar, val in zip(bars, sorted_vals):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_et_recall_precision_barchart(all_metrics, save_path='figures/et_recall_precision.png'):
    """Side-by-side bar chart: ET Recall & Precision for all models."""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    models = [m['model_name'] for m in all_metrics]
    recs  = [m.get('ET_Recall_mean', float('nan')) for m in all_metrics]
    precs = [m.get('ET_Precision_mean', float('nan')) for m in all_metrics]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 6))
    bars1 = ax.bar(x - width/2, recs, width, label='ET Recall',
                   color='#3498db', edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x + width/2, precs, width, label='ET Precision',
                   color='#e74c3c', edgecolor='white', linewidth=0.8)

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('ET Recall vs Precision (higher = better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Annotate
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# 2. Qualitative Visualizations — Case-level overlays
# ============================================================

def find_case_with_small_et(et_volumes, case_ids, percentile=25):
    """
    Find representative cases: one small-ET case and one average-ET case.
    Returns list of (case_index, description).
    """
    if not et_volumes:
        return []

    threshold = np.percentile(et_volumes, percentile)
    small_indices = [i for i, v in enumerate(et_volumes) if v <= threshold and v > 0]
    typical_indices = [i for i, v in enumerate(et_volumes)
                       if threshold < v <= np.percentile(et_volumes, 75)]

    selected = []
    # Pick 2 small-ET cases (first 2 in list)
    for idx in small_indices[:2]:
        selected.append((idx, f'small_ET_{case_ids[idx]}_{int(et_volumes[idx])}vox'))
    # Pick 1 typical case
    if typical_indices:
        mid = len(typical_indices) // 2
        idx = typical_indices[mid]
        selected.append((idx, f'typical_ET_{case_ids[idx]}_{int(et_volumes[idx])}vox'))

    return selected


def _get_best_slice(mask_3d, target_class=2):
    """Find axial slice with maximum tumor area for target class."""
    slice_sums = mask_3d[target_class].reshape(mask_3d[target_class].shape[0], -1).sum(axis=1)
    best = int(np.argmax(slice_sums)) if slice_sums.max() > 0 else mask_3d.shape[1] // 2
    return best


def _find_small_et_bbox(gt_mask, margin=8):
    """
    Find bounding box around small ET regions for zoom-in.
    Returns (z1, z2, y1, y2, x1, x2) or None if no ET.
    """
    et = gt_mask[2]  # ET channel
    if et.sum() == 0:
        return None
    coords = np.argwhere(et)
    z1, z2 = int(coords[:, 0].min()), int(coords[:, 0].max()) + 1
    y1, y2 = int(coords[:, 1].min()), int(coords[:, 1].max()) + 1
    x1, x2 = int(coords[:, 2].min()), int(coords[:, 2].max()) + 1
    # Add margin
    D, H, W = et.shape
    z1 = max(0, z1 - margin); z2 = min(D, z2 + margin)
    y1 = max(0, y1 - margin); y2 = min(H, y2 + margin)
    x1 = max(0, x1 - margin); x2 = min(W, x2 + margin)
    return (z1, z2, y1, y2, x1, x2)


def generate_case_overlay(dataloader, models_dict, case_indices,
                          save_dir='figures/cases',
                          max_models_per_fig=4):
    """
    Generate per-case overlay figures.

    Layout (per case):
      Row 1: MRI (FLAIR) | Ground Truth | Baseline Pred | Model2 Pred | ...
      Row 2: Small ET zoom panels (if applicable)

    Args:
        dataloader: test DataLoader
        models_dict: {'model_name': nn.Module, ...} — models already on device
        case_indices: list of (dataloader_index, description) to visualize
        save_dir: output directory
        max_models_per_fig: max prediction columns (auto-wrap if more)
    """
    os.makedirs(save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Collect case data from dataloader
    all_data = []
    for data in dataloader:
        all_data.append({
            'id':       data['Id'],
            'image':    data['image'].to(device),
            'mask':     data['mask'],
        })
        if len(all_data) >= 20:  # enough for selection
            break

    for case_idx, desc in case_indices:
        if case_idx >= len(all_data):
            continue

        case = all_data[case_idx]
        case_id = case['id'][0] if isinstance(case['id'], (list, tuple)) else case['id']
        flair = case['image'][0, 0].cpu().numpy()  # first modality = FLAIR
        gt = case['mask'][0].cpu().numpy()  # (3, D, H, W)

        best_slice = _get_best_slice(gt)
        et_vol = int(gt[2].sum())

        # Get predictions from all models
        preds = {}
        with torch.no_grad():
            for name, model in models_dict.items():
                model.eval()
                logits = model(case['image'])
                probs = torch.sigmoid(logits)
                pred = (probs >= 0.33).float()
                preds[name] = pred[0].cpu().numpy()  # (3, D, H, W)

        n_models = len(preds)
        n_cols = min(n_models, max_models_per_fig) + 2  # MRI + GT + preds
        n_rows = 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.5))
        if n_cols == 1:
            axes = [axes]

        # ── Column 1: MRI (FLAIR) ──
        ax = axes[0]
        mri_slice = (flair[best_slice] - flair[best_slice].min()) / \
                    (flair[best_slice].max() - flair[best_slice].min() + 1e-9)
        ax.imshow(mri_slice, cmap='gray')
        ax.set_title('MRI (FLAIR)', fontsize=10, fontweight='bold')
        ax.axis('off')

        # ── Column 2: Ground Truth ──
        ax = axes[1]
        _plot_mask_overlay(ax, flair[best_slice], gt[:, best_slice])
        ax.set_title('Ground Truth', fontsize=10, fontweight='bold')
        ax.axis('off')

        # ── Remaining columns: predictions ──
        col = 2
        for name, pred in preds.items():
            if col >= n_cols:
                break
            ax = axes[col]
            _plot_mask_overlay(ax, flair[best_slice], pred[:, best_slice])
            is_baseline = 'Baseline' in name
            title_color = '#2c3e50' if is_baseline else '#e74c3c'
            ax.set_title(name, fontsize=8, color=title_color,
                        fontweight='bold' if is_baseline else 'normal')
            ax.axis('off')
            col += 1

        fig.suptitle(f'Case: {case_id}  |  ET volume: {et_vol} vox  |  Slice: {best_slice}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'case_{desc}.png'),
                    dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()

        # ── Small ET zoom figure (separate, if ET is small) ──
        bbox = _find_small_et_bbox(gt)
        if bbox is not None:
            _generate_small_et_zoom(
                flair, gt, preds, bbox, case_id, desc, save_dir)

    print(f"Saved case overlays to: {save_dir}/")


def _plot_mask_overlay(ax, mri_slice, mask_slice):
    """
    Overlay 3-class mask on MRI slice.

    Colors: WT=red, TC=green, ET=blue (standard BraTS scheme)
    mask_slice: (3, H, W)
    """
    mri = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-9)
    ax.imshow(mri, cmap='gray')

    # Overlay each class with transparency
    # ET (blue)
    if mask_slice[2].sum() > 0:
        et_overlay = np.zeros((*mask_slice[2].shape, 4))
        et_overlay[..., 2] = 1.0  # blue
        et_overlay[..., 3] = mask_slice[2] * 0.7
        ax.imshow(et_overlay)

    # TC (green)
    if mask_slice[1].sum() > 0:
        tc_overlay = np.zeros((*mask_slice[1].shape, 4))
        tc_overlay[..., 1] = 1.0  # green
        tc_overlay[..., 3] = mask_slice[1] * 0.5
        ax.imshow(tc_overlay)

    # WT (red — but only where TC and ET are 0, to avoid covering)
    if mask_slice[0].sum() > 0:
        wt_only = mask_slice[0] * (1 - mask_slice[1]) * (1 - mask_slice[2])
        if wt_only.sum() > 0:
            wt_overlay = np.zeros((*wt_only.shape, 4))
            wt_overlay[..., 0] = 1.0  # red
            wt_overlay[..., 3] = wt_only * 0.4
            ax.imshow(wt_overlay)


def _generate_small_et_zoom(flair, gt, preds, bbox, case_id, desc, save_dir):
    """
    Generate a small-ET zoom figure: MRI | GT | Baseline | Best Others.
    Each row = one axial slice through the ET region.
    """
    z1, z2, y1, y2, x1, x2 = bbox
    n_z = z2 - z1
    if n_z == 0:
        return
    # Pick up to 3 slices evenly spaced through the small ET region
    slice_indices = [z1 + n_z // 4, z1 + n_z // 2, z1 + 3 * n_z // 4]
    slice_indices = sorted(set(min(s, z2 - 1) for s in slice_indices))

    # Pick top models by ET Dice (up to 4, excluding baseline)
    # For now just use first 4 non-baseline models
    non_bl_models = {k: v for k, v in preds.items() if 'Baseline' not in k}
    show_models = list(non_bl_models.items())[:3]  # show up to 3
    baseline_pred = {k: v for k, v in preds.items() if 'Baseline' in k}

    n_rows = len(slice_indices)
    n_cols = 2 + 1 + len(show_models)  # MRI | GT | Baseline | model2 | model3 | ...
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for r, z in enumerate(slice_indices):
        # Crop slice to bbox
        mri_crop = flair[z, y1:y2, x1:x2]
        gt_crop  = gt[:, z, y1:y2, x1:x2]

        mri_crop = (mri_crop - mri_crop.min()) / (mri_crop.max() - mri_crop.min() + 1e-9)

        # MRI
        axes[r, 0].imshow(mri_crop, cmap='gray')
        axes[r, 0].set_ylabel(f'z={z}', fontsize=8)
        if r == 0:
            axes[r, 0].set_title('MRI (FLAIR)', fontsize=9, fontweight='bold')
        axes[r, 0].axis('off')

        # GT
        _plot_mask_overlay(axes[r, 1], mri_crop, gt_crop if gt_crop.ndim == 3 else gt_crop)
        if r == 0:
            axes[r, 1].set_title('Ground Truth', fontsize=9, fontweight='bold')
        axes[r, 1].axis('off')

        # Baseline
        col = 2
        for bname, bpred in baseline_pred.items():
            bpred_crop = bpred[:, z, y1:y2, x1:x2]
            _plot_mask_overlay(axes[r, col], mri_crop, bpred_crop)
            if r == 0:
                axes[r, col].set_title('Baseline', fontsize=9, fontweight='bold', color='#2c3e50')
            axes[r, col].axis('off')
            col += 1

        # Other models
        for mname, mpred in show_models:
            mpred_crop = mpred[:, z, y1:y2, x1:x2]
            _plot_mask_overlay(axes[r, col], mri_crop, mpred_crop)
            if r == 0:
                short_name = mname[:25] + '..' if len(mname) > 25 else mname
                axes[r, col].set_title(short_name, fontsize=8, color='#e74c3c')
            axes[r, col].axis('off')
            col += 1

    fig.suptitle(f'Small ET Zoom — {case_id}  ({desc})',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'small_et_zoom_{desc}.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_legend_figure(save_path='figures/color_legend.png'):
    """Generate a standalone legend figure for mask colors."""
    fig, ax = plt.subplots(figsize=(4, 1.5))
    ax.set_xlim(0, 4); ax.set_ylim(0, 1)
    ax.axis('off')

    legend_elements = [
        mpatches.Patch(facecolor='red',   alpha=0.6, label='WT (Whole Tumor) — Edema'),
        mpatches.Patch(facecolor='green', alpha=0.6, label='TC (Tumor Core) — NCR/NET'),
        mpatches.Patch(facecolor='blue',  alpha=0.6, label='ET (Enhancing Tumor) — Active'),
    ]
    ax.legend(handles=legend_elements, loc='center', ncol=3,
              fontsize=10, frameon=False)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# 3. Combined multi-panel figure (paper Figure 2-3)
# ============================================================

def generate_multipanel_qualitative(dataloader, models_dict, case_idx,
                                   save_path='figures/qualitative_multipanel.png'):
    """
    Generate a 2-row multi-panel figure:
      Row 1: MRI | GT | Baseline | Best Model prediction (full slice)
      Row 2: Small ET zoom crops
    """
    # Similar to generate_case_overlay but as a single combined figure
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_data = []
    for data in dataloader:
        all_data.append({
            'id':    data['Id'],
            'image': data['image'].to(device),
            'mask':  data['mask'],
        })
        if len(all_data) > case_idx:
            break

    if case_idx >= len(all_data):
        print(f"[WARN] case_idx {case_idx} out of range ({len(all_data)} cases)")
        return

    case = all_data[case_idx]
    case_id = case['id'][0] if isinstance(case['id'], (list, tuple)) else case['id']
    flair = case['image'][0, 0].cpu().numpy()
    gt = case['mask'][0].cpu().numpy()

    best_slice = _get_best_slice(gt)
    et_vol = int(gt[2].sum())

    # Get predictions
    preds = {}
    with torch.no_grad():
        for name, model in models_dict.items():
            model.eval()
            logits = model(case['image'])
            probs = torch.sigmoid(logits)
            preds[name] = (probs >= 0.33).float()[0].cpu().numpy()

    # Layout: top row = 1 MRI + 1 GT + up to 4 predictions
    n_pred_cols = min(len(preds), 4)
    n_top_cols = 2 + n_pred_cols  # MRI + GT + preds
    bbox = _find_small_et_bbox(gt)

    if bbox:
        fig = plt.figure(figsize=(max(16, 4 * n_top_cols), 9))
        gs = fig.add_gridspec(2, max(n_top_cols, 3), hspace=0.25, wspace=0.05)
    else:
        fig = plt.figure(figsize=(4 * n_top_cols, 4.5))
        gs = fig.add_gridspec(1, n_top_cols)

    # Top row
    for col, (ax_name, ax_data) in enumerate(_top_row_axes(fig, gs, n_top_cols)):
        if col == 0:  # MRI
            mri_slice = (flair[best_slice] - flair[best_slice].min()) / \
                        (flair[best_slice].max() - flair[best_slice].min() + 1e-9)
            ax_data.imshow(mri_slice, cmap='gray')
            ax_data.set_title('MRI (FLAIR)', fontsize=10, fontweight='bold')
        elif col == 1:  # GT
            _plot_mask_overlay(ax_data, flair[best_slice], gt[:, best_slice])
            ax_data.set_title('Ground Truth', fontsize=10, fontweight='bold')
        else:  # Predictions
            pred_names = list(preds.keys())
            pi = col - 2
            if pi < len(pred_names):
                pname = pred_names[pi]
                _plot_mask_overlay(ax_data, flair[best_slice], preds[pname][:, best_slice])
                is_bl = 'Baseline' in pname
                ax_data.set_title(pname, fontsize=8,
                                 color='#2c3e50' if is_bl else '#e74c3c',
                                 fontweight='bold' if is_bl else 'normal')
        ax_data.axis('off')

    # Bottom row: small ET zoom
    if bbox:
        z1, z2, y1, y2, x1, x2 = bbox
        zoom_slices = [z1 + (z2 - z1)//3, z1 + (z2 - z1)//2, z1 + 2*(z2 - z1)//3]
        zoom_slices = sorted(set(min(s, z2-1) for s in zoom_slices))

        n_zoom_cols = min(2 + len(preds), n_top_cols)
        for zr, z in enumerate(zoom_slices):
            mri_crop = flair[z, y1:y2, x1:x2]
            mri_crop = (mri_crop - mri_crop.min()) / (mri_crop.max() - mri_crop.min() + 1e-9)
            gt_crop = gt[:, z, y1:y2, x1:x2]

            for col in range(n_zoom_cols):
                ax = fig.add_subplot(gs[1 + zr, col])
                if col == 0:
                    ax.imshow(mri_crop, cmap='gray')
                    if zr == 0: ax.set_title('MRI zoom', fontsize=8)
                elif col == 1:
                    _plot_mask_overlay(ax, mri_crop, gt_crop)
                    if zr == 0: ax.set_title('GT zoom', fontsize=8)
                else:
                    pnames = list(preds.keys())
                    pi = col - 2
                    if pi < len(pnames):
                        pname = pnames[pi]
                        pcrop = preds[pname][:, z, y1:y2, x1:x2]
                        _plot_mask_overlay(ax, mri_crop, pcrop)
                        if zr == 0:
                            short = pname[:20] + '..' if len(pname) > 20 else pname
                            ax.set_title(short, fontsize=7)
                ax.axis('off')

    fig.suptitle(f'Case {case_id}  |  ET volume: {et_vol} vox  |  Axial slice {best_slice}',
                 fontsize=11, fontweight='bold')
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def _top_row_axes(fig, gs, n_cols):
    """Helper: create axes objects for top row, handling grid spec."""
    axes = []
    for col in range(n_cols):
        axes.append((col, fig.add_subplot(gs[0, col])))
    return axes


# ============================================================
# 4. Master function — called from eval script
# ============================================================

def generate_all_figures(all_metrics, all_histories, dataloader,
                         models_dict, output_dir='figures'):
    """
    Generate ALL paper-ready figures.

    Args:
        all_metrics:   list of per-model metric dicts (from evaluate_experiments)
        all_histories: list of training history dicts
        dataloader:    test DataLoader
        models_dict:   {'model_name': nn.Module} (evaluated models on device)
        output_dir:    where to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Quantitative bar charts ──────────────────────────────
    print("\n--- Generating quantitative charts ---")
    plot_et_dice_barchart(all_metrics,
                          os.path.join(output_dir, 'et_dice_barchart.png'))
    plot_hd95_barchart(all_metrics,
                       os.path.join(output_dir, 'hd95_barchart.png'))
    plot_lesion_recall_barchart(all_metrics,
                                os.path.join(output_dir, 'lesion_recall_barchart.png'))
    plot_small_case_dice_barchart(all_metrics,
                                  os.path.join(output_dir, 'small_case_dice_barchart.png'))
    plot_et_recall_precision_barchart(all_metrics,
                                      os.path.join(output_dir, 'et_recall_precision.png'))

    # ── Qualitative case overlays ────────────────────────────
    print("\n--- Generating qualitative overlays ---")
    generate_legend_figure(os.path.join(output_dir, 'color_legend.png'))

    # Find cases with small ET
    et_vols = all_metrics[0].get('_et_volumes', []) if all_metrics else []
    if not et_vols and all_metrics:
        et_vols_raw = all_metrics[0].get('_raw_dice', {}).get('ET', [])
        et_vols = [0] * len(et_vols_raw)
    if not et_vols and all_metrics:
        et_vols = all_metrics[0].get('_et_volumes', [])

    case_ids = []
    if all_metrics:
        lesion_results = all_metrics[0].get('_lesion_results', [])
        if lesion_results:
            case_ids = [lr.get('case_id', f'case_{i}') for i, lr in enumerate(lesion_results)]

    # Select cases to visualize
    if et_vols and len(et_vols) > 0:
        case_indices = find_case_with_small_et(et_vols, case_ids or [f'case_{i}' for i in range(len(et_vols))])
        if case_indices:
            generate_case_overlay(
                dataloader, models_dict, case_indices,
                save_dir=os.path.join(output_dir, 'cases'))

            # Combined multi-panel for the first small case
            generate_multipanel_qualitative(
                dataloader, models_dict, case_indices[0][0],
                save_path=os.path.join(output_dir, 'qualitative_multipanel.png'))

    print(f"\nAll figures saved to: {output_dir}/")
