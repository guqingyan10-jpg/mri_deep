"""
=============================================================================
Advanced Evaluation Metrics for BraTS2020
=============================================================================
New metrics beyond original Dice/IoU:

  Pixel-level:
    - per_class_recall_precision()  — Recall & Precision per class (ET, TC, WT)
    - hd95()                        — Hausdorff Distance at 95th percentile
    - nsd()                         — Normalized Surface Distance (optional)

  Lesion-level (connected-component based):
    - lesion_wise_recall()          — % of GT ET lesions detected by prediction
    - lesion_wise_precision()       — % of predicted ET lesions that overlap GT
    - lesion_wise_f1()              — harmonic mean of lesion recall & precision

  Case-stratified:
    - small_case_dice()             — Dice computed only on small-ET subset
    - stratified_metrics()          — metrics by ET volume quartile

  Visualization:
    - boundary_overlay()            — GT vs Pred boundary contours on MRI slice
    - save_boundary_comparison()    — 4-model boundary comparison figure

Usage:
    from evaluation.advanced_metrics import (
        hd95, per_class_recall_precision, lesion_wise_recall,
        small_case_dice, compute_all_advanced_metrics
    )

Author: Generated for ResUNet enhancement project
Date:   2026-08-01
=============================================================================
"""

import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. Per-Class Recall & Precision
# ============================================================

def per_class_recall_precision(pred_binary, gt_binary, eps=1e-9):
    """
    Compute pixel-level Recall and Precision for binary mask.

    Recall = TP / (TP + FN)  — what % of real tumor was found?
    Precision = TP / (TP + FP) — what % of predicted tumor is correct?

    Args:
        pred_binary: (D, H, W) binary prediction
        gt_binary:   (D, H, W) binary ground truth

    Returns:
        recall, precision
    """
    tp = (pred_binary * gt_binary).sum()
    fn = ((1 - pred_binary) * gt_binary).sum()
    fp = (pred_binary * (1 - gt_binary)).sum()

    recall = (tp + eps) / (tp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)

    return float(recall), float(precision)


def compute_per_class_rp_all(pred, gt, threshold=0.33, classes=['WT', 'TC', 'ET']):
    """
    Compute per-class Recall and Precision across entire dataset.

    Args:
        pred:  (N, C, D, H, W) float logits or probabilities
        gt:    (N, C, D, H, W) float ground truth
        threshold: binarization threshold

    Returns:
        dict: {class: {'recall': [], 'precision': []}}
    """
    probs = torch.sigmoid(pred) if not isinstance(pred, np.ndarray) else pred
    if hasattr(probs, 'cpu'):
        probs = probs.cpu().numpy()
    if hasattr(gt, 'cpu'):
        gt = gt.cpu().numpy()

    pred_bin = (probs >= threshold).astype(np.float32)

    results = {cls: {'recall': [], 'precision': []} for cls in classes}
    N, C = pred_bin.shape[0], pred_bin.shape[1]

    for i in range(N):
        for c in range(min(C, len(classes))):
            r, p = per_class_recall_precision(pred_bin[i, c], gt[i, c])
            results[classes[c]]['recall'].append(r)
            results[classes[c]]['precision'].append(p)

    return results


# ============================================================
# 2. HD95 — Hausdorff Distance at 95th Percentile
# ============================================================

def surface_points(mask):
    """
    Extract surface voxel coordinates from a binary 3D mask.

    Surface = voxels where the mask is 1 AND at least one 6-neighbor is 0.
    """
    if mask.sum() == 0:
        return np.zeros((0, 3), dtype=np.int32)

    # Erosion with 6-connectivity structure
    struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    eroded = ndimage.binary_erosion(mask, structure=struct)
    surface = mask & (~eroded)
    coords = np.argwhere(surface)
    return coords


def hd95_single(pred_mask, gt_mask, voxel_spacing=(1, 1, 1)):
    """
    Compute Hausdorff Distance at 95th percentile between two binary masks.

    HD95 = 95th percentile of all minimum distances from each surface point
           of one mask to the surface of the other (symmetric).

    Uses scipy.ndimage.distance_transform_edt — vectorized, no for loops,
    efficient enough for 3D medical images.

    Args:
        pred_mask: (D, H, W) binary prediction
        gt_mask:   (D, H, W) binary ground truth
        voxel_spacing: (dz, dy, dx) in mm

    Returns:
        hd95 in mm, or NaN if either mask is empty
    """
    if gt_mask.sum() == 0:
        return float('nan')  # no GT — undefined

    if pred_mask.sum() == 0:
        return float('nan')  # no prediction — undefined

    # Compute distance transform of GT surface
    gt_surface = (gt_mask > 0) & ~ndimage.binary_erosion(
        gt_mask, structure=ndimage.generate_binary_structure(3, 1))
    gt_dist = ndimage.distance_transform_edt(
        ~gt_surface, sampling=voxel_spacing)

    # Compute distance transform of Pred surface
    pred_surface = (pred_mask > 0) & ~ndimage.binary_erosion(
        pred_mask, structure=ndimage.generate_binary_structure(3, 1))
    pred_dist = ndimage.distance_transform_edt(
        ~pred_surface, sampling=voxel_spacing)

    # GT boundary → nearest pred boundary
    if gt_surface.sum() > 0:
        d_gt_to_pred = gt_dist[pred_surface]
        hd95_gt_pred = np.percentile(d_gt_to_pred, 95) if len(d_gt_to_pred) > 0 else float('nan')
    else:
        hd95_gt_pred = float('nan')

    # Pred boundary → nearest GT boundary
    if pred_surface.sum() > 0:
        d_pred_to_gt = pred_dist[gt_surface]
        hd95_pred_gt = np.percentile(d_pred_to_gt, 95) if len(d_pred_to_gt) > 0 else float('nan')
    else:
        hd95_pred_gt = float('nan')

    if np.isnan(hd95_gt_pred) and np.isnan(hd95_pred_gt):
        return float('nan')
    elif np.isnan(hd95_gt_pred):
        return hd95_pred_gt
    elif np.isnan(hd95_pred_gt):
        return hd95_gt_pred

    return max(hd95_gt_pred, hd95_pred_gt)


def compute_hd95_all(model, dataloader, threshold=0.33,
                     voxel_spacing=(1, 1, 1), classes=['WT', 'TC', 'ET']):
    """
    Compute per-case HD95 for ET and TC across dataset.

    Returns:
        dict: {class: [hd95_values_per_case]}
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {cls: [] for cls in classes}

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Computing HD95"):
            imgs, targets = data['image'], data['mask']
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            if isinstance(logits, tuple):
                logits = logits[0]  # HF Boundary model returns (seg, boundary)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()

            B, C = preds_np.shape[0], preds_np.shape[1]
            for i in range(B):
                for c in range(min(C, len(classes))):
                    hd = hd95_single(preds_np[i, c], targets_np[i, c], voxel_spacing)
                    results[classes[c]].append(hd)

    return results


# ============================================================
# 3. Lesion-wise Recall (Connected Component based)
# ============================================================

def lesion_wise_detection(pred_et_mask, gt_et_mask, min_size=10, overlap_thresh=0.0):
    """
    Detect how many individual ET lesions in GT were found by prediction.

    Algorithm:
      1. Find connected components in GT ET mask (→ list of GT lesions)
      2. Find connected components in Pred ET mask (→ list of Pred lesions)
      3. Build the pairwise GT coverage matrix
      4. Use Hungarian assignment so each GT/Pred lesion is matched at most once
      5. Derive lesion-wise Precision, Recall, and F1 from TP/FP/FN

    Args:
        pred_et_mask: (D, H, W) binary prediction for ET
        gt_et_mask:   (D, H, W) binary ground truth for ET
        min_size:     minimum voxels to count as a real lesion
        overlap_thresh: minimum overlap ratio for detection (0 = any overlap)

    Returns:
        dict with lesion counts, TP/FP/FN, precision, recall, and F1
    """
    gt_labeled, gt_num = ndimage.label(np.asarray(gt_et_mask) > 0)

    # Filter small GT components
    gt_valid = []
    for comp_id in range(1, gt_num + 1):
        size = (gt_labeled == comp_id).sum()
        if size >= min_size:
            gt_valid.append(comp_id)
    gt_valid = sorted(gt_valid)

    pred_labeled, pred_num = ndimage.label(np.asarray(pred_et_mask) > 0)

    # Filter small Pred components
    pred_valid_map = {}
    for comp_id in range(1, pred_num + 1):
        size = (pred_labeled == comp_id).sum()
        if size >= min_size:
            pred_valid_map[comp_id] = size

    # --- One-to-one matching with Hungarian assignment ---
    pred_valid = list(pred_valid_map)
    overlap_ratios = np.zeros((len(gt_valid), len(pred_valid)), dtype=np.float64)
    for gt_idx, gt_id in enumerate(gt_valid):
        gt_comp_mask = gt_labeled == gt_id
        gt_size = gt_comp_mask.sum()
        for pred_idx, pred_id in enumerate(pred_valid):
            overlap = (gt_comp_mask & (pred_labeled == pred_id)).sum()
            overlap_ratios[gt_idx, pred_idx] = overlap / gt_size

    matched_pairs = []
    if overlap_ratios.size > 0:
        gt_indices, pred_indices = linear_sum_assignment(
            overlap_ratios, maximize=True)
        matched_pairs = [
            (int(gt_idx), int(pred_idx))
            for gt_idx, pred_idx in zip(gt_indices, pred_indices)
            if overlap_ratios[gt_idx, pred_idx] > overlap_thresh
        ]

    tp = len(matched_pairs)
    fp = len(pred_valid) - tp
    fn = len(gt_valid) - tp
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0 if fn > 0 else np.nan
    f1_denom = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denom if f1_denom > 0 else np.nan

    gt_sizes = [int((gt_labeled == g).sum()) for g in gt_valid]
    pred_sizes = [int((pred_labeled == p).sum()) for p in pred_valid_map]

    return {
        'gt_lesions':      len(gt_valid),
        'pred_lesions':    len(pred_valid_map),
        'detected':        tp,
        'tp_lesions':      tp,
        'fp_lesions':      fp,
        'fn_lesions':      fn,
        'lesion_recall':   recall,
        'lesion_precision': precision,
        'lesion_f1':       f1,
        'gt_sizes':        gt_sizes,
        'pred_sizes':      pred_sizes,
    }


def summarize_lesion_results(all_results):
    """Aggregate per-case lesion matches into macro and overall metrics."""
    precisions = [
        r['lesion_precision'] for r in all_results
        if not np.isnan(r['lesion_precision'])
    ]
    recalls = [
        r['lesion_recall'] for r in all_results
        if not np.isnan(r['lesion_recall'])
    ]
    f1_scores = [
        r['lesion_f1'] for r in all_results
        if not np.isnan(r['lesion_f1'])
    ]

    total_tp = sum(r['tp_lesions'] for r in all_results)
    total_fp = sum(r['fp_lesions'] for r in all_results)
    total_fn = sum(r['fn_lesions'] for r in all_results)

    precision_denom = total_tp + total_fp
    recall_denom = total_tp + total_fn
    f1_denom = 2 * total_tp + total_fp + total_fn

    return {
        'n_cases_with_et':          sum(r['gt_lesions'] > 0 for r in all_results),
        'total_gt_lesions':         sum(r['gt_lesions'] for r in all_results),
        'total_pred_lesions':       sum(r['pred_lesions'] for r in all_results),
        'total_detected':           total_tp,
        'total_tp_lesions':         total_tp,
        'total_fp_lesions':         total_fp,
        'total_fn_lesions':         total_fn,
        'mean_lesion_recall':       np.mean(recalls) if recalls else np.nan,
        'std_lesion_recall':        np.std(recalls) if recalls else np.nan,
        'mean_lesion_precision':    np.mean(precisions) if precisions else np.nan,
        'std_lesion_precision':     np.std(precisions) if precisions else np.nan,
        'mean_lesion_f1':           np.mean(f1_scores) if f1_scores else np.nan,
        'std_lesion_f1':            np.std(f1_scores) if f1_scores else np.nan,
        'overall_lesion_precision': total_tp / precision_denom if precision_denom > 0 else np.nan,
        'overall_lesion_recall':    total_tp / recall_denom if recall_denom > 0 else np.nan,
        'overall_lesion_f1':        2 * total_tp / f1_denom if f1_denom > 0 else np.nan,
        'per_case_recalls':         recalls,
        'per_case_precisions':      precisions,
        'per_case_f1_scores':       f1_scores,
    }


def compute_lesion_wise_all(model, dataloader, threshold=0.33, min_size=10):
    """
    Compute lesion-wise recall, precision, and F1 for all cases.

    Returns:
        results: list of per-case dicts
        summary: aggregated statistics
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Computing Lesion-wise metrics"):
            imgs, targets = data['image'], data['mask']
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            if isinstance(logits, tuple):
                logits = logits[0]  # HF Boundary model returns (seg, boundary)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            preds_np = preds.cpu().numpy()     # (B, 3, D, H, W)
            targets_np = targets.cpu().numpy()  # (B, 3, D, H, W)
            ids = data['Id']

            B = preds_np.shape[0]
            for i in range(B):
                # ET is channel index 2
                result = lesion_wise_detection(
                    preds_np[i, 2],
                    targets_np[i, 2],
                    min_size=min_size
                )
                result['case_id'] = ids[i] if isinstance(ids, list) else ids
                all_results.append(result)

    summary = summarize_lesion_results(all_results)

    return all_results, summary


# ============================================================
# 4. Small-case Dice
# ============================================================

def compute_small_case_dice(per_case_et_dice, et_volumes, percentile=25):
    """
    Compute mean ET Dice only on cases with small ET (bottom percentile).

    Args:
        per_case_et_dice: list/array of per-case ET Dice scores
        et_volumes:       list/array of per-case ET voxel counts
        percentile:       threshold percentile (default 25 = bottom quarter)

    Returns:
        small_dice_mean, small_dice_std, threshold, n_small
    """
    threshold = np.percentile(et_volumes, percentile)
    small_mask = np.array(et_volumes) <= threshold
    small_dice = np.array(per_case_et_dice)[small_mask]
    return {
        'small_case_dice_mean': np.mean(small_dice) if len(small_dice) > 0 else np.nan,
        'small_case_dice_std':  np.std(small_dice) if len(small_dice) > 0 else np.nan,
        'threshold_voxels':     threshold,
        'n_small_cases':        int(small_mask.sum()),
        'n_all_cases':          len(per_case_et_dice),
    }


# ============================================================
# 4b. NSD — Normalized Surface Dice
# ============================================================

def nsd_single(pred_mask, gt_mask, tau=1.0, voxel_spacing=(1, 1, 1)):
    """
    Normalized Surface Dice (NSD) between two binary 3D masks.

    NSD = (|S_gt ∩ B_pred^τ| + |S_pred ∩ B_gt^τ|) / (|S_gt| + |S_pred|)

    where S = surface, B^τ = band within distance τ.

    This tells you what fraction of the boundary is within τ mm of the
    other boundary — complementary to HD95. HD95 reports the worst-case
    distance; NSD reports the fraction of boundary that is "good enough."

    Args:
        pred_mask: (D, H, W) binary prediction
        gt_mask:   (D, H, W) binary ground truth
        tau:       tolerance distance in mm (default 1)
        voxel_spacing: (dz, dy, dx) in mm

    Returns:
        nsd: float, or NaN if either surface is empty
    """
    if gt_mask.sum() == 0 or pred_mask.sum() == 0:
        return float('nan')

    struct = ndimage.generate_binary_structure(3, 1)  # 6-connectivity

    # Ground truth surface
    gt_eroded = ndimage.binary_erosion(gt_mask, structure=struct)
    S_gt = (gt_mask > 0) & ~gt_eroded
    n_S_gt = S_gt.sum()

    # Prediction surface
    pred_eroded = ndimage.binary_erosion(pred_mask, structure=struct)
    S_pred = (pred_mask > 0) & ~pred_eroded
    n_S_pred = S_pred.sum()

    if n_S_gt == 0 or n_S_pred == 0:
        return float('nan')

    # Distance transform from pred surface
    dt_pred = ndimage.distance_transform_edt(~S_pred, sampling=voxel_spacing)
    dt_gt   = ndimage.distance_transform_edt(~S_gt,   sampling=voxel_spacing)

    # GT surface points within τ of pred surface
    matched_gt = (dt_pred[S_gt] <= tau).sum()

    # Pred surface points within τ of gt surface
    matched_pred = (dt_gt[S_pred] <= tau).sum()

    nsd = (matched_gt + matched_pred) / (n_S_gt + n_S_pred)
    return float(nsd)


# ============================================================
# 5. Boundary Overlay Visualization
# ============================================================

def boundary_overlay(mri_slice, gt_mask_slice, pred_mask_slice, ax=None, title=''):
    """
    Overlay GT and Pred boundaries on an MRI slice.

    Green = GT boundary (correct)
    Red   = Pred boundary

    Args:
        mri_slice:       (H, W) 2D array — MRI image (e.g., FLAIR)
        gt_mask_slice:   (H, W) 2D binary — GT mask
        pred_mask_slice: (H, W) 2D binary — Pred mask
        ax:              matplotlib axis (optional)
        title:           plot title
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Normalize MRI for display
    mri = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-9)

    # Extract boundaries using gradient magnitude
    from scipy.ndimage import sobel
    gt_edge = np.abs(sobel(gt_mask_slice.astype(float), axis=0)) + \
              np.abs(sobel(gt_mask_slice.astype(float), axis=1))
    gt_boundary = gt_edge > 0

    pred_edge = np.abs(sobel(pred_mask_slice.astype(float), axis=0)) + \
                np.abs(sobel(pred_mask_slice.astype(float), axis=1))
    pred_boundary = pred_edge > 0

    # Display
    ax.imshow(mri, cmap='gray')
    ax.imshow(np.ma.masked_where(~gt_boundary, gt_boundary),
              cmap='Greens', alpha=0.8, label='GT Boundary')
    ax.imshow(np.ma.masked_where(~pred_boundary, pred_boundary),
              cmap='Reds', alpha=0.6, label='Pred Boundary')

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='green', alpha=0.8, label='GT Boundary'),
        Patch(color='red', alpha=0.6, label='Pred Boundary'),
    ], loc='upper right', fontsize=8)

    if title:
        ax.set_title(title, fontsize=12)
    ax.axis('off')

    return ax


def save_boundary_comparison(models_dict, case_data, save_path,
                             slice_idx=None, classes=['ET', 'TC']):
    """
    Generate 4-model boundary comparison figure.

    Args:
        models_dict: {'UNet': model, 'ResUNet': model, ...}
        case_data:   dict with 'image' (4, D, H, W) and 'mask' (3, D, H, W)
        save_path:   path to save the figure
        slice_idx:   which slice to visualize (default: middle slice with max tumor)
        classes:     which classes to show

    Saves a figure with |Models| rows × |Classes| columns of boundary overlays.
    """
    import matplotlib.pyplot as plt
    from scipy.ndimage import sobel

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_models = len(models_dict)
    n_classes = len(classes)

    fig, axes = plt.subplots(n_models, n_classes,
                              figsize=(4 * n_classes, 4 * n_models))

    if n_models == 1:
        axes = axes.reshape(1, -1)

    # Get image and GT
    image_4d = case_data['image']     # (4, D, H, W) — modalities
    mask_3d = case_data['mask']       # (3, D, H, W) — WT, TC, ET

    # Use FLAIR (channel 0) for background
    flair = image_4d[0]  # (D, H, W)

    class_to_channel = {'WT': 0, 'TC': 1, 'ET': 2}

    for col, cls in enumerate(classes):
        c = class_to_channel[cls]
        gt_mask = mask_3d[c]

        # Find slice with max tumor for this class
        if slice_idx is None:
            slice_sums = gt_mask.reshape(gt_mask.shape[0], -1).sum(axis=1)
            best_slice = int(np.argmax(slice_sums)) if slice_sums.max() > 0 else gt_mask.shape[0] // 2
        else:
            best_slice = slice_idx

        flair_slice = flair[best_slice]
        gt_slice = gt_mask[best_slice]

        for row, (name, model) in enumerate(models_dict.items()):
            ax = axes[row, col]

            # Get model prediction
            model.eval()
            with torch.no_grad():
                img_t = torch.from_numpy(image_4d).unsqueeze(0).float().to(device)
                logits = model(img_t)
                probs = torch.sigmoid(logits)
                pred = (probs >= 0.33).float()
                pred_np = pred[0].cpu().numpy()

            pred_slice = pred_np[c, best_slice]

            # Boundary overlay
            boundary_overlay(flair_slice, gt_slice, pred_slice, ax=ax,
                           title=f'{name} — {cls} (slice {best_slice})')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Boundary comparison saved to: {save_path}")


# ============================================================
# 6. Master Function — Compute All Advanced Metrics
# ============================================================

def compute_all_advanced_metrics(model, dataloader, threshold=0.33,
                                  et_volumes=None, model_name='Model'):
    """
    One function to compute all advanced metrics for a single model.

    Args:
        model:      trained nn.Module
        dataloader: DataLoader for test/validation set
        threshold:  binarization threshold
        et_volumes: per-case ET voxel counts (for small-case Dice). If None, computed on the fly.
        model_name: label for printing

    Returns:
        dict with all metrics
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()

    # Accumulators
    all_dice = {'WT': [], 'TC': [], 'ET': []}
    all_recall = {'WT': [], 'TC': [], 'ET': []}
    all_precision = {'WT': [], 'TC': [], 'ET': []}
    all_hd95 = {'WT': [], 'TC': [], 'ET': []}
    all_nsd  = {'WT': [], 'TC': [], 'ET': []}
    et_vols = []
    case_ids = []

    # For lesion-wise
    lesion_results = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc=f"Evaluating {model_name}"):
            imgs, targets = data['image'], data['mask']
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            if isinstance(logits, tuple):
                logits = logits[0]  # HF Boundary model returns (seg, boundary)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            probs_np = probs.cpu().numpy()
            preds_np = preds.cpu().numpy()
            targets_np = targets.cpu().numpy()
            ids = data.get('Id', ['unknown'] * len(imgs))

            B, C = preds_np.shape[0], preds_np.shape[1]

            for i in range(B):
                # Per-class pixel-level metrics
                for c_idx, cls in enumerate(['WT', 'TC', 'ET']):
                    if c_idx < C:
                        # Dice
                        tp = (preds_np[i, c_idx] * targets_np[i, c_idx]).sum()
                        union = preds_np[i, c_idx].sum() + targets_np[i, c_idx].sum()
                        dice = 2 * tp / (union + 1e-9) if union > 0 else 1.0
                        all_dice[cls].append(float(dice))

                        # Recall & Precision
                        r, p = per_class_recall_precision(
                            preds_np[i, c_idx], targets_np[i, c_idx])
                        all_recall[cls].append(r)
                        all_precision[cls].append(p)

                        # HD95
                        hd = hd95_single(preds_np[i, c_idx], targets_np[i, c_idx])
                        all_hd95[cls].append(hd)

                        # NSD (Normalized Surface Dice, τ=1mm)
                        nsd_val = nsd_single(preds_np[i, c_idx], targets_np[i, c_idx], tau=1.0)
                        all_nsd[cls].append(nsd_val)

                # ET volume
                et_vols.append(float(targets_np[i, 2].sum()))
                case_ids.append(ids[i] if isinstance(ids, list) else ids)

                # Lesion-wise on ET (channel 2)
                lr = lesion_wise_detection(preds_np[i, 2], targets_np[i, 2])
                lr['case_id'] = case_ids[-1]
                lesion_results.append(lr)

    # --- Summarize ---
    metrics = {'model_name': model_name}

    for cls in ['ET', 'TC', 'WT']:
        metrics[f'{cls}_Dice_mean']  = np.mean(all_dice[cls])
        metrics[f'{cls}_Dice_std']   = np.std(all_dice[cls])
        metrics[f'{cls}_Recall_mean'] = np.mean(all_recall[cls])
        metrics[f'{cls}_Precision_mean'] = np.mean(all_precision[cls])

        hd_vals = [h for h in all_hd95[cls] if not np.isnan(h)]
        metrics[f'{cls}_HD95_mean'] = np.mean(hd_vals) if hd_vals else np.nan
        metrics[f'{cls}_HD95_std']  = np.std(hd_vals) if hd_vals else np.nan

        nsd_vals = [n for n in all_nsd[cls] if not np.isnan(n)]
        metrics[f'{cls}_NSD_mean'] = np.mean(nsd_vals) if nsd_vals else np.nan
        metrics[f'{cls}_NSD_std']  = np.std(nsd_vals) if nsd_vals else np.nan

    # Macro Dice — mean of the three sub-region Dice scores (WT/TC/ET)
    metrics['Macro_Dice_mean'] = np.mean([
        metrics['WT_Dice_mean'],
        metrics['TC_Dice_mean'],
        metrics['ET_Dice_mean'],
    ])

    # Lesion-wise summary from one-to-one TP/FP/FN matching
    lesion_summary = summarize_lesion_results(lesion_results)
    metrics['Lesion_Recall_mean'] = lesion_summary['mean_lesion_recall']
    metrics['Lesion_Recall_std'] = lesion_summary['std_lesion_recall']
    metrics['Lesion_Precision_mean'] = lesion_summary['mean_lesion_precision']
    metrics['Lesion_Precision_std'] = lesion_summary['std_lesion_precision']
    metrics['Lesion_F1_mean'] = lesion_summary['mean_lesion_f1']
    metrics['Lesion_F1_std'] = lesion_summary['std_lesion_f1']
    metrics['Total_GT_lesions'] = lesion_summary['total_gt_lesions']
    metrics['Total_Pred_lesions'] = lesion_summary['total_pred_lesions']
    metrics['Total_detected'] = lesion_summary['total_detected']
    metrics['Total_TP_lesions'] = lesion_summary['total_tp_lesions']
    metrics['Total_FP_lesions'] = lesion_summary['total_fp_lesions']
    metrics['Total_FN_lesions'] = lesion_summary['total_fn_lesions']
    metrics['Overall_lesion_precision'] = lesion_summary['overall_lesion_precision']
    metrics['Overall_lesion_recall'] = lesion_summary['overall_lesion_recall']
    metrics['Overall_lesion_f1'] = lesion_summary['overall_lesion_f1']

    # Small-case Dice
    valid_et = [(d, v) for d, v in zip(all_dice['ET'], et_vols) if v > 0]
    if valid_et:
        et_dice_vals, et_vol_vals = zip(*valid_et)
        small_info = compute_small_case_dice(list(et_dice_vals), list(et_vol_vals), percentile=25)
        metrics['Small_case_ET_Dice_mean'] = small_info['small_case_dice_mean']
        metrics['Small_case_ET_Dice_std']  = small_info['small_case_dice_std']
        metrics['Small_case_threshold']    = small_info['threshold_voxels']
        metrics['Small_case_n']            = small_info['n_small_cases']

    # Store raw data for later use
    metrics['_raw_dice'] = all_dice
    metrics['_raw_hd95'] = all_hd95
    metrics['_raw_nsd']  = all_nsd
    metrics['_raw_recall'] = all_recall
    metrics['_raw_precision'] = all_precision
    metrics['_et_volumes'] = et_vols
    metrics['_lesion_results'] = lesion_results

    return metrics


# ============================================================
# 7. Formatting — Comparison Table
# ============================================================

def print_comparison_table(all_metrics, metric_keys=None):
    """
    Print a formatted comparison table for multiple models.

    Args:
        all_metrics: list of dicts, each from compute_all_advanced_metrics()
    """
    if metric_keys is None:
        metric_keys = [
            'ET_Dice_mean', 'ET_Recall_mean', 'ET_Precision_mean', 'ET_HD95_mean',
            'TC_Dice_mean', 'TC_HD95_mean',
            'WT_Dice_mean',
            'Lesion_Recall_mean', 'Lesion_Precision_mean', 'Lesion_F1_mean',
            'Overall_lesion_precision', 'Overall_lesion_recall', 'Overall_lesion_f1',
            'Small_case_ET_Dice_mean',
        ]

    # Header
    print("\n" + "=" * 100)
    print("ADVANCED METRICS COMPARISON")
    print("=" * 100)

    col_width = max(len(k) for k in metric_keys) + 4
    header = f"{'Metric':<{col_width}}"
    for m in all_metrics:
        header += f" {m['model_name']:>18}"
    print(header)
    print("-" * len(header))

    # Best marker
    best_higher = {k for k in metric_keys if 'Recall' in k or 'Precision' in k or 'Dice' in k}
    best_lower = {k for k in metric_keys if 'HD95' in k}

    for key in metric_keys:
        values = []
        for m in all_metrics:
            v = m.get(key, np.nan)
            values.append(v)

        row = f"{key:<{col_width}}"
        for i, v in enumerate(values):
            if np.isnan(v):
                row += f" {'N/A':>18}"
            else:
                # Bold best value
                is_best = False
                valid = [x for x in values if not np.isnan(x)]
                if len(valid) >= 2:
                    if key in best_higher and v == max(valid):
                        is_best = True
                    elif key in best_lower and v == min(valid):
                        is_best = True

                if is_best:
                    row += f" \033[1m{v:>17.4f}\033[0m"  # bold via ANSI
                else:
                    row += f" {v:>18.4f}"
        print(row)

    print("=" * 100)
    print("Bold = best value. HD95: lower is better. Others: higher is better.")
    print("Small_case = bottom 25% ET volume subset.\n")
