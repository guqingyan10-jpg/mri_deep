"""
=============================================================================
BraTS2020 Comprehensive Model Evaluation Framework
=============================================================================
Covers ALL standard metrics from MultiModel_XAI_Brats2020_HFF.ipynb PLUS
the advanced metrics from the enhance_resu project.

=== METRICS COVERED ===

Part 1: Per-Pixel Classification (HFF-style, cells 70-85)
  - TP/FP/TN/FN per model
  - Accuracy, Precision, Recall, F1-Score (per-class: WT, TC, ET)
  - Confusion Matrix visualization

Part 2: Per-Class Segmentation (HFF-style, cells 98-115)
  - Dice Score (WT, TC, ET) — per-case mean ± std
  - Jaccard/IoU Score (WT, TC, ET) — per-case mean ± std
  - Bar charts + tables

Part 3: Advanced Boundary & Lesion Metrics (enhance_resu)
  - HD95 (ET, TC) — 95th percentile Hausdorff Distance
  - NSD (ET, TC) — Normalized Surface Dice (τ=1mm)
  - Lesion-wise Recall & Precision
  - Small-case ET Dice (bottom 25% ET volume)

Part 4: Training Analysis (HFF-style, cells 90-98)
  - Training time per epoch (mean ± std)
  - Validation time per epoch (mean ± std)
  - Total training time
  - Best epoch / convergence speed
  - Individual training curves (loss, dice, jaccard)

Part 5: Qualitative (HFF-style, cells 116-141)
  - Slice-wise overlays (MRI + GT + Prediction)
  - 3D tumor visualization (Plotly)
  - Boundary comparison overlays

Part 6: XAI / GradCAM (HFF-style, cells 142-151)
  - Per-class GradCAM attention maps (WT, TC, ET)
  - Overlay visualizations

=== USAGE ===
    # ⭐ RECOMMENDED: Merge with existing advanced eval results (skips HD95/NSD/Lesion etc.):
    python scripts/eval_comprehensive.py --existing-results all_experiments_results.json

    # Only compute the NEW metrics (Parts 1,2,4,5), skip Part 3:
    python scripts/eval_comprehensive.py --skip-advanced

    # Full evaluation including advanced (if you DON'T have existing results):
    python scripts/eval_comprehensive.py

    # Evaluate specific models:
    python scripts/eval_comprehensive.py --filter "Edge" --existing-results all_experiments_results.json

    # Skip timing + figures on CPU:
    python scripts/eval_comprehensive.py --existing-results all_experiments_results.json --no-timing --no-figures

=== OUTPUT ===
    comprehensive_results/
    ├── comprehensive_results.json     — ALL metrics per model
    ├── comprehensive_results.csv      — flattened table
    ├── paper_table_comprehensive.md   — Markdown tables
    ├── confusion_matrices/            — per-model confusion matrix PNGs
    ├── training_curves/               — individual + combined training plots
    ├── per_class_metrics/             — Dice + Jaccard bar charts
    ├── qualitative/                   — slice overlays
    └── gradcam/                       — XAI attention maps (if --xai)

Author: ResUNet Enhancement Project
Date:   2026-08-11
=============================================================================
"""

import os, sys, gc, json, time, argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from collections import defaultdict
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resunet3d import ResUNet3d
from models.resunet_edge import ResUNetEdge
from models.resunet_fgfe import ResUNetFGFE
from models.resunet_hf_boundary import ResUNetHFBoundary
from models.resunet_hf_concat_boundary import ResUNetHFConcatBoundary
from data.dataset import BratsDataset, get_dataloader
from training.metrics import (
    dice_coef_metric_per_classes,
    jaccard_coef_metric_per_classes,
)
from evaluation.advanced_metrics import (
    compute_all_advanced_metrics,
    hd95_single, nsd_single,
    lesion_wise_detection, compute_small_case_dice,
)
from evaluation.evaluator import (
    compute_metrics, metric, plot_confusion_matrix,
    compute_scores_per_classes, compute_scores_per_classes_mean,
    print_metrics_table,
)

# ============================================================
# Experiment Registry — SAME as eval_all_experiments.py
# ============================================================

EXPERIMENTS = [
    # ── Baseline ─────────────────────────────────────────────
    {
        'dir': '/root/autodl-tmp/ResUNet_model',
        'model_class': ResUNet3d,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'Baseline (BCEDice)',
        'category': 'Baseline',
        'description': 'Standard 3D ResUNet with BCEDiceLoss. '
                       'Architecture: ResBlock encoder/decoder, GroupNorm, '
                       'MaxPool3d down, Trilinear up, skip connections.',
        'key_remap': None,
    },

    # ── V1: Loss Function Ablation ──────────────────────────
    {
        'dir': '/root/autodl-tmp/ResUNet_Enhanced_lb0.1_model',
        'model_class': ResUNet3d,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'λb=0.1 (Dice+CE+0.1·BD)',
        'category': 'V1: Loss Function',
        'description': 'Replaces BCEDiceLoss with DiceCEBoundaryLoss (λb=0.1). '
                       'Boundary loss from Kervadec et al. 2019 using distance-transform '
                       'on GT boundaries. Class weights: WT=1, TC=3, ET=5.',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_Enhanced_lb0.3_model',
        'model_class': ResUNet3d,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'λb=0.3 (Dice+CE+0.3·BD)',
        'category': 'V1: Loss Function',
        'description': 'Same as λb=0.1 but with increased boundary weight λb=0.3. '
                       'Tests whether stronger boundary supervision improves HD95.',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_Enhanced_lb0.5_model',
        'model_class': ResUNet3d,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'λb=0.5 (Dice+CE+0.5·BD)',
        'category': 'V1: Loss Function',
        'description': 'Same as λb=0.1 but with λb=0.5. Tests whether boundary loss '
                       'dominates and degrades volume segmentation.',
        'key_remap': None,
    },

    # ── V2: Architecture Changes ────────────────────────────
    {
        'dir': '/root/autodl-tmp/ResUNet_Edge_concat_sobel_model',
        'model_class': ResUNetEdge,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24,
                         'fusion': 'concat', 'edge_type': 'sobel'},
        'label': 'Edge (Sobel, concat)',
        'category': 'V2: Architecture',
        'description': 'Adds Sobel edge detection branch (1st-derivative) to ResUNet. '
                       'Edge features concatenated with decoder features. '
                       'Sobel kernels operate on each MRI channel independently, '
                       'then fused via 1×1×1 conv into edge feature map.',
        'key_remap': 'edge',
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_Edge_add_sobel_model',
        'model_class': ResUNetEdge,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24,
                         'fusion': 'add', 'edge_type': 'sobel'},
        'label': 'Edge (Sobel, add)',
        'category': 'V2: Architecture',
        'description': 'Same Sobel edge branch but edge features ADDED (residual) '
                       'to decoder instead of concatenated. Tests fusion strategy.',
        'key_remap': 'edge',
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_Edge_concat_laplacian_model',
        'model_class': ResUNetEdge,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24,
                         'fusion': 'concat', 'edge_type': 'laplacian'},
        'label': 'Edge (Laplacian, concat)',
        'category': 'V2: Architecture',
        'description': 'Replaces Sobel with Laplacian edge operator (2nd-derivative, '
                       'I−blur(I)). Laplacian captures zero-crossings and finer edge '
                       'details. Concat fusion with decoder features.',
        'key_remap': 'edge',
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_FGFE_model',
        'model_class': ResUNetFGFE,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'FGFE (Freq. Enhancement)',
        'category': 'V2: Architecture',
        'description': 'Feature-level Frequency Enhancement (Yao et al., MICCAI 2025). '
                       'Operates on decoder feature maps: Laplacian decomposition into '
                       'F_h (high-freq) + F_l (low-freq), cross-attention to query '
                       'skip features, residual add back. Learns to enhance frequency '
                       'components important for segmentation.',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_HFBoundary_model',
        'model_class': ResUNetHFBoundary,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24,
                         'edge_type': 'laplacian'},
        'label': 'HF Boundary (Laplacian, w=0.2)',
        'category': 'V2: Architecture',
        'description': 'High-Frequency Boundary branch: Laplacian edge extraction + '
                       'separate decoder path for boundary prediction. Boundary output '
                       'supervised with BD loss (weight=0.2), fused with main decoder '
                       'via learnable attention gate. Combines explicit edge prior '
                       'with learnable boundary refinement.',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_HFBoundary_Plus_model',
        'model_class': ResUNetHFBoundary,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24,
                         'edge_type': 'laplacian'},
        'label': 'HF Boundary+ (Laplacian, w=0.3)',
        'category': 'V2: Architecture',
        'description': 'Same as HF Boundary but with increased boundary loss weight '
                       '(w=0.3). Tests whether stronger boundary supervision in the '
                       'auxiliary branch further improves boundary quality.',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_HFConcatBoundary_model',
        'model_class': ResUNetHFConcatBoundary,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'HF Concat Boundary (Laplacian, w=0.3)',
        'category': 'Final Combination',
        'description': 'Final combination model. Extracts Laplacian high-frequency '
                       'residuals from raw MRI, builds a four-level EdgePyramid, '
                       'concatenates edge features into every decoder stage, and '
                       'adds a boundary prediction head supervised with weight 0.3.',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_HFConcatBoundary_w0.2_model',
        'model_class': ResUNetHFConcatBoundary,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'HF Concat Boundary (Laplacian, w=0.2)',
        'category': 'Final Combination',
        'description': 'Final combination model with intermediate boundary loss '
                       'weight (w=0.2). Sweeps the boundary-supervision strength '
                       'between the original 0.3 and the lighter 0.1.',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_HFConcatBoundary_w0.15_model',
        'model_class': ResUNetHFConcatBoundary,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'HF Concat Boundary (Laplacian, w=0.15)',
        'category': 'Final Combination',
        'description': 'Final combination model with boundary loss weight w=0.15. '
                       'Adds a finer point to the sweep between 0.2 and 0.1.',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_HFConcatBoundary_w0.1_model',
        'model_class': ResUNetHFConcatBoundary,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'HF Concat Boundary (Laplacian, w=0.1)',
        'category': 'Final Combination',
        'description': 'Final combination model with reduced boundary loss weight '
                       '(w=0.1). Tests whether a lighter boundary supervision '
                       'preserves segmentation accuracy while still refining edges.',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_HFConcatBoundary_w0.05_model',
        'model_class': ResUNetHFConcatBoundary,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'HF Concat Boundary (Laplacian, w=0.05)',
        'category': 'Final Combination',
        'description': 'Final combination model with the lightest boundary loss '
                       'weight (w=0.05). Tests whether a very weak boundary '
                       'supervision still improves small-lesion edges.',
        'key_remap': None,
    },

    # ── SLA-FB: Data ────────────────────────────────────────
    {
        'dir': '/root/autodl-tmp/ResUNet_FG_Sampling_model',
        'model_class': ResUNet3d,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'FG Sampling (4-strategy)',
        'category': 'SLA-FB: Data',
        'description': 'Foreground-Aware 3D Patch Sampling. 4 weighted strategies: '
                       'random (20%), foreground/WT-centered (30%), ET-connected-component-'
                       'centered (30%), small-lesion-only <50 vox (20%). Inspired by '
                       'STSNet (Zhao et al., 2025). No model/loss changes — only data '
                       'sampling changed. Addresses ET class imbalance (~0.5% of voxels).',
        'key_remap': None,
    },

    # ── SLA-FB: Loss ────────────────────────────────────────
    {
        'dir': '/root/autodl-tmp/ResUNet_CCDice_model',
        'model_class': ResUNet3d,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'CC-Dice Loss',
        'category': 'SLA-FB: Loss',
        'description': 'Connected-Component Dice Loss. Computes Dice separately for '
                       'each ET connected component, then averages. Each lesion '
                       'receives equal loss weight regardless of size — directly '
                       'targets the "Dice dominated by large lesions" problem. '
                       '42% of ET lesions < 50 vox are invisible to global Dice.',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_PMDice_model',
        'model_class': ResUNet3d,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'PM-Dice Loss (γ=2)',
        'category': 'SLA-FB: Loss',
        'description': 'Power-Mean Dice Loss (Hosseini et al., 2025). Modulates Dice '
                       'by |y−p̂|^γ, upweighting misclassified pixels exponentially. '
                       'γ=2 gives stronger penalty to boundary/rare-class errors. '
                       'Operates at pixel level unlike CC-Dice (component level).',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_BCECCDice_model',
        'model_class': ResUNet3d,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'BCE+CC-Dice (no Global Dice)',
        'category': 'SLA-FB: Loss',
        'description': 'BCE + CC-Level Dice WITHOUT global Dice. Ablation: tests '
                       'whether the CC-level Dice alone (with BCE) suffices, or '
                       'whether global Dice is still needed for volume integrity.',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_BCEPMDice_model',
        'model_class': ResUNet3d,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'BCE+PM-Dice (no Global Dice)',
        'category': 'SLA-FB: Loss',
        'description': 'BCE + PM-Dice WITHOUT global Dice. Ablation: tests whether '
                       'PM-Dice alone handles pixel-level imbalance sufficiently.',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_FullCombined_model',
        'model_class': ResUNet3d,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'Full (Global+CC+PM)',
        'category': 'SLA-FB: Loss',
        'description': 'Combined loss: Global Dice + CC-Level Dice + PM-Dice. '
                       'Multi-scale supervision: global (volume), component (lesion), '
                       'pixel (hard examples). Tests whether the combination '
                       'outperforms any single loss.',
        'key_remap': None,
    },
]


# ============================================================
# Checkpoint discovery
# ============================================================

def find_checkpoint(model_dir, prefer='best'):
    """Find checkpoint in a directory. Returns (full_path, epoch_number) or (None, None)."""
    if not os.path.isdir(model_dir):
        return None, None
    best_files = [f for f in os.listdir(model_dir) if f.startswith('best_model_')]
    last_files = [f for f in os.listdir(model_dir) if f.startswith('last_epoch_model')]
    if prefer == 'best':
        primary, fallback = best_files, last_files
    else:
        primary, fallback = last_files, best_files
    files = primary if primary else fallback
    if not files:
        return None, None
    chosen = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
    epoch = int(chosen.split('_')[-1].split('.')[0])
    return os.path.join(model_dir, chosen), epoch


def load_model(spec, ckpt_path, device):
    """Load model with key remapping for checkpoint compatibility."""
    model = spec['model_class'](**spec['model_kwargs']).to(device)
    state = torch.load(ckpt_path, map_location=device)

    # Always: old Out(Sequential) → new Out(Conv3d)
    state = {k.replace('out.conv.0.', 'out.conv.'): v for k, v in state.items()}

    # Edge-specific: sobel.* → edge_extractor.*
    if spec.get('key_remap') == 'edge':
        new_state = {}
        for k, v in state.items():
            if k.startswith('sobel.'):
                k = k.replace('sobel.', 'edge_extractor.', 1)
            new_state[k] = v
        state = new_state

    # Match by name + shape
    model_state = model.state_dict()
    matched = {k: v for k, v in state.items()
               if k in model_state and v.shape == model_state[k].shape}
    model_state.update(matched)
    model.load_state_dict(model_state)
    return model, len(matched), len(model_state)


# ============================================================
# PART 1: Per-Pixel Classification Metrics (HFF cells 70-85)
# ============================================================

def compute_per_class_confusion_matrix(model, dataloader, device, threshold=0.33):
    """
    Compute TP/FP/TN/FN separately for each class (WT, TC, ET).
    HFF notebook cells 70-73 style, extended to per-class.

    Returns:
        dict: {class_name: {'TP': int, 'FP': int, 'TN': int, 'FN': int}}
    """
    CLASS_NAMES = ['WT', 'TC', 'ET']
    results = {cls: {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0} for cls in CLASS_NAMES}

    with torch.no_grad():
        for data in tqdm(dataloader, desc='  Confusion matrix'):
            imgs, targets = data['image'].to(device), data['mask'].to(device)
            logits = model(imgs)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            for c, cls_name in enumerate(CLASS_NAMES):
                p = preds[:, c]      # (B, D, H, W)
                t = targets[:, c]    # (B, D, H, W)
                results[cls_name]['TP'] += torch.sum((p == 1) & (t == 1)).item()
                results[cls_name]['FP'] += torch.sum((p == 1) & (t == 0)).item()
                results[cls_name]['TN'] += torch.sum((p == 0) & (t == 0)).item()
                results[cls_name]['FN'] += torch.sum((p == 0) & (t == 1)).item()

            del imgs, targets, logits, probs, preds
            torch.cuda.empty_cache()

    return results


def compute_classification_metrics(confusion_dict):
    """
    From per-class TP/FP/TN/FN, compute Accuracy, Precision, Recall, F1.
    HFF notebook cell 75 style.

    Returns:
        dict: {class_name: {'Accuracy':, 'Precision':, 'Recall':, 'F1':}}
    """
    metrics = {}
    for cls_name, counts in confusion_dict.items():
        tp, fp, tn, fn = counts['TP'], counts['FP'], counts['TN'], counts['FN']
        eps = 1e-9
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        metrics[cls_name] = {
            'Accuracy': round(accuracy, 6),
            'Precision': round(precision, 6),
            'Recall': round(recall, 6),
            'F1': round(f1, 6),
        }
    return metrics


def save_confusion_matrix_plots(all_confusion, output_dir):
    """Save per-model per-class confusion matrix heatmaps."""
    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)

    for entry in all_confusion:
        label = entry['label']
        safe_label = label.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace(',', '')
        cm = entry['confusion']

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for i, cls_name in enumerate(['WT', 'TC', 'ET']):
            tp, fp, tn, fn = cm[cls_name]['TP'], cm[cls_name]['FP'], cm[cls_name]['TN'], cm[cls_name]['FN']
            conf_mat = np.array([[tp, fp], [fn, tn]])
            im = axes[i].imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
            axes[i].set_title(f'{label}\n{cls_name} Confusion Matrix', fontsize=11)
            axes[i].set_xticks([0, 1]); axes[i].set_yticks([0, 1])
            axes[i].set_xticklabels(['Tumor', 'Background'])
            axes[i].set_yticklabels(['Tumor', 'Background'])
            axes[i].set_xlabel('Predicted'); axes[i].set_ylabel('True')
            # Annotate
            thresh = conf_mat.max() / 2.
            for (r, c), val in np.ndenumerate(conf_mat):
                axes[i].text(c, r, f'{int(val):,}', ha='center', va='center',
                           color='white' if val > thresh else 'black', fontsize=10)
            plt.colorbar(im, ax=axes[i], fraction=0.046)

        plt.tight_layout()
        plt.savefig(os.path.join(cm_dir, f'{safe_label}_confusion.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved: {cm_dir}/ ({len(all_confusion)} models)")


# ============================================================
# PART 2: Per-Class Segmentation Metrics (HFF cells 98-115)
# ============================================================

def compute_per_class_dice_jaccard(model, dataloader, device, threshold=0.33):
    """
    Compute per-case Dice and Jaccard for WT/TC/ET.
    HFF notebook cell 101 style, using training.metrics functions.

    Returns:
        tuple: (dice_dict, jaccard_dict)
            Each dict: {'WT': [per_case_values], 'TC': [...], 'ET': [...]}
    """
    CLASSES = ['WT', 'TC', 'ET']
    dice_scores = {key: [] for key in CLASSES}
    jaccard_scores = {key: [] for key in CLASSES}

    with torch.no_grad():
        for data in tqdm(dataloader, desc='  Dice+Jaccard'):
            imgs, targets = data['image'].to(device), data['mask'].to(device)
            logits = model(imgs)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits_np = logits.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()

            d_scores = dice_coef_metric_per_classes(logits_np, targets_np, threshold)
            j_scores = jaccard_coef_metric_per_classes(logits_np, targets_np, threshold)

            for cls_name in CLASSES:
                dice_scores[cls_name].extend(d_scores[cls_name])
                jaccard_scores[cls_name].extend(j_scores[cls_name])

            del imgs, targets, logits, logits_np, targets_np
            torch.cuda.empty_cache()

    return dice_scores, jaccard_scores


def compute_per_class_dice_jaccard_stats(dice_dict, jaccard_dict):
    """
    Compute mean ± std for per-class Dice and Jaccard.

    Returns:
        dict with keys: WT_Dice_mean, WT_Dice_std, ..., ET_Jaccard_mean, ET_Jaccard_std
    """
    stats = {}
    for cls_name in ['WT', 'TC', 'ET']:
        d_vals = np.array(dice_dict[cls_name])
        j_vals = np.array(jaccard_dict[cls_name])
        stats[f'{cls_name}_Dice_mean'] = float(np.mean(d_vals))
        stats[f'{cls_name}_Dice_std'] = float(np.std(d_vals))
        stats[f'{cls_name}_Jaccard_mean'] = float(np.mean(j_vals))
        stats[f'{cls_name}_Jaccard_std'] = float(np.std(j_vals))
    return stats


def save_per_class_bar_charts(all_seg_metrics, output_dir):
    """HFF-style bar charts: Dice + Jaccard per class per model."""
    bar_dir = os.path.join(output_dir, 'per_class_metrics')
    os.makedirs(bar_dir, exist_ok=True)

    models = [m['label'] for m in all_seg_metrics]
    x = np.arange(len(models))
    width = 0.25

    fig, axes = plt.subplots(2, 1, figsize=(max(16, len(models)*1.2), 14))

    for row, (metric_name, ax) in enumerate([('Dice', axes[0]), ('Jaccard', axes[1])]):
        for i, cls_name in enumerate(['WT', 'TC', 'ET']):
            key = f'{cls_name}_{metric_name}_mean'
            values = [m.get(key, 0) for m in all_seg_metrics]
            errors = [m.get(f'{cls_name}_{metric_name}_std', 0) for m in all_seg_metrics]
            offset = (i - 1) * width
            ax.bar(x + offset, values, width, yerr=errors, capsize=3,
                   label=cls_name, alpha=0.85)

        ax.set_ylabel(f'{metric_name} Score', fontsize=13)
        ax.set_title(f'Per-Class {metric_name} Score — All Models', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(bar_dir, 'per_class_dice_jaccard.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {bar_dir}/per_class_dice_jaccard.png")


# ============================================================
# PART 3: Advanced Metrics (wrap existing)
# ============================================================

def compute_advanced_metrics_wrapper(model, dataloader, device, label, threshold=0.33):
    """
    Wrapper around compute_all_advanced_metrics.
    Returns the standard advanced metrics dict.
    """
    print("  Computing advanced metrics (HD95, NSD, Lesion-wise, Small-case)...")
    metrics = compute_all_advanced_metrics(
        model, dataloader, threshold=threshold, model_name=label)
    return metrics


# ============================================================
# PART 4: Training Analysis (HFF cells 90-98)
# ============================================================

def compute_training_time_stats(model_dir):
    """
    From train_log.csv, compute training/validation time statistics.
    HFF notebook cell 92 style.

    Returns:
        dict with training time metrics, or None if no log found
    """
    log_path = os.path.join(model_dir, 'train_log.csv')
    if not os.path.exists(log_path):
        return None

    try:
        log = pd.read_csv(log_path)

        # Find time columns (may be named differently)
        train_time_col = None
        valid_time_col = None
        for col in log.columns:
            if 'train' in col.lower() and 'time' in col.lower():
                train_time_col = col
            if ('valid' in col.lower() or 'val' in col.lower()) and 'time' in col.lower():
                valid_time_col = col

        stats = {
            'total_epochs': len(log),
            'best_epoch': int(log['valid_loss'].idxmin()) + 1 if 'valid_loss' in log.columns else None,
            'best_val_loss': float(log['valid_loss'].min()) if 'valid_loss' in log.columns else None,
            'best_val_dice': float(log['valid_dice'].max()) if 'valid_dice' in log.columns else None,
            'final_val_loss': float(log['valid_loss'].iloc[-1]) if 'valid_loss' in log.columns else None,
        }

        if train_time_col:
            stats['train_time_mean_s'] = float(log[train_time_col].mean())
            stats['train_time_std_s'] = float(log[train_time_col].std())
            stats['train_time_total_s'] = float(log[train_time_col].sum())

        if valid_time_col:
            stats['valid_time_mean_s'] = float(log[valid_time_col].mean())
            stats['valid_time_std_s'] = float(log[valid_time_col].std())

        return stats
    except Exception as e:
        print(f"  [WARN] Could not parse train_log.csv: {e}")
        return None


def save_individual_training_curves(all_histories, output_dir):
    """HFF-style: individual per-model training curves (cell 94 style)."""
    curve_dir = os.path.join(output_dir, 'training_curves')
    os.makedirs(curve_dir, exist_ok=True)

    for h in all_histories:
        label = h.get('label', 'unknown')
        safe_label = label.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace(',', '')[:60]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Row 1: Loss
        for col, metric in enumerate(['train_loss', 'valid_loss']):
            ax = axes[0, col]
            if h.get(metric):
                epochs = list(range(1, len(h[metric]) + 1))
                ax.plot(epochs, h[metric], color='#2ecc71' if 'train' in metric else '#e74c3c', linewidth=1.5)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(metric.replace('_', ' ').title())
                ax.grid(alpha=0.3)
                if h.get('best_epoch') and metric == 'valid_loss':
                    ax.axvline(x=h['best_epoch'], color='red', linestyle='--', alpha=0.5,
                              label=f"Best epoch {h['best_epoch']}")
                    ax.legend()
            else:
                ax.set_title(f'{metric} — N/A')

        # Row 1 col 3: Combined loss
        ax = axes[0, 2]
        if h.get('train_loss') and h.get('valid_loss'):
            epochs = list(range(1, len(h['train_loss']) + 1))
            ax.plot(epochs, h['train_loss'], color='#2ecc71', alpha=0.7, label='Train Loss', linewidth=1.2)
            v_epochs = list(range(1, len(h['valid_loss']) + 1))
            ax.plot(v_epochs, h['valid_loss'], color='#e74c3c', alpha=0.7, label='Valid Loss', linewidth=1.2)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
            ax.set_title('Combined Loss')
            ax.legend(); ax.grid(alpha=0.3)
            if h.get('best_epoch'):
                ax.axvline(x=h['best_epoch'], color='red', linestyle='--', alpha=0.3)

        # Row 2: Dice + Jaccard
        for col, metric in enumerate(['train_dice', 'valid_dice', 'valid_jaccard' if h.get('valid_jaccard') else 'train_jaccard' if h.get('train_jaccard') else 'valid_dice']):
            ax = axes[1, col]
            if col == 2:
                actual_metric = 'train_jaccard' if h.get('train_jaccard') else ('valid_jaccard' if h.get('valid_jaccard') else None)
                if actual_metric and h.get(actual_metric):
                    epochs = list(range(1, len(h[actual_metric]) + 1))
                    ax.plot(epochs, h[actual_metric], color='#9b59b6', linewidth=1.5)
                    ax.set_title(actual_metric.replace('_', ' ').title())
                    ax.grid(alpha=0.3)
                else:
                    ax.set_title('Jaccard — N/A')
            elif h.get(metric):
                epochs = list(range(1, len(h[metric]) + 1))
                ax.plot(epochs, h[metric], color='#3498db' if 'train' in metric else '#f39c12', linewidth=1.5)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(metric.replace('_', ' ').title())
                ax.grid(alpha=0.3)
            else:
                ax.set_title(f'{metric} — N/A')
            ax.set_xlabel('Epoch')

        fig.suptitle(f'Training Curves — {label}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(curve_dir, f'{safe_label}_curves.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved: {curve_dir}/ ({len(all_histories)} models)")


def save_combined_training_curves(all_histories, output_dir):
    """HFF-style: combined multi-model plots (cell 96-98 style)."""
    curve_dir = os.path.join(output_dir, 'training_curves')
    os.makedirs(curve_dir, exist_ok=True)

    CATEGORY_COLORS = {
        'Baseline': '#000000',
        'V1: Loss Function': '#e74c3c',
        'V2: Architecture': '#2ecc71',
        'SLA-FB: Data': '#3498db',
        'SLA-FB: Loss': '#9b59b6',
    }
    palette = ['#f39c12', '#1abc9c', '#e67e22', '#34495e', '#c0392b']
    pi = 0
    for h in all_histories:
        cat = h.get('category', '')
        if cat not in CATEGORY_COLORS:
            CATEGORY_COLORS[cat] = palette[pi % len(palette)]; pi += 1

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    # Subplot 1: Validation Loss
    for h in all_histories:
        color = CATEGORY_COLORS.get(h.get('category', ''), '#95a5a6')
        if h.get('valid_loss'):
            epochs = list(range(1, len(h['valid_loss']) + 1))
            axes[0, 0].plot(epochs, h['valid_loss'], color=color, alpha=0.7, linewidth=1.0,
                          label=h.get('label', '?'))
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Validation Loss')
    axes[0, 0].set_title('Validation Loss — All Models'); axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend(fontsize=6, loc='upper right', ncol=2)

    # Subplot 2: Validation Dice
    for h in all_histories:
        color = CATEGORY_COLORS.get(h.get('category', ''), '#95a5a6')
        if h.get('valid_dice'):
            epochs = list(range(1, len(h['valid_dice']) + 1))
            axes[0, 1].plot(epochs, h['valid_dice'], color=color, alpha=0.7, linewidth=1.0)
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Validation Dice')
    axes[0, 1].set_title('Validation Dice — All Models'); axes[0, 1].grid(alpha=0.3)

    # Subplot 3: Combined Loss (train + valid)
    for h in all_histories:
        color = CATEGORY_COLORS.get(h.get('category', ''), '#95a5a6')
        if h.get('train_loss'):
            epochs = list(range(1, len(h['train_loss']) + 1))
            axes[1, 0].plot(epochs, h['train_loss'], color=color, alpha=0.4, linewidth=0.7, linestyle=':')
        if h.get('valid_loss'):
            epochs = list(range(1, len(h['valid_loss']) + 1))
            axes[1, 0].plot(epochs, h['valid_loss'], color=color, alpha=0.7, linewidth=1.0,
                          label=h.get('label', '?'))
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Train (dotted) + Valid (solid) Loss'); axes[1, 0].grid(alpha=0.3)

    # Subplot 4: Bar — best val_dice per model
    labels_short = [h.get('label', '?')[:30] for h in all_histories]
    best_dices = [h.get('best_val_dice', 0) or 0 for h in all_histories]
    colors = [CATEGORY_COLORS.get(h.get('category', ''), '#95a5a6') for h in all_histories]
    axes[1, 1].barh(range(len(labels_short)), best_dices, color=colors)
    axes[1, 1].set_yticks(range(len(labels_short)))
    axes[1, 1].set_yticklabels(labels_short, fontsize=7)
    axes[1, 1].set_xlabel('Best Validation Dice')
    axes[1, 1].set_title('Best Validation Dice — All Models')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(curve_dir, 'combined_training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {curve_dir}/combined_training_curves.png")


# ============================================================
# PART 5: Qualitative Visualization (HFF cells 116-141)
# ============================================================

def save_qualitative_overlays(model, dataloader, device, output_dir, label, threshold=0.33):
    """
    HFF-style slice overlays (cell 118 style).
    Saves MRI + GT + Prediction overlay for first 3 test cases.
    """
    qual_dir = os.path.join(output_dir, 'qualitative')
    os.makedirs(qual_dir, exist_ok=True)
    safe_label = label.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '').replace(',', '')[:40]

    model.eval()
    with torch.no_grad():
        for case_idx, data in enumerate(dataloader):
            if case_idx >= 3:
                break
            imgs, targets = data['image'].to(device), data['mask'].to(device)
            case_id = data['Id'][0] if isinstance(data['Id'], list) else data['Id']
            logits = model(imgs)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            img_np = imgs[0].cpu().numpy()        # (4, D, H, W)
            gt_np = targets[0].cpu().numpy()       # (3, D, H, W)
            pred_np = preds[0].cpu().numpy()       # (3, D, H, W)

            # Find middle slice with ET content, or just middle
            D = img_np.shape[1]
            et_slices = np.where(gt_np[2].sum(axis=(1, 2)) > 0)[0]
            mid_slice = et_slices[len(et_slices)//2] if len(et_slices) > 0 else D // 2

            fig, axes = plt.subplots(2, 4, figsize=(20, 10))

            # Row 1: MRI modalities
            modalities = ['T1', 'T1ce', 'T2', 'FLAIR']
            for i, mod_name in enumerate(modalities):
                axes[0, i].imshow(img_np[i, mid_slice], cmap='gray')
                axes[0, i].set_title(f'{mod_name} (z={mid_slice})')
                axes[0, i].axis('off')

            # Row 2: GT + Prediction overlays
            class_names = ['WT', 'TC', 'ET']
            class_colors = ['yellow', 'blue', 'red']
            mri_bg = img_np[2, mid_slice]  # T2 as background

            for i, (cls_name, color) in enumerate(zip(class_names, class_colors)):
                axes[1, i].imshow(mri_bg, cmap='gray')
                gt_mask = np.ma.masked_where(gt_np[i, mid_slice] < 0.5, gt_np[i, mid_slice])
                pred_mask = np.ma.masked_where(pred_np[i, mid_slice] < 0.5, pred_np[i, mid_slice])
                axes[1, i].imshow(gt_mask, cmap=LinearSegmentedColormap.from_list('gt', ['none', color]), alpha=0.6)
                axes[1, i].imshow(pred_mask, cmap=LinearSegmentedColormap.from_list('pred', ['none', 'cyan']), alpha=0.3)
                axes[1, i].set_title(f'{cls_name} — Yellow=GT, Cyan=Pred (z={mid_slice})')
                axes[1, i].axis('off')

            # Row 2 col 4: Combined overlay
            axes[1, 3].imshow(mri_bg, cmap='gray')
            for i, cls_color in enumerate(['yellow', 'blue', 'red']):
                mask = np.ma.masked_where(gt_np[i, mid_slice] < 0.5, gt_np[i, mid_slice])
                axes[1, 3].imshow(mask, cmap=LinearSegmentedColormap.from_list(f'gt{i}', ['none', cls_color]), alpha=0.5)
            axes[1, 3].set_title(f'All GT — Y:WT, B:TC, R:ET (z={mid_slice})')
            axes[1, 3].axis('off')

            fig.suptitle(f'{label} — Case {case_id}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(qual_dir, f'{safe_label}_case{case_idx+1}.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)

            del imgs, targets, logits, probs, preds
            torch.cuda.empty_cache()

    print(f"Saved: {qual_dir}/ ({min(3, len(dataloader))} cases × {label[:20]}...)")


# ============================================================
# PART 6: Training History Loading
# ============================================================

def load_training_history(model_dir):
    """Load train_log.csv into structured dict."""
    log_path = os.path.join(model_dir, 'train_log.csv')
    if not os.path.exists(log_path):
        return None
    try:
        log = pd.read_csv(log_path)
        valid_loss = log['valid_loss'].dropna() if 'valid_loss' in log.columns else pd.Series(dtype=float)
        valid_dice = log['valid_dice'].dropna() if 'valid_dice' in log.columns else pd.Series(dtype=float)

        history = {
            'epoch': list(range(1, len(log) + 1)),
            'train_loss': log.get('train_loss', pd.Series(dtype=float)).tolist(),
            'valid_loss': valid_loss.tolist(),
            'train_dice': log.get('train_dice', pd.Series(dtype=float)).tolist(),
            'valid_dice': valid_dice.tolist(),
            'train_jaccard': log.get('train_jaccard', pd.Series(dtype=float)).tolist() if 'train_jaccard' in log.columns else None,
            'valid_jaccard': log.get('valid_jaccard', pd.Series(dtype=float)).tolist() if 'valid_jaccard' in log.columns else None,
            'best_epoch': int(valid_loss.idxmin()) + 1 if len(valid_loss) > 0 else None,
            'best_val_loss': float(valid_loss.min()) if len(valid_loss) > 0 else None,
            'best_val_dice': float(valid_dice.max()) if len(valid_dice) > 0 else None,
        }
        return history
    except Exception as e:
        print(f"  [WARN] Could not read train_log.csv: {e}")
        return None


# ============================================================
# Output: Comprehensive Report Generation
# ============================================================

def generate_comprehensive_report(all_results, output_dir):
    """
    Generate all output files from the comprehensive results dict.
    Each entry in all_results contains ALL metrics for one model.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── JSON ─────────────────────────────────────────────────
    json_ready = []
    for r in all_results:
        clean = {}
        for k, v in r.items():
            if k.startswith('_') or k in ('per_case_dice', 'per_case_jaccard'):
                continue
            if isinstance(v, (np.floating,)): v = float(v)
            elif isinstance(v, (np.integer,)): v = int(v)
            clean[k] = v
        json_ready.append(clean)

    json_path = os.path.join(output_dir, 'comprehensive_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_ready, f, indent=2, default=str)
    print(f"Saved: {json_path}")

    # ── CSV ──────────────────────────────────────────────────
    csv_rows = []
    for r in all_results:
        row = {}
        # Basic info
        row['Model'] = r.get('model_name', '?')
        row['Category'] = r.get('category', '')
        row['Checkpoint_Epoch'] = r.get('checkpoint_epoch', '')
        row['n_params'] = r.get('n_params', 0)
        row['inference_time_s'] = round(r.get('inference_time_mean_s', 0), 3)

        # Classification metrics (Part 1)
        for cls_name in ['WT', 'TC', 'ET']:
            cm = r.get('classification_metrics', {}).get(cls_name, {})
            row[f'{cls_name}_Accuracy'] = f"{cm.get('Accuracy', 0):.4f}" if cm else ''
            row[f'{cls_name}_Precision'] = f"{cm.get('Precision', 0):.4f}" if cm else ''
            row[f'{cls_name}_Recall'] = f"{cm.get('Recall', 0):.4f}" if cm else ''
            row[f'{cls_name}_F1'] = f"{cm.get('F1', 0):.4f}" if cm else ''

        # Dice + Jaccard (Part 2)
        for cls_name in ['WT', 'TC', 'ET']:
            row[f'{cls_name}_Dice'] = f"{r.get(f'{cls_name}_Dice_mean', 0):.4f} ± {r.get(f'{cls_name}_Dice_std', 0):.4f}"
            row[f'{cls_name}_Jaccard'] = f"{r.get(f'{cls_name}_Jaccard_mean', 0):.4f} ± {r.get(f'{cls_name}_Jaccard_std', 0):.4f}"

        # Advanced metrics (Part 3)
        for key in ['ET_HD95_mean', 'ET_NSD_mean', 'TC_HD95_mean', 'TC_NSD_mean',
                     'ET_Recall_mean', 'ET_Precision_mean',
                     'Lesion_Recall_mean', 'Lesion_Precision_mean', 'Lesion_F1_mean',
                     'Overall_lesion_precision', 'Overall_lesion_recall',
                     'Overall_lesion_f1', 'Small_case_ET_Dice_mean']:
            val = r.get(key, 0)
            row[key] = f"{val:.4f}" if val else ''

        # Training time (Part 4)
        tstats = r.get('training_time_stats', {}) or {}
        row['Total_Epochs'] = tstats.get('total_epochs', '')
        row['Best_Epoch'] = tstats.get('best_epoch', '')
        row['Train_Time_per_Epoch_s'] = f"{tstats.get('train_time_mean_s', 0):.1f}" if tstats.get('train_time_mean_s') else ''
        row['Valid_Time_per_Epoch_s'] = f"{tstats.get('valid_time_mean_s', 0):.1f}" if tstats.get('valid_time_mean_s') else ''
        row['Total_Train_Time_h'] = f"{tstats.get('train_time_total_s', 0)/3600:.2f}" if tstats.get('train_time_total_s') else ''

        csv_rows.append(row)

    csv_path = os.path.join(output_dir, 'comprehensive_results.csv')
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # ── Markdown Paper Table ─────────────────────────────────
    _write_comprehensive_markdown(all_results, output_dir)


def _write_comprehensive_markdown(all_results, output_dir):
    """Generate paper-ready Markdown with ALL metrics."""
    lines = [
        "# ResUNet Enhancement — Comprehensive Evaluation Results\n",
        "> Generated by `eval_comprehensive.py`\n",
        "> Covers: pixel-level classification, per-class segmentation (Dice+Jaccard), "
        "boundary quality (HD95, NSD), lesion-level detection, training analysis.\n",
    ]

    # ── Table 1: Per-Class Segmentation (Dice + Jaccard) ────
    lines.append("## Table 1: Per-Class Dice & Jaccard Coefficients\n")
    lines.append("| Model | WT Dice | TC Dice | ET Dice | WT Jaccard | TC Jaccard | ET Jaccard |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in all_results:
        lines.append("| " + " | ".join([
            r.get('model_name', '?'),
            f"{r.get('WT_Dice_mean', 0):.4f} ± {r.get('WT_Dice_std', 0):.4f}",
            f"{r.get('TC_Dice_mean', 0):.4f} ± {r.get('TC_Dice_std', 0):.4f}",
            f"{r.get('ET_Dice_mean', 0):.4f} ± {r.get('ET_Dice_std', 0):.4f}",
            f"{r.get('WT_Jaccard_mean', 0):.4f} ± {r.get('WT_Jaccard_std', 0):.4f}",
            f"{r.get('TC_Jaccard_mean', 0):.4f} ± {r.get('TC_Jaccard_std', 0):.4f}",
            f"{r.get('ET_Jaccard_mean', 0):.4f} ± {r.get('ET_Jaccard_std', 0):.4f}",
        ]) + " |")

    # ── Table 2: Pixel-level Classification ──────────────────
    lines.append("\n\n## Table 2: Per-Pixel Classification Metrics\n")
    for cls_name in ['WT', 'TC', 'ET']:
        lines.append(f"\n### {cls_name} — Pixel Classification\n")
        lines.append("| Model | Accuracy | Precision | Recall | F1-Score | TP | FP | TN | FN |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in all_results:
            cm = r.get('classification_metrics', {}).get(cls_name, {})
            counts = r.get('confusion_matrix', {}).get(cls_name, {})
            lines.append("| " + " | ".join([
                r.get('model_name', '?'),
                f"{cm.get('Accuracy', 0):.4f}",
                f"{cm.get('Precision', 0):.4f}",
                f"{cm.get('Recall', 0):.4f}",
                f"{cm.get('F1', 0):.4f}",
                f"{int(counts.get('TP', 0)):,}",
                f"{int(counts.get('FP', 0)):,}",
                f"{int(counts.get('TN', 0)):,}",
                f"{int(counts.get('FN', 0)):,}",
            ]) + " |")

    # ── Table 3: Advanced Metrics ────────────────────────────
    lines.append("\n\n## Table 3: Advanced Boundary & Lesion Metrics\n")
    lines.append("| Model | ET HD95↓ | ET NSD↑ | TC HD95↓ | TC NSD↑ | "
                 "ET Recall | ET Prec. | Lesion Rec. | Lesion Prec. | Lesion F1 | Small ET Dice |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in all_results:
        lines.append("| " + " | ".join([
            r.get('model_name', '?'),
            f"{r.get('ET_HD95_mean', 0):.1f}",
            f"{r.get('ET_NSD_mean', 0):.4f}",
            f"{r.get('TC_HD95_mean', 0):.1f}",
            f"{r.get('TC_NSD_mean', 0):.4f}",
            f"{r.get('ET_Recall_mean', 0):.4f}",
            f"{r.get('ET_Precision_mean', 0):.4f}",
            f"{r.get('Lesion_Recall_mean', 0):.4f}",
            f"{r.get('Lesion_Precision_mean', 0):.4f}",
            f"{r.get('Lesion_F1_mean', 0):.4f}",
            f"{r.get('Small_case_ET_Dice_mean', 0):.4f}",
        ]) + " |")

    # ── Table 4: Training Analysis ───────────────────────────
    lines.append("\n\n## Table 4: Training Time Analysis\n")
    lines.append("| Model | Total Epochs | Best Epoch | Train Time/Epoch (s) | "
                 "Valid Time/Epoch (s) | Total Train Time (h) | Best Val Loss | Best Val Dice |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in all_results:
        ts = r.get('training_time_stats', {}) or {}
        lines.append("| " + " | ".join([
            r.get('model_name', '?'),
            f"{ts.get('total_epochs', '—')}",
            f"{ts.get('best_epoch', '—')}",
            f"{ts.get('train_time_mean_s', 0):.1f}" if ts.get('train_time_mean_s') else "—",
            f"{ts.get('valid_time_mean_s', 0):.1f}" if ts.get('valid_time_mean_s') else "—",
            f"{ts.get('train_time_total_s', 0)/3600:.2f}" if ts.get('train_time_total_s') else "—",
            f"{ts.get('best_val_loss', 0):.6f}" if ts.get('best_val_loss') else "—",
            f"{ts.get('best_val_dice', 0):.4f}" if ts.get('best_val_dice') else "—",
        ]) + " |")

    # ── Table 5: Infrastructure ──────────────────────────────
    lines.append("\n\n## Table 5: Model Infrastructure\n")
    lines.append("| Model | #Params | Inference Time (s/case) | Checkpoint Epoch |")
    lines.append("|---|---|---|---|")
    for r in all_results:
        lines.append("| " + " | ".join([
            r.get('model_name', '?'),
            f"{r.get('n_params', 0):,}",
            f"{r.get('inference_time_mean_s', 0):.3f}",
            f"{r.get('checkpoint_epoch', '—')}",
        ]) + " |")

    md_path = os.path.join(output_dir, 'paper_table_comprehensive.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {md_path}")


# ============================================================
# Main Evaluation Loop
# ============================================================

def evaluate_all(exps, test_loader, device, output_dir, args):
    """Run NEW evaluation parts (1,2,4,5) for each model, merging existing Part 3 results."""
    os.makedirs(output_dir, exist_ok=True)

    # ── Load existing advanced results if provided ──────────
    existing_metrics = {}
    if args.existing_results and os.path.exists(args.existing_results):
        print(f"Loading existing advanced metrics from: {args.existing_results}")
        with open(args.existing_results, 'r') as f:
            existing_list = json.load(f)
        existing_metrics = {item['model_name']: item for item in existing_list}
        print(f"  Found {len(existing_metrics)} models with existing advanced metrics")

    all_results = []
    all_histories = []
    errors = []
    total = len(exps)
    skipped_advanced = 0

    for i, spec in enumerate(exps):
        label = spec['label']
        print(f"\n{'='*80}")
        print(f"[{i+1}/{total}] {label}")
        print(f"      Category: {spec.get('category', '-')}")
        print(f"{'='*80}")

        # Try to find existing advanced metrics for this model
        existing = existing_metrics.get(label) if existing_metrics else None

        # ── Checkpoint ─────────────────────────────────────
        ckpt_path, epoch_num = find_checkpoint(spec['dir'])
        if ckpt_path is None:
            # If we have existing results but no checkpoint, still include
            if existing:
                print(f"  [WARN] No checkpoint found, but using existing advanced metrics")
                result = {
                    'model_name': label,
                    'category': spec.get('category', ''),
                    'description': spec.get('description', ''),
                    'checkpoint_epoch': existing.get('checkpoint_epoch', '?'),
                }
                # Copy advanced metrics from existing
                for key in ADVANCED_METRIC_KEYS:
                    if key in existing:
                        result[key] = existing[key]
                all_results.append(result)
                skipped_advanced += 1
                print(f"  ✓ Using existing advanced metrics (no model reload needed)")
                continue
            else:
                print(f"  [SKIP] No checkpoint found and no existing results")
                errors.append((label, 'no checkpoint'))
                continue
        print(f"  Checkpoint: {os.path.basename(ckpt_path)} (epoch {epoch_num})")

        try:
            # ── Load model ──────────────────────────────────
            model, n_matched, n_total = load_model(spec, ckpt_path, device)
            n_params = sum(p.numel() for p in model.parameters())
            model.eval()
            print(f"  Params: {n_params:,} (weights loaded: {n_matched}/{n_total})")

            result = {
                'model_name': label,
                'category': spec.get('category', ''),
                'description': spec.get('description', ''),
                'checkpoint_path': ckpt_path,
                'checkpoint_epoch': epoch_num,
                'n_params': n_params,
                'n_params_matched': n_matched,
            }

            # ── Inference time ───────────────────────────────
            if not args.no_timing:
                print("  Measuring inference time...")
                t_mean, t_std = measure_inference_time(model, test_loader, device)
                result['inference_time_mean_s'] = float(t_mean)
                result['inference_time_std_s'] = float(t_std)
                print(f"  Inference: {t_mean:.3f} ± {t_std:.3f} s/case")
            elif existing and 'inference_time_mean_s' in existing:
                result['inference_time_mean_s'] = existing['inference_time_mean_s']
                result['inference_time_std_s'] = existing.get('inference_time_std_s', 0)
            else:
                result['inference_time_mean_s'] = float('nan')
                result['inference_time_std_s'] = float('nan')

            # ── Part 1: Classification Metrics ────────────────
            print("  [Part 1/4] Computing classification metrics (NEW)...")
            cm_dict = compute_per_class_confusion_matrix(model, test_loader, device, args.threshold)
            cls_metrics = compute_classification_metrics(cm_dict)
            result['confusion_matrix'] = cm_dict
            result['classification_metrics'] = cls_metrics
            for cls_name in ['WT', 'TC', 'ET']:
                m = cls_metrics[cls_name]
                print(f"    {cls_name}: Acc={m['Accuracy']:.4f} Prec={m['Precision']:.4f} "
                      f"Rec={m['Recall']:.4f} F1={m['F1']:.4f}")

            # ── Part 2: Per-Class Dice + Jaccard ──────────────
            print("  [Part 2/4] Computing per-class Dice + Jaccard (NEW)...")
            dice_dict, jaccard_dict = compute_per_class_dice_jaccard(model, test_loader, device, args.threshold)
            dj_stats = compute_per_class_dice_jaccard_stats(dice_dict, jaccard_dict)
            result.update(dj_stats)
            result['_per_case_dice'] = dice_dict
            result['_per_case_jaccard'] = jaccard_dict
            for cls_name in ['WT', 'TC', 'ET']:
                print(f"    {cls_name}: Dice={dj_stats[f'{cls_name}_Dice_mean']:.4f}±{dj_stats[f'{cls_name}_Dice_std']:.4f}  "
                      f"Jaccard={dj_stats[f'{cls_name}_Jaccard_mean']:.4f}±{dj_stats[f'{cls_name}_Jaccard_std']:.4f}")

            # ── Part 3: Advanced Metrics (FROM EXISTING or COMPUTE) ──
            if args.skip_advanced:
                print("  [Part 3/4] SKIPPED (--skip-advanced)")
            elif existing:
                # Copy from existing results
                print(f"  [Part 3/4] Merging existing advanced metrics...")
                for key in ADVANCED_METRIC_KEYS:
                    if key in existing:
                        result[key] = existing[key]
                copied = sum(1 for k in ADVANCED_METRIC_KEYS if k in existing)
                print(f"    Copied {copied} advanced metrics from existing results")
                if existing.get('inference_time_mean_s') and np.isnan(result.get('inference_time_mean_s', float('nan'))):
                    result['inference_time_mean_s'] = existing['inference_time_mean_s']
            else:
                print("  [Part 3/4] Computing advanced metrics (no existing results)...")
                adv_metrics = compute_advanced_metrics_wrapper(model, test_loader, device, label, args.threshold)
                result.update(adv_metrics)
                print(f"    ET HD95={adv_metrics.get('ET_HD95_mean', 0):.2f}mm  "
                      f"NSD={adv_metrics.get('ET_NSD_mean', 0):.4f}  "
                      f"Lesion Rec={adv_metrics.get('Lesion_Recall_mean', 0):.4f}")

            if existing:
                # Show summary of merged advanced values
                et_hd = result.get('ET_HD95_mean', '?')
                et_nsd = result.get('ET_NSD_mean', '?')
                lr = result.get('Lesion_Recall_mean', '?')
                small_et = result.get('Small_case_ET_Dice_mean', '?')
                print(f"    [from existing] ET HD95={et_hd} NSD={et_nsd} "
                      f"Lesion Rec={lr} Small ET Dice={small_et}")

            # ── Part 4: Training Analysis ─────────────────────
            print("  [Part 4/4] Loading training history (NEW)...")
            tstats = compute_training_time_stats(spec['dir'])
            result['training_time_stats'] = tstats
            history = load_training_history(spec['dir'])
            if history:
                history['label'] = label
                history['category'] = spec.get('category', '')
                all_histories.append(history)
            if tstats:
                print(f"    Epochs={tstats.get('total_epochs')}  Best={tstats.get('best_epoch')}  "
                      f"Train time/epoch={tstats.get('train_time_mean_s', 0):.1f}s")

            all_results.append(result)

            # ── Qualitative (optional) ─────────────────────────
            if not args.no_figures:
                print("  [Qualitative] Generating slice overlays...")
                save_qualitative_overlays(model, test_loader, device, output_dir, label, args.threshold)

            # Cleanup
            del model
            gc.collect()
            torch.cuda.empty_cache()

            parts_done = "1,2," + ("merged-3" if existing else "3" if not args.skip_advanced else "skipped-3") + ",4"
            print(f"  ✓ DONE — {label} (Parts: {parts_done})")

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            errors.append((label, str(e)))

    return all_results, all_histories, errors


# Keys to copy from existing advanced results
ADVANCED_METRIC_KEYS = [
    'ET_Dice_mean', 'ET_Dice_std', 'TC_Dice_mean', 'TC_Dice_std', 'WT_Dice_mean', 'WT_Dice_std',
    'ET_Recall_mean', 'ET_Precision_mean',
    'ET_HD95_mean', 'ET_NSD_mean',
    'TC_HD95_mean', 'TC_NSD_mean',
    'Lesion_Recall_mean', 'Lesion_Precision_mean', 'Lesion_F1_mean',
    'Total_GT_lesions', 'Total_Pred_lesions',
    'Total_TP_lesions', 'Total_FP_lesions', 'Total_FN_lesions',
    'Overall_lesion_precision', 'Overall_lesion_recall', 'Overall_lesion_f1',
    'Small_case_ET_Dice_mean',
    'inference_time_mean_s', 'inference_time_std_s',
]


def measure_inference_time(model, dataloader, device, warmup=3, max_batches=53):
    """Per-case inference time measurement."""
    model.eval()
    times = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            imgs = data['image'].to(device)
            start = time.perf_counter()
            _ = model(imgs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if i >= warmup:
                times.append(elapsed / max(imgs.shape[0], 1))
            if i >= max_batches - 1:
                break
    return np.mean(times) if times else float('nan'), np.std(times) if times else float('nan')


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='BraTS2020 Comprehensive Model Evaluation — NEW metrics only (merge with existing)')
    parser.add_argument('--existing-results', type=str, default='all_experiments_results.json',
                        help='Path to existing advanced eval JSON (from eval_all_experiments.py). '
                             'If found, skips Part 3 and merges metrics. '
                             'Use --skip-advanced if you want to skip Part 3 without existing results.')
    parser.add_argument('--skip-advanced', action='store_true',
                        help='Skip Part 3 (HD95/NSD/Lesion-wise/Small-case Dice) entirely')
    parser.add_argument('--filter', type=str, default=None,
                        help='Only evaluate models matching this substring')
    parser.add_argument('--csv', type=str, default='tumourCSV.csv',
                        help='Path to data CSV')
    parser.add_argument('--threshold', type=float, default=0.33,
                        help='Binarization threshold')
    parser.add_argument('--no-timing', action='store_true',
                        help='Skip inference time measurement')
    parser.add_argument('--no-figures', action='store_true',
                        help='Skip qualitative overlays (faster)')
    parser.add_argument('--output-dir', type=str, default='comprehensive_results',
                        help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check existing results
    if args.existing_results:
        if os.path.exists(args.existing_results):
            print(f"[INFO] Found existing results: {args.existing_results}")
            print(f"[INFO] Part 3 (advanced metrics) will be MERGED from existing file, NOT recomputed.")
        else:
            print(f"[INFO] Existing results file not found: {args.existing_results}")
            print(f"[INFO] Will compute Part 3 (advanced metrics) from scratch.")
            args.existing_results = None
    if args.skip_advanced:
        print(f"[INFO] --skip-advanced: Part 3 (HD95/NSD/Lesion-wise) will be skipped entirely.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Filter experiments ───────────────────────────────────
    exps = EXPERIMENTS
    if args.filter:
        exps = [e for e in EXPERIMENTS
                if args.filter.lower() in e['label'].lower()
                or args.filter.lower() in e['dir'].lower()]
        if not exps:
            print(f"No experiments match filter '{args.filter}'")
            return

    print("=" * 80)
    print("BraTS2020 Comprehensive Model Evaluation")
    print(f"Device: {device} | Threshold: {args.threshold}")
    print(f"Models: {len(exps)}" + (f" (filtered by '{args.filter}')" if args.filter else ""))
    print(f"Output: {args.output_dir}/")
    print("=" * 80)
    print()
    print("Metrics coverage:")
    print("  Part 1: Per-Pixel Classification (Acc, Prec, Rec, F1, Confusion Matrix) [NEW]")
    print("  Part 2: Per-Class Segmentation (Dice + Jaccard per WT/TC/ET)           [NEW]")
    if args.skip_advanced:
        print("  Part 3: Advanced (HD95, NSD, Lesion Recall, Small-case Dice)     [SKIPPED]")
    elif args.existing_results:
        print("  Part 3: Advanced (HD95, NSD, Lesion Recall, Small-case Dice)     [MERGED from existing]")
    else:
        print("  Part 3: Advanced (HD95, NSD, Lesion Recall, Small-case Dice)     [COMPUTED fresh]")
    print("  Part 4: Training Analysis (time/epoch, convergence, curves)          [NEW]")
    print("  Part 5: Qualitative (slice overlays)" + ("                             [SKIPPED]" if args.no_figures else "                                [NEW]"))
    print("=" * 80)

    # ── Data ─────────────────────────────────────────────────
    print("\nLoading test dataloader...")
    test_loader = get_dataloader(BratsDataset, args.csv, phase='test')
    print(f"Test set: {len(test_loader.dataset)} cases, {len(test_loader)} batches")

    # ── Run evaluation ───────────────────────────────────────
    all_results, all_histories, errors = evaluate_all(
        exps, test_loader, device, args.output_dir, args)

    # ── Generate aggregate outputs ─────────────────────────────
    print(f"\n{'='*80}")
    print("GENERATING AGGREGATE OUTPUTS")
    print(f"{'='*80}")

    # JSON + CSV + MD
    generate_comprehensive_report(all_results, args.output_dir)

    # Confusion matrix plots
    save_confusion_matrix_plots(all_results, args.output_dir)

    # Per-class bar charts
    save_per_class_bar_charts(all_results, args.output_dir)

    # Training curves
    if all_histories:
        save_individual_training_curves(all_histories, args.output_dir)
        save_combined_training_curves(all_histories, args.output_dir)

    # ── Errors ──────────────────────────────────────────────
    if errors:
        print(f"\n[WARNING] {len(errors)} skipped/error:")
        for lab, msg in errors:
            print(f"  {lab}: {msg}")

    # ── Summary ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE EVALUATION COMPLETE")
    print(f"  Models evaluated: {len(all_results)}/{len(exps)}")
    print(f"  Output directory: {os.path.abspath(args.output_dir)}")
    print(f"  Files generated:")
    print(f"    - comprehensive_results.json")
    print(f"    - comprehensive_results.csv")
    print(f"    - paper_table_comprehensive.md")
    print(f"    - confusion_matrices/")
    print(f"    - per_class_metrics/")
    print(f"    - training_curves/")
    if not args.no_figures:
        print(f"    - qualitative/")
    print(f"{'='*80}")
    print("\nNext steps:")
    print("  1. Download comprehensive_results/ from server")
    print("  2. Open paper_table_comprehensive.md for paper tables")
    print("  3. Fill values into the paper draft from paper_draft.md")


if __name__ == '__main__':
    main()
