"""
=============================================================================
BraTS2020 Unified Model Evaluation Framework
=============================================================================
Extensible evaluation script — add new experiments by appending to the
registry at the bottom of this file. No core logic changes needed.

Metrics (per WT / TC / ET):
  Dice, Recall, Precision, HD95, NSD (Normalized Surface Dice, τ=1mm)

Diagnostic metrics:
  Lesion-wise Recall, Precision & F1 (ET connected-component level)
  Small-case ET Dice (bottom 25% ET volume subset)

Infrastructure:
  Trainable parameter count, inference time per case
  Training curves (val_loss + val_dice from train_log.csv)

Output files:
  all_experiments_results.json   — full per-model metrics dict
  all_experiments_results.csv    — paper-ready flattened table
  paper_table.md                 — 3 Markdown tables (complete, Δ-vs-baseline, by-category)
  training_curves.png            — val_loss + val_dice over epochs

Usage:
    # Evaluate all registered models:
    python scripts/eval_all_experiments.py

    # Evaluate specific models by name substring:
    python scripts/eval_all_experiments.py --filter "Edge"

    # Skip inference time measurement (offline / CPU):
    python scripts/eval_all_experiments.py --no-timing

    # Custom test CSV:
    python scripts/eval_all_experiments.py --csv my_test.csv

Extending:
    To add a new model, append an entry to EXPERIMENTS at the bottom
    of this file. The only required fields are `dir`, `model_class`,
    `model_kwargs`, and `label`. See examples below.

Author: ResUNet Enhancement Project
Date:   2026-08-06
=============================================================================
"""

import os, sys, gc, json, time, argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resunet3d import ResUNet3d
from models.resunet_edge import ResUNetEdge
from models.resunet_fgfe import ResUNetFGFE
from models.resunet_hf_boundary import ResUNetHFBoundary
from models.resunet_hf_concat_boundary import ResUNetHFConcatBoundary
from data.dataset import BratsDataset, get_dataloader
from evaluation.advanced_metrics import (
    compute_all_advanced_metrics,
    print_comparison_table,
)

# ============================================================
# Experiment Registry — ADD NEW MODELS HERE
# ============================================================
#
# Each entry is a dict with these keys:
#
#   dir          (str, required)  — checkpoint directory
#   model_class  (type, required) — e.g. ResUNet3d, ResUNetEdge
#   model_kwargs (dict, required) — kwargs passed to model_class()
#   label        (str, required)  — display name in tables
#   category     (str, optional)  — groups models in report tables
#   key_remap    (str|None)       — None = standard; 'edge' = sobel→edge_extractor
#   is_baseline  (bool)           — if True, used as Δ reference (only one)
# ============================================================

EXPERIMENTS = [
    # ── Baseline ─────────────────────────────────────────────
    {
        'dir': '/root/autodl-tmp/ResUNet_model',
        'model_class': ResUNet3d,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'Baseline (BCEDice)',
        'category': 'Baseline',
        'is_baseline': True,
        'key_remap': None,
    },

    # ── V1: Loss Function Ablation ──────────────────────────
    {
        'dir': '/root/autodl-tmp/ResUNet_Enhanced_lb0.1_model',
        'model_class': ResUNet3d,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'λb=0.1 (Dice+CE+0.1·BD)',
        'category': 'V1: Loss Function',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_Enhanced_lb0.3_model',
        'model_class': ResUNet3d,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'λb=0.3 (Dice+CE+0.3·BD)',
        'category': 'V1: Loss Function',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_Enhanced_lb0.5_model',
        'model_class': ResUNet3d,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'λb=0.5 (Dice+CE+0.5·BD)',
        'category': 'V1: Loss Function',
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
        'key_remap': 'edge',
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_Edge_add_sobel_model',
        'model_class': ResUNetEdge,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24,
                         'fusion': 'add', 'edge_type': 'sobel'},
        'label': 'Edge (Sobel, add)',
        'category': 'V2: Architecture',
        'key_remap': 'edge',
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_Edge_concat_laplacian_model',
        'model_class': ResUNetEdge,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24,
                         'fusion': 'concat', 'edge_type': 'laplacian'},
        'label': 'Edge (Laplacian, concat)',
        'category': 'V2: Architecture',
        'key_remap': 'edge',
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_FGFE_model',
        'model_class': ResUNetFGFE,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'FGFE (Freq. Enhancement)',
        'category': 'V2: Architecture',
        'key_remap': None,
    },

    # ── SLA-FB: Data & Loss ─────────────────────────────────
    {
        'dir': '/root/autodl-tmp/ResUNet_FG_Sampling_model',
        'model_class': ResUNet3d,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'FG Sampling (4-strategy)',
        'category': 'SLA-FB: Data',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_CCDice_model',
        'model_class': ResUNet3d,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'CC-Dice Loss',
        'category': 'SLA-FB: Loss',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_PMDice_model',
        'model_class': ResUNet3d,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'PM-Dice Loss (γ=2)',
        'category': 'SLA-FB: Loss',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_BCECCDice_model',
        'model_class': ResUNet3d,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'BCE+CC-Dice (no Global Dice)',
        'category': 'SLA-FB: Loss',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_BCEPMDice_model',
        'model_class': ResUNet3d,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'BCE+PM-Dice (no Global Dice)',
        'category': 'SLA-FB: Loss',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_FullCombined_model',
        'model_class': ResUNet3d,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'Full (Global+CC+PM)',
        'category': 'SLA-FB: Loss',
        'key_remap': None,
    },

    # ── HF Boundary Branch ─────────────────────────────────
    {
        'dir': '/root/autodl-tmp/ResUNet_HFBoundary_model',
        'model_class': ResUNetHFBoundary,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24,
                         'edge_type': 'laplacian'},
        'label': 'HF Boundary (Laplacian, w=0.2)',
        'category': 'V2: Architecture',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_HFBoundary_Plus_model',
        'model_class': ResUNetHFBoundary,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24,
                         'edge_type': 'laplacian'},
        'label': 'HF Boundary+ (Laplacian, w=0.3)',
        'category': 'V2: Architecture',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_HFConcatBoundary_model',
        'model_class': ResUNetHFConcatBoundary,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'HF Concat Boundary (Laplacian, w=0.3)',
        'category': 'Final Combination',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_HFConcatBoundary_w0.1_model',
        'model_class': ResUNetHFConcatBoundary,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'HF Concat Boundary (Laplacian, w=0.1)',
        'category': 'Final Combination',
        'key_remap': None,
    },
    {
        'dir': '/root/autodl-tmp/ResUNet_HFConcatBoundary_w0.2_model',
        'model_class': ResUNetHFConcatBoundary,
        'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
        'label': 'HF Concat Boundary (Laplacian, w=0.2)',
        'category': 'Final Combination',
        'key_remap': None,
    },
]


# ============================================================
# Checkpoint discovery
# ============================================================

def find_checkpoint(model_dir, prefer='best'):
    """
    Find checkpoint in a directory.

    Args:
        model_dir: directory to scan
        prefer: 'best' → best_model_*.pth first; 'last' → last_epoch_model first

    Returns:
        (full_path, epoch_number) or (None, None)
    """
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


# ============================================================
# Model loading (handles key remapping compat)
# ============================================================

def load_model(spec, ckpt_path, device):
    """
    Load a model from a registry entry + checkpoint path.

    Handles:
      - Standard key remap: out.conv.0.* → out.conv.*
      - Edge key remap:    sobel.* → edge_extractor.*
      - Partial weight match (for FGFE: baseline encoder + new decoder)
    """
    model = spec['model_class'](**spec['model_kwargs']).to(device)
    state = torch.load(ckpt_path, map_location=device)

    # Always apply: old Out(Sequential) → new Out(Conv3d)
    state = {k.replace('out.conv.0.', 'out.conv.'): v for k, v in state.items()}

    # Edge-specific: sobel.* → edge_extractor.*
    if spec.get('key_remap') == 'edge':
        new_state = {}
        for k, v in state.items():
            if k.startswith('sobel.'):
                k = k.replace('sobel.', 'edge_extractor.', 1)
            new_state[k] = v
        state = new_state

    # Match by name + shape (survives partial loads like FGFE)
    model_state = model.state_dict()
    matched = {k: v for k, v in state.items()
               if k in model_state and v.shape == model_state[k].shape}
    model_state.update(matched)
    model.load_state_dict(model_state)

    return model, len(matched), len(model_state)


# ============================================================
# Inference timing
# ============================================================

def measure_inference_time(model, dataloader, device, warmup=3, max_batches=53):
    """Measure per-case inference time (wall-clock, GPU-synchronized)."""
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
# Training history
# ============================================================

def load_training_history(model_dir):
    """Load train_log.csv → structured dict for plotting."""
    log_path = os.path.join(model_dir, 'train_log.csv')
    if not os.path.exists(log_path):
        return None
    try:
        log = pd.read_csv(log_path)
        valid_loss = log['valid_loss'].dropna() if 'valid_loss' in log.columns else pd.Series(dtype=float)
        return {
            'epoch':       list(range(1, len(log) + 1)),
            'train_loss':  log.get('train_loss',  pd.Series(dtype=float)).tolist(),
            'valid_loss':  valid_loss.tolist(),
            'train_dice':  log.get('train_dice',  pd.Series(dtype=float)).tolist(),
            'valid_dice':  log.get('valid_dice',  pd.Series(dtype=float)).tolist(),
            'best_epoch':  int(valid_loss.idxmin()) + 1 if len(valid_loss) > 0 else None,
            'best_loss':   float(valid_loss.min())  if len(valid_loss) > 0 else None,
        }
    except Exception as e:
        print(f"  [WARN] Could not read train_log.csv: {e}")
        return None


# ============================================================
# Output generation (format-agnostic)
# ============================================================

def _save_all_outputs(all_metrics, all_histories, baseline):
    """Generate JSON, CSV, Markdown, and training-curve PNG."""
    # ── JSON ─────────────────────────────────────────────────
    json_ready = []
    for m in all_metrics:
        clean = {}
        for k, v in m.items():
            if k.startswith('_'):
                continue
            if isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, (np.integer,)):
                v = int(v)
            clean[k] = v
        json_ready.append(clean)
    with open('all_experiments_results.json', 'w') as f:
        json.dump(json_ready, f, indent=2, default=str)
    print("Saved: all_experiments_results.json")

    # ── CSV ──────────────────────────────────────────────────
    csv_rows = []
    for m in all_metrics:
        csv_rows.append({
            'Model':                   m['model_name'],
            'Category':                m.get('category', ''),
            'Macro_Dice':              f"{m.get('Macro_Dice_mean', 0):.4f}",
            'Checkpoint_Epoch':        m.get('checkpoint_epoch', ''),
            'ET_Dice':                 f"{m.get('ET_Dice_mean', 0):.4f} ± {m.get('ET_Dice_std', 0):.4f}",
            'ET_Recall':               f"{m.get('ET_Recall_mean', 0):.4f}",
            'ET_Precision':            f"{m.get('ET_Precision_mean', 0):.4f}",
            'ET_HD95_mm':              f"{m.get('ET_HD95_mean', 0):.2f}",
            'ET_NSD':                  f"{m.get('ET_NSD_mean', 0):.4f}",
            'TC_Dice':                 f"{m.get('TC_Dice_mean', 0):.4f} ± {m.get('TC_Dice_std', 0):.4f}",
            'TC_HD95_mm':              f"{m.get('TC_HD95_mean', 0):.2f}",
            'TC_NSD':                  f"{m.get('TC_NSD_mean', 0):.4f}",
            'WT_Dice':                 f"{m.get('WT_Dice_mean', 0):.4f} ± {m.get('WT_Dice_std', 0):.4f}",
            'Lesion_Recall':           f"{m.get('Lesion_Recall_mean', 0):.4f}",
            'Lesion_Precision':        f"{m.get('Lesion_Precision_mean', 0):.4f}",
            'Lesion_F1':               f"{m.get('Lesion_F1_mean', 0):.4f}",
            'Lesion_TP':               m.get('Total_TP_lesions', 0),
            'Lesion_FP':               m.get('Total_FP_lesions', 0),
            'Lesion_FN':               m.get('Total_FN_lesions', 0),
            'Overall_Lesion_Precision': f"{m.get('Overall_lesion_precision', 0):.4f}",
            'Overall_Lesion_Recall':   f"{m.get('Overall_lesion_recall', 0):.4f}",
            'Overall_Lesion_F1':       f"{m.get('Overall_lesion_f1', 0):.4f}",
            'Small_case_ET_Dice':      f"{m.get('Small_case_ET_Dice_mean', 0):.4f}",
            'n_params':                m.get('n_params', 0),
            'Inference_time_s':        f"{m.get('inference_time_mean_s', 0):.3f}",
        })
    pd.DataFrame(csv_rows).to_csv('all_experiments_results.csv', index=False)
    print("Saved: all_experiments_results.csv")

    # ── Markdown ─────────────────────────────────────────────
    _write_markdown_tables(all_metrics, baseline)

    # ── Training curves ──────────────────────────────────────
    if all_histories:
        _plot_training_curves(all_histories)
        print("Saved: training_curves.png")


def _write_markdown_tables(all_metrics, baseline):
    """3 Markdown tables: complete, delta, by-category."""
    lines = ["# ResUNet Enhancement — Experimental Results\n"]

    # Table 0: Core metrics — the four primary indicators
    core_cols = ['Model', 'Macro Dice', 'ET Dice', 'ET HD95↓', 'Small-case ET Dice']
    lines.append("## Core Metrics (primary indicators)\n")
    lines.append("| " + " | ".join(core_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(core_cols)) + "|")
    for m in all_metrics:
        lines.append("| " + " | ".join([
            m['model_name'],
            f"{m.get('Macro_Dice_mean', 0):.3f}",
            f"{m.get('ET_Dice_mean', 0):.3f}",
            f"{m.get('ET_HD95_mean', 0):.1f}",
            f"{m.get('Small_case_ET_Dice_mean', 0):.3f}",
        ]) + " |")

    # Table 1: Complete metrics
    cols = ['Model', 'ET Dice', 'ET Recall', 'ET Prec.', 'ET HD95↓', 'ET NSD↑',
            'TC Dice', 'TC HD95↓', 'WT Dice',
            'Lesion Rec.', 'Lesion Prec.', 'Lesion F1',
            'Small ET Dice', '#Params', 'Infer.(s)']
    lines.append("## Table 1: Complete Evaluation Metrics\n")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for m in all_metrics:
        lines.append("| " + " | ".join([
            m['model_name'],
            f"{m.get('ET_Dice_mean', 0):.3f}",
            f"{m.get('ET_Recall_mean', 0):.3f}",
            f"{m.get('ET_Precision_mean', 0):.3f}",
            f"{m.get('ET_HD95_mean', 0):.1f}",
            f"{m.get('ET_NSD_mean', 0):.3f}",
            f"{m.get('TC_Dice_mean', 0):.3f}",
            f"{m.get('TC_HD95_mean', 0):.1f}",
            f"{m.get('WT_Dice_mean', 0):.3f}",
            f"{m.get('Lesion_Recall_mean', 0):.3f}",
            f"{m.get('Lesion_Precision_mean', 0):.3f}",
            f"{m.get('Lesion_F1_mean', 0):.3f}",
            f"{m.get('Small_case_ET_Dice_mean', 0):.3f}",
            f"{m.get('n_params', 0):,}",
            f"{m.get('inference_time_mean_s', 0):.2f}",
        ]) + " |")

    # Table 2: Δ vs Baseline
    if baseline:
        lines.append("\n\n## Table 2: Delta vs Baseline\n")
        dcols = ['Model', 'Δ ET Dice', 'Δ ET Recall', 'Δ ET Prec.',
                 'Δ ET HD95', 'Δ ET NSD', 'Δ Lesion Rec.', 'Δ Small ET Dice']
        lines.append("| " + " | ".join(dcols) + " |")
        lines.append("|" + "|".join(["---"] * len(dcols)) + "|")
        for m in all_metrics:
            if m is baseline:
                lines.append(f"| {m['model_name']} | (baseline) | — | — | — | — | — | — |")
                continue
            lines.append("| " + " | ".join([
                m['model_name'],
                f"{m.get('ET_Dice_mean', 0) - baseline.get('ET_Dice_mean', 0):+.4f}",
                f"{m.get('ET_Recall_mean', 0) - baseline.get('ET_Recall_mean', 0):+.4f}",
                f"{m.get('ET_Precision_mean', 0) - baseline.get('ET_Precision_mean', 0):+.4f}",
                f"{m.get('ET_HD95_mean', 0) - baseline.get('ET_HD95_mean', 0):+.2f}",
                f"{m.get('ET_NSD_mean', 0) - baseline.get('ET_NSD_mean', 0):+.4f}",
                f"{m.get('Lesion_Recall_mean', 0) - baseline.get('Lesion_Recall_mean', 0):+.4f}",
                f"{m.get('Small_case_ET_Dice_mean', 0) - baseline.get('Small_case_ET_Dice_mean', 0):+.4f}",
            ]) + " |")

    # Table 3: Best by category
    lines.append("\n\n## Table 3: Best Model by Category\n")
    lines.append("| Category | Best Model | ET Dice | ET HD95 | Small ET Dice |")
    lines.append("|---|---|---|---|---|")
    categories = {}
    for m in all_metrics:
        cat = m.get('category', 'Other')
        categories.setdefault(cat, []).append(m)
    for cat, models in categories.items():
        best = max(models, key=lambda x: x.get('ET_Dice_mean', 0))
        lines.append(
            f"| {cat} | {best['model_name']} | {best.get('ET_Dice_mean', 0):.3f} | "
            f"{best.get('ET_HD95_mean', 0):.1f} | {best.get('Small_case_ET_Dice_mean', 0):.3f} |")

    with open('paper_table.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print("Saved: paper_table.md")


def _plot_training_curves(all_histories):
    """Validation loss + Dice over epochs, colored by category."""
    CATEGORY_COLORS = {
        'Baseline':         '#000000',
        'V1: Loss Function':'#e74c3c',
        'V2: Architecture': '#2ecc71',
        'SLA-FB: Data':     '#3498db',
        'SLA-FB: Loss':     '#9b59b6',
    }
    # Assign colors to any unseen categories
    palette = ['#f39c12', '#1abc9c', '#e67e22', '#9b59b6', '#34495e', '#c0392b']
    pi = 0
    for h in all_histories:
        cat = h.get('category', '')
        if cat not in CATEGORY_COLORS:
            CATEGORY_COLORS[cat] = palette[pi % len(palette)]
            pi += 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    for h in all_histories:
        color = CATEGORY_COLORS.get(h.get('category', ''), '#95a5a6')
        if h.get('valid_loss'):
            epochs = list(range(1, len(h['valid_loss']) + 1))
            ax1.plot(epochs, h['valid_loss'], color=color, alpha=0.7,
                     linewidth=1.2, label=h.get('label', '?'))
            if h.get('best_epoch'):
                ax1.axvline(x=h['best_epoch'], color=color, alpha=0.25,
                           linestyle='--', linewidth=0.5)
        if h.get('valid_dice'):
            epochs = list(range(1, len(h['valid_dice']) + 1))
            ax2.plot(epochs, h['valid_dice'], color=color, alpha=0.7,
                     linewidth=1.2, label=h.get('label', '?'))

    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Validation Loss')
    ax1.set_title('Validation Loss (all experiments)'); ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7, loc='upper right', ncol=2)

    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Validation Dice')
    ax2.set_title('Validation Dice (all experiments)'); ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7, loc='lower right', ncol=2)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# Core evaluation loop
# ============================================================

def evaluate_experiments(experiments, dataloader, device,
                         threshold=0.33, measure_timing=True):
    """
    Evaluate a list of experiment specs on a given dataloader.

    Args:
        experiments: list of registry dicts
        dataloader:  DataLoader yielding {'image': (B,4,D,H,W), 'mask': (B,3,D,H,W)}
        device:      torch.device
        threshold:   binarization threshold
        measure_timing: if False, skip inference timing

    Returns:
        all_metrics:  list of per-model metric dicts
        all_histories: list of training history dicts
        errors:       list of (label, error_message)
    """
    all_metrics = []
    all_histories = []
    errors = []
    total = len(experiments)

    for i, spec in enumerate(experiments):
        label = spec['label']
        print(f"\n{'=' * 80}")
        print(f"[{i+1}/{total}] {label}")
        print(f"      Category: {spec.get('category', '-')}")
        print(f"{'=' * 80}")

        # ── Find checkpoint ──────────────────────────────────
        ckpt_path, epoch_num = find_checkpoint(spec['dir'])
        if ckpt_path is None:
            print(f"  [SKIP] No checkpoint found")
            errors.append((label, 'no checkpoint'))
            continue
        print(f"  Checkpoint: {os.path.basename(ckpt_path)} (epoch {epoch_num})")

        try:
            # ── Load model ───────────────────────────────────
            model, n_matched, n_total = load_model(spec, ckpt_path, device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Params: {n_params:,}  "
                  f"(weights loaded: {n_matched}/{n_total})")

            # ── Inference time ───────────────────────────────
            if measure_timing:
                print(f"  Measuring inference time...")
                t_mean, t_std = measure_inference_time(model, dataloader, device)
                print(f"  Inference: {t_mean:.3f} ± {t_std:.3f} s/case")
            else:
                t_mean, t_std = float('nan'), float('nan')

            # ── Compute all segmentation metrics ──────────────
            print(f"  Computing metrics...")
            metrics = compute_all_advanced_metrics(
                model, dataloader, threshold=threshold, model_name=label)

            # ── Attach metadata ──────────────────────────────
            metrics['category']            = spec.get('category', '')
            metrics['checkpoint_path']     = ckpt_path
            metrics['checkpoint_epoch']    = epoch_num
            metrics['n_params']            = n_params
            metrics['n_params_matched']    = n_matched
            metrics['inference_time_mean_s'] = float(t_mean)
            metrics['inference_time_std_s']  = float(t_std)

            # ── Training history ──────────────────────────────
            history = load_training_history(spec['dir'])
            if history:
                history['label']    = label
                history['category'] = spec.get('category', '')
                all_histories.append(history)

            all_metrics.append(metrics)
            gc.collect()
            torch.cuda.empty_cache()

            # Quick summary
            print(f"  ✓ ET Dice={metrics.get('ET_Dice_mean',0):.4f}  "
                  f"ET HD95={metrics.get('ET_HD95_mean',0):.2f}  "
                  f"Lesion Recall={metrics.get('Lesion_Recall_mean',0):.4f}  "
                  f"Small ET Dice={metrics.get('Small_case_ET_Dice_mean',0):.4f}")

        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            errors.append((label, str(e)))

    return all_metrics, all_histories, errors


# ============================================================
# Priority metric report
# ============================================================

PRIORITY_METRICS = [
    ('Macro_Dice_mean',           'Macro Dice ↑'),
    ('ET_Dice_mean',              'ET Dice ↑'),
    ('ET_Recall_mean',            'ET Recall ↑'),
    ('ET_Precision_mean',         'ET Precision ↑'),
    ('ET_HD95_mean',              'ET HD95 (mm) ↓'),
    ('ET_NSD_mean',               'ET NSD (τ=1mm) ↑'),
    ('TC_Dice_mean',              'TC Dice ↑'),
    ('TC_HD95_mean',              'TC HD95 (mm) ↓'),
    ('WT_Dice_mean',              'WT Dice ↑'),
    ('Lesion_Recall_mean',        'Lesion-wise Recall ↑'),
    ('Lesion_Precision_mean',     'Lesion-wise Precision ↑'),
    ('Lesion_F1_mean',            'Lesion-wise F1 ↑'),
    ('Small_case_ET_Dice_mean',   'Small-case ET Dice ↑'),
    ('n_params',                  '#Params'),
    ('inference_time_mean_s',     'Inference (s/case)'),
]

COMPARISON_TABLE_KEYS = [
    'Macro_Dice_mean',
    'ET_Dice_mean', 'ET_Recall_mean', 'ET_Precision_mean',
    'ET_HD95_mean', 'ET_NSD_mean',
    'TC_Dice_mean', 'TC_HD95_mean', 'TC_NSD_mean',
    'WT_Dice_mean',
    'Lesion_Recall_mean', 'Lesion_Precision_mean', 'Lesion_F1_mean',
    'Overall_lesion_precision', 'Overall_lesion_recall', 'Overall_lesion_f1',
    'Small_case_ET_Dice_mean',
]


def print_delta_report(all_metrics, baseline):
    """Print per-metric Δ vs baseline."""
    if baseline is None:
        return
    print("\n" + "=" * 100)
    print("PAPER-READY PRIORITY METRICS (Δ vs Baseline)")
    print("=" * 100)
    for key, label in PRIORITY_METRICS:
        base_v = baseline.get(key, float('nan'))
        print(f"\n  [{label}]")
        for m in all_metrics:
            v = m.get(key, float('nan'))
            if np.isnan(v):
                print(f"    {m['model_name']:<45} N/A")
            else:
                d = v - base_v if not np.isnan(base_v) and m is not baseline else 0
                arrow = '↑' if d > 0 else ('↓' if d < 0 else '=')
                delta_str = f"  Δ={d:+.4f} {arrow}" if m is not baseline else ""
                print(f"    {m['model_name']:<45} {v:.4f}{delta_str}")


# ============================================================
# Main entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='BraTS2020 Unified Model Evaluation')
    parser.add_argument('--filter', type=str, default=None,
                        help='Only evaluate models whose label/dir contains this substring')
    parser.add_argument('--csv', type=str, default='tumourCSV.csv',
                        help='Path to data CSV')
    parser.add_argument('--threshold', type=float, default=0.33,
                        help='Binarization threshold')
    parser.add_argument('--no-timing', action='store_true',
                        help='Skip inference time measurement')
    parser.add_argument('--no-save', action='store_true',
                        help='Skip saving output files')
    parser.add_argument('--figures', action='store_true',
                        help='Generate paper-ready figures (bar charts + case overlays)')
    parser.add_argument('--figures-dir', type=str, default='figures',
                        help='Output directory for figures (default: figures/)')
    args = parser.parse_args()

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
    print("BraTS2020 Unified Model Evaluation")
    print(f"Device: {device} | Threshold: {args.threshold}")
    print(f"Models: {len(exps)}" + (f" (filtered by '{args.filter}')" if args.filter else ""))
    print("=" * 80)

    # ── Data ─────────────────────────────────────────────────
    print("\nLoading test dataloader...")
    test_loader = get_dataloader(BratsDataset, args.csv, phase='test')
    print(f"Test set: {len(test_loader.dataset)} cases, {len(test_loader)} batches")

    # ── Evaluate ─────────────────────────────────────────────
    all_metrics, all_histories, errors = evaluate_experiments(
        exps, test_loader, device,
        threshold=args.threshold,
        measure_timing=not args.no_timing,
    )

    if not all_metrics:
        print("\nNo models evaluated successfully.")
        if errors:
            for label, msg in errors:
                print(f"  {label}: {msg}")
        return

    # ── Baseline for Δ ───────────────────────────────────────
    baseline = None
    for m in all_metrics:
        # A model flagged is_baseline in registry, or the first one
        spec = next((e for e in exps if e['label'] == m['model_name']), None)
        if spec and spec.get('is_baseline'):
            baseline = m
            break
    if baseline is None:
        baseline = all_metrics[0]  # fallback: first evaluated model

    # ── Reports ──────────────────────────────────────────────
    print_comparison_table(all_metrics, COMPARISON_TABLE_KEYS)
    print_delta_report(all_metrics, baseline)

    # ── Save outputs ─────────────────────────────────────────
    if not args.no_save:
        _save_all_outputs(all_metrics, all_histories, baseline)

    # ── Generate figures (optional) ───────────────────────────
    if args.figures:
        print("\n" + "=" * 80)
        print("GENERATING PAPER-READY FIGURES")
        print("=" * 80)
        try:
            from evaluation.visualize_report import generate_all_figures

            # Reload models for qualitative visualization
            print("\nReloading models for visualization...")
            models_dict = {}
            for spec in exps:
                ckpt_path, epoch = find_checkpoint(spec['dir'])
                if ckpt_path is None:
                    continue
                model, _, _ = load_model(spec, ckpt_path, device)
                model.eval()
                models_dict[spec['label']] = model
                print(f"  Loaded: {spec['label']}")

            generate_all_figures(
                all_metrics, all_histories,
                test_loader, models_dict,
                output_dir=args.figures_dir,
            )

            # Free models
            for m in models_dict.values():
                del m
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [WARN] Figure generation failed: {e}")
            import traceback
            traceback.print_exc()

    # ── Errors ───────────────────────────────────────────────
    if errors:
        print(f"\n[WARNING] {len(errors)} skipped/error:")
        for label, msg in errors:
            print(f"  {label}: {msg}")

    print(f"\n{'=' * 80}")
    print(f"DONE — {len(all_metrics)}/{len(exps)} models evaluated successfully")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
