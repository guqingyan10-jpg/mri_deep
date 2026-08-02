"""
=============================================================================
One-click evaluation: Baseline ResUNet vs lambda_b=0.1/0.3/0.5
=============================================================================
Evaluates all trained models on BraTS2020 test set with advanced metrics:
ET/TC/WT Dice, ET/TC Recall, ET/TC Precision, ET/TC HD95,
Lesion Recall, Lesion Precision, Small-case ET Dice.

Usage:
    python scripts/eval_lambda_experiments.py

Output:
    - Terminal: comparison table
    - lambda_results.csv: metrics for paper

Author: Generated for ResUNet enhancement project
Date:   2026-08-02
=============================================================================
"""

import os, sys, gc
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resunet3d import ResUNet3d
from data.dataset import BratsDataset, get_dataloader
from training.config import check_exist
from evaluation.advanced_metrics import compute_all_advanced_metrics, print_comparison_table

# ============================================================
# Configuration
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# All models to compare
MODELS = {
    'Baseline (BCEDice)': '/root/autodl-tmp/ResUNet_model/best_model_68.pth',
    'lb=0.1 (Dice+CE+0.1*BD)': '/root/autodl-tmp/ResUNet_Enhanced_lb0.1_model/best_model_6.pth',
    'lb=0.3 (Dice+CE+0.3*BD)': '/root/autodl-tmp/ResUNet_Enhanced_lb0.3_model/best_model_5.pth',
    'lb=0.5 (Dice+CE+0.5*BD)': '/root/autodl-tmp/ResUNet_Enhanced_lb0.5_model/best_model_5.pth',
}

# ============================================================
# Load data
# ============================================================

print('Loading test dataloader...')
test_loader = get_dataloader(BratsDataset, 'tumourCSV.csv', phase='test')
print(f'Test set: {len(test_loader)} batches')

# ============================================================
# Evaluate all models
# ============================================================

all_metrics = []
errors = []

for name, ckpt_path in MODELS.items():
    print(f'\n{"="*60}')
    print(f'Evaluating: {name}')
    print(f'Checkpoint: {ckpt_path}')
    print(f'{"="*60}')

    if not os.path.exists(ckpt_path):
        print(f'  [SKIP] Checkpoint not found: {ckpt_path}')
        errors.append((name, 'checkpoint missing'))
        continue

    try:
        model = ResUNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
        state = torch.load(ckpt_path, map_location=device)
        # Handle old Out(Sequential) key naming
        state = {k.replace('out.conv.0.', 'out.conv.'): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()
        print(f'  Model loaded: {sum(p.numel() for p in model.parameters()):,} params')

        metrics = compute_all_advanced_metrics(model, test_loader, model_name=name)
        all_metrics.append(metrics)
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'  [ERROR] {e}')
        errors.append((name, str(e)))

# ============================================================
# Results
# ============================================================

if errors:
    print(f'\n[WARNING] {len(errors)} model(s) had errors:')
    for name, err in errors:
        print(f'  {name}: {err}')

if len(all_metrics) == 0:
    print('No models evaluated. Exiting.')
    sys.exit(1)

# --- Terminal output ---
print_comparison_table(all_metrics)

# --- Priority metrics for paper ---
print("\n" + "=" * 80)
print("PRIORITY METRICS FOR PAPER (RESUNET BASELINE vs ENHANCED)")
print("=" * 80)

priority_keys = [
    ('ET_Dice_mean',        'ET Dice',              'higher'),
    ('TC_Dice_mean',        'TC Dice',              'higher'),
    ('ET_Recall_mean',      'ET Recall (pixel)',    'higher'),
    ('ET_HD95_mean',        'ET HD95 (mm)',         'lower'),
    ('TC_HD95_mean',        'TC HD95 (mm)',         'lower'),
    ('Lesion_Recall_mean',  'Lesion Recall',        'higher'),
    ('Small_case_ET_Dice_mean', 'Small-case ET Dice', 'higher'),
]

baseline = all_metrics[0]
for key, label, direction in priority_keys:
    print(f"\n  {label} ({direction} is better):")
    for m in all_metrics:
        v = m.get(key, float('nan'))
        delta = ''
        if m['model_name'] != baseline['model_name'] and not np.isnan(v) and not np.isnan(baseline.get(key, float('nan'))):
            d = v - baseline.get(key, 0)
            if direction == 'lower':
                delta = f"  ({'-' if d < 0 else '+'}{abs(d):.3f} vs baseline)"
            else:
                delta = f"  ({'+' if d > 0 else ''}{d:.3f} vs baseline)"
        print(f"    {m['model_name']:<40} {v:.4f}{delta}")

# --- Save CSV ---
rows = []
for m in all_metrics:
    rows.append({
        'Model': m['model_name'],
        'ET_Dice':        f"{m['ET_Dice_mean']:.4f}",
        'TC_Dice':        f"{m['TC_Dice_mean']:.4f}",
        'WT_Dice':        f"{m['WT_Dice_mean']:.4f}",
        'ET_Recall':      f"{m['ET_Recall_mean']:.4f}",
        'ET_Precision':   f"{m['ET_Precision_mean']:.4f}",
        'ET_HD95_mm':     f"{m['ET_HD95_mean']:.2f}",
        'TC_HD95_mm':     f"{m['TC_HD95_mean']:.2f}",
        'Lesion_Recall':  f"{m['Lesion_Recall_mean']:.4f}",
        'Lesion_Precision': f"{m['Lesion_Precision_mean']:.4f}",
        'Overall_Lesion_Recall': f"{m['Overall_lesion_recall']:.4f}",
        'Small_case_ET_Dice': f"{m.get('Small_case_ET_Dice_mean', 0):.4f}",
    })

pd.DataFrame(rows).to_csv('lambda_results.csv', index=False)
print('\nSaved: lambda_results.csv')
print('Done.')
