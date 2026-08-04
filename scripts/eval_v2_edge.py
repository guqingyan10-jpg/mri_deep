"""
=============================================================================
V2 Edge Branch Evaluation: Sobel-concat vs Sobel-add
=============================================================================
Compares the two fusion modes (concat/add) for the Sobel edge branch
with identical hyperparams. Picks the winner for all subsequent V2
experiments (Laplacian, random control, etc.).

Usage:
    python scripts/eval_v2_edge.py

Output:
    - Terminal: comparison table (concat vs add)
    - v2_edge_results.csv

Author: Generated for ResUNet enhancement project
Date:   2026-08-04
=============================================================================
"""

import os, sys, gc
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resunet_edge import ResUNetEdge
from data.dataset import BratsDataset, get_dataloader
from evaluation.advanced_metrics import compute_all_advanced_metrics, print_comparison_table

# ============================================================
# Configuration
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Model directories to evaluate
# These are the OLD directory names from before edge_type was added.
# Both used Sobel as the edge extractor.
MODEL_DIRS = {
    'V2 Edge (Sobel, concat)': '/root/autodl-tmp/ResUNet_Edge_concat_model',
    'V2 Edge (Sobel, add)':    '/root/autodl-tmp/ResUNet_Edge_add_model',
}


def find_best_checkpoint(model_dir):
    """Find best_model_*.pth in directory."""
    if not os.path.isdir(model_dir):
        return None
    files = [f for f in os.listdir(model_dir) if f.startswith('best_model_')]
    if not files:
        # Fallback: try last_epoch_model
        files = [f for f in os.listdir(model_dir) if f.startswith('last_epoch_model')]
    if not files:
        return None
    # Pick latest by epoch number
    best = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
    return os.path.join(model_dir, best)


# ============================================================
# Load data
# ============================================================

print('Loading test dataloader...')
test_loader = get_dataloader(BratsDataset, 'tumourCSV.csv', phase='test')
print(f'Test set: {len(test_loader)} batches')

# ============================================================
# Evaluate
# ============================================================

all_metrics = []
errors = []

for name, model_dir in MODEL_DIRS.items():
    print(f'\n{"="*60}')
    print(f'Evaluating: {name}')
    print(f'Directory:  {model_dir}')
    print(f'{"="*60}')

    ckpt_path = find_best_checkpoint(model_dir)
    if ckpt_path is None:
        print(f'  [SKIP] No checkpoint found in {model_dir}')
        errors.append((name, 'no checkpoint'))
        continue

    print(f'Checkpoint: {ckpt_path}')

    try:
        # Use edge_type='sobel' — matches what was actually trained
        model = ResUNetEdge(
            in_channels=4, n_classes=3, n_channels=24,
            fusion='concat' if 'concat' in name else 'add',
            edge_type='sobel',
        ).to(device)

        state = torch.load(ckpt_path, map_location=device)
        # Remap old key names → new key names (backward compat)
        # Old: sobel.kx → New: edge_extractor.kx
        # Old: use_random_edge → New: edge_type
        new_state = {}
        for k, v in state.items():
            if k.startswith('sobel.'):
                k = k.replace('sobel.', 'edge_extractor.', 1)
            new_state[k] = v
        # Handle old Out(Sequential) key naming
        new_state = {k.replace('out.conv.0.', 'out.conv.'): v for k, v in new_state.items()}
        model.load_state_dict(new_state)
        model.eval()
        print(f'  Model loaded: {sum(p.numel() for p in model.parameters()):,} params')

        metrics = compute_all_advanced_metrics(model, test_loader, model_name=name)
        all_metrics.append(metrics)
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'  [ERROR] {e}')
        import traceback
        traceback.print_exc()
        errors.append((name, str(e)))

# ============================================================
# Results
# ============================================================

if errors:
    print(f'\n[WARNING] {len(errors)} model(s) had errors:')
    for name, err in errors:
        print(f'  {name}: {err}')

if len(all_metrics) < 2:
    print(f'Only {len(all_metrics)} model(s) evaluated. Need both concat and add. Exiting.')
    sys.exit(1)

print_comparison_table(all_metrics)

# ============================================================
# Head-to-head: which fusion wins?
# ============================================================

print("\n" + "=" * 80)
print("CONCAT vs ADD — HEAD-TO-HEAD (which fusion wins?)")
print("=" * 80)

# Key metrics for brain tumor segmentation
METRICS = [
    ('ET_Dice_mean',        'ET Dice',              'higher'),
    ('TC_Dice_mean',        'TC Dice',              'higher'),
    ('WT_Dice_mean',        'WT Dice',              'higher'),
    ('ET_HD95_mean',        'ET HD95 (mm)',         'lower'),
    ('TC_HD95_mean',        'TC HD95 (mm)',         'lower'),
    ('Lesion_Recall_mean',  'Lesion Recall',        'higher'),
    ('Small_case_ET_Dice_mean', 'Small-case ET Dice', 'higher'),
]

concat_m = all_metrics[0] if 'concat' in all_metrics[0]['model_name'] else all_metrics[1]
add_m    = all_metrics[1] if 'add' in all_metrics[1]['model_name'] else all_metrics[0]

concat_wins = 0
add_wins = 0
ties = 0

for key, label, direction in METRICS:
    c_val = concat_m.get(key, float('nan'))
    a_val = add_m.get(key, float('nan'))
    if np.isnan(c_val) or np.isnan(a_val):
        continue

    diff = c_val - a_val
    if direction == 'higher':
        winner = 'concat' if diff > 0 else ('add' if diff < 0 else 'tie')
    else:
        winner = 'concat' if diff < 0 else ('add' if diff > 0 else 'tie')

    arrow = '→' if winner == 'concat' else ('←' if winner == 'add' else '=')
    print(f"  {label:<25} concat={c_val:.4f}  add={a_val:.4f}  Δ={diff:+.4f}  {arrow} {winner}")

    if winner == 'concat':
        concat_wins += 1
    elif winner == 'add':
        add_wins += 1
    else:
        ties += 1

print(f"\n{'='*60}")
print(f"  Concat wins: {concat_wins}  |  Add wins: {add_wins}  |  Ties: {ties}")
if concat_wins > add_wins:
    print(f"  >>> WINNER: CONCAT (use for all subsequent V2 experiments) <<<")
elif add_wins > concat_wins:
    print(f"  >>> WINNER: ADD (use for all subsequent V2 experiments) <<<")
else:
    print(f"  >>> TIE — pick concat (simpler, standard in literature)")
print(f"{'='*60}")

# ============================================================
# Save CSV
# ============================================================

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
        'Lesion_Precision':f"{m['Lesion_Precision_mean']:.4f}",
        'Overall_Lesion_Recall': f"{m['Overall_lesion_recall']:.4f}",
        'Small_case_ET_Dice': f"{m.get('Small_case_ET_Dice_mean', 0):.4f}",
    })

pd.DataFrame(rows).to_csv('v2_edge_results.csv', index=False)
print('\nSaved: v2_edge_results.csv')
print('Done.')
