"""Build notebooks/experiment_lambda_results.ipynb — compare lambda_b experiments"""
import json

cells = []

def md(s): cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': [s]})
def code(s): cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None, 'source': [s]})

md("""# lambda_b Tuning Experiment: ResUNet + Dice+CE+Boundary Loss

## Loss Formula
```
L = 1.0 * DiceLoss + 0.5 * CELoss + lambda_b * BoundaryLoss
```
Class weights: WT=1.0, TC=3.0, ET=5.0

## Models Compared
| Model | lambda_b | Checkpoint |
|---|---|---|
| Baseline | — | ResUNet (original BCEDiceLoss) |
| lb01 | 0.1 | ResUNet_Enhanced_lb0.1_model |
| lb03 | 0.3 | ResUNet_Enhanced_lb0.3_model |
| lb05 | 0.5 | ResUNet_Enhanced_lb0.5_model |

## Priority Metrics
- **ET/TC Dice** — main segmentation quality
- **ET/TC HD95** — boundary accuracy
- **Boundary Visualization** — qualitative edge comparison
""")

code("""import os, sys, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, '/root/autodl-tmp/mri_deep')

from models.resunet3d import ResUNet3d
from data.dataset import BratsDataset, get_dataloader
from training.metrics import dice_coef_metric_per_classes, jaccard_coef_metric_per_classes
from training.config import check_exist, config
from evaluation.advanced_metrics import (
    compute_all_advanced_metrics,
    print_comparison_table,
    save_boundary_comparison,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
""")

code("""# ============================================================
# Load Models
# ============================================================

print('Loading models...')

# Baseline ResUNet (BCEDiceLoss)
baseline = ResUNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
res_state = torch.load(check_exist(config.ResUNet_checkpoint_dir), map_location=device)
res_state = {k.replace('out.conv.0.', 'out.conv.'): v for k, v in res_state.items()}
baseline.load_state_dict(res_state)
baseline.eval()
print('  Baseline ResUNet (BCEDiceLoss) loaded')

# Enhanced models with different lambda_b
models = {}
for lb in [0.1, 0.3, 0.5]:
    ckpt_dir = f'/root/autodl-tmp/ResUNet_Enhanced_lb{lb}_model'
    path = check_exist(ckpt_dir)
    if path:
        m = ResUNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models[lb] = m
        print(f'  lb={lb}: loaded from {path}')
    else:
        print(f'  lb={lb}: [WARN] no checkpoint found at {ckpt_dir}')

print(f'\\nLoaded {len(models)} enhanced models + baseline')
""")

code("""# ============================================================
# Create Test Dataloader
# ============================================================
test_dataloader = get_dataloader(
    dataset=BratsDataset, path_to_csv='tumourCSV.csv',
    phase='test', batch_size=1, num_workers=0
)
print(f'Test set: {len(test_dataloader)} batches')
""")

code("""# ============================================================
# Run Advanced Evaluation on All Models
# ============================================================

all_metrics = []

print('='*60)
print('MODEL 1/4: Baseline ResUNet (BCEDiceLoss)')
print('='*60)
baseline_m = compute_all_advanced_metrics(baseline, test_dataloader, model_name='Baseline')
all_metrics.append(baseline_m)

for lb in [0.1, 0.3, 0.5]:
    if lb in models:
        print()
        print('='*60)
        print(f'MODEL: ResUNet + Enhanced Loss (lambda_b={lb})')
        print('='*60)
        m = compute_all_advanced_metrics(models[lb], test_dataloader, model_name=f'lb={lb}')
        all_metrics.append(m)

print()
print('All models evaluated!')
""")

code("""# ============================================================
# Comparison Table — All Metrics
# ============================================================
print_comparison_table(all_metrics)
""")

code("""# ============================================================
# Priority Metric Charts
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
names = [m['model_name'] for m in all_metrics]
colors = ['#607D8B', '#2196F3', '#4CAF50', '#FF9800']
n = len(names)

# ET Dice
ax = axes[0, 0]
vals = [m['ET_Dice_mean'] for m in all_metrics]
bars = ax.bar(names, vals, color=colors[:n])
ax.set_title('ET Dice (higher=better)', fontsize=13, fontweight='bold')
ax.set_ylim(0.6, max(vals)*1.05)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{v:.4f}', ha='center', fontweight='bold')

# TC Dice
ax = axes[0, 1]
vals = [m['TC_Dice_mean'] for m in all_metrics]
bars = ax.bar(names, vals, color=colors[:n])
ax.set_title('TC Dice (higher=better)', fontsize=13, fontweight='bold')
ax.set_ylim(0.6, max(vals)*1.05)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{v:.4f}', ha='center', fontweight='bold')

# ET Recall
ax = axes[0, 2]
vals = [m['ET_Recall_mean'] for m in all_metrics]
bars = ax.bar(names, vals, color=colors[:n])
ax.set_title('ET Recall (higher=better)', fontsize=13, fontweight='bold')
ax.set_ylim(0.6, max(vals)*1.05)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{v:.4f}', ha='center', fontweight='bold')

# ET HD95
ax = axes[1, 0]
vals = [m['ET_HD95_mean'] for m in all_metrics]
bars = ax.bar(names, vals, color=colors[:n])
ax.set_title('ET HD95 (lower=better)', fontsize=13, fontweight='bold')
baseline_val = vals[0] if len(vals)>0 else 10
ax.set_ylim(0, max(vals)*1.15)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, f'{v:.2f}', ha='center', fontweight='bold')
ax.axhline(y=baseline_val, color='gray', linestyle='--', alpha=0.5, label=f'Baseline={baseline_val:.1f}')

# TC HD95
ax = axes[1, 1]
vals = [m['TC_HD95_mean'] for m in all_metrics]
bars = ax.bar(names, vals, color=colors[:n])
ax.set_title('TC HD95 (lower=better)', fontsize=13, fontweight='bold')
baseline_val = vals[0] if len(vals)>0 else 10
ax.set_ylim(0, max(vals)*1.15)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, f'{v:.2f}', ha='center', fontweight='bold')
ax.axhline(y=baseline_val, color='gray', linestyle='--', alpha=0.5, label=f'Baseline={baseline_val:.1f}')

# Lesion Recall
ax = axes[1, 2]
vals = [m['Lesion_Recall_mean'] for m in all_metrics]
bars = ax.bar(names, vals, color=colors[:n])
ax.set_title('Lesion-wise Recall (higher=better)', fontsize=13, fontweight='bold')
ax.set_ylim(0.5, max(vals)*1.05)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('lambda_experiment_metrics.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: lambda_experiment_metrics.png')
""")

code("""# ============================================================
# ET Dice vs ET HD95 Tradeoff
# ============================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
names = [m['model_name'] for m in all_metrics]
colors = ['#607D8B', '#2196F3', '#4CAF50', '#FF9800']
markers = ['s', 'o', 'o', 'o']

for i, m in enumerate(all_metrics):
    ax.scatter(m['ET_HD95_mean'], m['ET_Dice_mean'],
              s=400, c=colors[i], marker=markers[i],
              label=m['model_name'], edgecolors='black', linewidth=1.5, zorder=5)
    ax.annotate(m['model_name'],
               (m['ET_HD95_mean'], m['ET_Dice_mean']),
               textcoords='offset points', xytext=(12, 8),
               fontsize=12, fontweight='bold')

ax.set_xlabel('ET HD95 (mm, lower=better)', fontsize=13)
ax.set_ylabel('ET Dice (higher=better)', fontsize=13)
ax.set_title('Boundary Quality vs Segmentation Accuracy Tradeoff', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower left')
ax.grid(True, alpha=0.3)

# Mark ideal corner
ax.annotate('BEST', xy=(min(m['ET_HD95_mean'] for m in all_metrics)*0.9,
                         max(m['ET_Dice_mean'] for m in all_metrics)*1.02),
           fontsize=11, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('lambda_dice_vs_hd95.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: lambda_dice_vs_hd95.png')
""")

code("""# ============================================================
# Boundary Visualization — All Models on Same Case
# ============================================================

print('Generating boundary comparisons...')

test_iter = iter(test_dataloader)
sample = next(test_iter)
case_id = sample['Id']
print(f'Case: {case_id}')

# Build models dict
models_dict = {'Baseline': baseline}
for lb in [0.1, 0.3, 0.5]:
    if lb in models:
        models_dict[f'lb={lb}'] = models[lb]

sample_data = {
    'image': sample['image'][0].numpy(),
    'mask': sample['mask'][0].numpy()
}

# ET boundary only
save_boundary_comparison(
    models_dict, sample_data,
    'lambda_boundary_ET.png',
    classes=['ET'], slice_idx=None
)

# ET + TC
save_boundary_comparison(
    models_dict, sample_data,
    'lambda_boundary_ET_TC.png',
    classes=['ET', 'TC'], slice_idx=None
)

print('Done!')
print('Saved: lambda_boundary_ET.png')
print('Saved: lambda_boundary_ET_TC.png')
""")

code("""# ============================================================
# Summary — Best lambda_b for Each Metric
# ============================================================

print('='*80)
print('BEST lambda_b PER METRIC')
print('='*80)

# Only consider the 3 enhanced models (skip baseline) for lambda selection
enhanced = [m for m in all_metrics if m['model_name'] != 'Baseline']

for key, direction in [
    ('ET_Dice_mean', 'higher'),
    ('TC_Dice_mean', 'higher'),
    ('ET_HD95_mean', 'lower'),
    ('TC_HD95_mean', 'lower'),
    ('ET_Recall_mean', 'higher'),
    ('ET_Precision_mean', 'higher'),
    ('Lesion_Recall_mean', 'higher'),
]:
    vals = [(m['model_name'], m.get(key, np.nan)) for m in enhanced]
    vals = [(n, v) for n, v in vals if not np.isnan(v)]
    if not vals: continue
    if direction == 'higher':
        best = max(vals, key=lambda x: x[1])
    else:
        best = min(vals, key=lambda x: x[1])
    print(f'  {key:<30} -> {best[0]}  {best[1]:.4f}')

print()
print('='*80)
print('RECOMMENDATION:')
print('='*80)

# Find lambda_b that wins most metrics
win_count = {}
for m in enhanced:
    win_count[m['model_name']] = 0

for key in ['ET_Dice_mean', 'TC_Dice_mean', 'ET_HD95_mean', 'TC_HD95_mean',
            'ET_Recall_mean', 'Lesion_Recall_mean']:
    vals = [(m['model_name'], m.get(key, np.nan)) for m in enhanced]
    vals = [(n, v) for n, v in vals if not np.isnan(v)]
    if not vals: continue
    if 'HD95' in key:
        best = min(vals, key=lambda x: x[1])[0]
    else:
        best = max(vals, key=lambda x: x[1])[0]
    win_count[best] += 1

for name, wins in sorted(win_count.items(), key=lambda x: -x[1]):
    print(f'  {name}: won {wins} / {len(win_count)} key metrics')

print()
best_lambda = max(win_count, key=win_count.get)
print(f'  >>> Recommended lambda_b = {best_lambda} <<<')
print('='*80)
""")

code("""# ============================================================
# Export metrics to CSV for paper
# ============================================================

rows = []
for m in all_metrics:
    rows.append({
        'Model': m['model_name'],
        'ET_Dice': f"{m['ET_Dice_mean']:.4f} +/- {m['ET_Dice_std']:.4f}",
        'TC_Dice': f"{m['TC_Dice_mean']:.4f} +/- {m['TC_Dice_std']:.4f}",
        'WT_Dice': f"{m['WT_Dice_mean']:.4f} +/- {m['WT_Dice_std']:.4f}",
        'ET_Recall': f"{m['ET_Recall_mean']:.4f}",
        'ET_Precision': f"{m['ET_Precision_mean']:.4f}",
        'ET_HD95': f"{m['ET_HD95_mean']:.2f} +/- {m['ET_HD95_std']:.2f}",
        'TC_HD95': f"{m['TC_HD95_mean']:.2f} +/- {m['TC_HD95_std']:.2f}",
        'Lesion_Recall': f"{m['Lesion_Recall_mean']:.4f}",
        'Small_case_Dice': f"{m.get('Small_case_ET_Dice_mean', 0):.4f}",
    })

df = pd.DataFrame(rows)
df.to_csv('lambda_experiment_results.csv', index=False)
print('Saved: lambda_experiment_results.csv')
df
""")

nb = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.0'}
    },
    'cells': cells
}

with open('notebooks/experiment_lambda_results.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Created: notebooks/experiment_lambda_results.ipynb ({len(cells)} cells)')
