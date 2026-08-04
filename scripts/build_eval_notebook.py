"""Build notebooks/evaluate_baselines.ipynb programmatically."""
import json

cells = []

def md(source):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': [source]})

def code(source):
    cells.append({'cell_type': 'code', 'metadata': {}, 'outputs': [], 'execution_count': None,
                  'source': [source]})

# ============================================================
md("""# Advanced Evaluation: 4 Baseline Models Comparison

## New Metrics Beyond Original Dice/IoU:
- **ET/TC Recall & Precision** — pixel-level detection quality
- **ET/TC HD95** — Hausdorff Distance at 95th percentile (boundary accuracy)
- **Lesion-wise Recall** — % of individual ET lesions detected
- **Small-case ET Dice** — Dice on bottom 25% ET-volume cases only
- **Boundary Visualization** — GT vs Pred contour overlays

## Usage:
- **Option A**: Run AFTER `MultiModel XAI Brats2020.ipynb` (models already in memory, skip to Cell 3)
- **Option B**: Run standalone — loads model checkpoints from disk (Cell 1-2)
""")

# ============================================================
code("""# ============================================================
# Cell 0: Imports
# ============================================================
import os, gc, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.base_blocks import DoubleConv, Down, Up, Out
from models.unet3d import UNet3d
from models.resunet3d import ResUNet3d
from models.attunet3d import AttUNet3d
from models.nnunet3d import nnUNet3d

from data.dataset import BratsDataset, get_dataloader
from training.metrics import dice_coef_metric_per_classes, jaccard_coef_metric_per_classes
from training.config import check_exist, config

from evaluation.advanced_metrics import (
    compute_all_advanced_metrics,
    print_comparison_table,
    save_boundary_comparison,
    boundary_overlay,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print('All imports OK.')
""")

# ============================================================
code("""# ============================================================
# Cell 1: Load 4 Baseline Models from Checkpoints
# ============================================================
# SKIP this cell if models already loaded from original notebook!

print('Loading 4 baseline models from checkpoints...')

UNet = UNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
UNet.load_state_dict(torch.load(check_exist(config.UNet_checkpoint_dir), map_location=device))
UNet.eval()

ResUNet = ResUNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
res_state = torch.load(check_exist(config.ResUNet_checkpoint_dir), map_location=device)
res_state = {k.replace('out.conv.0.', 'out.conv.'): v for k, v in res_state.items()}
ResUNet.load_state_dict(res_state)
ResUNet.eval()

AttUNet = AttUNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
att_state = torch.load(check_exist(config.Att_checkpoint_dir), map_location=device)
att_state = {k.replace('out.conv.0.', 'out.conv.'): v for k, v in att_state.items()}
AttUNet.load_state_dict(att_state)
AttUNet.eval()

nnUNet = nnUNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
nn_state = torch.load(check_exist(config.nnUNet_checkpoint_dir), map_location=device)
nnUNet.load_state_dict(nn_state)
nnUNet.eval()

print('All 4 models loaded and in eval mode.')
for name, m in [('UNet', UNet), ('ResUNet', ResUNet), ('AttUNet', AttUNet), ('nnUNet', nnUNet)]:
    print(f'  {name}: {sum(p.numel() for p in m.parameters()):,} params')
""")

# ============================================================
code("""# ============================================================
# Cell 2: Create Test Dataloader
# ============================================================
test_dataloader = get_dataloader(
    dataset=BratsDataset,
    path_to_csv='tumourCSV.csv',
    phase='test',
    batch_size=1,
    num_workers=0
)
print(f'Test set: {len(test_dataloader)} batches')
""")

# ============================================================
code("""# ============================================================
# Cell 3: Run Advanced Evaluation on All 4 Models
# ============================================================
# HD95 is computationally expensive — expect ~2-5 min per model

print('='*60)
print('MODEL 1/4: UNet')
print('='*60)
unet_metrics = compute_all_advanced_metrics(UNet, test_dataloader, model_name='UNet')

print()
print('='*60)
print('MODEL 2/4: ResUNet')
print('='*60)
resunet_metrics = compute_all_advanced_metrics(ResUNet, test_dataloader, model_name='ResUNet')

print()
print('='*60)
print('MODEL 3/4: AttUNet')
print('='*60)
attunet_metrics = compute_all_advanced_metrics(AttUNet, test_dataloader, model_name='AttUNet')

print()
print('='*60)
print('MODEL 4/4: nnUNet')
print('='*60)
nnunet_metrics = compute_all_advanced_metrics(nnUNet, test_dataloader, model_name='nnUNet')

print()
print('All 4 models evaluated!')
""")

# ============================================================
code("""# ============================================================
# Cell 4: Main Comparison Table
# ============================================================
all_metrics = [unet_metrics, resunet_metrics, attunet_metrics, nnunet_metrics]
print_comparison_table(all_metrics)
""")

# ============================================================
code("""# ============================================================
# Cell 5: Lesion-wise Recall Detail
# ============================================================
print('='*80)
print('LESION-WISE ET DETECTION ANALYSIS')
print('='*80)

for m in all_metrics:
    print(f"\\n{m['model_name']}:")
    print(f"  Total GT ET lesions:     {m['Total_GT_lesions']}")
    print(f"  Total detected:          {m['Total_detected']}")
    print(f"  Overall Lesion Recall:   {m['Overall_lesion_recall']:.4f}")
    print(f"  Mean Lesion Recall:      {m['Lesion_Recall_mean']:.4f} +/- {m['Lesion_Recall_std']:.4f}")
    print(f"  Mean Lesion Precision:   {m['Lesion_Precision_mean']:.4f} +/- {m['Lesion_Precision_std']:.4f}")

# Bar chart
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
names = [m['model_name'] for m in all_metrics]
colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0']

lr_vals = [m['Lesion_Recall_mean'] for m in all_metrics]
axes[0].bar(names, lr_vals, color=colors)
axes[0].set_title('Lesion-wise Recall (higher=better)', fontsize=13, fontweight='bold')
axes[0].set_ylim(0, 1)
for i, v in enumerate(lr_vals):
    axes[0].text(i, v+0.03, f'{v:.3f}', ha='center', fontweight='bold', fontsize=12)

lp_vals = [m['Lesion_Precision_mean'] for m in all_metrics]
axes[1].bar(names, lp_vals, color=colors)
axes[1].set_title('Lesion-wise Precision (higher=better)', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, 1)
for i, v in enumerate(lp_vals):
    axes[1].text(i, v+0.03, f'{v:.3f}', ha='center', fontweight='bold', fontsize=12)

sd_vals = [m.get('Small_case_ET_Dice_mean', 0) for m in all_metrics]
axes[2].bar(names, sd_vals, color=colors)
axes[2].set_title(f"Small-case ET Dice\\n(bottom 25%, <{all_metrics[0].get('Small_case_threshold',0):.0f} vox)", fontsize=13, fontweight='bold')
axes[2].set_ylim(0, 1)
for i, v in enumerate(sd_vals):
    axes[2].text(i, v+0.03, f'{v:.3f}', ha='center', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('lesion_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: lesion_metrics_comparison.png')
""")

# ============================================================
code("""# ============================================================
# Cell 6: HD95 — Boundary Quality
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
names = [m['model_name'] for m in all_metrics]
colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0']

et_hd = [m['ET_HD95_mean'] for m in all_metrics]
axes[0].bar(names, et_hd, color=colors)
axes[0].set_title('ET HD95 (mm, lower=better)', fontsize=13, fontweight='bold')
for i, v in enumerate(et_hd):
    axes[0].text(i, v+0.1, f'{v:.2f}', ha='center', fontweight='bold', fontsize=12)

tc_hd = [m['TC_HD95_mean'] for m in all_metrics]
axes[1].bar(names, tc_hd, color=colors)
axes[1].set_title('TC HD95 (mm, lower=better)', fontsize=13, fontweight='bold')
for i, v in enumerate(tc_hd):
    axes[1].text(i, v+0.1, f'{v:.2f}', ha='center', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('hd95_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: hd95_comparison.png')
""")

# ============================================================
code("""# ============================================================
# Cell 7: ET Detection — Recall vs Precision
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0']
markers = ['o', 's', '^', 'D']

for i, m in enumerate(all_metrics):
    ax.scatter(m['ET_Precision_mean'], m['ET_Recall_mean'],
              s=400, c=colors[i], marker=markers[i],
              label=m['model_name'], edgecolors='black', linewidth=1.5, zorder=5)
    ax.annotate(m['model_name'],
               (m['ET_Precision_mean'], m['ET_Recall_mean']),
               textcoords='offset points', xytext=(14, 10),
               fontsize=13, fontweight='bold')

# F1 contours
for f1 in [0.3, 0.5, 0.7, 0.8, 0.9]:
    x = np.linspace(0.01, 1, 100)
    y = f1 * x / (2 * x - f1)
    valid = (y > 0) & (y < 1)
    ax.plot(x[valid], y[valid], 'gray', alpha=0.2, linestyle='--')
    mid = len(x) // 2
    if valid[mid]:
        ax.annotate(f'F1={f1}', (x[mid], y[mid]), alpha=0.35, fontsize=8)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('ET Precision (pixel-level)', fontsize=13)
ax.set_ylabel('ET Recall (pixel-level)', fontsize=13)
ax.set_title('ET Detection: Precision vs Recall', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower left')
ax.grid(True, alpha=0.3)
ax.annotate('IDEAL', (0.96, 0.96), fontsize=11, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('et_recall_precision.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: et_recall_precision.png')
""")

# ============================================================
code("""# ============================================================
# Cell 8: Boundary Visualization — 4 Models on First Test Case
# ============================================================
print('Generating boundary overlay visualizations...')

test_iter = iter(test_dataloader)
sample = next(test_iter)
print(f"Case: {sample['Id']}")

models_dict = {'UNet': UNet, 'ResUNet': ResUNet, 'AttUNet': AttUNet, 'nnUNet': nnUNet}

sample_data = {
    'image': sample['image'][0].numpy(),
    'mask': sample['mask'][0].numpy()
}

# ET boundary
save_boundary_comparison(models_dict, sample_data,
                         'boundary_ET_comparison.png',
                         classes=['ET'], slice_idx=None)

# ET + TC
save_boundary_comparison(models_dict, sample_data,
                         'boundary_ET_TC_comparison.png',
                         classes=['ET', 'TC'], slice_idx=None)

print('Done!')
""")

# ============================================================
code("""# ============================================================
# Cell 9: Summary — Best Model Per Metric
# ============================================================
print('='*80)
print('BEST MODEL PER METRIC')
print('='*80)

higher_better = ['ET_Dice_mean', 'TC_Dice_mean', 'WT_Dice_mean',
                 'ET_Recall_mean', 'ET_Precision_mean',
                 'Lesion_Recall_mean', 'Lesion_Precision_mean',
                 'Small_case_ET_Dice_mean']
lower_better  = ['ET_HD95_mean', 'TC_HD95_mean']

for key in higher_better + lower_better:
    vals = [(m['model_name'], m.get(key, np.nan)) for m in all_metrics]
    vals = [(n, v) for n, v in vals if not np.isnan(v)]
    if not vals:
        continue
    if key in higher_better:
        best = max(vals, key=lambda x: x[1])
        direction = '(higher=better)'
    else:
        best = min(vals, key=lambda x: x[1])
        direction = '(lower=better)'
    print(f'  {key:<35} -> {best[0]:>10}  {best[1]:.4f}  {direction}')

print()
print('='*80)
print('Evaluation complete!')
print('='*80)
""")

# ============================================================
# Write notebook JSON
# ============================================================
nb = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10.0'}
    },
    'cells': cells
}

with open('notebooks/evaluate_baselines.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Created: notebooks/evaluate_baselines.ipynb ({len(cells)} cells)')
