"""
=============================================================================
Key Model Comparison — 8 selected models
=============================================================================
Evaluates only the eight models relevant to the current comparison:

  1. Baseline (BCEDice)                     — ResUNet3d
  2. Edge (Laplacian, concat)               — ResUNetEdge, multi-scale edge
  3. HF Boundary+ (Laplacian, w=0.3)        — ResUNetHFBoundary, single-scale add
  4. HF Concat Boundary (Laplacian, w=0.3)  — ResUNetHFConcatBoundary, final combo
  5. HF Concat Boundary (Laplacian, w=0.2)  — ResUNetHFConcatBoundary, final combo
  6. HF Concat Boundary (Laplacian, w=0.15) — ResUNetHFConcatBoundary, final combo
  7. HF Concat Boundary (Laplacian, w=0.1)  — ResUNetHFConcatBoundary, final combo
  8. HF Concat Boundary (Laplacian, w=0.05) — ResUNetHFConcatBoundary, final combo

Focuses on the four primary indicators (Macro Dice / ET Dice / ET HD95 /
Small-case ET Dice) and emits:
  - terminal + markdown core-metric comparison table
  - key_comparison_results.json
  - key_composite_rank.png  (composite-score ranking bar chart)
  - key_radar.png           (single radar, all eight models overlaid)

Reuses the registry and evaluation loop from eval_all_experiments.py so the
metrics and checkpoint handling stay identical to the full pipeline.

Usage:
    python scripts/eval_key_comparison.py              # full eval on test set
    python scripts/eval_key_comparison.py --no-timing  # skip inference timing
    python scripts/eval_key_comparison.py --no-cache   # force full re-evaluation
    python scripts/eval_key_comparison.py --seed 123   # seed-matched comparison models

Results are cached per checkpoint (key_comparison_cache.json): re-running only
re-evaluates models whose checkpoint changed and reuses the rest. Change the
threshold/test CSV or want a timing refresh? Pass --no-cache.

Author: ResUNet Enhancement Project
=============================================================================
"""

import os, sys, json, argparse, importlib.util

import numpy as np

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

# --- Reuse eval_all_experiments.py as a module (scripts/ is not a package) ---
_EA_PATH = os.path.join(PROJ_ROOT, 'scripts', 'eval_all_experiments.py')
_spec = importlib.util.spec_from_file_location('eval_all', _EA_PATH)
eval_all = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_all)

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.dataset import BratsDataset, get_dataloader
from evaluation.visualize_report import (
    PRIMARY_INDICATORS,
    _normalize_indicators,
    plot_composite_rank_barchart,
)

# --- The models to evaluate (must match labels in eval_all.EXPERIMENTS) ---
KEY_LABELS = [
    'Baseline (BCEDice)',
    'Edge (Laplacian, concat)',
    'HF Boundary+ (Laplacian, w=0.3)',
    'HF Concat Boundary (Laplacian, w=0.3)',
    'HF Concat Boundary (Laplacian, w=0.2)',
    'HF Concat Boundary (Laplacian, w=0.15)',
    'HF Concat Boundary (Laplacian, w=0.1)',
    'HF Concat Boundary (Laplacian, w=0.05)',
]

# Fixed colors, aligned with KEY_LABELS order.
KEY_COLORS = ['#2c3e50', '#2ecc71', '#3498db', '#e74c3c', '#1abc9c',
              '#9b59b6', '#e67e22', '#e84393']

# Main-model comparison (seed55): the five variants the primary experiment
# actually trains. Labels must match eval_all.EXPERIMENTS exactly.
MAIN_LABELS = [
    'Baseline (BCEDice)',
    'Edge (Laplacian, concat)',
    'HF Concat Boundary (Laplacian, w=0.1)',
    'HF Concat Boundary + Multi-scale V2 (w=0.1, seed55)',
    'HF Concat Boundary (Laplacian, w=0.05)',
]

# Primary indicators: (metric_key, display, direction)  direction: 'high'|'low'
PRIMARY = [
    ('Macro_Dice_mean',         'Macro Dice',      'high'),
    ('ET_Dice_mean',            'ET Dice',         'high'),
    ('Small_case_ET_Dice_mean', 'Small-case Dice', 'high'),
    ('ET_HD95_mean',            'ET HD95',         'low'),
    ('Lesion_F1_mean',          'Lesion F1',       'high'),
]

# Cached metrics keyed by checkpoint path, so unchanged models are reused
# across runs instead of re-running inference on the whole test set.
CACHE_FILE = 'key_comparison_cache.json'


def build_seed_experiments(seed, stability_root):
    """Build existing and gated stability experiments for one random seed."""
    seed_root = os.path.join(stability_root, f'seed{seed}')
    return [
        {
            'dir': os.path.join(seed_root, 'baseline'),
            'model_class': eval_all.ResUNet3d,
            'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
            'label': f'Seed{seed} Baseline (BCEDice)',
            'category': 'Seed Stability',
            'is_baseline': True,
            'key_remap': None,
        },
        {
            'dir': os.path.join(seed_root, 'edge_laplacian_concat'),
            'model_class': eval_all.ResUNetEdge,
            'model_kwargs': {
                'in_channels': 4, 'n_classes': 3, 'n_channels': 24,
                'fusion': 'concat', 'edge_type': 'laplacian',
            },
            'label': f'Seed{seed} Edge (Laplacian, concat)',
            'category': 'Seed Stability',
            'key_remap': 'edge',
        },
        {
            'dir': os.path.join(seed_root, 'edge_laplacian_gated_concat'),
            'model_class': eval_all.ResUNetEdge,
            'model_kwargs': {
                'in_channels': 4, 'n_classes': 3, 'n_channels': 24,
                'fusion': 'gated_concat', 'edge_type': 'laplacian',
            },
            'label': f'Seed{seed} Edge (Laplacian, gated concat)',
            'category': 'Seed Stability',
            'key_remap': 'edge',
        },
        {
            'dir': os.path.join(seed_root, 'hf_concat_boundary_w0.1'),
            'model_class': eval_all.ResUNetHFConcatBoundary,
            'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
            'label': f'Seed{seed} HF Concat Boundary (w=0.1)',
            'category': 'Seed Stability',
            'key_remap': None,
        },
        {
            'dir': os.path.join(
                seed_root, 'hf_concat_boundary_w0.1_multiscale'
            ),
            'model_class': eval_all.ResUNetHFConcatBoundary,
            'model_kwargs': {
                'in_channels': 4, 'n_classes': 3, 'n_channels': 24,
                'multiscale_context': True,
            },
            'label': f'Seed{seed} HF Concat Boundary + Multi-scale (w=0.1)',
            'category': 'Seed Stability',
            'key_remap': None,
        },
        {
            'dir': os.path.join(
                seed_root, 'hf_concat_boundary_w0.1_multiscale_v2'
            ),
            'model_class': eval_all.ResUNetHFConcatBoundary,
            'model_kwargs': {
                'in_channels': 4, 'n_classes': 3, 'n_channels': 24,
                'multiscale_context_v2': True,
            },
            'label': f'Seed{seed} HF Concat Boundary + Multi-scale V2 (w=0.1)',
            'category': 'Seed Stability',
            'key_remap': None,
        },
        {
            'dir': os.path.join(
                seed_root, 'hf_concat_boundary_w0.1_multiscale_v3'
            ),
            'model_class': eval_all.ResUNetHFConcatBoundary,
            'model_kwargs': {
                'in_channels': 4, 'n_classes': 3, 'n_channels': 24,
                'multiscale_context_v3': True,
            },
            'label': f'Seed{seed} HF Concat Boundary + Multi-scale V3 (w=0.1)',
            'category': 'Seed Stability',
            'key_remap': None,
        },
        {
            'dir': os.path.join(seed_root, 'hf_gated_concat_boundary_w0.1'),
            'model_class': eval_all.ResUNetHFConcatBoundary,
            'model_kwargs': {
                'in_channels': 4, 'n_classes': 3, 'n_channels': 24,
                'fusion': 'gated_concat',
            },
            'label': f'Seed{seed} HF Gated Concat Boundary (w=0.1)',
            'category': 'Seed Stability',
            'key_remap': None,
        },
        {
            'dir': os.path.join(seed_root, 'hf_concat_boundary_w0.05'),
            'model_class': eval_all.ResUNetHFConcatBoundary,
            'model_kwargs': {'in_channels': 4, 'n_classes': 3, 'n_channels': 24},
            'label': f'Seed{seed} HF Concat Boundary (w=0.05)',
            'category': 'Seed Stability',
            'key_remap': None,
        },
    ]


def _json_safe(m):
    """Return a JSON-safe copy of a metrics dict (numpy scalars -> native)."""
    clean = {}
    for k, v in m.items():
        if k.startswith('_'):
            continue
        if isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (np.integer,)):
            v = int(v)
        clean[k] = v
    return clean


def _load_cache(path=CACHE_FILE):
    """Load {checkpoint_path: metrics} cache, tolerating a missing/corrupt file."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        print(f"[WARN] Could not read cache {path}; re-evaluating from scratch.")
        return {}


def _save_cache(cache, path=CACHE_FILE):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({ckpt: _json_safe(m) for ckpt, m in cache.items()},
                  f, indent=2, default=str)


def _fmt(v, key):
    """Format a metric value; HD95 gets 2 decimals, Dice 3."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'N/A'
    return f'{v:.2f}' if 'HD95' in key else f'{v:.3f}'


def _best_value(vals, direction):
    clean = [v for v in vals if v is not None and not np.isnan(v)]
    if not clean:
        return None
    return max(clean) if direction == 'high' else min(clean)


def print_core_table(all_metrics):
    """Terminal comparison table over core and lesion indicators."""
    best = [_best_value([m.get(k, float('nan')) for m in all_metrics], d)
            for k, _, d in PRIMARY]

    header = f"{'Model':<40}" + ''.join(f"{disp:>16}" for _, disp, _ in PRIMARY)
    print('\n' + header)
    print('-' * len(header))
    for m in all_metrics:
        cells = [f"{m['model_name']:<40}"]
        for j, (key, _, _) in enumerate(PRIMARY):
            v = m.get(key, float('nan'))
            s = _fmt(v, key)
            if best[j] is not None and not np.isnan(v) and v == best[j]:
                cells.append(f"\033[1m{s:>16}\033[0m")
            else:
                cells.append(f"{s:>16}")
        print(''.join(cells))
    print('Bold = best per column. HD95 lower is better; others higher.\n')


def write_key_table(all_metrics, baseline):
    """Markdown core-indicator table + delta-vs-baseline table."""
    lines = ['# Key Model Comparison — Core Indicators\n']

    lines.append('| Model | Macro Dice | ET Dice | ET HD95↓ | Small-case Dice | Lesion F1 |')
    lines.append('|---|---|---|---|---|---|')
    for m in all_metrics:
        lines.append(
            f"| {m['model_name']} | "
            f"{_fmt(m.get('Macro_Dice_mean', float('nan')), 'Dice')} | "
            f"{_fmt(m.get('ET_Dice_mean', float('nan')), 'Dice')} | "
            f"{_fmt(m.get('ET_HD95_mean', float('nan')), 'HD95')} | "
            f"{_fmt(m.get('Small_case_ET_Dice_mean', float('nan')), 'Dice')} | "
            f"{_fmt(m.get('Lesion_F1_mean', float('nan')), 'Dice')} |")

    if baseline is not None:
        lines.append('\n## Delta vs Baseline\n')
        lines.append('| Model | Δ Macro Dice | Δ ET Dice | Δ ET HD95 | Δ Small-case Dice | Δ Lesion F1 |')
        lines.append('|---|---|---|---|---|---|')
        for m in all_metrics:
            if m is baseline:
                lines.append(f"| {m['model_name']} | (baseline) | — | — | — | — |")
                continue
            lines.append(
                f"| {m['model_name']} | "
                f"{m.get('Macro_Dice_mean', 0) - baseline.get('Macro_Dice_mean', 0):+.4f} | "
                f"{m.get('ET_Dice_mean', 0) - baseline.get('ET_Dice_mean', 0):+.4f} | "
                f"{m.get('ET_HD95_mean', 0) - baseline.get('ET_HD95_mean', 0):+.2f} | "
                f"{m.get('Small_case_ET_Dice_mean', 0) - baseline.get('Small_case_ET_Dice_mean', 0):+.4f} | "
                f"{m.get('Lesion_F1_mean', 0) - baseline.get('Lesion_F1_mean', 0):+.4f} |")

    with open('key_comparison_table.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('Saved: key_comparison_table.md')


def plot_key_radar(all_metrics, save_path='figures/key_radar.png'):
    """
    Single radar chart overlaying all eight models (no category grouping).
    Axes are min-max normalized across these models; ET HD95 is reversed,
    so a larger polygon = better overall.
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    norm = _normalize_indicators(all_metrics)
    keys = [k for k, _, _ in PRIMARY_INDICATORS]
    n = len(keys)
    angles = [a / n * 2 * np.pi for a in range(n)]
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([lab for _, lab, _ in PRIMARY_INDICATORS], fontsize=10)
    ax.set_ylim(0, 1.05)

    for i, m in enumerate(all_metrics):
        vals = [norm[k][i] for k in keys]
        vals += vals[:1]
        color = KEY_COLORS[i % len(KEY_COLORS)]
        ax.plot(angles, vals, color=color, linewidth=2.2,
                label=m['model_name'], alpha=0.95)
        ax.fill(angles, vals, color=color, alpha=0.10)

    ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.12), fontsize=8)
    ax.set_title('Four Primary Indicators — Normalized (larger = better)',
                 fontsize=13, fontweight='bold', pad=24)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Key model comparison (8 models)')
    parser.add_argument('--main', action='store_true',
                        help='Compare the five main-model variants (seed55) instead of the 8 KEY_LABELS')
    parser.add_argument('--seed', type=int, default=None,
                        help='Evaluate seed-matched stability models using four core metrics')
    parser.add_argument('--stability-root', type=str,
                        default='/root/autodl-tmp/stability',
                        help='Root directory containing seed<id> stability checkpoints')
    parser.add_argument('--csv', type=str, default='tumourCSV.csv',
                        help='Path to data CSV')
    parser.add_argument('--threshold', type=float, default=0.33,
                        help='Binarization threshold')
    parser.add_argument('--no-timing', action='store_true',
                        help='Skip inference timing measurement')
    parser.add_argument('--no-cache', action='store_true',
                        help='Ignore cached results and re-evaluate all models')
    parser.add_argument('--figures-dir', type=str, default='figures',
                        help='Output directory for figures (default: figures/)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.seed is not None:
        exps = build_seed_experiments(args.seed, args.stability_root)
    else:
        # Select experiments from the shared registry, keeping the requested order.
        labels = MAIN_LABELS if args.main else KEY_LABELS
        by_label = {e['label']: e for e in eval_all.EXPERIMENTS}
        missing = [l for l in labels if l not in by_label]
        if missing:
            print(f'[ERROR] These labels are missing from the registry:\n  {missing}')
            return
        exps = [by_label[l] for l in labels]

    print('=' * 80)
    title = (f'Key Model Comparison (seed {args.seed}, {len(exps)} models)'
             if args.seed is not None else f'Key Model Comparison ({len(exps)} models)')
    print(title)
    print(f'Device: {device} | Threshold: {args.threshold}')
    for spec in exps:
        print(f"  - {spec['label']}")
    print('=' * 80)

    print('\nLoading test dataloader...')
    test_loader = get_dataloader(BratsDataset, args.csv, phase='test')
    print(f'Test set: {len(test_loader.dataset)} cases')

    # ── Reuse cached metrics for checkpoints already evaluated ──────────
    cache = {} if args.no_cache else _load_cache()
    to_compute = []
    cached_by_label = {}
    for spec in exps:
        ckpt, _epoch = eval_all.find_checkpoint(spec['dir'])
        if ckpt is not None and ckpt in cache:
            cached_by_label[spec['label']] = cache[ckpt]
        else:
            to_compute.append(spec)

    if to_compute:
        print(f'\nReusing cached: {len(cached_by_label)}/{len(exps)} models. '
              f'Evaluating {len(to_compute)} newly.')
    else:
        print(f'\nAll {len(exps)} models cached — no inference needed.')

    new_metrics, all_histories, errors = eval_all.evaluate_experiments(
        to_compute, test_loader, device,
        threshold=args.threshold,
        measure_timing=not args.no_timing,
    )

    # Update cache with freshly computed metrics.
    new_by_label = {m['model_name']: m for m in new_metrics}
    for m in new_metrics:
        cache[m['checkpoint_path']] = m
    if not args.no_cache:
        _save_cache(cache)

    # Merge cached + freshly computed, preserving requested experiment order.
    all_metrics = []
    for l in [spec['label'] for spec in exps]:
        if l in cached_by_label:
            all_metrics.append(cached_by_label[l])
        elif l in new_by_label:
            all_metrics.append(new_by_label[l])

    if not all_metrics:
        print('\nNo models evaluated successfully.')
        for lbl, msg in errors:
            print(f'  {lbl}: {msg}')
        return

    baseline = next((m for m in all_metrics if 'Baseline' in m['model_name']), None)

    # Core table (terminal + markdown)
    print_core_table(all_metrics)
    write_key_table(all_metrics, baseline)

    # JSON dump
    json_ready = [_json_safe(m) for m in all_metrics]
    with open('key_comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(json_ready, f, indent=2, default=str)
    print('Saved: key_comparison_results.json')

    # Figures — composite ranking + single radar
    os.makedirs(args.figures_dir, exist_ok=True)
    plot_composite_rank_barchart(
        all_metrics,
        save_path=os.path.join(args.figures_dir, 'key_composite_rank.png'))
    plot_key_radar(
        all_metrics,
        save_path=os.path.join(args.figures_dir, 'key_radar.png'))

    if errors:
        print(f'\n[WARNING] {len(errors)} skipped/error:')
        for lbl, msg in errors:
            print(f'  {lbl}: {msg}')

    print('\nDONE')


if __name__ == '__main__':
    main()
