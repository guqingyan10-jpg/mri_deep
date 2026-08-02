"""
BraTS2020 Evaluation Package.

Contains:
  - compute_metrics              — TP/FP/TN/FN pixel-level metrics
  - metric                        — accuracy, precision, recall, f1
  - plot_confusion_matrix         — confusion matrix visualization
  - compute_scores_per_classes    — per-class Dice/IoU
  - compute_scores_per_classes_mean — mean per-class Dice/IoU
  - compute_results               — prediction collection for visualization
  - print_metrics_table           — formatted metrics output

Visualization:
  - Image3dToGIF3d               — 3D GIF generation from MRI volumes
  - ShowResult                   — ground truth vs prediction overlay
  - tumour_graphics              — interactive slice viewer
  - generate_3d_plotly           — 3D Plotly scatter visualization
  - merging_two_gif              — side-by-side GIF merge

Usage:
    from evaluation.evaluator import compute_metrics, compute_scores_per_classes_mean
    from evaluation.visualization import ShowResult, Image3dToGIF3d
"""

from evaluation.evaluator import (
    compute_metrics,
    metric,
    plot_confusion_matrix,
    compute_scores_per_classes,
    compute_scores_per_classes_mean,
    compute_results,
    print_metrics_table,
)

from evaluation.visualization import (
    Image3dToGIF3d,
    ShowResult,
    tumour_graphics,
    generate_3d_plotly,
    merging_two_gif,
    get_all_csv_file,
)

from evaluation.advanced_metrics import (
    per_class_recall_precision,
    hd95_single,
    compute_hd95_all,
    lesion_wise_detection,
    compute_lesion_wise_all,
    compute_small_case_dice,
    boundary_overlay,
    save_boundary_comparison,
    compute_all_advanced_metrics,
    print_comparison_table,
)

__all__ = [
    # Evaluator (original)
    'compute_metrics', 'metric', 'plot_confusion_matrix',
    'compute_scores_per_classes', 'compute_scores_per_classes_mean',
    'compute_results', 'print_metrics_table',
    # Visualization (original)
    'Image3dToGIF3d', 'ShowResult', 'tumour_graphics',
    'generate_3d_plotly', 'merging_two_gif', 'get_all_csv_file',
    # Advanced metrics (new)
    'per_class_recall_precision', 'hd95_single', 'compute_hd95_all',
    'lesion_wise_detection', 'compute_lesion_wise_all',
    'compute_small_case_dice',
    'boundary_overlay', 'save_boundary_comparison',
    'compute_all_advanced_metrics', 'print_comparison_table',
]
