"""BraTS2020 evaluation utilities.

Import metric and visualization modules only when their public functions are
requested. Lesion evaluation should not need optional notebook or plotting
packages merely to import :mod:`evaluation.wt_lesion_stratified`.
"""

from importlib import import_module


_EXPORT_MODULES = {
    # Original evaluator
    "compute_metrics": "evaluator",
    "metric": "evaluator",
    "plot_confusion_matrix": "evaluator",
    "compute_scores_per_classes": "evaluator",
    "compute_scores_per_classes_mean": "evaluator",
    "compute_results": "evaluator",
    "print_metrics_table": "evaluator",
    # Optional visualization utilities
    "Image3dToGIF3d": "visualization",
    "ShowResult": "visualization",
    "tumour_graphics": "visualization",
    "generate_3d_plotly": "visualization",
    "merging_two_gif": "visualization",
    "get_all_csv_file": "visualization",
    # Advanced metrics
    "per_class_recall_precision": "advanced_metrics",
    "hd95_single": "advanced_metrics",
    "nsd_single": "advanced_metrics",
    "compute_hd95_all": "advanced_metrics",
    "lesion_wise_detection": "advanced_metrics",
    "compute_lesion_wise_all": "advanced_metrics",
    "compute_small_case_dice": "advanced_metrics",
    "boundary_overlay": "advanced_metrics",
    "save_boundary_comparison": "advanced_metrics",
    "compute_all_advanced_metrics": "advanced_metrics",
    "print_comparison_table": "advanced_metrics",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
