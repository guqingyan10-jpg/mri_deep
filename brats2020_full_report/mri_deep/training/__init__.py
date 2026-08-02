"""
BraTS2020 Training Infrastructure Package.

Contains:
  - Trainer            — training loop with checkpointing, early stopping
  - GlobalConfig       — path and hyperparameter configuration
  - seed_everything, check_exist  — utilities
  - Meter, dice_coef_metric, jaccard_coef_metric, etc. — metrics

Usage:
    from training.trainer import Trainer
    from training.config import GlobalConfig, seed_everything, config, check_exist
    from training.metrics import Meter, dice_coef_metric, jaccard_coef_metric
"""

from training.trainer import Trainer
from training.config import GlobalConfig, seed_everything, config, check_exist
from training.metrics import (
    Meter,
    dice_coef_metric,
    jaccard_coef_metric,
    dice_coef_metric_per_classes,
    jaccard_coef_metric_per_classes,
)

__all__ = [
    'Trainer',
    'GlobalConfig', 'seed_everything', 'config', 'check_exist',
    'Meter',
    'dice_coef_metric', 'jaccard_coef_metric',
    'dice_coef_metric_per_classes', 'jaccard_coef_metric_per_classes',
]
