"""
BraTS2020 Data Pipeline Package.

Contains:
  - BratsDataset                  — BraTS2020 3D MRI dataset loader
  - BratsDatasetWithFGSampling    — foreground-aware patch sampling variant
  - get_dataloader                — train/valid/test dataloader factory
  - get_augmentations             — albumentations augmentation pipeline
  - ForegroundAwarePatchSampler   — STSNet-inspired 4-strategy 3D sampler

Usage:
    from data.dataset import BratsDataset, BratsDatasetWithFGSampling
    from data import get_dataloader
    from data.foreground_sampler import ForegroundAwarePatchSampler
"""

from data.dataset import BratsDataset, BratsDatasetWithFGSampling, get_dataloader, get_augmentations
from data.foreground_sampler import ForegroundAwarePatchSampler

__all__ = [
    'BratsDataset', 'BratsDatasetWithFGSampling',
    'get_dataloader', 'get_augmentations',
    'ForegroundAwarePatchSampler',
]
