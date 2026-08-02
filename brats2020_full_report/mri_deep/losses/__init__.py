"""
BraTS2020 Loss Functions Package.

Available losses:
  - DiceLoss, BCEDiceLoss  (from basics) — original baseline loss
  - CELoss, BoundaryLoss, DiceCEBoundaryLoss  (from enhanced) — improved losses

Usage:
    from losses import BCEDiceLoss, DiceCEBoundaryLoss
"""

from losses.basics import DiceLoss, BCEDiceLoss
from losses.enhanced import CELoss, BoundaryLoss, DiceCEBoundaryLoss

__all__ = [
    'DiceLoss', 'BCEDiceLoss',
    'CELoss', 'BoundaryLoss', 'DiceCEBoundaryLoss',
]
