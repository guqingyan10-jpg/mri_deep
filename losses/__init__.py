"""
BraTS2020 Loss Functions Package.

Available losses:
  - DiceLoss, BCEDiceLoss          (from basics) — original baseline loss
  - CELoss, BoundaryLoss, DiceCEBoundaryLoss (from enhanced) — V1 losses
  - CCLevelDiceLoss, BCEDiceCCLoss (from enhanced) — SLA-FB Step 2 losses

Usage:
    from losses import BCEDiceLoss, BCEDiceCCLoss, CCLevelDiceLoss
"""

from losses.basics import DiceLoss, BCEDiceLoss
from losses.enhanced import (CELoss, BoundaryLoss, DiceCEBoundaryLoss,
                             CCLevelDiceLoss, BCEDiceCCLoss, BCECCDiceLoss,
                             PMDiceLoss, BCEDicePMLoss, BCEPMDiceLoss,
                             BCEDiceCCPMLoss, BCEDiceWithBoundaryLoss)

__all__ = [
    'DiceLoss', 'BCEDiceLoss',
    'CELoss', 'BoundaryLoss', 'DiceCEBoundaryLoss',
    'CCLevelDiceLoss', 'BCEDiceCCLoss', 'BCECCDiceLoss',
    'PMDiceLoss', 'BCEDicePMLoss', 'BCEPMDiceLoss', 'BCEDiceCCPMLoss',
    'BCEDiceWithBoundaryLoss',
]
