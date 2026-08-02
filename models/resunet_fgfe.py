"""
=============================================================================
ResUNet3d + FGFE (Frequency Guidance Feature Enhancement)
=============================================================================
Based on Yao et al., BraTS-UMamba, MICCAI 2025.

Architecture:
  ResUNet encoder (unchanged)
  + ResUpFGFE decoder blocks (replaces ResUp)
    Each block: Upsample → Cat → ResBlock → FGFE module

FGFE module:
  Laplacian pyramid: F_s → F_h (high-freq) + F_l (low-freq)
  Cross-attention:   Q_h(F_h), Q_l(F_l) attend to K,V(F_s)
  Residual:          F_s + projection(concat(attn_h, attn_l))

This is a CLEAN ablation:
  - Loss: BCEDiceLoss (same as baseline)
  - Weights: no class weights (same as baseline)
  - Only change: ResUNet + FGFE decoder

Usage:
    from models.resunet_fgfe import ResUNetFGFE
    model = ResUNetFGFE(in_channels=4, n_classes=3, n_channels=24)

Author: Generated for ResUNet enhancement project
Date:   2026-08-02
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resunet3d import ResBlock, ResDown, FirstLayer, Out
from models.fgfe_module import ResUpFGFE


class ResUNetFGFE(nn.Module):
    """
    ResUNet3d + FGFE modules at each decoder stage.

    Same encoder as ResUNet3d. Decoder uses ResUpFGFE which applies
    frequency-guided feature enhancement after each upsampling block.

    Param difference vs baseline ResUNet: ~+0.5M (from FGFE mapping layers)
    """

    def __init__(self, in_channels=4, n_classes=3, n_channels=24):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        # Encoder (identical to ResUNet3d)
        self.conv = FirstLayer(in_channels, n_channels)
        self.enc1 = ResDown(n_channels, 2 * n_channels)
        self.enc2 = ResDown(2 * n_channels, 4 * n_channels)
        self.enc3 = ResDown(4 * n_channels, 8 * n_channels)
        self.enc4 = ResDown(8 * n_channels, 8 * n_channels)

        # Decoder with FGFE
        self.dec1 = ResUpFGFE(16 * n_channels, 4 * n_channels)
        self.dec2 = ResUpFGFE(8 * n_channels, 2 * n_channels)
        self.dec3 = ResUpFGFE(4 * n_channels, n_channels)
        self.dec4 = ResUpFGFE(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        # Decoder with FGFE
        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)

        return mask
