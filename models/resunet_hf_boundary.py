"""
=============================================================================
ResUNet + HF Boundary Auxiliary Branch
=============================================================================
Strict single-variable change from Exp-0 baseline (ResUNet3d + BCEDiceLoss):

  100% Copy: Encoder, Decoder, main loss, data, lr, scheduler, seed, batch
  ONLY ADD: Fixed Sobel/Laplacian edge extractor, 1x1x1 channel aligner,
            boundary auxiliary prediction head

Architecture:

  MRI (B,4,128^3)
    |
    +--> ResUNet Encoder -> Decoder (dec1..dec4)    <-- 100% baseline copy
    |                        |
    |                  dec4_out (B,24,128^3)
    |                        |
    +--> Sobel/Laplacian (FIXED, no grad)            <-- NEW
    |     hf_raw (B,4,128^3)                         |
    |       |                                         |
    |     Conv3d(4->n_channels, 1x1x1)               <-- NEW: channel align
    |     hf_aligned                                  |
    |       |                                         |
    |     F.interpolate (if size mismatch)            |
    |       |                                         |
    |       +---------- ADD --------------------------+
    |                      |
    |              fused (B,n_channels,128^3)
    |                 |
    |         +-------+-------+
    |         |               |
    |     seg_head        boundary_head              <-- seg: original Out
    |     (Out: 1x1)      (Conv->GN->ReLU->Conv)     <-- boundary: NEW
    |         |               |
    |     seg_out          boundary_out
    |     (B,3,128^3)      (B,3,128^3)

Loss:
    Total = BCEDiceLoss(seg_out, GT) + 0.2 * BCE(boundary_out, boundary_GT)

    boundary_GT = GT XOR eroded(GT)  (extracted on GPU via avg_pool3d)

Usage:
    from models.resunet_hf_boundary import ResUNetHFBoundary
    model = ResUNetHFBoundary(in_channels=4, n_classes=3, n_channels=24,
                               edge_type='sobel')  # 'sobel' or 'laplacian'

Author: Generated for ResUNet enhancement project
Date:   2026-08-07
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resunet3d import ResBlock, ResDown, ResUp, FirstLayer
from models.base_blocks import Out
from models.resunet_edge import SobelEdge3d, LaplacianEdge3d


class ResUNetHFBoundary(nn.Module):
    """
    ResUNet3d with HF Boundary auxiliary branch.

    100% baseline ResUNet3d encoder + decoder.
    Adds: fixed edge extractor, 1x1 channel aligner, boundary prediction head.
    """

    def __init__(self, in_channels=4, n_classes=3, n_channels=24,
                 edge_type='sobel'):
        super().__init__()
        if edge_type not in ('sobel', 'laplacian'):
            raise ValueError(f"edge_type must be 'sobel' or 'laplacian', got {edge_type}")

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.edge_type = edge_type

        # ============================================================
        # Encoder (100% identical to ResUNet3d)
        # ============================================================
        self.conv = FirstLayer(in_channels, n_channels)
        self.enc1 = ResDown(n_channels, 2 * n_channels)
        self.enc2 = ResDown(2 * n_channels, 4 * n_channels)
        self.enc3 = ResDown(4 * n_channels, 8 * n_channels)
        self.enc4 = ResDown(8 * n_channels, 8 * n_channels)

        # ============================================================
        # Decoder (100% identical to ResUNet3d)
        # ============================================================
        self.dec1 = ResUp(16 * n_channels, 4 * n_channels)
        self.dec2 = ResUp(8 * n_channels, 2 * n_channels)
        self.dec3 = ResUp(4 * n_channels, n_channels)
        self.dec4 = ResUp(2 * n_channels, n_channels)

        # ============================================================
        # HF Branch (NEW — only additions)
        # ============================================================

        # Fixed edge extractor (non-trainable)
        if edge_type == 'sobel':
            self.edge_extractor = SobelEdge3d()
        else:
            self.edge_extractor = LaplacianEdge3d()
        # Freeze: this is a fixed dataset transform, not learned
        for p in self.edge_extractor.parameters():
            p.requires_grad = False

        # 1x1x1 Conv for channel alignment: 4 -> n_channels
        self.hf_align = nn.Conv3d(in_channels, n_channels, kernel_size=1, bias=False)

        # ============================================================
        # Prediction Heads (seg_head = original; boundary_head = NEW)
        # ============================================================
        self.seg_head = Out(n_channels, n_classes)
        self.boundary_head = nn.Sequential(
            nn.Conv3d(n_channels, n_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, n_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_channels, n_classes, kernel_size=1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 4, D, H, W) — 4-modal MRI input

        Returns:
            (seg, boundary): tuple of tensors, both (B, 3, D, H, W)
        """
        # ---- Encoder (identical to ResUNet3d) ----
        x1 = self.conv(x)                     # (B, 24, 128^3)
        x2 = self.enc1(x1)                    # (B, 48,  64^3)
        x3 = self.enc2(x2)                    # (B, 96,  32^3)
        x4 = self.enc3(x3)                    # (B, 192, 16^3)
        x5 = self.enc4(x4)                    # (B, 192,  8^3)

        # ---- Decoder (identical to ResUNet3d) ----
        mask = self.dec1(x5, x4)              # (B, 96,  16^3)
        mask = self.dec2(mask, x3)            # (B, 48,  32^3)
        mask = self.dec3(mask, x2)            # (B, 24,  64^3)
        dec4_out = self.dec4(mask, x1)        # (B, 24, 128^3)

        # ---- HF Branch: extract edges from raw MRI ----
        with torch.no_grad():
            hf_raw = self.edge_extractor(x)    # (B, 4, 128^3) — fixed, no grad
        hf_aligned = self.hf_align(hf_raw)     # (B, n_channels, 128^3)

        # Size matching
        if hf_aligned.shape[2:] != dec4_out.shape[2:]:
            hf_aligned = F.interpolate(hf_aligned, size=dec4_out.shape[2:],
                                       mode='trilinear', align_corners=True)

        # ---- Debug shape check ----
        if not hasattr(self, '_debug_printed'):
            print(f"[HF-Boundary] dec4_out: {dec4_out.shape}, "
                  f"hf_raw: {hf_raw.shape}, hf_aligned: {hf_aligned.shape}")
            self._debug_printed = True

        # ---- Fusion ----
        fused = dec4_out + hf_aligned           # (B, n_channels, D, H, W)

        # ---- Dual heads ----
        seg = self.seg_head(fused)              # (B, 3, D, H, W)
        boundary = self.boundary_head(fused)    # (B, 3, D, H, W)

        return seg, boundary
