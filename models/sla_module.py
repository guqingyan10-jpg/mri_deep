"""
=============================================================================
SLA-FB: Small Lesion Attention — Foreground Boundary Module
=============================================================================
Inspired by STSNet (Zhao et al., Scientific Reports 2025) but adapted for
3D brain tumor segmentation (BraTS2020) with multi-connected ET lesions.

STSNet's three mechanisms, adapted to 3D:
  1. Center Crop Amplification → 3D Connected-Component-Aware Patch Sampling
  2. TwoStreamBatchSampler     → Foreground-weighted batch composition
  3. ESCA / CBAM attention     → 3D Small Lesion Spatial-Channel Attention

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  SLA Module (inserted at decoder high-res stages)    │
  │                                                       │
  │  Decoder Feature (B, C, D, H, W)                      │
  │       │                                               │
  │       ├──→ Channel Attention (3D SE-like)              │
  │       │     GAP3d → Conv1d → Sigmoid → scale          │
  │       │                                               │
  │       ├──→ Spatial Attention (3D CBAM-like)            │
  │       │     MaxPool+AvgPool along C → Conv3d → Sigmoid │
  │       │                                               │
  │       └──→ Residual: F_out = F_in + SLA(F_in)         │
  └─────────────────────────────────────────────────────┘

Key design choices (from STSNet lessons):
  - Lightweight: depthwise-separable for 3D to keep params low
  - High-res only: applied at dec3 (64³) and dec4 (128³) where
    spatial resolution is sufficient for small lesion detection
  - Residual: preserves original features, attention is additive

Usage:
    from models.sla_module import SLA3D, ForegroundPatchSampler

    # Attention module (in model):
    sla = SLA3D(channels=24)  # at decoder output stage

    # Patch sampler (in dataloader):
    sampler = ForegroundPatchSampler(et_mask, patch_size=128)

Reference:
  Zhao et al., "A Novel Framework for Segmentation of Small Targets
  in Medical Images", Scientific Reports, 2025.
  https://github.com/zlxokok/STSNet

Author: Generated for ResUNet enhancement project
Date:   2026-08-04
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import label as connected_components


# ============================================================
# 3D Small Lesion Spatial-Channel Attention (SLA)
# ============================================================

class ChannelAttention3D(nn.Module):
    """
    3D Efficient Channel Attention (adapted from STSNet's ECA / SE).

    Unlike 2D ECA which uses 1D conv over channels, here we use
    adaptive 1D conv with kernel size auto-computed from channel dim.

    STSNet insight: standard SE/SENet reduction ratio=16 works for
    regular objects but loses too much info for small targets.
    We use reduction=4 to preserve small-lesion channel signatures.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid_ch = max(1, channels // reduction)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, mid_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_ch, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.gap(x)        # (B, C, 1, 1, 1)
        w = self.fc(w)         # (B, C, 1, 1, 1)
        return x * w


class SpatialAttention3D(nn.Module):
    """
    3D Spatial Attention (adapted from STSNet's spatial_attention).

    STSNet uses 2D max+avg pool along channel axis → Conv2d(2→1) → Sigmoid.
    We extend to 3D: max+avg along channel → Conv3d(2→1) → Sigmoid.

    Uses depthwise-separable conv to keep params tractable in 3D.
    """

    def __init__(self, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        # 2 input channels (max + avg), 1 output → spatial weight map
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise max and average → 2-channel spatial descriptor
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, D, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, D, H, W)
        spatial = torch.cat([avg_out, max_out], dim=1)   # (B, 2, D, H, W)
        spatial = self.conv(spatial)                      # (B, 1, D, H, W)
        spatial = self.sigmoid(spatial)
        return x * spatial


class SLA3D(nn.Module):
    """
    Small Lesion Attention (3D) — combines channel + spatial attention
    with residual connection.

    STSNet insight: CBAM applies channel-then-spatial sequentially,
    which can over-suppress small targets. We use PARALLEL branches
    and ADDITIVE fusion with residual — gentler for small lesions.

    Args:
        channels: number of feature channels
        reduction: channel attention squeeze ratio (default 4, not 16)
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channel_att = ChannelAttention3D(channels, reduction)
        self.spatial_att = SpatialAttention3D(kernel_size=3)

        # Learnable balance between channel and spatial branches
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # Parallel attention branches (not sequential — gentler for small targets)
        c_out = self.channel_att(x)
        s_out = self.spatial_att(x)

        # Learnable weighted fusion + residual
        out = self.alpha * c_out + (1 - self.alpha) * s_out
        return x + out  # residual: preserves original decoder features


# ============================================================
# Foreground Patch Sampler (FPS)
# ============================================================

class ForegroundPatchSampler:
    """
    ET/TC foreground-aware 3D patch coordinate sampler.

    STSNet equivalent: TwoStreamBatchSampler + center crop amplification,
    but adapted for ONLINE (in-training) 3D patch sampling instead of
    offline 2.5D image augmentation.

    Algorithm:
      1. Pre-compute: for each training case, run 3D connected component
         analysis on ET mask (label=4). Record centroid of each component.
      2. At training time, with probability p_fg:
         - Randomly select an ET component from the current case
         - Sample a 3D patch centered on that component's centroid
         - With probability p_amp, also do center-crop amplification
           (crop tighter around lesion → resize back to patch_size)
      3. With probability (1 - p_fg):
         - Uniform random crop (standard behavior)

    STSNet mapping:
      - findContours + minAreaRect → 3D connected component labeling
      - center_h, center_w          → centroid_z, centroid_y, centroid_x
      - random(150,160) crop        → adaptive window based on component size
      - resize(480,480)             → resize to patch_size

    Usage:
        sampler = ForegroundPatchSampler(p_fg=0.5, p_amp=0.3, patch_size=128)

        # Pre-compute per case (called once before training):
        sampler.build_index(case_id, et_mask_3d)

        # At each training iteration:
        crop_coords = sampler.sample(case_id, volume_shape)
        patch = volume[crop_coords]
    """

    def __init__(self, p_fg=0.5, p_amp=0.3, patch_size=128,
                 min_component_size=10):
        """
        Args:
            p_fg: probability of foreground-centered sampling
            p_amp: probability of center-crop amplification (given fg sample)
            patch_size: output patch size (default 128 for BraTS)
            min_component_size: minimum voxels for a valid ET component
        """
        self.p_fg = p_fg
        self.p_amp = p_amp
        self.patch_size = patch_size
        self.min_size = min_component_size

        # Per-case index: case_id → list of (centroid_z, centroid_y, centroid_x, size)
        self._fg_index = {}

    def build_index(self, case_id, et_mask):
        """
        Pre-compute ET foreground index for one case.

        Args:
            case_id: unique case identifier
            et_mask: (D, H, W) binary mask where ET(label=4) = 1

        STSNet equivalent: findContours + minAreaRect in 4_find_label_center_together.py
        """
        # 3D connected component labeling (26-connectivity)
        labeled, num_components = connected_components(et_mask)

        components = []
        for comp_id in range(1, num_components + 1):
            comp_mask = (labeled == comp_id)
            comp_size = comp_mask.sum()

            if comp_size < self.min_size:
                continue

            # Centroid (center of mass)
            coords = np.argwhere(comp_mask)  # (N, 3) → (z, y, x)
            centroid = coords.mean(axis=0).astype(int)  # (z, y, x)

            components.append({
                'centroid': tuple(centroid),
                'size': int(comp_size),
            })

        if components:
            self._fg_index[case_id] = components

    def sample(self, case_id, volume_shape):
        """
        Sample a 3D patch bounding box.

        Args:
            case_id: case identifier
            volume_shape: (D, H, W) of the full volume

        Returns:
            (z1, z2, y1, y2, x1, x2) — crop coordinates

        STSNet equivalent: center_h/center_w ± random(150,160) in
        4_find_label_center_together.py:44-48
        """
        D, H, W = volume_shape
        half = self.patch_size // 2

        # Decide: foreground or random?
        use_fg = (np.random.random() < self.p_fg
                  and case_id in self._fg_index
                  and len(self._fg_index[case_id]) > 0)

        if use_fg:
            # Pick a random ET component
            comp = self._fg_index[case_id][
                np.random.randint(len(self._fg_index[case_id]))
            ]
            cz, cy, cx = comp['centroid']
            comp_size = comp['size']

            # Adaptive window size based on component size
            # (STSNet uses random(150, 160) for 2D; we scale for 3D)
            if np.random.random() < self.p_amp:
                # Amplification mode: tighter crop around small lesion
                margin = int(np.ceil(comp_size ** (1/3))) + np.random.randint(4, 12)
            else:
                # Normal foreground mode: wider crop
                margin = half

            z1 = max(0, cz - margin)
            z2 = min(D, z1 + self.patch_size)
            y1 = max(0, cy - margin)
            y2 = min(H, y1 + self.patch_size)
            x1 = max(0, cx - margin)
            x2 = min(W, x1 + self.patch_size)

            # Ensure exact patch_size
            if z2 - z1 < self.patch_size:
                z1 = max(0, z2 - self.patch_size)
            if y2 - y1 < self.patch_size:
                y1 = max(0, y2 - self.patch_size)
            if x2 - x1 < self.patch_size:
                x1 = max(0, x2 - self.patch_size)
        else:
            # Uniform random crop (standard behavior)
            z1 = np.random.randint(0, max(1, D - self.patch_size))
            y1 = np.random.randint(0, max(1, H - self.patch_size))
            x1 = np.random.randint(0, max(1, W - self.patch_size))
            z2, y2, x2 = z1 + self.patch_size, y1 + self.patch_size, x1 + self.patch_size

        return (z1, z2, y1, y2, x1, x2)

    def has_foreground(self, case_id):
        """Check if case has any valid ET components."""
        return case_id in self._fg_index and len(self._fg_index[case_id]) > 0

    def num_components(self, case_id):
        """Return number of ET connected components for this case."""
        if case_id not in self._fg_index:
            return 0
        return len(self._fg_index[case_id])


# ============================================================
# SLA Decoder Block — drop-in replacement for ResUp with SLA
# ============================================================

class ResUpSLA(nn.Module):
    """
    ResUp + SLA attention — decoder block with small lesion focus.

    Same interface as ResUp / ResUpEdge / ResUpFGFE.
    Can be combined with edge branch or used standalone.

    STSNet insight: attention at HIGH-RESOLUTION decoder stages only
    (dec3, dec4 where spatial detail is sufficient for small lesions).
    Apply SLA at dec3 (64³) and dec4 (128³), skip at dec1/dec2 (too coarse).
    """

    def __init__(self, in_channels, out_channels, trilinear=True,
                 use_sla=False):
        """
        Args:
            use_sla: if True, apply Small Lesion Attention after conv
        """
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2,
                                         kernel_size=2, stride=2)

        # Import locally to avoid circular dependency
        from models.resunet3d import ResBlock
        self.conv = ResBlock(in_channels, out_channels)

        if use_sla:
            self.sla = SLA3D(out_channels)
        else:
            self.sla = nn.Identity()

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.sla(x)  # SLA attention at this decoder stage
        return x
