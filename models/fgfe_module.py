"""
=============================================================================
FGFE Module — Frequency Guidance based Feature Enhancement
=============================================================================
Faithful implementation based on Yao et al., BraTS-UMamba, MICCAI 2025.

Core idea (from paper Section 2.2, Fig. 1):
  Given a decoder spatial feature F_s:
    1. Apply 3D Laplacian pyramid decomposition to F_s
       → F_h (high-freq: edges, textures, boundaries)
       → F_l (low-freq: global structure, smooth regions)
    2. Cross-domain attention fusion:
       F_h, F_l → mapping → Q_h, Q_l  (queries from frequency domain)
       F_s      → mapping → K, V      (keys/values from spatial domain)
       Attention: softmax(QK^T/sqrt(d)) * V
    3. Concat(attended_low, attended_high) + F_s → output

Why this is better than Sobel-on-raw-MRI:
  - Operates on LEARNED features, not raw pixels
  - Jointly uses LF (global structure) AND HF (boundary detail)
  - Cross-attention SELECTS informative features from both bands
  - Paper evidence: LF helps Dice, HF helps HD95, both together best

Reference:
  Yao et al., "BraTS-UMamba: Adaptive Mamba UNet with Dual-Band
  Frequency based Feature Enhancement for Brain Tumor Segmentation",
  MICCAI 2025.

Author: Generated for ResUNet enhancement project
Date:   2026-08-02
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resunet3d import ResBlock

# ============================================================
# 3D Laplacian Pyramid Decomposition
# ============================================================

class LaplacianPyramid3d(nn.Module):
    """
    Decompose a 3D feature map into high-freq and low-freq components.

    Algorithm (Dippel et al., IEEE TMI 2002, as cited in the paper):
      1. Gaussian blur: F_smooth = GaussianFilter(F)  (3x3x3, sigma=1)
      2. Low-freq:  F_l = F_smooth
      3. High-freq: F_h = F - F_smooth  (residual = edges/textures)

    This is a SINGLE level decomposition (not multi-level as in the
    full Laplacian pyramid). The paper uses this on each decoder
    feature map independently.

    Note: We use a simple 3x3x3 avg pool as the "Gaussian blur"
          approximation, following common practice in deep learning
          (exact Gaussian is expensive and not differentiable).
    """

    def __init__(self):
        super().__init__()
        # 3D Gaussian-like smoothing via 3x3x3 convolution
        # Kernel: 3D binomial approximation of Gaussian (sigma≈1)
        k3d = torch.ones(1, 1, 3, 3, 3, dtype=torch.float32) / 27.0
        self.register_buffer('blur_kernel', k3d)

    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W) — decoder feature map

        Returns:
            F_l: (B, C, D, H, W) — low-freq (smoothed)
            F_h: (B, C, D, H, W) — high-freq (residual = x - F_l)
        """
        B, C, D, H, W = x.shape
        # Apply per-channel 3D blur
        x_flat = x.reshape(B * C, 1, D, H, W)
        F_l_flat = F.conv3d(x_flat, self.blur_kernel, padding=1)
        F_l = F_l_flat.reshape(B, C, D, H, W)
        F_h = x - F_l  # high-freq = original - smoothed
        return F_h, F_l


# ============================================================
# FGFE Module (Frequency Guidance Feature Enhancement)
# ============================================================

class FGFE(nn.Module):
    """
    Frequency Guidance based Feature Enhancement module.

    Diagram:
        F_s (decoder spatial feature)
         │
         ├──> LaplacianPyramid ──> F_h ──> φ ──> Q_h ──┐
         │                       ──> F_l ──> φ ──> Q_l ──┤
         │                                                ├──> Attention ──> concat ──> output
         └──> φ ──> K ────────────────────────────────────┤
              φ ──> V ────────────────────────────────────┘

    Where each φ is: GroupNorm → ReLU → Conv1x1x1

    Args:
        channels: number of channels in F_s
        reduction: reduction ratio for mapping layers (default 2)
    """

    def __init__(self, channels, reduction=2):
        super().__init__()
        mid_ch = channels // reduction

        self.lap = LaplacianPyramid3d()

        # Mapping layers for queries (from frequency domain)
        self.qh_map = nn.Sequential(
            nn.GroupNorm(min(8, mid_ch), channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, mid_ch, kernel_size=1),
        )
        self.ql_map = nn.Sequential(
            nn.GroupNorm(min(8, mid_ch), channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, mid_ch, kernel_size=1),
        )

        # Mapping layers for keys and values (from spatial domain)
        self.k_map = nn.Sequential(
            nn.GroupNorm(min(8, mid_ch), channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, mid_ch, kernel_size=1),
        )
        self.v_map = nn.Sequential(
            nn.GroupNorm(min(8, mid_ch), channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, mid_ch, kernel_size=1),
        )

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv3d(2 * mid_ch, channels, kernel_size=1),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True),
        )

        self.scale = mid_ch ** 0.5  # 1/sqrt(d) for attention scaling

    def forward(self, F_s):
        """
        Args:
            F_s: (B, C, D, H, W) — spatial decoder feature

        Returns:
            out: (B, C, D, H, W) — enhanced spatial feature
        """
        B, C, D, H, W = F_s.shape

        # Step 1: Laplacian decomposition
        F_h, F_l = self.lap(F_s)

        # Step 2: Map to query/key/value spaces
        Q_h_raw = self.qh_map(F_h)  # (B, d, D, H, W)
        Q_l_raw = self.ql_map(F_l)  # (B, d, D, H, W)
        K_raw   = self.k_map(F_s)   # (B, d, D, H, W)
        V_raw   = self.v_map(F_s)   # (B, d, D, H, W)

        # Step 3: Spatial pooling for memory-efficient attention
        #   Full 3D attention (D*H*W)² is infeasible for 128³ data:
        #     dec1 (16³): 67 MB ✓         dec2 (32³): 4.3 GB ✗
        #     dec3 (64³): 274 GB ✗         dec4 (128³): 17 TB ✗
        #   Fix: pool K,V to max 16³; Q stays at original resolution.
        #   Q(N_full, d) @ K(N_pool, d)^T → (N_full, N_pool) ≤ 67 MB.
        max_spatial = 16
        pool_size = tuple(min(s, max_spatial) for s in (D, H, W))
        need_pool = any(s > max_spatial for s in (D, H, W))

        if need_pool:
            K = F.adaptive_avg_pool3d(K_raw, pool_size)
            V = F.adaptive_avg_pool3d(V_raw, pool_size)
        else:
            K, V = K_raw, V_raw

        # Reshape to (B, N, d) for attention
        N_full = D * H * W
        N_pool = pool_size[0] * pool_size[1] * pool_size[2]
        mid_c = Q_h_raw.shape[1]

        Q_h = Q_h_raw.reshape(B, mid_c, N_full).permute(0, 2, 1)  # (B, N_full, d)
        Q_l = Q_l_raw.reshape(B, mid_c, N_full).permute(0, 2, 1)  # (B, N_full, d)
        K   = K.reshape(B, mid_c, N_pool).permute(0, 2, 1)         # (B, N_pool, d)
        V   = V.reshape(B, mid_c, N_pool).permute(0, 2, 1)         # (B, N_pool, d)

        # Step 4: Scaled dot-product attention (memory-efficient)
        attn_h = F.softmax((Q_h @ K.transpose(-2, -1)) / self.scale, dim=-1) @ V
        attn_l = F.softmax((Q_l @ K.transpose(-2, -1)) / self.scale, dim=-1) @ V

        # Step 5: Reshape back & concat → project
        attn_h = attn_h.permute(0, 2, 1).reshape(B, mid_c, D, H, W)
        attn_l = attn_l.permute(0, 2, 1).reshape(B, mid_c, D, H, W)

        out = self.out_proj(torch.cat([attn_h, attn_l], dim=1))

        # Residual connection (paper: "add it back to F_s")
        return F_s + out


# ============================================================
# Decoder Block with FGFE
# ============================================================

class ResUpFGFE(nn.Module):
    """
    ResUp + FGFE module at decoder stage.
    Same interface as ResUp but applies FGFE after convolution.
    """

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2,
                                         kernel_size=2, stride=2)

        self.conv = ResBlock(in_channels, out_channels)

        # FGFE after the conv block
        self.fgfe = FGFE(out_channels)

    def forward(self, x1, x2):
        """
        x1: deeper feature
        x2: skip feature
        """
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2,
                         diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        # Frequency enhancement
        x = self.fgfe(x)

        return x
