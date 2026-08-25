"""
=============================================================================
ResUNet3d + High-Frequency Edge Auxiliary Branch (V2)
=============================================================================
Adds an explicit edge feature branch to ResUNet3d:

  1. Edge Extraction (Dataset Transform, on raw MRI):
     - Sobel (1st derivative): directional gradient magnitude |∇I|
     - Laplacian (2nd derivative): I - Gaussian(I) = high-freq residual
     - Random noise: fairness control (isolates param count from edge info)
     - Output → (B, 4, D, H, W) edge volume

  2. Multi-scale Edge Pyramid:
     - Downsample edge volume via strided conv (or avgpool)
     - Match decoder feature resolutions at each stage

  3. Fusion into Decoder (3 options):
     - Concat: torch.cat([decoder_feat, edge_feat], dim=1) → DoubleConv
     - Gated concat: adaptively scale edge features before concatenation
     - Add:    decoder_feat + edge_feat (after 1x1x1 conv alignment)

  4. Sobel vs Laplacian:
     - Sobel (1st deriv): directional gradient, preserves edge orientation
     - Laplacian (2nd deriv): isotropic, captures ALL edges regardless of direction
     - Both are valid dataset transforms; comparison isolates derivative order
     - FGFE uses Laplacian on decoder FEATURES (not raw MRI) — separate mechanism

  5. Why inject at decoder last 2 stages:
     - dec3 (64^3) and dec4 (128^3) are closest to output resolution
     - Boundary refinement needs high spatial resolution
     - Earlier decoder stages (16^3, 32^3) have too coarse resolution

Reference:
  - Yi et al., "Frequency-Aware Ensemble", BraTS 2025, arXiv:2509.19353
  - Yao et al., "BraTS-UMamba: Dual-Band Frequency Feature Enhancement", MICCAI 2025

Architecture diagram:

  4-modal MRI (B, 4, 128³)
       │
       ├──> Encoder (ResUNet, unchanged)
       │         │
       │         └──> Bottleneck (B, 192, 8³)
       │                    │
       │                    └──> Decoder ──> Output (B, 3, 128³)
       │                         │    ▲
       │                         │    │
       └──> Sobel/Laplacian    │    │
            Edge Extraction  Edge │    │
            (B,4,128³)    Pyramid │    │
                          (B,C,   │    │
                           D,H,W)│    │
                                 │    │
                    Concat or Add ────┘
                    (at dec3, dec4)

Usage:
    from models.resunet_edge import ResUNetEdge, SobelEdge3d, LaplacianEdge3d

    model = ResUNetEdge(
        in_channels=4, n_classes=3, n_channels=24,
        fusion='concat',   # 'concat', 'gated_concat', or 'add'
        edge_type='sobel', # 'sobel', 'laplacian', or 'random'
    )

Author: Generated for ResUNet enhancement project
Date:   2026-08-02
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from models.resunet3d import ResBlock, ResDown, ResUp, FirstLayer, Out


class FusionMode(Enum):
    CONCAT = "concat"
    GATED_CONCAT = "gated_concat"
    ADD = "add"


# ============================================================
# 3D Sobel Edge Extraction
# ============================================================

class SobelEdge3d(nn.Module):
    """
    Extract gradient magnitude from each input channel using 3D Sobel.

    Sobel kernels (3x3x3) for x, y, z directions:
      - Gx: detects horizontal edges (intensity change along x)
      - Gy: detects vertical edges (intensity change along y)
      - Gz: detects depth edges (intensity change along z)

    Gradient magnitude: |nabla I| = sqrt(Gx^2 + Gy^2 + Gz^2)

    This is applied PER MODALITY (4 times), then stacked → (B, 4, D, H, W).

    Why per-modality not per-channel of feature map:
      - Raw MRI has physical meaning: T1ce enhancement = real tissue property
      - Applying edge detection on semantic features loses this interpretability
      - The edge branch should provide PRIOR knowledge, not learned features
    """

    def __init__(self):
        super().__init__()

        # ---- Sobel-x kernel: detects gradients along x-axis ----
        # Central column is positive, side columns negative
        kx = torch.zeros(1, 1, 3, 3, 3)
        kx[0, 0, 1, :, :] = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ], dtype=torch.float32)

        # ---- Sobel-y kernel: detects gradients along y-axis ----
        ky = torch.zeros(1, 1, 3, 3, 3)
        ky[0, 0, :, 1, :] = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1],
        ], dtype=torch.float32).T

        # ---- Sobel-z kernel: detects gradients along z-axis ----
        kz = torch.zeros(1, 1, 3, 3, 3)
        kz[0, 0, :, :, 1] = torch.tensor([
            [-1, -2, -1],
            [-2, -4, -2],   # This isn't quite right for 3D...
            [-1, -2, -1],
        ], dtype=torch.float32)

        # Actually, let me fix the Sobel kernels properly
        # Standard 3D Sobel decomposes as: outer product of 1D Sobel and 1D smoothing

        # 1D Sobel operator (derivative): [1, 0, -1] or [-1, 0, 1]
        # 1D Smoothing operator: [1, 2, 1]

        sobel_1d = torch.tensor([1, 0, -1], dtype=torch.float32)  # derivative
        smooth_1d = torch.tensor([1, 2, 1], dtype=torch.float32)   # smoothing

        # Gx = sobel_1d ⊗ smooth_1d ⊗ smooth_1d (derivative along x, smooth y,z)
        gx_kernel = torch.einsum('i,j,k->ijk', sobel_1d, smooth_1d, smooth_1d)
        # Gy = smooth_1d ⊗ sobel_1d ⊗ smooth_1d (derivative along y)
        gy_kernel = torch.einsum('i,j,k->ijk', smooth_1d, sobel_1d, smooth_1d)
        # Gz = smooth_1d ⊗ smooth_1d ⊗ sobel_1d (derivative along z)
        gz_kernel = torch.einsum('i,j,k->ijk', smooth_1d, smooth_1d, sobel_1d)

        self.register_buffer('kx', gx_kernel.view(1, 1, 3, 3, 3))
        self.register_buffer('ky', gy_kernel.view(1, 1, 3, 3, 3))
        self.register_buffer('kz', gz_kernel.view(1, 1, 3, 3, 3))

    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W) — 4-modal MRI input

        Returns:
            edge: (B, C, D, H, W) — gradient magnitude per channel
        """
        B, C, D, H, W = x.shape
        x_flat = x.view(B * C, 1, D, H, W)

        # Apply 3 directional Sobel kernels
        gx = F.conv3d(x_flat, self.kx, padding=1)
        gy = F.conv3d(x_flat, self.ky, padding=1)
        gz = F.conv3d(x_flat, self.kz, padding=1)

        # Gradient magnitude
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + 1e-9)
        grad_mag = grad_mag.view(B, C, D, H, W)

        return grad_mag


# ============================================================
# 3D Laplacian Edge Extraction
# ============================================================

class LaplacianEdge3d(nn.Module):
    """
    Extract high-frequency residual from raw MRI using Laplacian decomposition.

    Algorithm (Dippel et al., IEEE TMI 2002):
      1. Gaussian blur: I_smooth = GaussianFilter(I)  (3x3x3, sigma≈1)
      2. High-freq:  I_h = I - I_smooth  (residual = edges/textures)

    This is the 2nd-derivative counterpart to SobelEdge3d (1st derivative).
    Laplacian is isotropic — it detects edges in ALL directions equally,
    while Sobel preserves directional information per axis.

    Applied PER MODALITY (4 times), stacked → (B, 4, D, H, W).
    Same interface and output shape as SobelEdge3d → drop-in replacement.

    Relationship to FGFE:
      - LaplacianEdge3d: operates on RAW MRI pixels (dataset transform)
      - FGFE.LaplacianPyramid3d: operates on DECODER FEATURES (learned repr.)
      - Both use the same mathematical operation (I - blur(I))
      - But apply it at DIFFERENT stages of the pipeline
    """

    def __init__(self):
        super().__init__()
        # 3D box filter (binomial approximation of Gaussian, sigma≈1)
        k3d = torch.ones(1, 1, 3, 3, 3, dtype=torch.float32) / 27.0
        self.register_buffer('blur_kernel', k3d)

    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W) — 4-modal MRI input

        Returns:
            high_freq: (B, C, D, H, W) — Laplacian high-freq residual
        """
        B, C, D, H, W = x.shape
        x_flat = x.reshape(B * C, 1, D, H, W)
        x_smooth = F.conv3d(x_flat, self.blur_kernel, padding=1)
        x_smooth = x_smooth.reshape(B, C, D, H, W)
        high_freq = x - x_smooth  # Laplacian: residual = edges
        return high_freq


# ============================================================
# Edge Pyramid: multi-scale edge features
# ============================================================

class EdgePyramid(nn.Module):
    """
    Build a multi-scale pyramid of edge features.

    Input:  (B, 4, 128, 128, 128) — Sobel edge maps
    Output: dict of (B, C_out, D_s, H_s, W_s) at each decoder resolution

    Uses AvgPool3d(2,2) downsampling (NOT stride-2 conv) to match
    the exact rounding behavior of encoder's MaxPool3d(2,2).
    Both use floor division → identical spatial sizes at every stage.

    Channel progression matches decoder stages:
      level_4: (B,  24, 128³) — matches dec4 input (skip from conv)
      level_3: (B,  24,  64³) — matches dec3 input (skip from enc1)
      level_2: (B,  48,  32³) — matches dec2 input (skip from enc2)
      level_1: (B,  96,  16³) — matches dec1 input (skip from enc3)

    Safety: ResUpEdge applies F.interpolate if sizes still differ by 1-2 voxels.
    """

    def __init__(self, in_channels=4, base_channels=24):
        super().__init__()

        # Edge processing at full resolution
        self.edge_conv0 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.ReLU(inplace=True),
        )

        # Downsampling: MUST use AvgPool not stride-2 conv.
        # Reason: Encoder uses MaxPool3d(2,2) which truncates (floor),
        #          while Conv3d(stride=2) rounds differently for odd dims.
        #          AvgPool3d(2,2) gives SAME output size as MaxPool3d(2,2).
        def make_down(in_ch, out_ch):
            return nn.Sequential(
                nn.AvgPool3d(2, 2),                         # same rounding as MaxPool
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.ReLU(inplace=True),
            )

        self.edge_down1 = make_down(base_channels, base_channels)
        self.edge_down2 = make_down(base_channels, 2 * base_channels)
        self.edge_down3 = make_down(2 * base_channels, 4 * base_channels)

    def forward(self, edge_input):
        """
        Returns:
            dict: {
                'dec4': (B, base_channels, 128³),
                'dec3': (B, base_channels, 64³),
                'dec2': (B, 2*base_channels, 32³),
                'dec1': (B, 4*base_channels, 16³),
            }
        """
        e0 = self.edge_conv0(edge_input)   # (B, 24, 128³)
        e1 = self.edge_down1(e0)           # (B, 24,  64³)
        e2 = self.edge_down2(e1)           # (B, 48,  32³)
        e3 = self.edge_down3(e2)           # (B, 96,  16³)

        return {
            'dec4': e0,  # matches conv output
            'dec3': e1,  # matches enc1 skip
            'dec2': e2,  # matches enc2 skip
            'dec1': e3,  # matches enc3 skip
        }


# ============================================================
# Decoder Blocks with Edge Fusion
# ============================================================

class ResUpEdge(nn.Module):
    """
    ResUp + Edge Feature Fusion.

    Args:
        fusion: 'concat', 'gated_concat', or 'add'
          - concat: cat([upsampled, skip, edge]) → ResBlock
          - gated_concat: scale edge with a residual gate, then concat
          - add:    ResBlock(cat([upsampled, skip])) + edge (residual)
    """

    def __init__(self, in_channels, out_channels, edge_channels,
                 trilinear=True, fusion='concat'):
        super().__init__()

        self.fusion = fusion

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2,
                                         kernel_size=2, stride=2)

        if fusion in ('concat', 'gated_concat'):
            # concat: [upsampled | skip | edge] → ResBlock
            # in_channels already includes skip (in_ch = deeper + skip_ch)
            # add edge_channels
            total_in = in_channels + edge_channels
            self.conv = ResBlock(total_in, out_channels)
            if fusion == 'gated_concat':
                # Creating a Conv3d normally advances the global RNG and would
                # change later shared-layer initialization versus concat.
                rng_state = torch.get_rng_state()
                self.edge_gate = nn.Conv3d(
                    total_in, edge_channels, kernel_size=1, bias=True,
                )
                torch.set_rng_state(rng_state)
                nn.init.zeros_(self.edge_gate.weight)
                nn.init.zeros_(self.edge_gate.bias)
        elif fusion == 'add':
            # Residual add: ResBlock([upsampled | skip]) + edge
            self.conv = ResBlock(in_channels, out_channels)
            # 1x1 conv to align edge channels to output
            self.edge_proj = nn.Conv3d(edge_channels, out_channels,
                                       kernel_size=1, bias=False)
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

    def forward(self, x1, x2, edge_feat=None):
        """
        x1: deeper feature (C, smaller spatial)
        x2: skip feature (C, larger spatial)
        edge_feat: edge pyramid feature at this resolution (may differ by 1-2 voxels)
        """
        x1 = self.up(x1)

        # Size matching for x1 → x2
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2,
                         diffZ // 2, diffZ - diffZ // 2])

        # Size matching for edge_feat → x2 (critical: AvgPool vs MaxPool rounding)
        if edge_feat is not None:
            if edge_feat.shape[2:] != x2.shape[2:]:
                edge_feat = F.interpolate(edge_feat, size=x2.shape[2:],
                                          mode='trilinear', align_corners=True)

        if self.fusion in ('concat', 'gated_concat'):
            if edge_feat is not None:
                if self.fusion == 'gated_concat':
                    gate_input = torch.cat([x2, x1, edge_feat], dim=1)
                    edge_scale = 1.0 + torch.tanh(self.edge_gate(gate_input))
                    edge_feat = edge_feat * edge_scale
                x = torch.cat([x2, x1, edge_feat], dim=1)
            else:
                x = torch.cat([x2, x1], dim=1)
            return self.conv(x)

        elif self.fusion == 'add':
            x = torch.cat([x2, x1], dim=1)
            out = self.conv(x)
            if edge_feat is not None:
                out = out + self.edge_proj(edge_feat)
            return out


# ============================================================
# Lightweight multi-scale context at the bottleneck
# ============================================================

class MultiScaleContext3d(nn.Module):
    """Aggregate local and dilated 3D context through a residual block."""

    def __init__(self, channels, identity_start=False):
        super().__init__()
        self.identity_start = identity_start
        branch_channels = max(4, channels // 4)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels, branch_channels, kernel_size=3,
                          padding=1, dilation=1),
                nn.GroupNorm(4, branch_channels),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv3d(channels, branch_channels, kernel_size=3,
                          padding=2, dilation=2),
                nn.GroupNorm(4, branch_channels),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv3d(channels, branch_channels, kernel_size=3,
                          padding=3, dilation=3),
                nn.GroupNorm(4, branch_channels),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv3d(channels, branch_channels, kernel_size=1),
                nn.GroupNorm(4, branch_channels),
                nn.ReLU(inplace=True),
            ),
        ])
        self.project = nn.Sequential(
            nn.Conv3d(4 * branch_channels, channels, kernel_size=1),
            nn.GroupNorm(8, channels),
        )
        if identity_start:
            self.alpha = nn.Parameter(torch.zeros(1))
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        multi_scale = torch.cat([branch(x) for branch in self.branches], dim=1)
        context = self.project(multi_scale)
        if self.identity_start:
            return x + self.alpha * context
        return self.activation(x + context)


# ============================================================
# Full ResUNet + Edge Branch
# ============================================================

class ResUNetEdge(nn.Module):
    """
    ResUNet3d with High-Frequency Edge Auxiliary Branch.

    Args:
        edge_type: 'sobel' (1st derivative), 'laplacian' (2nd derivative),
                   or 'random' (fairness control — noise instead of edges)

    Usage:
        model = ResUNetEdge(in_channels=4, n_classes=3, n_channels=24,
                            fusion='concat', edge_type='sobel')

        # Random edge control (fairness ablation):
        model = ResUNetEdge(..., edge_type='random')
    """

    def __init__(self, in_channels=4, n_classes=3, n_channels=24,
                 fusion='concat', edge_type='sobel',
                 multiscale_context=False, multiscale_context_v2=False):
        super().__init__()

        if fusion not in ('concat', 'gated_concat', 'add'):
            raise ValueError(
                "fusion must be 'concat', 'gated_concat', or 'add', "
                f"got {fusion}"
            )
        if edge_type not in ('sobel', 'laplacian', 'random'):
            raise ValueError(f"edge_type must be 'sobel', 'laplacian', or 'random', got {edge_type}")

        self.fusion = fusion
        self.edge_type = edge_type
        if multiscale_context and multiscale_context_v2:
            raise ValueError("Choose either multiscale_context or multiscale_context_v2")
        self.multiscale_context_enabled = (
            multiscale_context or multiscale_context_v2
        )
        self.multiscale_context_v2_enabled = multiscale_context_v2
        self.in_channels = in_channels
        self.n_channels = n_channels

        # ---- Edge Extraction (dataset transform) ----
        if edge_type == 'sobel':
            self.edge_extractor = SobelEdge3d()
        elif edge_type == 'laplacian':
            self.edge_extractor = LaplacianEdge3d()
        else:  # 'random' — extractor is unused, but keep for param count parity
            self.edge_extractor = None

        # ---- Edge Pyramid ----
        self.edge_pyramid = EdgePyramid(in_channels, n_channels)

        # ---- Encoder (unchanged from ResUNet3d) ----
        self.conv = FirstLayer(in_channels, n_channels)
        self.enc1 = ResDown(n_channels, 2 * n_channels)
        self.enc2 = ResDown(2 * n_channels, 4 * n_channels)
        self.enc3 = ResDown(4 * n_channels, 8 * n_channels)
        self.enc4 = ResDown(8 * n_channels, 8 * n_channels)

        # ---- Decoder (with edge fusion) ----
        # Edge channels at each level: 4*n_ch, 2*n_ch, n_ch, n_ch
        edge_ch_list = {
            'dec1': 4 * n_channels,  # matches enc3 skip
            'dec2': 2 * n_channels,  # matches enc2 skip
            'dec3': n_channels,      # matches enc1 skip
            'dec4': n_channels,      # matches conv skip
        }

        self.dec1 = ResUpEdge(16 * n_channels, 4 * n_channels,
                              edge_ch_list['dec1'], fusion=fusion)
        self.dec2 = ResUpEdge(8 * n_channels, 2 * n_channels,
                              edge_ch_list['dec2'], fusion=fusion)
        self.dec3 = ResUpEdge(4 * n_channels, n_channels,
                              edge_ch_list['dec3'], fusion=fusion)
        self.dec4 = ResUpEdge(2 * n_channels, n_channels,
                              edge_ch_list['dec4'], fusion=fusion)
        self.out = Out(n_channels, n_classes)
        if multiscale_context:
            # The extra branch must not perturb shared-layer initialization.
            with torch.random.fork_rng(devices=[]):
                self.multiscale_context = MultiScaleContext3d(8 * n_channels)
        elif multiscale_context_v2:
            # V2 starts as an exact identity on the pretrained backbone.
            with torch.random.fork_rng(devices=[]):
                self.multiscale_context = MultiScaleContext3d(
                    8 * n_channels,
                    identity_start=True,
                )

    def forward(self, x):
        """
        Args:
            x: (B, 4, D, H, W) — 4-modal MRI

        Returns:
            mask: (B, 3, D, H, W) — WT, TC, ET predictions
        """
        # ---- Edge branch ----
        if self.edge_type == 'random':
            # Fairness control: random noise instead of edge info
            # Same dimensions, same param count, different information content
            edge_input = torch.randn_like(x)
        else:
            edge_input = self.edge_extractor(x)      # (B, 4, D, H, W)
        edge_dict = self.edge_pyramid(edge_input)  # multi-scale edge features

        # ---- Encoder ----
        x1 = self.conv(x)                     # (B, 24, 128³)
        x2 = self.enc1(x1)                    # (B, 48,  64³)
        x3 = self.enc2(x2)                    # (B, 96,  32³)
        x4 = self.enc3(x3)                    # (B, 192, 16³)
        x5 = self.enc4(x4)                    # (B, 192,  8³)
        if self.multiscale_context_enabled:
            x5 = self.multiscale_context(x5)

        # ---- Decoder (with edge injection) ----
        mask = self.dec1(x5, x4, edge_dict['dec1'])   # → (B, 96, 16³)
        mask = self.dec2(mask, x3, edge_dict['dec2'])  # → (B, 48, 32³)
        mask = self.dec3(mask, x2, edge_dict['dec3'])  # → (B, 24, 64³)
        mask = self.dec4(mask, x1, edge_dict['dec4'])  # → (B, 24, 128³)
        mask = self.out(mask)

        return mask
