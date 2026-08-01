"""
nnUNet3d Model Architecture for BraTS2020
==========================================
Pure model module — no Dataset, Trainer, or training script.
Import into your Jupyter notebook and use with the existing Trainer.

Usage (in notebook):
    from nnunet_model import nnUNet3d

    model = nnUNet3d(in_channels=4, n_classes=3, n_channels=24).to('cuda')
    trainer = Trainer(net=model, ..., model_type=config.nnUNet_checkpoint_dir)
    trainer.run(config.nnUNet_checkpoint_dir)

Architecture differences vs. existing models:
---------------------------------------------------------------------------
| Component        | UNet3d        | ResUNet3d     | AttUNet3d     | nnUNet3d      |
|------------------|---------------|---------------|---------------|---------------|
| Normalization    | GroupNorm     | GroupNorm     | GroupNorm     | InstanceNorm  |
| Activation       | ReLU          | ReLU          | ReLU          | LeakyReLU     |
| Downsampling     | MaxPool3d     | MaxPool3d     | MaxPool3d     | Strided Conv  |
| Upsampling       | Trilinear     | Trilinear     | Trilinear     | ConvTranspose |
| Skip Connection  | Concat        | Concat+Res    | Concat+Att    | Concat        |
---------------------------------------------------------------------------

Reference: Isensee et al., Nature Methods 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# nnU-Net Building Blocks
# ============================================================

class nnDoubleConv(nn.Module):
    """
    (Conv3D -> InstanceNorm3d -> LeakyReLU) * 2

    Why InstanceNorm instead of GroupNorm?
    - 3D medical imaging uses batch_size=1 (GPU memory constraint)
    - BatchNorm is unstable with batch_size=1
    - InstanceNorm treats each channel independently: most stable for small batches

    Why LeakyReLU instead of ReLU?
    - Prevents "dying ReLU" (permanently inactive neurons)
    - negative_slope=1e-2 allows small gradients for negative values
    """
    def __init__(self, in_channels, out_channels, leaky_slope=1e-2):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class nnDown(nn.Module):
    """
    Strided Conv for downsampling (NO MaxPool).

    Conv3d(kernel=3, stride=2, padding=1) halves spatial dims.
    Learnable weights adapt to data — unlike fixed MaxPool which
    discards 87.5% of spatial information per 2x2x2 window.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nnDoubleConv(out_channels, out_channels),
        )

    def forward(self, x):
        return self.encoder(x)


class nnUp(nn.Module):
    """
    Transposed Conv for upsampling + nnDoubleConv.

    Channel math (SAME interface as original Up/ResUp/AttUp):
        in_channels = deeper_ch + skip_ch = 2 * deeper_ch
        ConvTranspose3d(C, C, kernel=2, stride=2) — preserves channels
        cat(C, C) -> 2C -> nnDoubleConv(2C, out)

    Why ConvTranspose3d instead of trilinear upsample?
    - Trilinear uses fixed interpolation weights
    - ConvTranspose3d learns optimal upsampling per feature map
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        deeper_channels = in_channels // 2
        self.up = nn.ConvTranspose3d(
            deeper_channels, deeper_channels,
            kernel_size=2, stride=2
        )
        self.conv = nnDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: deeper feature (C channels, smaller spatial)
        x2: skip feature  (C channels, larger spatial)
        """
        x1 = self.up(x1)

        # Handle size mismatches (same logic as original Up)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2,
                         diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ============================================================
# nnUNet3d — The Full Architecture
# ============================================================

class nnUNet3d(nn.Module):
    """
    nnU-Net 3D for Brain Tumor Segmentation.

    Input:  (B, 4,  128, 128, 128) — FLAIR, T1, T1ce, T2
    Output: (B, 3,  128, 128, 128) — WT, TC, ET

    Encoder (strided conv downsampling):
        conv  (4->24):   nnDoubleConv            -> (24,  128^3)
        enc1  (24->48):  StridedConv + nnDouble  -> (48,   64^3)
        enc2  (48->96):  StridedConv + nnDouble  -> (96,   32^3)
        enc3  (96->192): StridedConv + nnDouble  -> (192,  16^3)
        enc4  (192->192):StridedConv + nnDouble  -> (192,   8^3)  [bottleneck]

    Decoder (ConvTranspose3d upsampling):
        dec1  (384->96): ConvTranspose + skip    -> (96,   16^3)
        dec2  (192->48): ConvTranspose + skip    -> (48,   32^3)
        dec3  (96->24):  ConvTranspose + skip    -> (24,   64^3)
        dec4  (48->24):  ConvTranspose + skip    -> (24,  128^3)
        out   (24->3):   1x1x1 Conv3d            -> (3,   128^3)

    ~8.35M params (n_channels=24) — comparable scale to existing models.
    """
    def __init__(self, in_channels=4, n_classes=3, n_channels=24):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        # Encoder
        self.conv = nnDoubleConv(in_channels, n_channels)           # 4->24, 128^3
        self.enc1 = nnDown(n_channels, 2 * n_channels)              # 24->48, 64^3
        self.enc2 = nnDown(2 * n_channels, 4 * n_channels)          # 48->96, 32^3
        self.enc3 = nnDown(4 * n_channels, 8 * n_channels)          # 96->192, 16^3
        self.enc4 = nnDown(8 * n_channels, 8 * n_channels)          # 192->192, 8^3

        # Decoder
        self.dec1 = nnUp(16 * n_channels, 4 * n_channels)           # 384->96
        self.dec2 = nnUp(8 * n_channels, 2 * n_channels)            # 192->48
        self.dec3 = nnUp(4 * n_channels, n_channels)                # 96->24
        self.dec4 = nnUp(2 * n_channels, n_channels)                # 48->24
        self.out = nn.Conv3d(n_channels, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        # Decoder with skip connections
        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask
