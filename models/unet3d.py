"""
UNet3d — Baseline 3D U-Net for BraTS2020.
Extracted from: MultiModel XAI Brats2020.ipynb (cell 42)

Uses: GroupNorm + ReLU, MaxPool3d down, Trilinear up, Concat skip.
"""

import torch.nn as nn
from models.base_blocks import DoubleConv, Down, Up, Out


class UNet3d(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        # extracting the features by incrementally multiplying the no.of channels
        self.conv = DoubleConv(in_channels, n_channels) #64
        self.enc1 = Down(n_channels, 2 * n_channels) #64,128
        self.enc2 = Down(2 * n_channels, 4 * n_channels) #128, 256
        self.enc3 = Down(4 * n_channels, 8 * n_channels) #256, 512
        self.enc4 = Down(8 * n_channels, 8 * n_channels) #512, 512

        self.dec1 = Up(16 * n_channels, 4 * n_channels) # 512+512, 256
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):

        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)

        """
        After a series of either Upsampling / 3d Transpose
        a segmented image of the input image is generated
        & returned
        """
        #print(mask.shape)
        return mask
