"""
AttUNet3d — Attention 3D U-Net for BraTS2020.
Extracted from: MultiModel XAI Brats2020.ipynb (cells 51, 53)

Attention modules: CBAM (Channel + Spatial) + Attention Gate on skip connections.
Uses: DoubleConv, Down, Out from base_blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_blocks import DoubleConv, Out


# ============================================================
# Attention Modules (cell 51)
# ============================================================

class ChannelAttention(nn.Module):
    def __init__(self, ch, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.ratio = ratio
        self.sigmoid = nn.Sigmoid()
        self.channel = ch

        self.mlp = nn.Sequential(
            nn.Conv3d(self.channel, self.channel // self.ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv3d(self.channel // self.ratio, self.channel, kernel_size=1, bias=False)
        )
    def forward(self, x):
        x1 = self.avg_pool(x)


        x1 = self.mlp(x1)
        x2 = self.max_pool(x)
        x2 = self.mlp(x2)

        feats = x1 + x2
        feats = self.sigmoid(feats)
        refined_feats = x * feats
        return refined_feats


class SpatialAttention(nn.Module):
    def __init__(self, ch, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(2, ch, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        feats = self.sigmoid(x)
        refined_feats = x* feats
        return refined_feats


class cbam(nn.Module):
    def __init__(self, channel):
        super().__init__()


        self.ca = ChannelAttention(channel)

        self.sa = SpatialAttention(channel)

    def forward(self, x):
        x = self.ca(x)

        x = self.sa(x)

        return x

# https://idiotdeveloper.com/attention-unet-in-pytorch/
class attention_gate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm3d(out_c)
        )
        self.Ws = nn.Sequential(

            nn.Conv3d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm3d(out_c)

        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv3d(out_c, out_c, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, g, s):

        Wg = self.Wg(g) #from attention gate


        Ws = self.Ws(s) #from skip connection

        out = self.relu(Wg + Ws)

        out = self.output(out)

        return out * Ws


# ============================================================
# AttUNet3d Architecture (cell 53)
# ============================================================

class AttUp(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

        self.num_channels = in_channels //2
        self.cbam_module =  cbam(self.num_channels)
        self.attention_gate = attention_gate(self.num_channels, self.num_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x2 = self.cbam_module(x2)

        x2 = self.attention_gate(x1,x2)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Down(nn.Module): # Move downwards

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        # max pooling 3d + doubleConv
        return self.encoder(x)

class AttUNet3d(nn.Module):

    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        # extracting the features by incrementally multiplying the no.of channels
        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)

        self.dec1 = AttUp(16 * n_channels, 4 * n_channels)
        self.dec2 = AttUp(8* n_channels, 2 * n_channels)
        self.dec3 = AttUp(4 * n_channels, n_channels)
        self.dec4 = AttUp(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)

        #Bridge

        x5 = self.enc4(x4)

        #Decoder
        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)

        return mask
