"""
ResUNet with multi-scale Laplacian residual concatenation and a boundary head.

This is the final combination of two existing ablations:
  - ResUNetEdge: Laplacian residual -> EdgePyramid -> decoder concat
  - ResUNetHFBoundary: segmentation head + auxiliary boundary head

The encoder, multi-scale edge branch, and concat decoder are inherited from
ResUNetEdge so the combined experiment stays identical to the edge-only model
except for the auxiliary boundary head and its loss.
"""

import torch.nn as nn

from models.resunet_edge import ResUNetEdge


class ResUNetHFConcatBoundary(ResUNetEdge):
    """Laplacian multi-scale (optionally gated) ResUNet with dual heads."""

    def __init__(self, in_channels=4, n_classes=3, n_channels=24,
                 fusion="concat", multiscale_context=False):
        super().__init__(
            in_channels=in_channels,
            n_classes=n_classes,
            n_channels=n_channels,
            fusion=fusion,
            edge_type="laplacian",
            multiscale_context=multiscale_context,
        )

        self.n_classes = n_classes
        self.boundary_head = nn.Sequential(
            nn.Conv3d(n_channels, n_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, n_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_channels, n_classes, kernel_size=1),
        )

    def forward(self, x):
        """Return segmentation and boundary logits at the input resolution."""
        edge_input = self.edge_extractor(x)
        edge_dict = self.edge_pyramid(edge_input)

        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        if self.multiscale_context_enabled:
            x5 = self.multiscale_context(x5)

        decoded = self.dec1(x5, x4, edge_dict["dec1"])
        decoded = self.dec2(decoded, x3, edge_dict["dec2"])
        decoded = self.dec3(decoded, x2, edge_dict["dec3"])
        decoded = self.dec4(decoded, x1, edge_dict["dec4"])

        seg = self.out(decoded)
        boundary = self.boundary_head(decoded)
        return seg, boundary
