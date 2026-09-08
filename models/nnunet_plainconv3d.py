"""nnU-Net v2 PlainConvUNet architecture for the unified baseline protocol.

The network implementation comes directly from the official nnU-Net v2
architecture backend, ``dynamic-network-architectures``.  Data loading, loss,
optimizer and model selection deliberately remain those used by this
repository's U-Net and ResUNet baselines.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
except ImportError as exc:  # pragma: no cover - exercised on AutoDL if setup was skipped
    raise ImportError(
        "NNUNetV2PlainConv3D requires dynamic-network-architectures==0.4.4. "
        "Install it with: python -m pip install -r "
        "requirements_nnunet_architecture.txt"
    ) from exc


DEFAULT_FEATURES = (32, 64, 128, 256, 320)


class NNUNetV2PlainConv3D(nn.Module):
    """Official nnU-Net v2 PlainConvUNet backend with shape-safe I/O.

    Four stride-2 stages require spatial sizes divisible by 16.  The existing
    BraTS loader yields ``(D, H, W) = (100, 170, 170)``, so the adapter pads
    only for the forward pass and crops raw logits back to the exact target
    shape.  No sigmoid is applied because ``BCEDiceLoss`` consumes logits.
    """

    def __init__(
        self,
        in_channels: int = 4,
        n_classes: int = 3,
        features_per_stage: Sequence[int] = DEFAULT_FEATURES,
    ) -> None:
        super().__init__()
        features = tuple(int(value) for value in features_per_stage)
        if len(features) != 5 or any(value <= 0 for value in features):
            raise ValueError("features_per_stage must contain five positive integers")

        self.in_channels = int(in_channels)
        self.n_classes = int(n_classes)
        self.features_per_stage = features
        self.required_divisibility = 2 ** (len(features) - 1)
        self.network = PlainConvUNet(
            input_channels=self.in_channels,
            n_stages=len(features),
            features_per_stage=features,
            conv_op=nn.Conv3d,
            kernel_sizes=((3, 3, 3),) * len(features),
            strides=((1, 1, 1),) + ((2, 2, 2),) * (len(features) - 1),
            n_conv_per_stage=(2,) * len(features),
            num_classes=self.n_classes,
            n_conv_per_stage_decoder=(2,) * (len(features) - 1),
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=False,
            nonlin_first=False,
        )
        # This is the same He initialization hook used by nnU-Net when it
        # creates the architecture from a plans file.
        self.network.apply(self.network.initialize)

    def _pad_to_valid_shape(self, tensor: torch.Tensor):
        before_after = []
        for size in tensor.shape[-3:]:
            total = (-int(size)) % self.required_divisibility
            before = total // 2
            before_after.append((before, total - before))
        pad = tuple(
            value
            for pair in reversed(before_after)
            for value in pair
        )
        return F.pad(tensor, pad, mode="constant", value=0), tuple(before_after)

    @staticmethod
    def _crop_to_original_shape(
        tensor: torch.Tensor,
        padding: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
        spatial_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        slices = tuple(
            slice(before, before + size)
            for (before, _), size in zip(padding, spatial_shape)
        )
        return tensor[(slice(None), slice(None), *slices)]

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5:
            raise ValueError(f"expected B,C,D,H,W input, got shape {tuple(image.shape)}")
        if image.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} MRI channels, got {image.shape[1]}"
            )
        spatial_shape = tuple(int(value) for value in image.shape[-3:])
        padded, padding = self._pad_to_valid_shape(image)
        logits = self.network(padded)
        return self._crop_to_original_shape(logits, padding, spatial_shape)

__all__ = ["NNUNetV2PlainConv3D"]
