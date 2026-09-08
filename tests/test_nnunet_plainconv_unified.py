from pathlib import Path

import torch
import pytest

from models.nnunet_plainconv3d import NNUNetV2PlainConv3D
from scripts.train_nnunet_plainconv_unified import (
    build_arg_parser,
    validate_registered_protocol,
)


def test_plainconv_adapter_preserves_odd_brats_spatial_shape():
    """Catches missing pad/crop logic at the four official downsampling stages."""
    model = NNUNetV2PlainConv3D(
        in_channels=4,
        n_classes=3,
        features_per_stage=(8, 16, 24, 32, 40),
    ).eval()
    image = torch.randn(1, 4, 17, 33, 35)

    with torch.no_grad():
        logits = model(image)

    assert logits.shape == (1, 3, 17, 33, 35)
    assert torch.isfinite(logits).all()


def test_plainconv_uses_official_backend_and_raw_logits():
    """Catches replacement by another hand-written surrogate or output sigmoid."""
    model = NNUNetV2PlainConv3D(
        in_channels=4,
        n_classes=3,
        features_per_stage=(8, 16, 24, 32, 40),
    )

    assert model.network.__class__.__module__.startswith(
        "dynamic_network_architectures.architectures.unet"
    )
    assert model.network.__class__.__name__ == "PlainConvUNet"
    assert model.network.decoder.deep_supervision is False
    assert not any(isinstance(module, torch.nn.Sigmoid) for module in model.modules())


def test_training_entry_defaults_to_existing_baseline_protocol():
    """Catches accidental reuse of nnU-Net's SGD/0.01/1000-epoch recipe."""
    args = build_arg_parser().parse_args([])

    assert args.seed == 55
    assert args.lr == 5e-4
    assert args.epochs == 200
    assert args.accumulation_steps == 4
    assert args.batch_size == 1
    assert args.early_stopping_patience == 25
    assert args.min_delta == 1e-4
    assert args.checkpoint_dir == Path(
        "/root/autodl-tmp/nnUNet_v2_PlainConv_unified_model"
    )


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("--epochs", "201"),
        ("--lr", "0.001"),
        ("--batch-size", "2"),
        ("--accumulation-steps", "2"),
        ("--early-stopping-patience", "24"),
        ("--min-delta", "0.001"),
    ],
)
def test_registered_protocol_rejects_accidental_hyperparameter_changes(
    argument, value
):
    args = build_arg_parser().parse_args([argument, value])

    with pytest.raises(SystemExit, match="registered fair baseline"):
        validate_registered_protocol(args)
