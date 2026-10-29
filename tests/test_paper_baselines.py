"""Check the pinned author architecture through the project's training adapter."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
pytest.importorskip("einops")

from losses.basics import BCEDiceLoss
from models.paper_baselines import PaperBaseline


def test_doubleblock_padding_region_order_and_training_step(tmp_path):
    # A missing external checkout also checks that the bundled author source
    # is sufficient; construction verifies the pinned source checksum.
    model = PaperBaseline("doubleblock", tmp_path / "no_external_checkout")
    model.eval()
    image = torch.randn(1, 4, 31, 29, 27)
    padded = torch.nn.functional.pad(image, (0, 5, 0, 3, 0, 1))
    with torch.no_grad():
        author_logits = model.architecture(padded)
        actual = model(image)
    expected = author_logits[:, [2, 1, 0], :31, :29, :27]
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (1, 3, 31, 29, 27)
    assert torch.isfinite(actual).all()

    # Run the real baseline loss and optimizer, not a replacement network.
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    prediction = model(torch.randn(1, 4, 32, 32, 32)).contiguous()
    targets = torch.zeros_like(prediction)
    targets[..., 12:20, 12:20, 12:20] = 1
    loss = BCEDiceLoss()(prediction, targets)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    parameter = next(p for p in model.parameters() if p.grad is not None)
    before = parameter.detach().clone()
    optimizer.step()
    assert not torch.equal(parameter.detach(), before)

    checkpoint = tmp_path / "best_model_001.pth"
    torch.save(model.state_dict(), checkpoint)
    restored = PaperBaseline("doubleblock", tmp_path / "no_external_checkout")
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
