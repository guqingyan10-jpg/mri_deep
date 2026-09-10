"""Thin trainer adapter for UCSF models with an auxiliary boundary head."""

from training.trainer import Trainer


class UCSFBoundaryTrainer(Trainer):
    """Optimize both heads while reporting metrics from segmentation logits."""

    def _compute_loss_and_outputs(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        segmentation, boundary = self.net(images)
        loss = self.criterion((segmentation, boundary), targets)
        return loss, segmentation
