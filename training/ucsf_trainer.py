"""UCSF trainers with bounded checkpoint storage."""

from pathlib import Path

from training.trainer import Trainer


class UCSFTrainer(Trainer):
    """Keep only the newest recovery checkpoint plus the best checkpoint."""

    def _save_train_history(self, epoch):
        super()._save_train_history(epoch)
        current = (Path(self.model_type) / f"last_epoch_model_{epoch}.pth").resolve()
        for checkpoint in Path(self.model_type).glob("last_epoch_model_*.pth"):
            if checkpoint.resolve() != current:
                checkpoint.unlink()


class UCSFBoundaryTrainer(UCSFTrainer):
    """Optimize both heads while reporting metrics from segmentation logits."""

    def _compute_loss_and_outputs(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        segmentation, boundary = self.net(images)
        loss = self.criterion((segmentation, boundary), targets)
        return loss, segmentation
