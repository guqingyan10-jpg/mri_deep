"""
Trainer Class for BraTS2020 Model Training.
Extracted from: MultiModel XAI Brats2020.ipynb (cell 39)

Handles: training loop, validation, checkpointing, early stopping,
         ReduceLROnPlateau scheduling, gradient accumulation.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output

from data.dataset import get_dataloader
from training.metrics import Meter


class Trainer():
    """
    Factory for training proccess.

    Args:
        net:                 neural network for mask prediction.
        dataset:             BratsDataset class reference.
        criterion:           loss function (e.g., BCEDiceLoss, DiceCEBoundaryLoss).
        lr:                  learning rate (default: 5e-4).
        accumulation_steps:  gradient accumulation steps (default: 4).
        batch_size:          data batch size (default: 1).
        fold:                fold number for cross-validation tracking.
        num_epochs:          maximum number of training epochs.
        path_to_csv:         path to tumourCSV.csv.
        model_type:          checkpoint directory path.
        display_plot:        if True, plot train history after last epoch.

        early_stopping_patience: number of epochs without val_loss
            improvement before stopping. Default 25.
            - All experiments MUST use the same patience for fair comparison.
            - Warm-start fine-tuning typically converges in 10-30 epochs,
              so 25 is a safe default that catches convergence while
              preventing excessive overfitting.
            - The monitor metric is always val_loss (BCEDiceLoss or
              equivalent combined loss).

        min_delta: minimum absolute improvement in val_loss to count
            as a meaningful improvement. Default 1e-4.
    """
    def __init__(self,
                 net: nn.Module,
                 dataset: torch.utils.data.Dataset,
                 criterion: nn.Module,
                 lr: float,
                 accumulation_steps: int,
                 batch_size: int,
                 fold: int,
                 num_epochs: int,
                 path_to_csv: str,
                 model_type: str,
                 display_plot: bool = True,
                 early_stopping_patience: int = 25,
                 min_delta: float = 1e-4,
                ):

        """Initialization."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("device:", self.device)
        self.display_plot = display_plot
        self.net = net
        self.net = self.net.to(self.device)
        self.criterion = criterion
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min",
                                           patience=2)
        self.accumulation_steps = accumulation_steps // batch_size
        self.phases = ["train", "valid"]
        self.num_epochs = num_epochs
        self.model_type = model_type
        self.epoch_value = self.check_epoch_number(self.model_type)

        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.epochs_without_improvement = 0
        self.best_epoch = 0

        self.dataloaders = {
            phase: get_dataloader(
                dataset = dataset,
                path_to_csv = path_to_csv,
                phase = phase,
                fold = fold,
                batch_size = batch_size,
                num_workers = 0
            )
            for phase in self.phases
        }

        self.best_loss = float("inf")

        # calculating the list of losses for both train & validation phases
        self.losses = {phase: [] for phase in self.phases}

        # calculating the dice scores for both train & validation phases
        self.dice_scores = {phase: [] for phase in self.phases}

        # calculating the jaccard scores for both train & validation phases
        self.jaccard_scores = {phase: [] for phase in self.phases}

        # calculating the time for both train & validation phases
        self.time = {phase: [] for phase in self.phases}

        # --- Restore history from previous training if resuming ---
        if self.epoch_value > 0:
            self._restore_history()

    def _restore_history(self):
        """Load previous training logs so train_log.csv is not overwritten."""
        log_path = f"{self.model_type}/train_log.csv"
        if not os.path.exists(log_path):
            return
        try:
            old_log = pd.read_csv(log_path)
            # Map old column names back to our dicts
            phase_map = {'train': 'train', 'valid': 'valid'}
            metric_map = {
                '_loss': 'losses',
                '_dice': 'dice_scores',
                '_jaccard': 'jaccard_scores',
                '_time': 'time',
            }
            for col in old_log.columns:
                for phase_key in ['train', 'valid']:
                    for suffix, attr in metric_map.items():
                        if col == phase_key + suffix:
                            values = old_log[col].dropna().tolist()
                            if len(values) > 0:
                                getattr(self, attr)[phase_key] = values

            # Restore best_loss from the loaded valid losses
            if len(self.losses['valid']) > 0:
                self.best_loss = min(self.losses['valid'])
                self.best_epoch = self.losses['valid'].index(self.best_loss)
                print(f"Restored training history: {len(self.losses['train'])} epochs, "
                      f"best val_loss={self.best_loss:.6f} at epoch {self.best_epoch}")
        except Exception as e:
            print(f"[WARN] Could not restore training history: {e}")

    def _compute_loss_and_outputs(self,
                                  images: torch.Tensor,
                                  targets: torch.Tensor):
        images = images.to(self.device)
        targets = targets.to(self.device)

        # making images predictions symmetric using logits
        logits = self.net(images)

        # calculating the loss bce loss / dice loss / jaccard loss / combined loss
        # as defined calcluating the mean square error loss
        loss = self.criterion(logits, targets)
        return loss, logits

    def _do_epoch(self, epoch: int, phase: str):
        start_time = time.time()
        meter = Meter()
        dataloader = self.dataloaders[phase]

        total_batches = len(dataloader)
        running_loss = 0.0

        # Initialize tqdm progress bar
        progress_bar = tqdm(dataloader, desc=f"{phase} epoch: {epoch}", unit="batch", dynamic_ncols=True)

        self.net.train() if phase == "train" else self.net.eval()

        for itr, data_batch in enumerate(progress_bar):
            images, targets = data_batch['image'], data_batch['mask']


            # BCEDiceLoss & raw prediction( logits ) are calculated

            loss, logits = self._compute_loss_and_outputs(images, targets)
            loss = loss / self.accumulation_steps

            if phase == "train":
                # Backpropagating the losses generated to train the Unet
                loss.backward()

                # if a certain no. is reached then all the gradient accumulated will be given to the optimizer & it gets trained
                # after giving, gradient gets reset to 0
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            running_loss += loss.item()
            progress_bar.set_postfix({"loss": running_loss / (itr + 1)})  # Update loss in progress bar
            meter.update(logits.detach().cpu(), targets.detach().cpu())

        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        epoch_dice, epoch_iou = meter.get_metrics()

        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(epoch_dice)

        self.jaccard_scores[phase].append(epoch_iou)


        # self.haus_scores[phase].append(epoch_haus)
        end_time = time.time()

        total_time = end_time - start_time

        total_time = round(total_time, 2)
        self.time[phase].append(total_time)
        return epoch_loss

    def run(self, check_path):
        epoch = self.epoch_value

        for epoch in range(int(self.epoch_value) + 1, self.num_epochs):
            self._do_epoch(epoch, "train")
            with torch.no_grad():
                val_loss = self._do_epoch(epoch, "valid")
                print(f"BCEDiceLoss for epoch {epoch} is : " , val_loss )
                self.scheduler.step(val_loss)
            if self.display_plot and epoch == self.num_epochs:
                self._plot_train_history()

            # --- Early Stopping Check ---
            # Only counts as improvement if val_loss drops by at least min_delta
            if val_loss < (self.best_loss - self.min_delta):
                print(f"\n{'#'*20}\nSaved new checkpoint\n{'#'*20}\n")
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0

                # Remove old best_model and save new one
                checkpoint_dir = check_path
                all_files = os.listdir(checkpoint_dir)
                best_model_current = [file for file in all_files if file.startswith("best_model_")]
                for best_model in best_model_current:
                    os.remove(checkpoint_dir + "/" + best_model)
                torch.save(self.net.state_dict(), f"{self.model_type}/best_model_{epoch}.pth")
            else:
                self.epochs_without_improvement += 1

            if epoch % 1 == 0:
                self._save_train_history(epoch)
            print()

            # --- Early Stopping Trigger ---
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n{'='*60}")
                print(f"EARLY STOPPING triggered at epoch {epoch}")
                print(f"Best val_loss: {self.best_loss:.6f} at epoch {self.best_epoch}")
                print(f"No improvement for {self.epochs_without_improvement} epochs")
                print(f"Best model saved at: {self.model_type}/best_model_{self.best_epoch}.pth")
                print(f"{'='*60}\n")
                break

        # Final save at exit (either max epochs or early stop)
        self._save_train_history(epoch)

    def _plot_train_history(self):
        data = [self.losses, self.dice_scores, self.jaccard_scores]
        colors = ['deepskyblue', "crimson"]
        labels = [
            f"""
            train loss {self.losses['train'][-1]}
            val loss {self.losses['val'][-1]}
            """,

            f"""
            train dice score {self.dice_scores['train'][-1]}
            val dice score {self.dice_scores['val'][-1]}
            """,

            f"""
            train jaccard score {self.jaccard_scores['train'][-1]}
            val jaccard score {self.jaccard_scores['val'][-1]}
            """
        ]

        clear_output(True)

        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        for i, ax in enumerate(axes):
            ax.plot(data[i]['val'], c=colors[0], label="val")
            ax.plot(data[i]['train'], c=colors[-1], label="train")
            ax.set_title(labels[i])
            ax.legend(loc="upper right")

        plt.tight_layout()
        plt.show()

    def load_pretrain_model(self,
                             state_path: str):

        pretrain = torch.load(state_path, weights_only=False)
        if isinstance(pretrain, dict):
            self.net.load_state_dict(pretrain)
        else:
            self.net.load_state_dict(pretrain.state_dict())
        print("Pretrain model loaded")

    def check_epoch_number(self, checkpoint_dir):
        value_of_hash = 0
        # Get a list of all files in the checkpoint directory
        all_files = os.listdir(checkpoint_dir)

        # Filter the files to get only the model checkpoint files
        model_checkpoint_files = [file for file in all_files if file.startswith("last_epoch_model")]

        # Sort the model checkpoint files based on their names (assuming they contain the epoch number)

        if model_checkpoint_files:

            sorted_file_names = sorted(model_checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

            # Get the latest model checkpoint file
            latest_checkpoint_file = sorted_file_names[-1]


            # Construct the full path to the latest model checkpoint
            pretrained_model_path = os.path.join(checkpoint_dir, latest_checkpoint_file)
            latest = pretrained_model_path.split("_")
            value_of_hash = latest[-1].split(".")[0]
            return value_of_hash
        else:
            return value_of_hash

    def _save_train_history(self, epoch):
        """writing model weights and training logs to files."""
        torch.save(self.net.state_dict(),
                   f"{self.model_type}/last_epoch_model_{epoch}.pth")

        logs_ = [self.losses, self.dice_scores, self.jaccard_scores, self.time]

        log_names_ = ["_loss", "_dice", "_jaccard", "_time"]
        logs = [logs_[i][key] for i in list(range(len(logs_)))
                         for key in logs_[i]]
        log_names = [key+log_names_[i]
                     for i in list(range(len(logs_)))
                     for key in logs_[i]
                    ]
        pd.DataFrame(
            dict(zip(log_names, logs))
        ).to_csv(f"{self.model_type}/train_log.csv", index=False)
