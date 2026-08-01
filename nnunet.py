"""
nnUNet3d for BraTS2020 Brain Tumor Segmentation
================================================
A new baseline model for comparison with UNet, ResUNet, AttUNet.

Architecture differences from existing models (ALL use same Trainer):
---------------------------------------------------------------------------
| Component        | UNet3d        | ResUNet3d     | AttUNet3d     | nnUNet3d      |
|------------------|---------------|---------------|---------------|---------------|
| Normalization    | GroupNorm     | GroupNorm     | GroupNorm     | InstanceNorm  |
| Activation       | ReLU          | ReLU          | ReLU          | LeakyReLU     |
| Downsampling     | MaxPool3d     | MaxPool3d     | MaxPool3d     | Strided Conv  |
| Upsampling       | Trilinear     | Trilinear     | Trilinear     | ConvTranspose |
| Skip Connection  | Concat        | Concat+Res    | Concat+Att    | Concat        |
| Residual         | No            | Yes           | No            | No            |
| Attention        | No            | No            | CBAM+AG       | No            |
---------------------------------------------------------------------------

Training fairness guarantee:
- Same Trainer class (exact same code)
- Same lr=5e-4
- Same BCEDiceLoss
- Same Adam optimizer
- Same ReduceLROnPlateau scheduler (patience=2)
- Same batch_size=1, accumulation_steps=4
- Same n_channels=24 (matching parameter budget across models)
- Same num_epochs=200
- Same seed=55, same data split

Reference: Isensee et al., "nnU-Net: a self-configuring method for deep
learning-based biomedical image segmentation", Nature Methods 2021.
https://doi.org/10.1038/s41592-020-01008-z

Author: Extended from MultiModel XAI Brats2020 notebook
"""

import os
import time
import gc
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

import nibabel as nib
import albumentations as A
from albumentations import Compose
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# Global Configuration (IDENTICAL to original notebook)
# ============================================================

class GlobalConfig:
    root_dir = r'/root/autodl-tmp'
    train_root_dir = r'/root/autodl-tmp/brats_project/MICCAI_BraTS2020_TrainingData/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    test_root_dir = r'/root/autodl-tmp/test_df'
    path_to_csv = 'tumourCSV.csv'

    UNet_checkpoint_dir = r"/root/autodl-tmp/UNet_model"
    ResUNet_checkpoint_dir = r"/root/autodl-tmp/ResUNet_model"
    Att_checkpoint_dir = r"/root/autodl-tmp/AttUNet_model"
    nnUNet_checkpoint_dir = r"/root/autodl-tmp/nnUNet_model"

    train_logs_path = r'/root/autodl-tmp/UNet_model/train_log.csv'
    ResUNet_train_logs_path = r'/root/autodl-tmp/ResUNet_model/train_log.csv'
    AttUNet_train_logs_path = r'/root/autodl-tmp/AttUNet_model/train_log.csv'
    nnUNet_train_logs_path = r'/root/autodl-tmp/nnUNet_model/train_log.csv'

    seed = 55


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


config = GlobalConfig()
seed_everything(config.seed)


# ============================================================
# Dataset & DataLoader (IDENTICAL to original notebook)
# ============================================================

def get_augmentations(phase):
    list_transforms = []
    list_trfms = Compose(list_transforms, is_check_shapes=False)
    return list_trfms


def get_dataloader(
    dataset: torch.utils.data.Dataset,
    path_to_csv: str,
    phase: str,
    fold: int = 0,
    batch_size: int = 1,
    num_workers: int = 0
):
    df = pd.read_csv(path_to_csv)
    train_df_copy, test_df = train_test_split(df, test_size=0.3, random_state=10, shuffle=True)
    train_df_copy, test_df_copy = train_df_copy.reset_index(drop=True), test_df.reset_index(drop=True)
    test_df = test_df_copy.iloc[len(test_df_copy) * 2 // 3:].reset_index(drop=True)
    val_df = test_df_copy.iloc[:len(test_df_copy) * 2 // 3].reset_index(drop=True)

    if phase != 'test':
        if phase == "train":
            df = train_df_copy
        elif phase == "valid":
            df = val_df
        dataset = dataset(df, phase)
    else:
        df = test_df
        dataset = dataset(df, phase)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )
    return dataloader


class BratsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, phase: str = "test", is_resize: bool = True):
        self.df = df
        self.phase = phase
        self.augmentations = get_augmentations(phase)
        self.data_types = ['_flair.nii', '_t1.nii', '_t1ce.nii', '_t2.nii']
        self.is_resize = is_resize

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        id_ = self.df.loc[idx, 'Brats20ID']
        root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0]

        images = []
        for data_type in self.data_types:
            img_path = os.path.join(root_path, id_ + data_type)
            img = self.load_img(img_path)
            if self.is_resize:
                img = self.resize(img)
            img = self.normalize(img)
            images.append(img)

        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))

        mask_path = os.path.join(root_path, id_ + "_seg.nii")
        mask = self.load_img(mask_path)
        if self.is_resize:
            mask = self.resize(mask)
        mask = self.preprocess_mask_labels(mask)

        augmented = self.augmentations(image=img.astype(np.float32),
                                       mask=mask.astype(np.float32))
        img = augmented['image']
        mask = augmented['mask']

        return {
            "Id": id_,
            "image": img,
            "mask": mask,
        }

    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

    def resize(self, data: np.ndarray):
        data = data[40:210, 40:210, 20:120]
        return data

    def preprocess_mask_labels(self, mask: np.ndarray):
        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))
        return mask


# ============================================================
# nnU-Net Building Blocks
# ============================================================
#
# Key design decisions (from Isensee et al. 2021):
#
# 1. InstanceNorm3d instead of GroupNorm/BatchNorm
#    Why: Invariant to batch size. Critical for 3D medical imaging
#    where batch_size=1 is common due to GPU memory constraints.
#
# 2. LeakyReLU instead of ReLU
#    Why: Prevents "dying ReLU" where neurons become permanently
#    inactive. negative_slope=1e-2 allows small negative gradients.
#
# 3. Strided Conv3d for downsampling (NO MaxPool)
#    Why: Learnable downsampling adapts to the data, unlike fixed
#    MaxPool which discards 87.5% of spatial information per 2x2x2 window.
#    Conv3d(kernel=3, stride=2, padding=1) halves spatial dims.
#
# 4. ConvTranspose3d for upsampling (NO trilinear interpolation)
#    Why: Learnable upsampling recovers finer structural details.
#    Trilinear uses fixed weights that cannot adapt to the data.
#


class nnDoubleConv(nn.Module):
    """
    (Conv3D -> InstanceNorm3d -> LeakyReLU) * 2

    Core building block of nnU-Net.
    Replaces GroupNorm+ReLU with InstanceNorm+LeakyReLU.

    Why InstanceNorm for 3D medical images?
    - Batch_size=1 makes BatchNorm unstable
    - GroupNorm divides channels into arbitrary groups
    - InstanceNorm treats each channel independently: most stable for small batches
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
    Strided Convolution for downsampling (NO MaxPool).

    Conv3d(stride=2) halves spatial dims and doubles channels.
    Learnable weights adapt to the data distribution.

    Reference: Springenberg et al., "Striving for Simplicity:
    The All Convolutional Net", ICLR 2015.
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
    Transposed Convolution for upsampling + nnDoubleConv.

    Channel math (matches original Up class convention exactly):
    -------------------------------------------------------------
    Original Up:
      Upsample(trilinear) preserves channels
      x1(deeper=C) -> Upsample -> C -> cat(skip=C, up=C) -> 2C -> DoubleConv(2C, out)

    nnUp:
      ConvTranspose3d preserves channels, doubles spatial
      x1(deeper=C) -> ConvTranspose3d(C,C,stride=2) -> C -> cat(skip=C, up=C) -> 2C -> nnDoubleConv(2C, out)

    Same __init__(in_channels, out_channels) interface as Up/ResUp/AttUp.
    """
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: channels AFTER skip concat (= 2 * deeper_channels)
            out_channels: desired output channels
        """
        super().__init__()
        deeper_channels = in_channels // 2
        self.up = nn.ConvTranspose3d(
            deeper_channels, deeper_channels,
            kernel_size=2, stride=2
        )
        self.conv = nnDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: deeper feature (C channels, smaller spatial)
            x2: skip feature (C channels, larger spatial)
        """
        x1 = self.up(x1)

        # Size matching (same logic as original Up)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2,
                         diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Out(nn.Module):
    """1x1x1 Conv3d output layer — IDENTICAL across all models."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ============================================================
# nnUNet3d -- The Full Architecture
# ============================================================

class nnUNet3d(nn.Module):
    """
    nnU-Net 3D for Brain Tumor Segmentation.

    Input:  (B, 4,  128, 128, 128) -- 4 MRI modalities
    Output: (B, 3,  128, 128, 128) -- WT, TC, ET

    Encoder (strided conv downsampling):
        conv  (4 -> 24):  nnDoubleConv             -> (24,  128^3)
        enc1  (24 -> 48): StridedConv + nnDouble   -> (48,   64^3)
        enc2  (48 -> 96): StridedConv + nnDouble   -> (96,   32^3)
        enc3  (96 -> 192):StridedConv + nnDouble   -> (192,  16^3)
        enc4  (192-> 192):StridedConv + nnDouble   -> (192,   8^3) [bottleneck]

    Decoder (ConvTranspose3d upsampling):
        dec1  (384-> 96): ConvTranspose + skip     -> (96,   16^3)
        dec2  (192-> 48): ConvTranspose + skip     -> (48,   32^3)
        dec3  (96 -> 24): ConvTranspose + skip     -> (24,   64^3)
        dec4  (48 -> 24): ConvTranspose + skip     -> (24,  128^3)
        out   (24 -> 3):  1x1x1 Conv3d             -> (3,   128^3)

    Key differences from UNet3d:
        | Component      | UNet3d            | nnUNet3d           |
        |----------------|-------------------|--------------------|
        | Normalization  | GroupNorm(8)      | InstanceNorm       |
        | Activation     | ReLU              | LeakyReLU(0.01)    |
        | Downsampling   | MaxPool3d         | Conv3d(stride=2)   |
        | Upsampling     | Upsample(trilinear)| ConvTranspose3d   |
        | Base channels  | 24                | 24 (matched)       |

    Approx parameter count (n_channels=24): ~2.0M (comparable to UNet3d ~1.9M)

    Reference: Isensee et al., Nature Methods 2021.
    """
    def __init__(self, in_channels=4, n_classes=3, n_channels=24):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        # Encoder
        self.conv = nnDoubleConv(in_channels, n_channels)
        self.enc1 = nnDown(n_channels, 2 * n_channels)
        self.enc2 = nnDown(2 * n_channels, 4 * n_channels)
        self.enc3 = nnDown(4 * n_channels, 8 * n_channels)
        self.enc4 = nnDown(8 * n_channels, 8 * n_channels)

        # Decoder (same in_channels convention as original Up/ResUp/AttUp)
        self.dec1 = nnUp(16 * n_channels, 4 * n_channels)
        self.dec2 = nnUp(8 * n_channels, 2 * n_channels)
        self.dec3 = nnUp(4 * n_channels, n_channels)
        self.dec4 = nnUp(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

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


# ============================================================
# Metrics (IDENTICAL to original notebook)
# ============================================================

def dice_coef_metric(probabilities, truth, treshold=0.5, eps=1e-9):
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = 2.0 * (truth_ * prediction).sum()
        union = truth_.sum() + prediction.sum()
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)


def jaccard_coef_metric(probabilities, truth, treshold=0.5, eps=1e-9):
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = (prediction * truth_).sum()
        union = (prediction.sum() + truth_.sum()) - intersection + eps
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)


class Meter:
    def __init__(self, treshold=0.5):
        self.threshold = treshold
        self.dice_scores = []
        self.iou_scores = []

    def update(self, logits, targets):
        probs = torch.sigmoid(logits)
        dice = dice_coef_metric(probs, targets, self.threshold)
        iou = jaccard_coef_metric(probs, targets, self.threshold)
        self.dice_scores.append(dice)
        self.iou_scores.append(iou)

    def get_metrics(self):
        return np.mean(self.dice_scores), np.mean(self.iou_scores)


def dice_coef_metric_per_classes(probabilities, truth, treshold=0.33, eps=1e-9,
                                  classes=['WT', 'TC', 'ET']):
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= treshold).astype(np.float32)
    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = 2.0 * (truth_ * prediction).sum()
            union = truth_.sum() + prediction.sum()
            if truth_.sum() == 0 and prediction.sum() == 0:
                scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)
    return scores


def jaccard_coef_metric_per_classes(probabilities, truth, treshold=0.33, eps=1e-9,
                                     classes=['WT', 'TC', 'ET']):
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= treshold).astype(np.float32)
    for i in range(num):
        for class_ in range(num_classes):
            prediction = predictions[i][class_]
            truth_ = truth[i][class_]
            intersection = (prediction * truth_).sum()
            union = (prediction.sum() + truth_.sum()) - intersection + eps
            if truth_.sum() == 0 and prediction.sum() == 0:
                scores[classes[class_]].append(1.0)
            else:
                scores[classes[class_]].append((intersection + eps) / union)
    return scores


# ============================================================
# Loss Functions (IDENTICAL to original notebook)
# ============================================================

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        num = targets.size(0)
        probability = torch.sigmoid(logits)
        probability = probability.view(num, -1)
        targets = targets.view(num, -1)
        intersection = 2.0 * (probability * targets).sum()
        union = probability.sum() + targets.sum()
        dice_score = (intersection + self.eps) / union
        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """BCE + Dice loss -- SAME loss used by UNet/ResUNet/AttUNet."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        assert logits.shape == targets.shape
        return self.bce(logits, targets) + self.dice(logits, targets)


# ============================================================
# Trainer Class (IDENTICAL to original notebook)
# ============================================================
# This is the EXACT same Trainer used for UNet3d, ResUNet3d, AttUNet3d.
# Using identical Trainer guarantees: same optimizer, scheduler,
# accumulation logic, checkpoint saving, and logging.
# The ONLY variable across models is the architecture.


class Trainer:
    def __init__(self, net, dataset, criterion, lr, accumulation_steps,
                 batch_size, fold, num_epochs, path_to_csv, model_type,
                 display_plot=True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("device:", self.device)
        self.display_plot = display_plot
        self.net = net.to(self.device)
        self.criterion = criterion
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=2)
        self.accumulation_steps = accumulation_steps // batch_size
        self.phases = ["train", "valid"]
        self.num_epochs = num_epochs
        self.model_type = model_type
        self.epoch_value = self._check_epoch_number(self.model_type)

        self.dataloaders = {
            phase: get_dataloader(
                dataset=dataset, path_to_csv=path_to_csv,
                phase=phase, fold=fold,
                batch_size=batch_size, num_workers=0
            )
            for phase in self.phases
        }

        self.best_loss = float("inf")
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.jaccard_scores = {phase: [] for phase in self.phases}
        self.time = {phase: [] for phase in self.phases}

    def _compute_loss_and_outputs(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        logits = self.net(images)
        loss = self.criterion(logits, targets)
        return loss, logits

    def _do_epoch(self, epoch, phase):
        start_time = time.time()
        meter = Meter()
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"{phase} epoch: {epoch}",
                           unit="batch", dynamic_ncols=True)
        self.net.train() if phase == "train" else self.net.eval()

        for itr, data_batch in enumerate(progress_bar):
            images, targets = data_batch['image'], data_batch['mask']
            loss, logits = self._compute_loss_and_outputs(images, targets)
            loss = loss / self.accumulation_steps

            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            running_loss += loss.item()
            progress_bar.set_postfix({"loss": running_loss / (itr + 1)})
            meter.update(logits.detach().cpu(), targets.detach().cpu())

        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        epoch_dice, epoch_iou = meter.get_metrics()

        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(epoch_dice)
        self.jaccard_scores[phase].append(epoch_iou)
        self.time[phase].append(round(time.time() - start_time, 2))
        return epoch_loss

    def run(self, check_path):
        for epoch in range(int(self.epoch_value) + 1, self.num_epochs):
            self._do_epoch(epoch, "train")
            with torch.no_grad():
                val_loss = self._do_epoch(epoch, "valid")
                print(f"BCEDiceLoss for epoch {epoch} is : ", val_loss)
                self.scheduler.step(val_loss)

            if val_loss < self.best_loss:
                print(f"\n{'#'*20}\nSaved new checkpoint\n{'#'*20}\n")
                self.best_loss = val_loss
                checkpoint_dir = check_path
                all_files = os.listdir(checkpoint_dir)
                for f in [x for x in all_files if x.startswith("best_model_")]:
                    os.remove(os.path.join(checkpoint_dir, f))
                torch.save(self.net.state_dict(),
                          f"{self.model_type}/best_model_{epoch}.pth")

            if epoch % 1 == 0:
                self._save_train_history(epoch)
            print()
        self._save_train_history(epoch)

    def _plot_train_history(self):
        data = [self.losses, self.dice_scores, self.jaccard_scores]
        colors = ['deepskyblue', "crimson"]
        labels = [
            f"train loss {self.losses['train'][-1]}\nval loss {self.losses['valid'][-1]}",
            f"train dice {self.dice_scores['train'][-1]}\nval dice {self.dice_scores['valid'][-1]}",
            f"train jaccard {self.jaccard_scores['train'][-1]}\nval jaccard {self.jaccard_scores['valid'][-1]}"
        ]
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        for i, ax in enumerate(axes):
            ax.plot(data[i]['valid'], c=colors[0], label="valid")
            ax.plot(data[i]['train'], c=colors[-1], label="train")
            ax.set_title(labels[i])
            ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    def load_pretrain_model(self, state_path):
        pretrain = torch.load(state_path, weights_only=False)
        if isinstance(pretrain, dict):
            self.net.load_state_dict(pretrain)
        else:
            self.net.load_state_dict(pretrain.state_dict())
        print("Pretrain model loaded")

    def _check_epoch_number(self, checkpoint_dir):
        if not os.path.exists(checkpoint_dir):
            return 0
        all_files = os.listdir(checkpoint_dir)
        models = [f for f in all_files if f.startswith("last_epoch_model")]
        if models:
            latest = sorted(models, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
            return latest.split("_")[-1].split(".")[0]
        return 0

    def _save_train_history(self, epoch=None):
        torch.save(self.net.state_dict(),
                   f"{self.model_type}/last_epoch_model_{epoch}.pth")
        logs_ = [self.losses, self.dice_scores, self.jaccard_scores, self.time]
        log_names_ = ["_loss", "_dice", "_jaccard", "_time"]
        logs = [logs_[i][key] for i in range(len(logs_)) for key in logs_[i]]
        log_names = [key + log_names_[i] for i in range(len(logs_)) for key in logs_[i]]
        pd.DataFrame(dict(zip(log_names, logs))).to_csv(
            f"{self.model_type}/train_log.csv", index=False)


# ============================================================
# Evaluation Functions
# ============================================================

def compute_scores_per_classes(model, dataloader, classes=['WT', 'TC', 'ET']):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dice_scores_per_classes = {key: list() for key in classes}
    iou_scores_per_classes = {key: list() for key in classes}

    with torch.no_grad():
        for data in dataloader:
            imgs, targets = data['image'], data['mask']
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            logits = logits.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

            dice_scores = dice_coef_metric_per_classes(logits, targets)
            iou_scores = jaccard_coef_metric_per_classes(logits, targets)
            for key in dice_scores.keys():
                dice_scores_per_classes[key].extend(dice_scores[key])
            for key in iou_scores.keys():
                iou_scores_per_classes[key].extend(iou_scores[key])

    return dice_scores_per_classes, iou_scores_per_classes


def compute_scores_per_classes_mean(model, dataloader, classes=['WT', 'TC', 'ET']):
    dice_scores_per_classes, iou_scores_per_classes = compute_scores_per_classes(
        model, dataloader, classes)
    return ({key: np.mean(values) for key, values in dice_scores_per_classes.items()},
            {key: np.mean(values) for key, values in iou_scores_per_classes.items()})


def print_metrics_table(dice_means, iou_means, model_name="Model"):
    print(f"\n{'='*60}")
    print(f"  {model_name} - Evaluation Results")
    print(f"{'='*60}")
    print(f"{'Class':<10} {'Dice Score':<15} {'IoU/Jaccard':<15}")
    print(f"{'-'*40}")
    for cls in dice_means.keys():
        print(f"{cls:<10} {dice_means[cls]:<15.4f} {iou_means[cls]:<15.4f}")
    print(f"{'-'*40}")
    avg_dice = np.mean(list(dice_means.values()))
    avg_iou = np.mean(list(iou_means.values()))
    print(f"{'Average':<10} {avg_dice:<15.4f} {avg_iou:<15.4f}")
    print(f"{'='*60}\n")
    return avg_dice, avg_iou


def check_exist(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    all_files = os.listdir(checkpoint_dir)
    models = [f for f in all_files if f.startswith("best_model_")]
    if models:
        latest = sorted(models, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        return os.path.join(checkpoint_dir, latest)
    return None


# ============================================================
# MAIN TRAINING SCRIPT
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Training nnUNet3d (InstanceNorm + LeakyReLU + StridedConv)")
    print("Hyperparameters: IDENTICAL to UNet/ResUNet/AttUNet")
    print(f"  lr={5e-4}, BCEDiceLoss, Adam, ReduceLROnPlateau")
    print(f"  batch_size=1, accumulation_steps=4, num_epochs=200")
    print(f"  n_channels=24 (matched parameter budget)")
    print("=" * 60)

    model = nnUNet3d(in_channels=4, n_classes=3, n_channels=24).to('cuda')

    trainer = Trainer(
        net=model,
        dataset=BratsDataset,
        criterion=BCEDiceLoss(),
        lr=5e-4,
        accumulation_steps=4,
        batch_size=1,
        fold=0,
        num_epochs=200,
        path_to_csv=config.path_to_csv,
        model_type=config.nnUNet_checkpoint_dir
    )

    os.makedirs(config.nnUNet_checkpoint_dir, exist_ok=True)

    # Resume from checkpoint if exists
    if check_exist(config.nnUNet_checkpoint_dir) is not None:
        trainer.load_pretrain_model(check_exist(config.nnUNet_checkpoint_dir))

    # Load previous logs if resuming
    if os.path.exists(config.nnUNet_train_logs_path):
        train_logs = pd.read_csv(config.nnUNet_train_logs_path)
    else:
        cols = ["train_loss", "valid_loss", "train_dice", "valid_dice",
                "train_jaccard", "valid_jaccard", "train_time", "valid_time"]
        train_logs = pd.DataFrame({c: [] for c in cols})

    trainer.losses["train"] = train_logs.loc[:, "train_loss"].to_list()
    trainer.losses["valid"] = train_logs.loc[:, "valid_loss"].to_list()
    trainer.dice_scores["train"] = train_logs.loc[:, "train_dice"].to_list()
    trainer.dice_scores["valid"] = train_logs.loc[:, "valid_dice"].to_list()
    trainer.jaccard_scores["train"] = train_logs.loc[:, "train_jaccard"].to_list()
    trainer.jaccard_scores["valid"] = train_logs.loc[:, "valid_jaccard"].to_list()
    trainer.time["train"] = train_logs.loc[:, "train_time"].to_list()
    trainer.time["valid"] = train_logs.loc[:, "valid_time"].to_list()

    trainer.run(config.nnUNet_checkpoint_dir)

    # ----------------------------------------------------------
    # Evaluation on validation set
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION: nnUNet3d on Validation Set")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataloader = get_dataloader(
        dataset=BratsDataset, path_to_csv=config.path_to_csv,
        phase='valid', batch_size=1
    )

    nnunet_model = nnUNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
    best_path = check_exist(config.nnUNet_checkpoint_dir)
    if best_path:
        nnunet_model.load_state_dict(torch.load(best_path, map_location=device))
        nnunet_model.eval()

    dice_means, iou_means = compute_scores_per_classes_mean(nnunet_model, val_dataloader)
    print_metrics_table(dice_means, iou_means, "nnUNet3d")
