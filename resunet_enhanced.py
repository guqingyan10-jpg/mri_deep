"""
ResUNet3d for BraTS2020 Brain Tumor Segmentation (Enhanced Version)
=========================================================
Extracted from MultiModel_XAI_Brats2020_HFF.ipynb
Modifications:
  - Loss: Dice + CE + Boundary Loss (replaces BCEDiceLoss)
  - Class weights: ET (highest) > TC > WT
  - Clean standalone structure matching original notebook style

Original Author: (from MultiModel XAI Brats2020 notebook)
"""

import os
import time
import gc
import numpy as np
import pandas as pd
from random import randint

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
# Global Configuration
# ============================================================

class GlobalConfig:
    # --- Paths (ADAPT THESE TO YOUR ENVIRONMENT) ---
    root_dir = r'/root/autodl-tmp'
    train_root_dir = r'/root/autodl-tmp/brats_project/MICCAI_BraTS2020_TrainingData/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    test_root_dir = r'/root/autodl-tmp/test_df'
    path_to_csv = 'tumourCSV.csv'

    # Checkpoint directories
    ResUNet_checkpoint_dir = r"/root/autodl-tmp/ResUNet_model"
    ResUNet_Enhanced_checkpoint_dir = r"/root/autodl-tmp/ResUNet_Enhanced_model"

    # Log paths
    ResUNet_train_logs_path = r'/root/autodl-tmp/ResUNet_model/train_log.csv'
    ResUNet_Enhanced_train_logs_path = r'/root/autodl-tmp/ResUNet_Enhanced_model/train_log.csv'

    seed = 55


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


config = GlobalConfig()
seed_everything(config.seed)

# ============================================================
# Dataset & DataLoader (cells 27, 243-358)
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

        # Load all 4 modalities
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

        # Load mask
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
        # WT: labels 1,2,4 -> 1
        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1

        # TC: labels 1,4 -> 1 (exclude edema=2)
        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        # ET: label 4 only
        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))
        return mask


# ============================================================
# Base Building Blocks (cell 42)
# ============================================================

class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2,
                         diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ============================================================
# ResUNet Building Blocks (cell 53)
# ============================================================

class ResBlock(nn.Module):
    """(GroupNorm -> ReLU -> Conv3D) * 2 + Residual"""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        s = self.conv(x)
        x = self.residual_block(x)
        return x + s


class ResDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            ResBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class ResUp(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ResBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2,
                         diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class FirstLayer(nn.Module):
    """First layer of ResUNet: Conv3D + Residual (no GroupNorm before first conv)"""
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        s = self.conv(x)
        x = self.residual_block(x)
        return x + s


class ResUNet3d(nn.Module):
    """3D ResUNet for Brain Tumor Segmentation.
    Architecture: ResBlock encoder-decoder with skip connections.
    Input:  (B, 4, 128, 128, 128)  — 4 MRI modalities
    Output: (B, 3, 128, 128, 128)  — WT, TC, ET
    """
    def __init__(self, in_channels=4, n_classes=3, n_channels=24):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        # Encoder
        self.conv = FirstLayer(in_channels, n_channels)          # 24
        self.enc1 = ResDown(n_channels, 2 * n_channels)          # 48
        self.enc2 = ResDown(2 * n_channels, 4 * n_channels)      # 96
        self.enc3 = ResDown(4 * n_channels, 8 * n_channels)      # 192
        self.enc4 = ResDown(8 * n_channels, 8 * n_channels)      # 192

        # Decoder
        self.dec1 = ResUp(16 * n_channels, 4 * n_channels)       # 192+192 -> 96
        self.dec2 = ResUp(8 * n_channels, 2 * n_channels)        # 96+96 -> 48
        self.dec3 = ResUp(4 * n_channels, n_channels)            # 48+48 -> 24
        self.dec4 = ResUp(2 * n_channels, n_channels)            # 24+24 -> 24
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
        return mask


# ============================================================
# Metrics (cell 37)
# ============================================================

def dice_coef_metric(probabilities: torch.Tensor,
                     truth: torch.Tensor,
                     treshold: float = 0.5,
                     eps: float = 1e-9) -> np.ndarray:
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert predictions.shape == truth.shape
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


def jaccard_coef_metric(probabilities: torch.Tensor,
                         truth: torch.Tensor,
                         treshold: float = 0.5,
                         eps: float = 1e-9) -> np.ndarray:
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert predictions.shape == truth.shape
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
    """Factory for storing and updating iou and dice scores."""
    def __init__(self, treshold: float = 0.5):
        self.threshold: float = treshold
        self.dice_scores: list = []
        self.iou_scores: list = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits)
        dice = dice_coef_metric(probs, targets, self.threshold)
        iou = jaccard_coef_metric(probs, targets, self.threshold)
        self.dice_scores.append(dice)
        self.iou_scores.append(iou)

    def get_metrics(self) -> np.ndarray:
        dice = np.mean(self.dice_scores)
        iou = np.mean(self.iou_scores)
        return dice, iou


def dice_coef_metric_per_classes(probabilities: np.ndarray,
                                   truth: np.ndarray,
                                   treshold: float = 0.33,
                                   eps: float = 1e-9,
                                   classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= treshold).astype(np.float32)
    assert predictions.shape == truth.shape
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


def jaccard_coef_metric_per_classes(probabilities: np.ndarray,
                                      truth: np.ndarray,
                                      treshold: float = 0.33,
                                      eps: float = 1e-9,
                                      classes: list = ['WT', 'TC', 'ET']) -> np.ndarray:
    scores = {key: list() for key in classes}
    num = probabilities.shape[0]
    num_classes = probabilities.shape[1]
    predictions = (probabilities >= treshold).astype(np.float32)
    assert predictions.shape == truth.shape
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
# LOSS FUNCTIONS
# ============================================================
# Original: BCEDiceLoss
# Enhanced: DiceLoss + CELoss + BoundaryLoss with class weights
# Class weights: ET (highest) > TC > WT
# ============================================================

# --- Original Loss (for reference / comparison) ---

class DiceLoss(nn.Module):
    """Calculate dice loss."""
    def __init__(self, eps: float = 1e-9):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num = targets.size(0)
        probability = torch.sigmoid(logits)
        probability = probability.view(num, -1)
        targets = targets.view(num, -1)
        intersection = 2.0 * (probability * targets).sum()
        union = probability.sum() + targets.sum()
        dice_score = (intersection + self.eps) / union
        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """Original: BCE + Dice loss."""
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert logits.shape == targets.shape
        return self.bce(logits, targets) + self.dice(logits, targets)


# --- NEW: Enhanced Loss Functions ---

class CELoss(nn.Module):
    """
    Cross-Entropy Loss for multi-label segmentation.

    What is CE Loss?
    ----------------
    Cross-Entropy measures the difference between predicted probability
    distribution and the true distribution, pixel by pixel:

        L_CE = -1/N * sum( y_i * log(p_i) + (1-y_i) * log(1-p_i) )

    For BraTS multi-class (WT, TC, ET), we apply BCEWithLogitsLoss
    per channel, which is equivalent to per-class binary CE.

    Unlike Dice loss (which optimizes overlap), CE penalizes every
    misclassified pixel equally, providing stable gradients.
    """
    def __init__(self, class_weights=None):
        """
        Args:
            class_weights: tensor of shape (n_classes,) — higher weight
                          means higher penalty for that class.
                          Recommended: [1.0, 2.0, 4.0] for [WT, TC, ET]
        """
        super(CELoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.class_weights is not None:
            # Apply per-class weights: weight each channel differently
            weights = self.class_weights.to(logits.device)
            # Expand weights to match target spatial dims
            # weights: (C,) -> (1, C, 1, 1, 1)
            w = weights.view(1, -1, 1, 1, 1)
            # Weighted BCE: -[w*y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x))]
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            bce = (bce * w).mean()
            return bce
        else:
            return F.binary_cross_entropy_with_logits(logits, targets)


class BoundaryLoss(nn.Module):
    """
    Boundary Loss using Distance Transform.

    What boundary losses do people commonly use?
    --------------------------------------------
    1. BD Loss (Kervadec et al., MIDL 2019):
       - Uses distance transform to weight CE loss
       - Boundary pixels get higher weight based on distance to contour
       - Simplest and most widely adopted for BraTS

    2. Surface Loss (Kervadec et al., MIDL 2019):
       - Integral approximation over boundary distance map
       - Mathematically elegant but complex implementation

    3. Hausdorff Distance Loss (Karimi et al., MICCAI 2019):
       - Directly optimizes Hausdorff distance
       - Computationally expensive, can be unstable

    THIS IMPLEMENTATION: Edge-aware Boundary Loss
    - Uses 3D Laplacian operator to detect edge regions
    - Computes BCE only on boundary pixels (weighted higher)
    - Simple, no pre-computation needed, differentiable

    Reference: Many BraTS papers use edge-aware weighting combined
    with Dice+CE for improved boundary delineation.
    """
    def __init__(self, edge_weight: float = 5.0):
        """
        Args:
            edge_weight: multiplier for boundary pixel loss.
                         Higher = more emphasis on boundaries.
        """
        super(BoundaryLoss, self).__init__()
        self.edge_weight = edge_weight

        # 3D Laplacian kernel for edge detection
        # This is a 3x3x3 kernel that highlights boundaries
        laplacian_kernel = torch.ones(1, 1, 3, 3, 3) * -1.0
        laplacian_kernel[0, 0, 1, 1, 1] = 26.0  # center = sum of neighbors
        self.register_buffer('laplacian_kernel', laplacian_kernel)

    def get_boundary_mask(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Extract boundary regions from ground truth masks using Laplacian.
        Boundary = pixels where Laplacian of mask != 0 (i.e., near edges).
        """
        B, C, D, H, W = targets.shape
        # Process each class channel
        boundaries = []
        for c in range(C):
            # (B, 1, D, H, W) -> apply Laplacian
            ch = targets[:, c:c+1, :, :, :]
            lap = F.conv3d(ch, self.laplacian_kernel, padding=1)
            # Boundary = where Laplacian != 0
            boundary = (lap.abs() > 1e-6).float()
            boundaries.append(boundary)
        boundary_mask = torch.cat(boundaries, dim=1)  # (B, C, D, H, W)
        return boundary_mask

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Get boundary mask from ground truth
        boundary_mask = self.get_boundary_mask(targets)

        # Compute BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Weight: edge_weight on boundaries, 1.0 elsewhere
        weight = 1.0 + boundary_mask * (self.edge_weight - 1.0)

        return (bce * weight).mean()


class DiceCEBoundaryLoss(nn.Module):
    """
    Combined Loss: Dice + Weighted CE + Boundary

    Loss = alpha * DiceLoss
         + beta  * CELoss (with class weights: ET > TC > WT)
         + gamma * BoundaryLoss

    Default weights tuned for BraTS:
        alpha=1.0 (dice), beta=0.5 (CE), gamma=0.3 (boundary)
        Class weights: WT=1.0, TC=2.0, ET=4.0
    """
    def __init__(
        self,
        alpha: float = 1.0,       # Dice weight
        beta: float = 0.5,        # CE weight
        gamma: float = 0.3,       # Boundary weight
        class_weights: list = None,  # [WT, TC, ET] weights
        edge_weight: float = 5.0,    # Boundary emphasis
    ):
        super(DiceCEBoundaryLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if class_weights is None:
            # Default: ET highest (4x), TC medium (2x), WT baseline (1x)
            class_weights = [1.0, 2.0, 4.0]

        self.dice = DiceLoss()
        self.ce = CELoss(class_weights=torch.tensor(class_weights))
        self.boundary = BoundaryLoss(edge_weight=edge_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert logits.shape == targets.shape

        loss_dice = self.dice(logits, targets)
        loss_ce = self.ce(logits, targets)
        loss_boundary = self.boundary(logits, targets)

        total = self.alpha * loss_dice + self.beta * loss_ce + self.gamma * loss_boundary
        return total

    def log_components(self, logits, targets):
        """Return individual loss components for logging."""
        with torch.no_grad():
            d = self.dice(logits, targets).item()
            c = self.ce(logits, targets).item()
            b = self.boundary(logits, targets).item()
        return {'dice': d, 'ce': c, 'boundary': b}


# ============================================================
# Trainer Class (cell 39, adapted)
# ============================================================

class Trainer():
    """
    Factory for training process.
    Args:
        net: neural network for mask prediction.
        dataset: BratsDataset class reference.
        criterion: loss function (e.g., DiceCEBoundaryLoss).
        lr: learning rate.
        accumulation_steps: gradient accumulation steps.
        batch_size: data batch size.
        fold: fold number (for cross-validation tracking).
        num_epochs: number of training epochs.
        path_to_csv: path to CSV file.
        model_type: checkpoint directory path.
        display_plot: if True, plot train history after each epoch.
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
                 early_stopping_patience: int = 40  # NEW: early stopping
                 ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("device:", self.device)
        self.display_plot = display_plot
        self.net = net
        self.net = self.net.to(self.device)
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
                dataset=dataset,
                path_to_csv=path_to_csv,
                phase=phase,
                fold=fold,
                batch_size=batch_size,
                num_workers=0
            )
            for phase in self.phases
        }

        self.early_stopping_patience = early_stopping_patience
        self.epochs_without_improvement = 0
        self.best_epoch = 0

        self.best_loss = float("inf")
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.jaccard_scores = {phase: [] for phase in self.phases}
        self.time = {phase: [] for phase in self.phases}

    def _compute_loss_and_outputs(self, images: torch.Tensor, targets: torch.Tensor):
        images = images.to(self.device)
        targets = targets.to(self.device)
        logits = self.net(images)
        loss = self.criterion(logits, targets)
        return loss, logits

    def _do_epoch(self, epoch: int, phase: str):
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

        end_time = time.time()
        total_time = round(end_time - start_time, 2)
        self.time[phase].append(total_time)
        return epoch_loss

    def run(self, check_path):
        epoch = self.epoch_value
        for epoch in range(int(self.epoch_value) + 1, self.num_epochs):
            self._do_epoch(epoch, "train")
            with torch.no_grad():
                val_loss = self._do_epoch(epoch, "valid")
                print(f"Loss for epoch {epoch} is: ", val_loss)
                self.scheduler.step(val_loss)
            if self.display_plot and epoch == self.num_epochs:
                self._plot_train_history()

            if val_loss < self.best_loss:
                print(f"\n{'#'*20}\nSaved new checkpoint\n{'#'*20}\n")
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                # Remove old best_model
                checkpoint_dir = check_path
                all_files = os.listdir(checkpoint_dir)
                best_model_current = [file for file in all_files if file.startswith("best_model_")]
                for best_model in best_model_current:
                    os.remove(os.path.join(checkpoint_dir, best_model))
                torch.save(self.net.state_dict(),
                          f"{self.model_type}/best_model_{epoch}.pth")
            else:
                self.epochs_without_improvement += 1

            if epoch % 1 == 0:
                self._save_train_history(epoch)

            # Early stopping check
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"Best val_loss: {self.best_loss:.6f} at epoch {self.best_epoch}")
                print(f"No improvement for {self.epochs_without_improvement} epochs")
                print(f"{'='*60}\n")
                break

            print()
        self._save_train_history(epoch)

    def _plot_train_history(self):
        data = [self.losses, self.dice_scores, self.jaccard_scores]
        colors = ['deepskyblue', "crimson"]
        labels = [
            f"train loss {self.losses['train'][-1]}\nval loss {self.losses['valid'][-1]}",
            f"train dice score {self.dice_scores['train'][-1]}\nval dice score {self.dice_scores['valid'][-1]}",
            f"train jaccard score {self.jaccard_scores['train'][-1]}\nval jaccard score {self.jaccard_scores['valid'][-1]}"
        ]
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        for i, ax in enumerate(axes):
            ax.plot(data[i]['valid'], c=colors[0], label="valid")
            ax.plot(data[i]['train'], c=colors[-1], label="train")
            ax.set_title(labels[i])
            ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    def load_pretrain_model(self, state_path: str):
        pretrain = torch.load(state_path, weights_only=False)
        if isinstance(pretrain, dict):
            self.net.load_state_dict(pretrain)
        else:
            self.net.load_state_dict(pretrain.state_dict())
        print("Pretrain model loaded")

    def _check_epoch_number(self, checkpoint_dir):
        value_of_hash = 0
        if not os.path.exists(checkpoint_dir):
            return value_of_hash
        all_files = os.listdir(checkpoint_dir)
        model_checkpoint_files = [file for file in all_files if file.startswith("last_epoch_model")]
        if model_checkpoint_files:
            sorted_file_names = sorted(model_checkpoint_files,
                                       key=lambda x: int(x.split('_')[-1].split('.')[0]))
            latest_checkpoint_file = sorted_file_names[-1]
            latest = latest_checkpoint_file.split("_")
            value_of_hash = latest[-1].split(".")[0]
            return value_of_hash
        else:
            return value_of_hash

    def _save_train_history(self, epoch=None):
        """writing model weights and training logs to files."""
        torch.save(self.net.state_dict(),
                   f"{self.model_type}/last_epoch_model_{epoch}.pth")

        logs_ = [self.losses, self.dice_scores, self.jaccard_scores, self.time]
        log_names_ = ["_loss", "_dice", "_jaccard", "_time"]
        logs = [logs_[i][key] for i in list(range(len(logs_)))
                for key in logs_[i]]
        log_names = [key + log_names_[i]
                     for i in list(range(len(logs_)))
                     for key in logs_[i]]
        pd.DataFrame(
            dict(zip(log_names, logs))
        ).to_csv(f"{self.model_type}/train_log.csv", index=False)


# ============================================================
# Evaluation Functions
# ============================================================

def compute_scores_per_classes(model, dataloader, classes=['WT', 'TC', 'ET']):
    """
    Compute Dice and Jaccard coefficients for each class.
    Returns: (dice_scores_per_classes, iou_scores_per_classes)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dice_scores_per_classes = {key: list() for key in classes}
    iou_scores_per_classes = {key: list() for key in classes}

    with torch.no_grad():
        for i, data in enumerate(dataloader):
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
    """Return mean dice and iou per class."""
    dice_scores_per_classes, iou_scores_per_classes = compute_scores_per_classes(
        model, dataloader, classes)
    dice_means = {key: np.mean(values) for key, values in dice_scores_per_classes.items()}
    iou_means = {key: np.mean(values) for key, values in iou_scores_per_classes.items()}
    return dice_means, iou_means


def print_metrics_table(dice_means, iou_means, model_name="Model"):
    """Print a formatted table of per-class metrics."""
    print(f"\n{'='*60}")
    print(f"  {model_name} — Evaluation Results")
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


# ============================================================
# Utility: Check if model exists
# ============================================================

def check_exist(checkpoint_dir):
    """Check if a pretrained model exists in checkpoint_dir."""
    if not os.path.exists(checkpoint_dir):
        return None
    all_files = os.listdir(checkpoint_dir)
    model_checkpoint_files = [file for file in all_files if file.startswith("best_model_")]
    if model_checkpoint_files:
        sorted_file_names = sorted(model_checkpoint_files,
                                   key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_checkpoint_file = sorted_file_names[-1]
        return os.path.join(checkpoint_dir, latest_checkpoint_file)
    return None


# ============================================================
# MAIN TRAINING SCRIPT
# ============================================================

if __name__ == "__main__":
    # ----------------------------------------------------------
    # 1. Train Enhanced ResUNet with new loss function
    # ----------------------------------------------------------
    print("=" * 60)
    print("Training ResUNet with Dice + CE + Boundary Loss")
    print("Class weights: WT=1.0, TC=2.0, ET=4.0")
    print("=" * 60)

    model = ResUNet3d(in_channels=4, n_classes=3, n_channels=24).to('cuda')

    # NEW loss function with class weights
    criterion = DiceCEBoundaryLoss(
        alpha=1.0,          # Dice weight
        beta=0.5,           # CE weight
        gamma=0.3,          # Boundary weight
        class_weights=[1.0, 2.0, 4.0],  # WT=1.0, TC=2.0, ET=4.0
        edge_weight=5.0,    # Boundary emphasis
    )

    trainer = Trainer(
        net=model,
        dataset=BratsDataset,
        criterion=criterion,
        lr=5e-4,                          # same as original
        accumulation_steps=4,             # same as original
        batch_size=1,                     # same as original
        fold=0,
        num_epochs=200,                   # same as original
        path_to_csv=config.path_to_csv,
        model_type=config.ResUNet_Enhanced_checkpoint_dir,
        early_stopping_patience=40        # NEW: stop if no improvement for 40 epochs
    )

    # Optionally load pretrained ResUNet as starting point
    pretrained_path = check_exist(config.ResUNet_checkpoint_dir)
    if pretrained_path is not None:
        print(f"Loading pretrained ResUNet from: {pretrained_path}")
        trainer.load_pretrain_model(pretrained_path)

    # Create checkpoint dir if not exists
    os.makedirs(config.ResUNet_Enhanced_checkpoint_dir, exist_ok=True)

    if check_exist(config.ResUNet_Enhanced_checkpoint_dir) is not None:
        trainer.load_pretrain_model(check_exist(config.ResUNet_Enhanced_checkpoint_dir))

    trainer.run(check_path=config.ResUNet_Enhanced_checkpoint_dir)

    # ----------------------------------------------------------
    # 2. Evaluation: Compare with original ResUNet
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION: Comparing Original vs Enhanced ResUNet")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load original ResUNet
    original_model = ResUNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
    orig_path = check_exist(config.ResUNet_checkpoint_dir)
    if orig_path:
        original_model.load_state_dict(torch.load(orig_path, map_location=device))
        original_model.eval()
        print(f"Original ResUNet loaded from: {orig_path}")

    # Load enhanced ResUNet
    enhanced_model = ResUNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)
    enh_path = check_exist(config.ResUNet_Enhanced_checkpoint_dir)
    if enh_path:
        enhanced_model.load_state_dict(torch.load(enh_path, map_location=device))
        enhanced_model.eval()
        print(f"Enhanced ResUNet loaded from: {enh_path}")

    # Evaluate on validation set
    val_dataloader = get_dataloader(
        dataset=BratsDataset,
        path_to_csv=config.path_to_csv,
        phase='valid',
        batch_size=1
    )

    print("\n--- Original ResUNet (BCEDiceLoss) ---")
    orig_dice, orig_iou = compute_scores_per_classes_mean(original_model, val_dataloader)
    print_metrics_table(orig_dice, orig_iou, "Original ResUNet")

    print("\n--- Enhanced ResUNet (Dice+CE+Boundary) ---")
    enh_dice, enh_iou = compute_scores_per_classes_mean(enhanced_model, val_dataloader)
    print_metrics_table(enh_dice, enh_iou, "Enhanced ResUNet")

    # Comparison summary
    print("\n--- Improvement ---")
    for cls in ['WT', 'TC', 'ET']:
        delta = enh_dice[cls] - orig_dice[cls]
        print(f"  {cls} Dice: {orig_dice[cls]:.4f} -> {enh_dice[cls]:.4f} ({delta:+.4f})")
