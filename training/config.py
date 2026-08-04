"""
Global Configuration for BraTS2020 Training.
Extracted from: MultiModel XAI Brats2020.ipynb (cells 10, 15)

Contains: GlobalConfig, seed_everything, check_exist
"""

import os
import numpy as np
import torch


class GlobalConfig:
    root_dir = r'/root/autodl-tmp'
    train_root_dir = r'/root/autodl-tmp/brats_project/MICCAI_BraTS2020_TrainingData/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    test_root_dir = r'/root/autodl-tmp/test_df'
    path_to_csv = 'tumourCSV.csv'


    # Define the directory where the model checkpoints are saved

    UNet_checkpoint_dir = r"/root/autodl-tmp/UNet_model"
    ResUNet_checkpoint_dir =  r"/root/autodl-tmp/ResUNet_model"
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


def check_exist(checkpoint_dir):
    """
    Find checkpoint for WARM-START (from another model's best weights).

    Priority:
      1. best_model_*.pth (lowest val_loss)
      2. last_epoch_model_*.pth (fallback, e.g. before any best_model saved)
    """
    all_files = os.listdir(checkpoint_dir)

    best_files = [f for f in all_files if f.startswith("best_model_")]
    if best_files:
        best = sorted(best_files,
                      key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        return os.path.join(checkpoint_dir, best)

    last_files = [f for f in all_files if f.startswith("last_epoch_model")]
    if last_files:
        last = sorted(last_files,
                      key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        return os.path.join(checkpoint_dir, last)

    return None


def check_exist_last(checkpoint_dir):
    """
    Find checkpoint for RESUME (continue training from where it stopped).

    Priority:
      1. last_epoch_model_*.pth (latest epoch, keep training)
      2. best_model_*.pth (fallback if no last_epoch_model exists)

    This is DIFFERENT from check_exist() which prefers best_model
    for warm-starting new experiments from baseline.
    """
    all_files = os.listdir(checkpoint_dir)

    last_files = [f for f in all_files if f.startswith("last_epoch_model")]
    if last_files:
        last = sorted(last_files,
                      key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        return os.path.join(checkpoint_dir, last)

    best_files = [f for f in all_files if f.startswith("best_model_")]
    if best_files:
        best = sorted(best_files,
                      key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
        return os.path.join(checkpoint_dir, best)

    return None
