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
    """Check if a pretrained model exists in checkpoint_dir."""
    # Get a list of all files in the checkpoint directory
    all_files = os.listdir(checkpoint_dir)

    # Filter the files to get only the model checkpoint files
    model_checkpoint_files = [file for file in all_files if file.startswith("last_epoch_model")]

    if model_checkpoint_files:

        sorted_file_names = sorted(model_checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        # Get the latest model checkpoint file
        latest_checkpoint_file = sorted_file_names[-1]


        # Construct the full path to the latest model checkpoint
        pretrained_model_path = os.path.join(checkpoint_dir, latest_checkpoint_file)

        return pretrained_model_path
    else:
        pretrained_model_path = None
