"""
Brats2020 Dataset and DataLoader.
Extracted from: MultiModel XAI Brats2020.ipynb (cell 27)

Contains: get_augmentations, get_dataloader, BratsDataset
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from albumentations import Compose
import nibabel as nib


def get_augmentations(phase):
    list_transforms = []

    # Does data augmentations & tranformation required for IMAGES & MASKS
    # they include cropping, padding, flipping , rotating
    list_trfms = Compose(list_transforms,  is_check_shapes=False)
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
    test_df = test_df_copy.iloc[len(test_df_copy)*2//3:].reset_index(drop=True)
    val_df = test_df_copy.iloc[:len(test_df_copy)*2//3].reset_index(drop=True)



    if phase != 'test':



    # selection a particluar fold while calling the get_dataloader function

        '''Returns: dataloader for the model training'''

        if phase == "train" :

            df = train_df_copy
        elif phase == "valid" :

            df = val_df

        dataset = dataset(df, phase)

    else:

        df = test_df
        dataset = dataset(df, phase)
    """
    DataLoader iteratively goes through every id in the df & gets all the individual tuples for individual ids & appends all of them
    like this :
    { id : ['BraTS20_Training_235'] ,
      image : [] ,
      tensor : [] ,
    }
    { id : ['BraTS20_Training_236'] ,
      image : [] ,
      tensor : [] ,
    }
    { id : ['BraTS20_Training_237'] ,
      image : [] ,
      tensor : [] ,
    }
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )

    return dataloader


class BratsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, phase: str="test", is_resize: bool=True):
        self.df = df
        self.phase = phase
        self.augmentations = get_augmentations(phase)
        self.data_types = ['_flair.nii', '_t1.nii', '_t1ce.nii', '_t2.nii']
        self.is_resize = is_resize

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        # at a specified index ( idx ) select the value under 'Brats20ID' & asssign it to id_
        id_ = self.df.loc[idx, 'Brats20ID']


        root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0]

        # load all modalities
        images = []

        for data_type in self.data_types:
            img_path = os.path.join(root_path, id_ + data_type)
            img = self.load_img(img_path)#.transpose(2, 0, 1)

            if self.is_resize:
                img = self.resize(img)

            img = self.normalize(img)
            images.append(img)

        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))

        # if self.phase != "test":
        mask_path =  os.path.join(root_path, id_ + "_seg.nii")
        mask = self.load_img(mask_path)

        if self.is_resize:

            mask = self.resize(mask)

        mask = self.preprocess_mask_labels(mask)
        # setting the mask labels 1 , 2 , 4 for the mask file ( _seg.ii )


        augmented = self.augmentations(image=img.astype(np.float32),
                                        mask=mask.astype(np.float32))
        # Several augmentations / transformations like flipping, rotating, padding will be applied to both the images
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
        # normalization = (each element - min element) / ( max - min )
        return (data - data_min) / (np.max(data) - data_min)

    def resize(self, data: np.ndarray):

        data = data[ 40:210, 40:210, 20:120]
        # The selected indices do not remove the slices that contain the brain tumour
        #40:210
        #40:210
        #20:120
        return data

    def preprocess_mask_labels(self, mask: np.ndarray):

        # whole tumour
        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1
        # include all tumours

        # NCR / NET - LABEL 1
        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1
        # exclude 2 / 4 labelled tumour

        # ET - LABEL 4
        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1
        # exclude 2 / 1 labelled tumour

        # # ED - LABEL 2
        # # mask_ED = mask.copy()
        # # mask_ED[mask_ED == 1] = 0
        # # mask_ED[mask_ED == 2] = 1
        # # mask_ED[mask_ED == 4] = 0


        # mask = np.stack([mask_WT, mask_TC, mask_ET, mask_ED])
        mask = np.stack([mask_WT, mask_TC, mask_ET])

        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

        return mask


# ============================================================
# Foreground-Aware Patch Sampling Dataset Wrapper
# ============================================================

import pickle

from data.foreground_sampler import ForegroundAwarePatchSampler


class BratsDatasetWithFGSampling(BratsDataset):
    """
    BraTS2020 dataset with foreground-aware 3D patch sampling.

    Wraps BratsDataset, adding STSNet-inspired patch sampling strategies:
      random (20%) + foreground (30%) + et_centered (30%) + small_lesion (20%)

    Workflow:
      1. Pre-indexing: scan training masks → build ET connected-component index
      2. Training: each __getitem__ samples a patch using the configured strategy
         mix, then crops the full volume to that patch

    STSNet mapping:
      BratsDataset.resize() → (170, 170, 100) → corresponds to 480×480 in STSNet
      patch_size=(128,128,128) → corresponds to STSNet's random(150,160) crop window
      build_index() → corresponds to 4_find_label_center_together.py
      4-strategy mix → corresponds to TwoStreamBatchSampler

    Usage:
        dataset = BratsDatasetWithFGSampling(df, phase='train', patch_size=(128,128,96))
        dataset.build_foreground_index()  # once before training
        patch = dataset[0]  # returns {'image': (4,D,H,W), 'mask': (3,D,H,W)}
    """

    def __init__(self, df, phase='train', is_resize=True,
                 patch_size=(128, 128, 96),
                 ratios=None,
                 small_threshold=50):
        """
        Args:
            df: training dataframe
            phase: 'train' (sampling enabled), 'valid'/'test' (sampling disabled)
            is_resize: whether to apply BratsDataset.resize()
            patch_size: (D, H, W) of patches to sample
            ratios: sampling strategy ratios dict
            small_threshold: max ET voxels for 'small_lesion' category
        """
        super().__init__(df, phase, is_resize)
        self.patch_size = tuple(patch_size)

        if phase == 'train':
            self.sampler = ForegroundAwarePatchSampler(
                patch_size=self.patch_size,
                ratios=ratios,
                small_threshold=small_threshold,
            )
            self._index_built = False
        else:
            self.sampler = None
            self._index_built = True

    # ── Pre-indexing ────────────────────────────────────────────

    def build_foreground_index(self, cache_path=None):
        """
        Scan all training cases and build ET connected-component index.

        Must be called ONCE before training starts.
        If cache_path is provided, saves/loads the index from disk.

        STSNet equivalent:
          4_find_label_center_together.py iterates over all images
          in the dataset, runs findContours+minAreaRect for each.

        Args:
            cache_path: optional path to save/load cached index (.pkl)
        """
        if self.phase != 'train':
            print("[SKIP] Not in train phase, no index needed.")
            return

        # Try loading from cache
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached foreground index from: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            self.sampler._fg_index = cache.get('fg_index', {})
            self.sampler._small_index = cache.get('small_index', {})
            self.sampler._fg_coords = cache.get('fg_coords', {})
            self.sampler.stats = cache.get('stats', {})
            self._index_built = True
            self.sampler.print_stats()
            return

        print(f"Building foreground index for {len(self.df)} training cases...")
        for idx in range(len(self.df)):
            id_ = self.df.loc[idx, 'Brats20ID']
            root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0]

            mask_path = os.path.join(root_path, id_ + "_seg.nii")
            mask = self.load_img(mask_path)
            if self.is_resize:
                mask = self.resize(mask)
            mask = self.preprocess_mask_labels(mask)  # (3, D, H, W)

            self.sampler.build_index(id_, mask)

            if (idx + 1) % 50 == 0:
                print(f"  ... {idx + 1}/{len(self.df)} cases indexed")

        self._index_built = True
        self.sampler.print_stats()

        # Save to cache
        if cache_path:
            print(f"Saving foreground index to: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'fg_index': self.sampler._fg_index,
                    'small_index': self.sampler._small_index,
                    'fg_coords': self.sampler._fg_coords,
                    'stats': self.sampler.stats,
                }, f)

    # ── Patch Sampling ──────────────────────────────────────────

    def get_sampler_stats(self):
        """Return foreground sampler statistics (for logging)."""
        if self.sampler is not None:
            return self.sampler.get_stats()
        return {}

    def __getitem__(self, idx):
        """
        Returns a patch from the case, sampled using the strategy mix.

        For train phase: samples a patch using ForegroundAwarePatchSampler,
        then crops both image and mask to that patch.

        For valid/test phase: returns the full volume (same as BratsDataset).
        """
        # Load full volume using parent class logic
        id_ = self.df.loc[idx, 'Brats20ID']
        root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0]

        # Load all modalities
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

        # ── Patch Sampling (train only) ──
        if self.phase == 'train' and self.sampler is not None:
            if not self._index_built:
                raise RuntimeError(
                    "Foreground index not built! Call build_foreground_index() before training."
                )

            volume_shape = img.shape[1:]  # (D, H, W)
            z1, z2, y1, y2, x1, x2 = self.sampler.sample(id_, volume_shape)

            # Crop to patch
            img = img[:, z1:z2, y1:y2, x1:x2]
            mask = mask[:, z1:z2, y1:y2, x1:x2]

            # If patch is smaller than expected (edge case near boundary), pad
            pD, pH, pW = self.patch_size
            if img.shape[1:] != (pD, pH, pW):
                pad_d = max(0, pD - img.shape[1])
                pad_h = max(0, pH - img.shape[2])
                pad_w = max(0, pW - img.shape[3])
                if pad_d > 0 or pad_h > 0 or pad_w > 0:
                    img = np.pad(img, ((0,0),(0,pad_d),(0,pad_h),(0,pad_w)),
                                 mode='constant')
                    mask = np.pad(mask, ((0,0),(0,pad_d),(0,pad_h),(0,pad_w)),
                                  mode='constant')

        # Augmentations
        augmented = self.augmentations(
            image=img.astype(np.float32),
            mask=mask.astype(np.float32),
        )
        img = augmented['image']
        mask = augmented['mask']

        return {
            "Id": id_,
            "image": img,
            "mask": mask,
        }
