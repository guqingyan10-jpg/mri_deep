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
