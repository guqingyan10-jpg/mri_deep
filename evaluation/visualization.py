"""
Visualization utilities for BraTS2020.
Extracted from: MultiModel XAI Brats2020.ipynb (cells 8, 117, 118)

Contains:
  - Image3dToGIF3d: 3D GIF generation
  - ShowResult: ground truth vs prediction overlay
  - tumour_graphics: interactive widget visualization
  - generate_3d_plotly: 3D Plotly visualization
  - merging_two_gif: side-by-side GIF merge
  - get_all_csv_file: CSV file discovery
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.transform import resize
from skimage.util import montage
import imageio
from tqdm import tqdm

import ipywidgets as widgets
from ipywidgets import interact, fixed
import plotly.graph_objs as go
import plotly


# ============================================================
# Image3dToGIF3d (cell 8)
# ============================================================

class Image3dToGIF3d:
    """
    Displaying 3D images in 3d axes.
    Parameters:
        img_dim: shape of cube for resizing.
        figsize: figure size for plotting in inches.
    """
    def __init__(self,
                 img_dim: tuple = (55, 55, 55), #Image dimension size
                 figsize: tuple = (15, 10), #size of image output
                 binary: bool = False,
                 normalizing: bool = True,
                ):
        """Initialization."""
        self.img_dim = img_dim
        print(img_dim)
        self.figsize = figsize
        self.binary = binary
        self.normalizing = normalizing

    def _explode(self, data: np.ndarray):
        """
        Takes: array and return an array twice as large in each dimension,
        with an extra space between each voxel.
        """
        shape_arr = np.array(data.shape)
        size = shape_arr[:3] * 2 - 1
        exploded = np.zeros(np.concatenate([size, shape_arr[3:]]),
                            dtype=data.dtype)
        exploded[::2, ::2, ::2] = data
        return exploded

    def _expand_coordinates(self, indices: np.ndarray):
        x, y, z = indices
        x[1::2, :, :] += 1
        y[:, 1::2, :] += 1
        z[:, :, 1::2] += 1
        return x, y, z

    def _normalize(self, arr: np.ndarray):
        """Normilize image value between 0 and 1."""
        arr_min = np.min(arr)
        return (arr - arr_min) / (np.max(arr) - arr_min)


    def _scale_by(self, arr: np.ndarray, factor: int):
        """
        Scale 3d Image to factor.
        Parameters:
            arr: 3d image for scalling.
            factor: factor for scalling.
        """
        mean = np.mean(arr)
        return (arr - mean) * factor + mean
        # the mean is added back to the scaled array ((arr - mean) * factor + mean).
        # This step ensures that the mean value of the resulting array remains unchanged after scaling.

    def get_transformed_data(self, data: np.ndarray):
        """Data transformation: normalization, scaling, resizing."""
        if self.binary:
            resized_data = resize(data, self.img_dim, preserve_range=True)
            return np.clip(resized_data.astype(np.uint8), 0, 1).astype(np.float32)

        norm_data = np.clip(self._normalize(data)-0.1, 0, 1) ** 0.4
        scaled_data = np.clip(self._scale_by(norm_data, 2) - 0.1, 0, 1)
        resized_data = resize(scaled_data, self.img_dim, preserve_range=True)

        return resized_data

    def plot_cube(self,
                  cube,
                  title: str = '',
                  init_angle: int = 0,
                  make_gif: bool = False,
                  path_to_save: str = 'filename.gif'
                 ):
        """
        Plot 3d data.
        Parameters:
            cube: 3d data
            title: title for figure.
            init_angle: angle for image plot (from 0-360).
            make_gif: if True create gif from every 5th frames from 3d image plot.
            path_to_save: path to save GIF file.
        """
        if self.binary:
            facecolors = cm.winter(cube)
            print("binary")
        else:
            if self.normalizing:
                cube = self._normalize(cube)
            facecolors = cm.gist_stern(cube)
            print("not binary")

        facecolors[:,:,:,-1] = cube
        facecolors = self._explode(facecolors)

        filled = facecolors[:,:,:,-1] != 0
        x, y, z = self._expand_coordinates(np.indices(np.array(filled.shape) + 1))

        with plt.style.context("dark_background"):

            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(projection = '3d')

            ax.view_init(30, init_angle)
            ax.set_xlim(right = self.img_dim[0] * 2)
            ax.set_ylim(top = self.img_dim[1] * 2)
            ax.set_zlim(top = self.img_dim[2] * 2)
            ax.set_title(title, fontsize=18, y=1.05)

            ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)

            if make_gif:
                images = []
                for angle in tqdm(range(0, 360, 5)):
                    ax.view_init(30, angle)
                    fname = str(angle) + '.png'

                    plt.savefig(fname, dpi=120, format='png', bbox_inches='tight')
                    images.append(imageio.imread(fname))
                    #os.remove(fname)
                imageio.mimsave(path_to_save, images)
                plt.close()

            else:
                plt.show()


# ============================================================
# ShowResult (cell 8)
# ============================================================

class ShowResult:

    def mask_preprocessing(self, mask):
        """
        Test.
        """
        # removing all the ones in the tensor --> using cpu --> removing the tensor from its computational graph --> tensor to numpy conversion

        print(mask.shape)
        mask_crop1 = mask[0,0,:,:,:]
        mask_crop2 = mask[0,1,:,:,:]
        mask_crop3 = mask[0,2,:,:,:]

        mask_WT = montage(mask_crop1)
        mask_TC = montage(mask_crop2)
        mask_ET = montage(mask_crop3)

        return mask_WT, mask_TC, mask_ET


    def image_preprocessing(self, image):
        """
        Returns image flair as mask for overlaping gt and predictions.
        """
        image = image.squeeze().cpu().detach().numpy()

        # image = np.moveaxis(image, (0, 1, 2, 3), (0, 3, 2, 1))

        img_crop = image[0, :,:,:]
        flair_img = montage(img_crop)

        return flair_img

    def plot(self, image, ground_truth, prediction):
        image = self.image_preprocessing(image)
        gt_mask_WT, gt_mask_TC, gt_mask_ET = self.mask_preprocessing(ground_truth)
        pr_mask_WT, pr_mask_TC, pr_mask_ET = self.mask_preprocessing(prediction)

        fig, axes = plt.subplots(1, 2, figsize = (35, 30))

        [ax.axis("off") for ax in axes]
        axes[0].set_title("Ground Truth", fontsize=35, weight='bold')
        axes[0].imshow(image, cmap ='bone')
        axes[0].imshow(np.ma.masked_where(gt_mask_WT == False, gt_mask_WT),
                  cmap='summer', alpha=0.6)
        axes[0].imshow(np.ma.masked_where(gt_mask_TC == False, gt_mask_TC),
                  cmap='rainbow', alpha=0.6)
        axes[0].imshow(np.ma.masked_where(gt_mask_ET == False, gt_mask_ET),
                  cmap='Wistia', alpha=0.6)



        axes[1].set_title("Prediction", fontsize=35, weight='bold')
        axes[1].imshow(image, cmap ='bone')
        axes[1].imshow(np.ma.masked_where(pr_mask_WT == False, pr_mask_WT),
                   cmap='summer', alpha=0.6)
        axes[1].imshow(np.ma.masked_where(pr_mask_TC == False, pr_mask_TC),
                   cmap='rainbow', alpha=0.6)
        axes[1].imshow(np.ma.masked_where(pr_mask_ET == False, pr_mask_ET),
                  cmap='Wistia', alpha=0.6)

        plt.tight_layout()

        plt.show()


# ============================================================
# Interactive Tumour Graphics (cell 117)
# ============================================================

def tumour_graphics(n_slice, img, gt, prediction):
    print("Image Shape:", img.shape)
    print("GT Shape:", gt.shape)
    print("Prediction Shape:", prediction.shape)

    # Convert to NumPy for visualization
    img_np = img.cpu().numpy() if torch.is_tensor(img) else img
    gt_np = gt.cpu().numpy() if torch.is_tensor(gt) else gt
    prediction_np = prediction.cpu().numpy() if torch.is_tensor(prediction) else prediction

    # Select a slice
    img_slice = img_np[0, 0, :, :, n_slice]
    gt_slice = gt_np[0, 0, :, :, n_slice]
    pr_slice = prediction_np[0, 0, :, :, n_slice]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title('MRI Image')
    axes[0].axis('off')

    axes[1].imshow(img_slice, cmap='gray')
    axes[1].imshow(gt_slice, cmap='jet', alpha=0.5)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    axes[2].imshow(img_slice, cmap='gray')
    axes[2].imshow(pr_slice, cmap='jet', alpha=0.5)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================
# 3D Plotly Visualization (cell 118)
# ============================================================

def generate_3d_plotly(img, prediction, text):
    data = img[0, 0, :, :, :]
    data1 = prediction[0, 0, :, :, :]
    data2 = prediction[0,1,:,:,:]
    data3 = prediction[0,2,:,:,:]

    # Threshold value
    threshold = 0.2

    # Extract coordinates and values for the first tensor
    coords = (data > threshold).nonzero(as_tuple=False)
    z = coords[:, 0].tolist()
    y = coords[:, 1].tolist()
    x = coords[:, 2].tolist()
    values = data[coords[:, 0], coords[:, 1], coords[:, 2]].tolist()

    # Extract coordinates and values for the second tensor
    coords1 = (data1 > threshold).nonzero(as_tuple=False)
    z1 = coords1[:, 0].tolist()
    y1 = coords1[:, 1].tolist()
    x1 = coords1[:, 2].tolist()
    values1 = data1[coords1[:, 0], coords1[:, 1], coords1[:, 2]].tolist()

    coords2 = (data2 > threshold).nonzero(as_tuple=False)
    z2 = coords2[:, 0].tolist()
    y2 = coords2[:, 1].tolist()
    x2 = coords2[:, 2].tolist()
    values2 = data2[coords2[:, 0], coords2[:, 1], coords2[:, 2]].tolist()

    coords3 = (data3 > threshold).nonzero(as_tuple=False)
    z3 = coords3[:, 0].tolist()
    y3 = coords3[:, 1].tolist()
    x3 = coords3[:, 2].tolist()
    values3 = data3[coords3[:, 0], coords3[:, 1], coords3[:, 2]].tolist()

    # Create scatter plots for each tensor
    scatter1 = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=1,
            color=values,
            colorscale='Greys',
            opacity=0.5
        ),
        name="MRI Image (single channel)"
    )

    scatter2 = go.Scatter3d(
        x=x1, y=y1, z=z1,
        mode='markers',
        marker=dict(
            size=1,
            color=values1,
            colorscale='Reds',
            opacity=0.5
        ),
        name="WT Prediction"
    )

    scatter3 = go.Scatter3d(
        x=x2, y=y2, z=z2,
        mode='markers',
        marker=dict(
            size=1,
            color=values2,
            colorscale='Greens',
            opacity=0.5
        ),
        name="TC Prediction"
    )

    scatter4 = go.Scatter3d(
        x=x3, y=y3, z=z3,
        mode='markers',
        marker=dict(
            size=1,
            color=values3,
            colorscale='Blues',
            opacity=0.5
        ),
        name="ET Prediction"
    )

    # Combine all scatter plots
    data = [scatter1, scatter2, scatter3, scatter4]

    # Create the 3D plot
    fig = go.Figure(data=data)
    fig.update_layout(
        title=text,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    fig.show()


# ============================================================
# Utility Functions (cell 8)
# ============================================================

def merging_two_gif(path1: str, path2: str, name_to_save: str):
    """
    Merging GIFs side by side.
    Parameters:
        path1: path to gif with ground truth.
        path2: path to gif with prediction.
        name_to_save: name for saving new GIF.
    """
    #Create reader object for the gif
    gif1 = imageio.get_reader(path1)
    gif2 = imageio.get_reader(path2)

    #If they don't have the same number of frame take the shorter
    number_of_frames = min(gif1.get_length(), gif2.get_length())

    #Create writer object
    new_gif = imageio.get_writer(name_to_save)

    for frame_number in range(number_of_frames):
        img1 = gif1.get_next_data()
        img2 = gif2.get_next_data()
        #here is the magic
        new_image = np.hstack((img1, img2))
        new_gif.append_data(new_image)

    gif1.close()
    gif2.close()
    new_gif.close()

#merging_two_gif('BraTS20_Training_001_flair_3d.gif',
#                'BraTS20_Training_001_flair_3d.gif',
#                'result.gif')

def get_all_csv_file(root: str) -> list:
    """Extraction all unique ids from file names."""
    ids = []
    for dirname, _, filenames in os.walk(root):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            if path.endswith(".csv"):
                ids.append(path)
    ids = list(set(filter(None, ids)))
    print(f"Extracted {len(ids)} csv files.")
    return ids
