# ABS real-label visualization

This script exports real MRI/GT illustrations for Auxiliary Boundary Supervision (ABS). It does not train a model, alter evaluation settings, or require a checkpoint. The maps labelled GT, eroded GT and boundary are annotation-derived targets, not predictions.

## UCSF

For an example using the **exact preprocessed training arrays**, prefer the existing prepared cache:

```bash
python scripts/visualize_abs_module.py \
  --prepared /root/autodl-tmp/ucsf_bmsr_prepared/cases/100106A.npz \
  --dataset ucsf --verify-torch --out outputs/abs_detail_ucsf_training_grid
```

The cache is read exactly as `UCSFPreparedDataset` reads it: image is converted from float16 to float32, the binary mask is unchanged, and no extra resize or normalization is performed. Channel 0 is T1post. The default axial axis is 0 for cached DHW arrays. Substitute your actual preprocessing output directory if different.

For a high-resolution **native-grid illustration** of the same operator:

Run from the Git repository, not an evaluation output directory:

```bash
cd /root/autodl-tmp/mri_deep
git pull --rebase origin master
python scripts/visualize_abs_module.py \
  --image /root/autodl-tmp/UCSF_BrainMetastases_TRAIN/100106A/100106A_T1post.nii.gz \
  --mask /root/autodl-tmp/UCSF_BrainMetastases_TRAIN/100106A/100106A_seg.nii.gz \
  --dataset ucsf --region lesion --verify-torch \
  --out outputs/abs_detail_ucsf
```

Use an available patient ID if 100106A is absent. UCSF training uses one binary channel, obtained from all nonzero values of the original `_seg.nii.gz` mask. This is not the BraTS ET/TC/WT region setting. The image and mask must have matching shape and affine.

Native examples are explicitly tagged in `metadata.json`. UCSF training first reorients to RAS and resizes the field of view to (100,170,170) DHW. Consequently a boundary extracted on native NIfTI data is an illustration of the operator, rather than the exact resized training target. Do not mix the two grids in quantitative mechanism analyses. The `--prepared` route avoids this mismatch.

## BraTS2020

```bash
python scripts/visualize_abs_module.py \
  --image /path/to/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii.gz \
  --mask /path/to/BraTS20_Training_001/BraTS20_Training_001_seg.nii.gz \
  --dataset brats --region ET --et-label 4 --verify-torch \
  --out outputs/abs_detail_brats_et
```

Both `.nii` and `.nii.gz` are accepted. Replace the paths with the intact server files. For TC or WT change `--region` and the output directory. Original labels: WT = {1,2,4}; TC = {1,4}; ET = {4}. `--et-label 3` is only for explicitly remapped labels. Boundaries are computed independently for these overlapping binary channels, exactly as in training.

Optional `--axis 2 --slice 43` specifies a zero-based native slice. By default the script chooses the slice with most foreground and crops around its largest 8-connected component with a 12-voxel margin. Label construction always precedes slice selection and ROI cropping. Display-only MRI normalization is whole-volume min-max, not training preprocessing.

## Exact implementation

```text
E3D(Y) = avg_pool3d(Y, kernel_size=3, stride=1, padding=1,
                  count_include_pad=True) > 0.999
Yb = Y - E3D(Y)
Ltotal = Lseg + 0.1 Lboundary
Lseg = BCEWithLogits(seg_logits, Y) + baseline global Dice loss
Lboundary = BCEWithLogits(boundary_logits, Yb), mean over all voxels
```

The visualizer implements the same binary erosion through 27 boolean neighbour intersections and always checks against SciPy binary erosion. `--verify-torch` additionally compares it with the project's actual `_extract_boundary_gt` method. The 3-voxel kernel is specified in voxel units; anisotropic spacing means its physical width varies between axes. A 2D slice contour is not the implemented 3D target. Also, auxiliary BCE is averaged over all voxels: a larger boundary fraction within a small lesion does **not** by itself imply a greater total penalty than a large lesion.

The segmentation head is a 1×1×1 convolution. The auxiliary head receives the same 24-channel final decoder tensor and applies Conv3D(24→24,3³) → GroupNorm(8 groups) → ReLU → Conv3D(24→K,1³). Outputs are logits. K=1 for UCSF, K=3 for BraTS. No edge-gating or segmentation–boundary consistency loss is present in this ABS implementation.

## Send back

Zip the output directory and send it back. The useful files are `abs_arrays.npz`, `metadata.json`, `ABS_real_examples.svg/pdf/png`, four ROI PNGs, and `3d_vs_2d_diagnostic.png`. Numerical arrays are retained at original slice/ROI resolution. PDF/SVG illustrations consist of vector cells and editable text; the configurable display grid does not change the underlying arrays.

Dependencies are numpy, nibabel, matplotlib, scipy (and torch for the optional exact implementation check). Existing training environments normally already contain them. No retraining is needed.

## Paper figure references

Hatamizadeh et al., *Edge-Gated CNNs for Volumetric Semantic Segmentation of Medical Images*, arXiv:2002.04207, Figure 1, provides a useful reference for showing region and edge branches. Its multi-level gating, 3D Sobel targets, balanced edge loss, and consistency objective differ from this project's simple auxiliary head. Use it as related work and layout reference, not evidence for this model's measured performance. https://arxiv.org/abs/2002.04207
