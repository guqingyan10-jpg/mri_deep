# High-frequency module visualization

`scripts/visualize_hf_module.py` generates real MRI before/after comparisons and
an editable LHFC module diagram. No checkpoint or GPU is required. Dependencies:
`numpy`, `nibabel`, and `matplotlib`; reuse the existing environment when available.

## AutoDL: UCSF example

Run from the repository root after pulling `master`:

```bash
python scripts/visualize_hf_module.py --image /root/autodl-tmp/UCSF_BrainMetastases_TRAIN/100101A/100101A_T1post.nii.gz --mask /root/autodl-tmp/UCSF_BrainMetastases_TRAIN/100101A/100101A_seg.nii.gz --modality T1post --out outputs/hf_detail
```

Use literal underscores and double-hyphen options. Do not paste HTML entities
or extra backslashes before underscores/options from a rich-text editor.

## Outputs

- `hf_before_after.pdf/svg/png`: input, low-pass, signed high-pass residual,
  residual magnitude, and the same region enlarged below each image.
- `lhfc_module_detail.pdf/svg/png`: fixed filtering, learned feature-pyramid
  structure, and decoder concatenation.
- Individual input/filter PNGs, original-resolution numerical slices in
  `source_slice_values.npz`, and provenance in `visualization_metadata.json`.

SVG text and diagram objects remain editable. MRI panels use vector cells
(at most 128 per axis by default), with full-resolution slice values saved
separately. `--vector-resolution 256` increases visual sampling and file size.
These are large editable masters; check labels again at final journal size.

## Interpretation and preprocessing

The script applies a 3D uniform 3x3x3 kernel (1/27 per element) with zero padding,
then subtracts the low-pass image from the input. This matches the fixed operator
in `models/resunet_edge.py::LaplacianEdge3d`. It does not use a Gaussian kernel,
FFT, or wavelets. Filtering precedes slice/ROI extraction.

The default min-max normalization illustrates this operator on one modality; it
does not reproduce all UCSF training resampling/normalization or checkpoint
activations. The model consumes the signed residual. Absolute residual is shown
only for display. High-frequency responses include normal tissue and noise as
well as lesion boundaries; this figure alone is not evidence of detection gains.

Learned pyramid tensors are structural illustrations for base channels=24, not
measured activations. Actual feature maps require trained weights and matching
preprocessing. `--verify-torch` checks the fixed filter against the repository's
PyTorch implementation on CPU when run inside the complete repository.

## Slice and ROI selection

With a mask, the default slice has maximum selected GT cross-sectional area.
Without a mask, the middle slice is used. `--slice` is zero-based and `--axis`
selects the native NIfTI axis. `--roi-size 48` controls the crop width;
`--roi-center ROW COL` refers to the rotated displayed slice.

Input/low-pass share the [0,1] window. Signed residual uses a symmetric window;
absolute residual uses [0,limit]. The default limit is the 99th percentile of
absolute residual within the nonzero input on the selected slice. Full images
and their ROIs use identical windows. Masks must match image shape and affine.

## BraTS example

```bash
python scripts/visualize_hf_module.py --image /path/to/CASE/CASE_t1ce.nii --mask /path/to/CASE/CASE_seg.nii --mask-label 4 --modality T1ce --crop brats-legacy --out outputs/hf_detail_brats
```

Use `--crop brats-legacy` only if the original training used the
`data/dataset.py` crop `[40:210,40:210,20:120]` before min-max normalization;
otherwise omit it. Label 4 is ET in BraTS 2020; check other dataset versions.
Do not apply this label or crop convention to the original UCSF masks.
