# DoubleBlock-ViT training on the fixed BraTS2020 split

The model adapter and training entry point are available on `master`.
The DoubleBlock-ViT architecture is loaded from the authors' unchanged source
at commit `5b617d72608986f38e602f4dd69b95d45443f5c7` of
[DoubleBlock-ViT-Unet-segment](https://github.com/Laptq201/DoubleBlock-ViT-Unet-segment).
The exact source and its MIT license are bundled in
`third_party/paper_baselines/doubleblock/`; the loader verifies the source's
SHA-256 and makes no network request during training. Only the two import-time
CUDA demonstration statements are skipped; the model definitions are unchanged.

The shared `PaperBaseline` adapter and training entry point also retain the
SuperLightNet option from the source branch. Its pinned public source and
GPL-3.0 license are included so that the shared entry point is self-contained.
Author revisions and checksums are recorded in
[`third_party/paper_baselines/README.md`](../third_party/paper_baselines/README.md).

## Update and install

Run from the AutoDL Git checkout, not from a results or checkpoint directory:

```bash
cd /root/autodl-tmp/mri_deep
git switch master
git pull --ff-only origin master
python -m pip install -r requirements_paper_baselines.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Use Python 3.10 and a CUDA-enabled PyTorch build compatible with the AutoDL
machine. The dependency file pins MONAI 1.4.0 and einops 0.8.1; it does not
replace the machine's PyTorch/CUDA installation. A default run requires CUDA
and fails immediately if CUDA is unavailable. CPU is allowed only for preflight.

The `path` column in `tumourCSV.csv` must point to the existing BraTS case
directories. This entry point uses the project's existing `.nii` loader; each
case must contain `_flair.nii`, `_t1.nii`, `_t1ce.nii`, `_t2.nii`, and `_seg.nii`.
Keep the case IDs, CSV row order, and fixed split unchanged when remapping paths.

## Preflight and training

```bash
python scripts/train_paper_baselines.py --model doubleblock --preflight
python -u scripts/train_paper_baselines.py --model doubleblock
```

Preflight checks the split and file paths, reads the first training case,
verifies its finite values and `[4,100,170,170]` image / `[3,100,170,170]` mask
shapes, constructs the pinned model, and tests a synthetic `32 x 32 x 32`
forward pass. It does not start training or evaluate validation/test images.
A successful small forward pass does not establish that a full volume fits
in GPU memory. Use `--preflight --device cpu` for an architecture check on CPU.

The default output directory is
`/root/autodl-tmp/paper_baselines/doubleblock/seed_55/`, containing:

- `protocol.json`: case IDs, CSV checksum, author revision, initialization,
  optimizer, training settings, region order, and measured parameter count;
- `best_model_NNN.pth`: the checkpoint selected by validation loss;
- `last_state.pth`: model, optimizer, scheduler, random states, and pending
  accumulated gradients for resuming the same run;
- `train_log.csv`: epoch, training/validation losses, learning rate, and selection.

Resume the same interrupted run with:

```bash
python -u scripts/train_paper_baselines.py --model doubleblock --resume
```

Use `--csv /path/to/tumourCSV.csv`, `--seed 55`, or
`--output-dir /path/to/new_run` when needed. A new training run starts from
random author-model initialization; it does not load ResUNet or paper weights.
Keep an existing completed run intact and select a new output directory for
a separate experiment.

## Training protocol and comparison scope

The entry point reuses `data.dataset.get_dataloader` and `BCEDiceLoss`:

| Setting | Value |
| --- | --- |
| Cases and split | Existing 368-case CSV; `random_state=10`; 257 train / 74 validation / 37 held-out test cases |
| Input | FLAIR, T1, T1ce, T2; fixed source crop `[40:210,40:210,20:120]`; modality min-max normalization; no augmentation |
| Target | Project `[WT,TC,ET]`; DoubleBlock's `[ET,TC,WT]` logits are reordered by the adapter |
| Model | Author `Unet(in_channels=4, n_classes=3, n_channels=16)` |
| Spatial adapter | High-end zero padding to multiples of 32, then crop logits back to the input shape |
| Optimization | Seed 55; Adam `5e-4`; batch size 1; gradient accumulation 4 |
| Scheduling | `ReduceLROnPlateau(mode="min", patience=2)` |
| Selection | Lowest validation loss; up to 200 epochs; early stopping patience 25, `min_delta=1e-4` |

The legacy project trainer carries an incomplete accumulation group into the
next epoch; this entry point preserves that behavior and saves pending
gradients for resume. Training and checkpoint selection do not infer on test images.

If a full volume exceeds GPU memory, `--crop-size 96 128 128` is an explicit
alternative for a **new** run. Validation then uses 25%-overlap sliding-window
inference. This changes the training protocol and must be reported; it is not
identical to an existing full-volume experiment. Crop dimensions must be
multiples of 32 and fit within the project's `100 x 170 x 170` volume.

This is an author-architecture comparison under the project's training
protocol, not a reproduction of the authors' published scores or their
complete training pipeline. The historical AFBMS checkpoint was warm-started
from ResUNet, whereas these baselines start from scratch; disclose that
initialization and training-budget difference. No test metrics are supplied
by this training-code port. Use the selected checkpoint and a separately
verified common evaluation protocol before adding a result to a paper table.
