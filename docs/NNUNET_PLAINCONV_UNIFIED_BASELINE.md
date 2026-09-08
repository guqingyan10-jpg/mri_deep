# nnU-Net v2 PlainConvUNet unified baseline

This is the comparison requested for the paper: the network is the official
nnU-Net v2 `PlainConvUNet` implementation, while training is performed under
the same repository protocol as U-Net and ResUNet. It is therefore reported
as **nnU-Net v2 PlainConvUNet architecture under the unified training
protocol**, not as the complete self-configuring nnU-Net pipeline.

Fixed protocol:

- split: 257 train / 74 validation / 37 test, `random_state=10`;
- input: the same four modalities and `(100, 170, 170)` crop;
- targets: WT, TC and ET region logits;
- loss: BCE-Dice;
- optimizer: Adam, learning rate `5e-4`;
- batch size 1, gradient accumulation 4;
- scheduler: ReduceLROnPlateau, patience 2;
- maximum setting: 200 epochs;
- early stopping: patience 25, `min_delta=1e-4`;
- model selection: lowest validation BCE-Dice loss;
- test threshold: 0.33.

The model uses five stages with features `32/64/128/256/320`, InstanceNorm3d,
LeakyReLU, strided convolutions and transposed-convolution decoding. The
official backend requires spatial divisibility by 16, so the adapter pads the
existing crop during the forward pass and crops logits back to the original
shape before loss/evaluation.

## AutoDL training

```bash
cd /root/autodl-tmp/mri_deep
git checkout master
git pull origin master

python -m pip install -r requirements_nnunet_architecture.txt

python scripts/train_nnunet_plainconv_unified.py
```

The new run has its own directory and does not overwrite the historical
hand-written nnUNet checkpoint:

```text
/root/autodl-tmp/nnUNet_v2_PlainConv_unified_model/
```

For the formal result, prefer one uninterrupted run. The shared historical
`Trainer` can reload the most recent model weights, but its legacy resume path
does not restore Adam, scheduler, early-stopping or RNG state exactly.
