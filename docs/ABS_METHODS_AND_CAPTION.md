# ABS methods and figure caption

## Suggested Methods text (implementation-verified)

**Auxiliary boundary supervision.** To provide explicit boundary-related supervision, we attach an auxiliary prediction head to the final decoder feature map. The head comprises a 3×3×3 convolution with 24 output channels, group normalization with eight groups, a rectified linear activation, and a 1×1×1 convolution producing K boundary-logit channels. K is one for binary brain-metastasis segmentation and three for the overlapping WT, TC, and ET regions in BraTS. The segmentation and boundary heads share the final decoder features.

For each binary region mask Y, the boundary target is generated as Y_b = Y − E_3D(Y), where E_3D denotes binary erosion with a full 3×3×3 neighbourhood and zero-valued padding. In implementation, erosion is obtained by thresholding three-dimensional average pooling at 0.999. Boundary targets are generated independently for each region channel. The auxiliary objective is binary cross-entropy with logits, averaged over all voxels. The total objective is L_total = L_seg + 0.1 L_bnd, where L_seg retains the baseline binary-cross-entropy and global Dice objective. The final segmentation is obtained from the segmentation head; the auxiliary boundary prediction is used as a training objective.

The auxiliary task requires no additional manual annotations. It encourages the shared decoder representation to retain boundary-related information. Its contribution to small-lesion detection and segmentation is evaluated through controlled ablation, rather than inferred from the boundary illustrations alone.

## English figure caption

**Figure X. Auxiliary boundary supervision (ABS).** (a) Boundary targets are generated from each binary ground-truth region by subtracting three-dimensional erosion from the original mask. MRI and label illustrations are from UCSF case 100106A, native axial slice 13, with a common lesion-centred ROI. These native-grid maps demonstrate the implemented operator; actual training targets are generated after dataset preprocessing. The displayed boundary is a slice through a three-dimensional surface target, rather than a contour extracted from the displayed two-dimensional slice. (b) The segmentation and auxiliary boundary heads receive the same final decoder tensor. Their outputs are logits; illustrated output tensors are schematic. (c) The segmentation objective is combined with the auxiliary boundary binary-cross-entropy objective using a fixed weight of 0.1. K = 1 for UCSF and K = 3 for BraTS WT/TC/ET. GN, group normalization; BCE, binary cross-entropy implemented with logits.

## 中文说明与证据边界

- 当前实图为 UCSF 100106A，slice 13；另保留 100101A，slice 43 作为三维腐蚀后内部可能消失的实例。数据均为真实 MRI 和真实标签，没有生成或伪造预测图。
- `volume_boundary_fraction` 是某一病例整个所选区域的边界体素占比，不是单病灶统计，也不是检出率或分割性能。病例包含多个病灶时，不可把该值当作代表性单个小病灶指标。
- “小病灶边界占比高”是一个可检验的几何解释，但当前边界 BCE 是全体素平均，因此不能直接写成“小病灶总惩罚更大”。还须测量按病灶分组的梯度/损失与控制变量消融。
- 当前代码的 ABS 会增加训练参数：UCSF 边界头 15,649 个，BraTS 边界头 15,699 个。仅在单独导出不计算边界分支的推理模型时，可省去该分支的参数和计算。当前 forward 仍返回两个头，图中不应声称其天然不增加参数量。
- 本地 BraTS MRI 文件曾检测出体素数据截断，未拿损坏数据出图；服务器脚本支持完整 BraTS2020 原始标签的 WT/TC/ET 三种区域。
- 别人的边界分支图只用于参考布局和相关工作，不能替代自己的模型路径，也不能用其结果证明本模型的机制。

## Related-work reference

Hatamizadeh A, Terzopoulos D, Myronenko A. Edge-Gated CNNs for Volumetric Semantic Segmentation of Medical Images. arXiv:2002.04207 (2020). https://arxiv.org/abs/2002.04207

Its Figure 1 shows region/edge streams and their objectives. Unlike ABS here, EG-CNN uses a multiscale gated stream, 3D Sobel-derived targets, and consistency supervision. The downloaded paper, cropped reference figure, and an extracted vector convolution-block icon are supplied in `reference_assets`, with their origin documented. The ABS drawing is independently constructed from this repository's implementation.
