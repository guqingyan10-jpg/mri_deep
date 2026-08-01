"""
Patch MultiModel XAI Brats2020.ipynb to add nnUNet3d as 4th model.
Run: python patch_notebook.py
"""
import json, copy

NB_PATH = 'MultiModel XAI Brats2020.ipynb'

with open(NB_PATH, 'r', encoding='utf-8-sig') as f:
    nb = json.load(f)

# ── Helper: make a code cell ──
def code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines[:-1]] + [source_lines[-1]]
    }

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text]
    }

def get_src(cell):
    return ''.join(cell['source'])

def set_src(cell, new_text):
    cell['source'] = [new_text]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 1: Insert new cells after Cell 55 (AttUNet training)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INSERT_AFTER = 55  # index of AttUNet training code cell

new_cells = []

# --- nnUNet Architecture markdown ---
new_cells.append(md_cell(
    "### nnUNet\n\n"
    "**A new baseline for comparison.** Uses **InstanceNorm + LeakyReLU + StridedConv + ConvTranspose3d**, "
    "replacing GroupNorm/ReLU/MaxPool/Trilinear in the original UNet.\n\n"
    "| Component | UNet3d | ResUNet3d | AttUNet3d | **nnUNet3d** |\n"
    "|-----------|--------|-----------|-----------|-------------|\n"
    "| Normalization | GroupNorm | GroupNorm | GroupNorm | **InstanceNorm** |\n"
    "| Activation | ReLU | ReLU | ReLU | **LeakyReLU** |\n"
    "| Downsampling | MaxPool3d | MaxPool3d | MaxPool3d | **Conv3d(stride=2)** |\n"
    "| Upsampling | Trilinear | Trilinear | Trilinear | **ConvTranspose3d** |\n"
    "| Skip Connection | Concat | Concat+Res | Concat+Att | **Concat** |\n\n"
    "Reference: Isensee et al., Nature Methods 2021."
))

# --- nnUNet Architecture code ---
new_cells.append(code_cell([
    "# ============================================================",
    "# nnU-Net Building Blocks",
    "# ============================================================",
    "# Key differences from original UNet:",
    "#   1. InstanceNorm3d (not GroupNorm) — stable with batch_size=1",
    "#   2. LeakyReLU (not ReLU) — prevents dying neurons",
    "#   3. Conv3d(stride=2) (not MaxPool) — learnable downsampling",
    "#   4. ConvTranspose3d (not trilinear) — learnable upsampling",
    "",
    "class nnDoubleConv(nn.Module):",
    '    """(Conv3D -> InstanceNorm3d -> LeakyReLU) * 2"""',
    "    def __init__(self, in_channels, out_channels, leaky_slope=1e-2):",
    "        super().__init__()",
    "        self.double_conv = nn.Sequential(",
    "            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),",
    "            nn.InstanceNorm3d(out_channels, affine=True),",
    "            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),",
    "",
    "            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),",
    "            nn.InstanceNorm3d(out_channels, affine=True),",
    "            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),",
    "        )",
    "",
    "    def forward(self, x):",
    "        return self.double_conv(x)",
    "",
    "",
    "class nnDown(nn.Module):  # Strided Conv downsampling (NO MaxPool)",
    '    """Conv3d(stride=2) + nnDoubleConv — learnable downsampling"""',
    "    def __init__(self, in_channels, out_channels):",
    "        super().__init__()",
    "        self.encoder = nn.Sequential(",
    "            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),",
    "            nn.InstanceNorm3d(out_channels, affine=True),",
    "            nn.LeakyReLU(negative_slope=1e-2, inplace=True),",
    "            nnDoubleConv(out_channels, out_channels),",
    "        )",
    "",
    "    def forward(self, x):",
    "        return self.encoder(x)",
    "",
    "",
    "class nnUp(nn.Module):",
    '    """ConvTranspose3d + nnDoubleConv — learnable upsampling',
    "    ",
    "    Matches original Up class interface: in_channels = skip_ch + deeper_ch",
    "    ConvTranspose3d preserves channels, 2x spatial dims.    ",
    '    """',
    "    def __init__(self, in_channels, out_channels):",
    "        super().__init__()",
    "        deeper_channels = in_channels // 2",
    "        self.up = nn.ConvTranspose3d(deeper_channels, deeper_channels, kernel_size=2, stride=2)",
    "        self.conv = nnDoubleConv(in_channels, out_channels)",
    "",
    "    def forward(self, x1, x2):",
    "        x1 = self.up(x1)",
    "        # Size matching (same logic as original Up)",
    "        diffZ = x2.size()[2] - x1.size()[2]",
    "        diffY = x2.size()[3] - x1.size()[3]",
    "        diffX = x2.size()[4] - x1.size()[4]",
    "        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,",
    "                         diffY // 2, diffY - diffY // 2,",
    "                         diffZ // 2, diffZ - diffZ // 2])",
    "        x = torch.cat([x2, x1], dim=1)",
    "        return self.conv(x)",
    "",
    "",
    "# ============================================================",
    "# nnUNet3d — The Full Architecture",
    "# ============================================================",
    "",
    "class nnUNet3d(nn.Module):",
    '    """',
    "    nnU-Net 3D for Brain Tumor Segmentation.",
    "    Input:  (B, 4, 128, 128, 128) — FLAIR, T1, T1ce, T2",
    "    Output: (B, 3, 128, 128, 128) — WT, TC, ET",
    '    """',
    "    def __init__(self, in_channels=4, n_classes=3, n_channels=24):",
    "        super().__init__()",
    "        self.in_channels = in_channels",
    "        self.n_classes = n_classes",
    "        self.n_channels = n_channels",
    "",
    "        # Encoder (strided conv downsampling)",
    "        self.conv = nnDoubleConv(in_channels, n_channels)           # 4->24, 128^3",
    "        self.enc1 = nnDown(n_channels, 2 * n_channels)              # 24->48, 64^3",
    "        self.enc2 = nnDown(2 * n_channels, 4 * n_channels)          # 48->96, 32^3",
    "        self.enc3 = nnDown(4 * n_channels, 8 * n_channels)          # 96->192, 16^3",
    "        self.enc4 = nnDown(8 * n_channels, 8 * n_channels)          # 192->192, 8^3",
    "",
    "        # Decoder (ConvTranspose3d upsampling, same interface as Up/ResUp/AttUp)",
    "        self.dec1 = nnUp(16 * n_channels, 4 * n_channels)           # 384->96",
    "        self.dec2 = nnUp(8 * n_channels, 2 * n_channels)            # 192->48",
    "        self.dec3 = nnUp(4 * n_channels, n_channels)                # 96->24",
    "        self.dec4 = nnUp(2 * n_channels, n_channels)                # 48->24",
    "        self.out = nn.Conv3d(n_channels, n_classes, kernel_size=1)",
    "",
    "    def forward(self, x):",
    "        # Encoder",
    "        x1 = self.conv(x)",
    "        x2 = self.enc1(x1)",
    "        x3 = self.enc2(x2)",
    "        x4 = self.enc3(x3)",
    "        x5 = self.enc4(x4)",
    "",
    "        # Decoder with skip connections",
    "        mask = self.dec1(x5, x4)",
    "        mask = self.dec2(mask, x3)",
    "        mask = self.dec3(mask, x2)",
    "        mask = self.dec4(mask, x1)",
    "        mask = self.out(mask)",
    "        return mask",
]))

# --- Train nnUNet markdown ---
new_cells.append(md_cell(
    "#### Train the nnUNet model\n\n"
    "**Identical hyperparameters to UNet/ResUNet/AttUNet for fair comparison:**\n"
    "- lr=5e-4, BCEDiceLoss, Adam, ReduceLROnPlateau(patience=2)\n"
    "- batch_size=1, accumulation_steps=4, num_epochs=200\n"
    "- n_channels=24 (matched parameter budget across all 4 models)"
))

# --- Train nnUNet code ---
new_cells.append(code_cell([
    "model4 = nnUNet3d(in_channels=4, n_classes=3, n_channels=24).to('cuda')",
    "",
    "trainer = Trainer(net=model4,",
    "                  dataset=BratsDataset,",
    "                  criterion=BCEDiceLoss(),",
    "                  lr=5e-4,                    # 🟢 SAME as UNet/ResUNet/AttUNet",
    "                  accumulation_steps=4,        # 🟢 SAME",
    "                  batch_size=1,                # 🟢 SAME",
    "                  fold=0,",
    "                  num_epochs=200,              # 🟢 SAME as UNet/ResUNet",
    "                  path_to_csv=config.path_to_csv,",
    "                  model_type=config.nnUNet_checkpoint_dir",
    "                  )",
    "",
    "# Create checkpoint directory",
    "import os",
    "os.makedirs(config.nnUNet_checkpoint_dir, exist_ok=True)",
    "",
    "# Resume from checkpoint if exists",
    "if check_exist(config.nnUNet_checkpoint_dir) is not None:",
    "    trainer.load_pretrain_model(check_exist(config.nnUNet_checkpoint_dir))",
    "",
    "# Load previous logs if resuming",
    "if os.path.exists(config.nnUNet_train_logs_path):",
    "    train_logs = pd.read_csv(config.nnUNet_train_logs_path)",
    "else:",
    '    cols = ["train_loss","valid_loss","train_dice","valid_dice","train_jaccard","valid_jaccard","train_time","valid_time"]',
    "    train_logs = pd.DataFrame({c: [] for c in cols})",
    'trainer.losses["train"] =  train_logs.loc[:, "train_loss"].to_list()',
    'trainer.losses["valid"] =  train_logs.loc[:, "valid_loss"].to_list()',
    'trainer.dice_scores["train"] = train_logs.loc[:, "train_dice"].to_list()',
    'trainer.dice_scores["valid"] = train_logs.loc[:, "valid_dice"].to_list()',
    'trainer.jaccard_scores["train"] = train_logs.loc[:, "train_jaccard"].to_list()',
    'trainer.jaccard_scores["valid"] = train_logs.loc[:, "valid_jaccard"].to_list()',
    'trainer.time["train"] = train_logs.loc[:, "train_time"].to_list()',
    'trainer.time["valid"] = train_logs.loc[:, "valid_time"].to_list()',
    "",
    "trainer.run(config.nnUNet_checkpoint_dir)",
]))

# Insert after Cell 55
nb['cells'][INSERT_AFTER+1:INSERT_AFTER+1] = new_cells
SHIFT = len(new_cells)
print(f"Inserted {SHIFT} new cells after cell {INSERT_AFTER} (AttUNet training)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 2: Modify existing cells
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Helper: apply replacement to shifted cell
def patch(old_idx, old_str, new_str):
    """Replace old_str with new_str in cell at original index old_idx (shift is applied automatically)"""
    actual_idx = old_idx + SHIFT
    src = get_src(nb['cells'][actual_idx])
    if old_str in src:
        nb['cells'][actual_idx]['source'] = [src.replace(old_str, new_str)]
        print(f"  Cell {old_idx}→{actual_idx}: OK")
    else:
        print(f"  Cell {old_idx}→{actual_idx}: WARNING - pattern NOT FOUND!")
        # Show what we're looking for
        print(f"    Find: {repr(old_str[:120])}")

print("\n--- Phase 2: Modify existing cells ---")

# ── Cell 59: Load 4 models instead of 3 ──
patch(59,
    '# 创建三个模型并加载最佳权重\n',
    '# 创建四个模型并加载最佳权重\n')
patch(59,
    'print("三个模型加载完成，均已置为 eval 模式")',
    'print("四个模型加载完成，均已置为 eval 模式")')
# Add nnUNet loading after AttUNet loading
patch(59,
    'UNet.eval()\nResUNet.eval()\nAttUNet.eval()',
    '# 🆕 加载 nnUNet\n'
    'nnUNet = nnUNet3d(in_channels=4, n_classes=3, n_channels=24).to(device)\n'
    'nn_state = torch.load(check_exist(config.nnUNet_checkpoint_dir), map_location=device)\n'
    'nnUNet.load_state_dict(nn_state)\n'
    '\n'
    'UNet.eval()\n'
    'ResUNet.eval()\n'
    'AttUNet.eval()\n'
    'nnUNet.eval()   # 🆕')

# ── Cell 70 (val): add nnUNet compute_metrics ──
patch(70,
    'attTP, attFP, attTN, attFN = compute_metrics(AttUNet, val_dataloader, threshold=0.33)',
    'attTP, attFP, attTN, attFN = compute_metrics(AttUNet, val_dataloader, threshold=0.33)\n'
    'nnTP, nnFP, nnTN, nnFN = compute_metrics(nnUNet, val_dataloader, threshold=0.33)  # 🆕')

# ── Cell 71 (val): confusion matrix 1x3 → 1x4 ──
patch(71,
    'fig, axes = plt.subplots(1, 3, figsize=(18, 6))',
    'fig, axes = plt.subplots(1, 4, figsize=(24, 6))  # 🆕 3→4 models')
patch(71,
    'plot_confusion_matrix(axes[2], attTP, attFP, attTN, attFN, "Confusion Matrix AttUNet (Validation)")',
    'plot_confusion_matrix(axes[2], attTP, attFP, attTN, attFN, "Confusion Matrix AttUNet (Validation)")\n'
    'plot_confusion_matrix(axes[3], nnTP, nnFP, nnTN, nnFN, "Confusion Matrix nnUNet (Validation)")  # 🆕')

# ── Cell 72 (val): add nnUNetMetric to dictionary ──
patch(72,
    'AttUNetMetric = metric(attTP, attTN, attFP, attFN)\n',
    'AttUNetMetric = metric(attTP, attTN, attFP, attFN)\n'
    'nnUNetMetric = metric(nnTP, nnTN, nnFP, nnFN)  # 🆕\n')
patch(72,
    "dictionary = {'UNet':UNetMetric,\n"
    "        'ResUNet':ResUNetMetric,\n"
    "        'AttUNet':AttUNetMetric\n"
    "        }",
    "dictionary = {'UNet':UNetMetric,\n"
    "        'ResUNet':ResUNetMetric,\n"
    "        'AttUNet':AttUNetMetric,\n"
    "        'nnUNet':nnUNetMetric  # 🆕\n"
    "        }")

# ── Cell 75 (test): add nnUNet compute_metrics ──
patch(75,
    'attTP, attFP, attTN, attFN = compute_metrics(AttUNet, test_dataloader, threshold=0.33)',
    'attTP, attFP, attTN, attFN = compute_metrics(AttUNet, test_dataloader, threshold=0.33)\n'
    'nnTP, nnFP, nnTN, nnFN = compute_metrics(nnUNet, test_dataloader, threshold=0.33)  # 🆕')

# ── Cell 76 (test): confusion matrix 1x3 → 1x4 ──
patch(76,
    'fig, axes = plt.subplots(1, 3, figsize=(18, 6))',
    'fig, axes = plt.subplots(1, 4, figsize=(24, 6))  # 🆕 3→4 models')
patch(76,
    'plot_confusion_matrix(axes[2], attTP, attFP, attTN, attFN, "Confusion Matrix AttUNet (Testing)")',
    'plot_confusion_matrix(axes[2], attTP, attFP, attTN, attFN, "Confusion Matrix AttUNet (Testing)")\n'
    'plot_confusion_matrix(axes[3], nnTP, nnFP, nnTN, nnFN, "Confusion Matrix nnUNet (Testing)")  # 🆕')

# ── Cell 77 (test): add nnUNetMetric to dictionary ──
patch(77,
    'AttUNetMetric = metric(attTP, attTN, attFP, attFN)\n',
    'AttUNetMetric = metric(attTP, attTN, attFP, attFN)\n'
    'nnUNetMetric = metric(nnTP, nnTN, nnFP, nnFN)  # 🆕\n')
patch(77,
    "dictionary = {'UNet':UNetMetric,\n"
    "        'ResUNet':ResUNetMetric,\n"
    "        'AttUNet':AttUNetMetric\n"
    "        }",
    "dictionary = {'UNet':UNetMetric,\n"
    "        'ResUNet':ResUNetMetric,\n"
    "        'AttUNet':AttUNetMetric,\n"
    "        'nnUNet':nnUNetMetric  # 🆕\n"
    "        }")

# ── Cell 85: add nnUNet train log ──
patch(85,
    "att_train_data = pd.read_csv(r'/root/autodl-tmp/AttUNet_model\\train_log.csv')",
    "att_train_data = pd.read_csv(r'/root/autodl-tmp/AttUNet_model\\train_log.csv')\n"
    "nn_train_data = pd.read_csv(r'/root/autodl-tmp/nnUNet_model\\train_log.csv')  # 🆕")
patch(85,
    'df_time = pd.concat([gettime(base_train_data, "UNet"),gettime(res_train_data, "ResUNet"),gettime(att_train_data, "AttUNet")])',
    'df_time = pd.concat([gettime(base_train_data, "UNet"),gettime(res_train_data, "ResUNet"),gettime(att_train_data, "AttUNet"),gettime(nn_train_data, "nnUNet")])  # 🆕')

# ── Cell 87: add nnUNet to default model lists ──
patch(87,
    "def plotScoresindi(metric, model=['UNet', 'ResUNet', 'AttUNet']):",
    "def plotScoresindi(metric, model=['UNet', 'ResUNet', 'AttUNet', 'nnUNet']):  # 🆕")
patch(87,
    "def plotScores(metric, models=['UNet', 'ResUNet', 'AttUNet']):",
    "def plotScores(metric, models=['UNet', 'ResUNet', 'AttUNet', 'nnUNet']):  # 🆕")

# ── Cell 96-98: Already fine (UNet/ResUNet/AttUNet val per-class)
# ── Insert new cell after 98: nnUNet val per-class ──
nn_val_cell = code_cell([
    "nndice_scores_per_classes, nniou_scores_per_classes = compute_scores_per_classes(",
    "    nnUNet, val_dataloader, ['WT', 'TC', 'ET']  # 🆕",
    "    )",
    "",
    "nndice_df = pd.DataFrame(nndice_scores_per_classes)",
    "nndice_df.columns = ['WT dice', 'TC dice', 'ET dice']",
    "",
    "nniou_df = pd.DataFrame(nniou_scores_per_classes)",
    "nniou_df.columns = ['WT jaccard', 'TC jaccard', 'ET jaccard']",
    "",
    "val_metrics_df = pd.concat([dice_df, iou_df], axis=1, sort=False)",
    "resval_metrics_df = pd.concat([resdice_df, resiou_df], axis =1, sort=False)",
    "attval_metrics_df = pd.concat([attdice_df, attiou_df], axis=1, sort=False)",
    "nnval_metrics_df = pd.concat([nndice_df, nniou_df], axis=1, sort=False)  # 🆕",
])
# 98 shifted to 98+SHIFT
nb['cells'].insert(98 + SHIFT + 1, nn_val_cell)
SHIFT += 1
print(f"  Inserted nnUNet val per-class cell after 98→{98+SHIFT-1} (SHIFT now {SHIFT})")

# ── Cell 99: add nnUNet to summary tables ──
patch(99,
    "dice_avg =pd.DataFrame([np.mean(dice_df.values, axis =0).round(6), np.mean(resdice_df.values, axis =0).round(6), np.mean(attdice_df.values, axis =0).round(6)], index = ['UNet', 'ResUNet', 'AttUNet'], columns = dice_df.columns)",
    "dice_avg =pd.DataFrame([np.mean(dice_df.values, axis =0).round(6), np.mean(resdice_df.values, axis =0).round(6), np.mean(attdice_df.values, axis =0).round(6), np.mean(nndice_df.values, axis =0).round(6)], index = ['UNet', 'ResUNet', 'AttUNet', 'nnUNet'], columns = dice_df.columns)  # 🆕")
patch(99,
    "jac_avg = pd.DataFrame([np.mean(iou_df.values, axis =0).round(6), np.mean(resiou_df.values, axis =0).round(6), np.mean(attiou_df.values, axis =0).round(6)], index = ['UNet', 'ResUNet', 'AttUNet'], columns = iou_df.columns)",
    "jac_avg = pd.DataFrame([np.mean(iou_df.values, axis =0).round(6), np.mean(resiou_df.values, axis =0).round(6), np.mean(attiou_df.values, axis =0).round(6), np.mean(nniou_df.values, axis =0).round(6)], index = ['UNet', 'ResUNet', 'AttUNet', 'nnUNet'], columns = iou_df.columns)  # 🆕")

# ── Cell 100: add nnUNet to rowLabels ──
patch(100,
    "axs[0].table(cellText=dice_avg.values, colLabels=dice_df.columns, cellLoc='center', loc='center', rowLabels = ['UNet', 'ResUNet', 'AttUNet'])",
    "axs[0].table(cellText=dice_avg.values, colLabels=dice_df.columns, cellLoc='center', loc='center', rowLabels = ['UNet', 'ResUNet', 'AttUNet', 'nnUNet'])  # 🆕")
patch(100,
    "axs[1].table(cellText=jac_avg.values, colLabels=iou_df.columns, cellLoc='center', loc='center', rowLabels = ['UNet', 'ResUNet', 'AttUNet'])",
    "axs[1].table(cellText=jac_avg.values, colLabels=iou_df.columns, cellLoc='center', loc='center', rowLabels = ['UNet', 'ResUNet', 'AttUNet', 'nnUNet'])  # 🆕")

# ── Cell 101: bar plot 3→4 subplots ──
patch(101,
    'fig, axs = plt.subplots(3, 1, figsize=(10, 20))',
    'fig, axs = plt.subplots(4, 1, figsize=(10, 26))  # 🆕 3→4 models')
patch(101,
    '# Plotting AttUNet metrics\nbar(attval_metrics_df, "AttUNet", type=\'Validation\',ax=axs[2])',
    '# Plotting AttUNet metrics\nbar(attval_metrics_df, "AttUNet", type=\'Validation\',ax=axs[2])\n\n'
    '# Plotting nnUNet metrics\nbar(nnval_metrics_df, "nnUNet", type=\'Validation\',ax=axs[3])  # 🆕')

# ── Cell 103-105: test per-class (fine as-is)
# ── Insert new cell after 105: nnUNet test per-class ──
nn_test_cell = code_cell([
    "nndice_scores_per_classes, nniou_scores_per_classes = compute_scores_per_classes(",
    "    nnUNet, test_dataloader, ['WT', 'TC', 'ET']  # 🆕",
    "    )",
    "",
    "nndice_df = pd.DataFrame(nndice_scores_per_classes)",
    "nndice_df.columns = ['WT dice', 'TC dice', 'ET dice']",
    "",
    "nniou_df = pd.DataFrame(nniou_scores_per_classes)",
    "nniou_df.columns = ['WT jaccard', 'TC jaccard', 'ET jaccard']",
    "",
    "test_metrics_df = pd.concat([dice_df, iou_df], axis=1, sort=False)",
    "restest_metrics_df = pd.concat([resdice_df, resiou_df], axis =1, sort=False)",
    "attest_metrics_df = pd.concat([attdice_df, attiou_df], axis=1, sort=False)",
    "nntest_metrics_df = pd.concat([nndice_df, nniou_df], axis=1, sort=False)  # 🆕",
])
nb['cells'].insert(105 + SHIFT + 1, nn_test_cell)
SHIFT += 1
print(f"  Inserted nnUNet test per-class cell after 105→{105+SHIFT-1} (SHIFT now {SHIFT})")

# ── Cell 106: add nnUNet to test summary tables ──
patch(106,
    "dice_avg =pd.DataFrame([np.mean(dice_df.values, axis =0).round(6), np.mean(resdice_df.values, axis =0).round(6), np.mean(attdice_df.values, axis =0).round(6)], index = ['UNet', 'ResUNet', 'AttUNet'], columns = dice_df.columns)",
    "dice_avg =pd.DataFrame([np.mean(dice_df.values, axis =0).round(6), np.mean(resdice_df.values, axis =0).round(6), np.mean(attdice_df.values, axis =0).round(6), np.mean(nndice_df.values, axis =0).round(6)], index = ['UNet', 'ResUNet', 'AttUNet', 'nnUNet'], columns = dice_df.columns)  # 🆕")
patch(106,
    "jac_avg = pd.DataFrame([np.mean(iou_df.values, axis =0).round(6), np.mean(resiou_df.values, axis =0).round(6), np.mean(attiou_df.values, axis =0).round(6)], index = ['UNet', 'ResUNet', 'AttUNet'], columns = iou_df.columns)",
    "jac_avg = pd.DataFrame([np.mean(iou_df.values, axis =0).round(6), np.mean(resiou_df.values, axis =0).round(6), np.mean(attiou_df.values, axis =0).round(6), np.mean(nniou_df.values, axis =0).round(6)], index = ['UNet', 'ResUNet', 'AttUNet', 'nnUNet'], columns = iou_df.columns)  # 🆕")

# ── Cell 107: add nnUNet to rowLabels ──
patch(107,
    "axs[0].table(cellText=dice_avg.values, colLabels=dice_df.columns, cellLoc='center', loc='center', rowLabels = ['UNet', 'ResUNet', 'AttUNet'])",
    "axs[0].table(cellText=dice_avg.values, colLabels=dice_df.columns, cellLoc='center', loc='center', rowLabels = ['UNet', 'ResUNet', 'AttUNet', 'nnUNet'])  # 🆕")
patch(107,
    "axs[1].table(cellText=jac_avg.values, colLabels=iou_df.columns, cellLoc='center', loc='center', rowLabels = ['UNet', 'ResUNet', 'AttUNet'])",
    "axs[1].table(cellText=jac_avg.values, colLabels=iou_df.columns, cellLoc='center', loc='center', rowLabels = ['UNet', 'ResUNet', 'AttUNet', 'nnUNet'])  # 🆕")

# ── Cell 108: bar plot test 3→4 subplots ──
patch(108,
    'fig, axs = plt.subplots(3, 1, figsize=(10, 20))',
    'fig, axs = plt.subplots(4, 1, figsize=(10, 26))  # 🆕 3→4 models')
patch(108,
    '# Plotting AttUNet metrics\nbar(atttest_metrics_df, "AttUNet", type = \'Testing\', ax=axs[2])',
    '# Plotting AttUNet metrics\nbar(atttest_metrics_df, "AttUNet", type = \'Testing\', ax=axs[2])\n\n'
    '# Plotting nnUNet metrics\nbar(nntest_metrics_df, "nnUNet", type = \'Testing\', ax=axs[3])  # 🆕')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PHASE 3: Add nnUNet prediction visualization section
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# After the AttUNet prediction cells (127-133 original), add nnUNet prediction cells

print("\n--- Phase 3: Add nnUNet prediction visualization ---")

# Insert after cell 133 (AttUNet prediction GIF cell, shifted)
nn_pred_cells = [
    md_cell("### Prediction for nnUNet"),
    code_cell([
        "nnUNet_results = compute_results(nnUNet, test_dataloader, 0.33)  # 🆕",
    ]),
    code_cell([
        "# Visualize nnUNet predictions interactively",
        "nn_img_list = nnUNet_results['image']",
        "nn_gt_list = nnUNet_results['GT']",
        "nn_prediction_list = nnUNet_results['Prediction']",
        "n_slices = 100",
        "interact(tumour_graphics, n_slice=widgets.IntSlider(min=0, max=n_slices-1, step=1, value=0),",
        "         img=fixed(nn_img_list), gt=fixed(nn_gt_list), prediction=fixed(nn_prediction_list), n_slices=fixed(n_slices))",
    ]),
    code_cell([
        "generate_3d_plotly(nn_img_list[BRAIN_INDEX], nn_prediction_list[BRAIN_INDEX], 'nnUNet Prediction')",
    ]),
    code_cell([
        "show_result.plot(nn_img_list[BRAIN_INDEX], nn_gt_list[BRAIN_INDEX], nn_prediction_list[BRAIN_INDEX])",
    ]),
]

# 133 + SHIFT
insert_pos = 133 + SHIFT + 1
nb['cells'][insert_pos:insert_pos] = nn_pred_cells
SHIFT += len(nn_pred_cells)
print(f"Inserted {len(nn_pred_cells)} nnUNet prediction cells after 133→{insert_pos} (final SHIFT {SHIFT})")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SAVE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\n{'='*60}")
print("DONE! Notebook updated with nnUNet3d as 4th model.")
print(f"Saved to: {NB_PATH}")
print(f"{'='*60}")
