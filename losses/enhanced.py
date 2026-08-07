"""
Enhanced Loss Functions for BraTS2020 Segmentation.
Extracted from: resunet_enhanced.py

New losses beyond the original BCEDiceLoss:
  - CELoss: Weighted Cross-Entropy with class weights (ET > TC > WT)
  - BoundaryLoss: Edge-aware loss using distance transform (Kervadec 2019)
  - DiceCEBoundaryLoss: Combined loss = alpha*Dice + beta*CE + gamma*Boundary
  - CCLevelDiceLoss: Instance-level Dice — per-connected-component Dice
    averaged equally (small lesions get same vote as large ones)
  - DiceCCELoss: Global Dice + CC-level Dice + weighted CE — SLA-FB
    Step 2 loss, single-variable change from BCEDiceLoss

Reference for CC-level Dice:
    "Instance-level Dice Loss for Brain Tumor Segmentation"
    — Each ET connected component contributes equally to the loss,
      preventing large tumors from dominating small-lesion gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage
from losses.basics import DiceLoss


# --- NEW: Enhanced Loss Functions ---

class CELoss(nn.Module):
    """
    Cross-Entropy Loss for multi-label segmentation.

    What is CE Loss?
    ----------------
    Cross-Entropy measures the difference between predicted probability
    distribution and the true distribution, pixel by pixel:

        L_CE = -1/N * sum( y_i * log(p_i) + (1-y_i) * log(1-p_i) )

    For BraTS multi-class (WT, TC, ET), we apply BCEWithLogitsLoss
    per channel, which is equivalent to per-class binary CE.

    Unlike Dice loss (which optimizes overlap), CE penalizes every
    misclassified pixel equally, providing stable gradients.
    """
    def __init__(self, class_weights=None):
        """
        Args:
            class_weights: tensor of shape (n_classes,) — higher weight
                          means higher penalty for that class.
                          Recommended: [1.0, 2.0, 4.0] for [WT, TC, ET]
        """
        super(CELoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.class_weights is not None:
            # Apply per-class weights: weight each channel differently
            weights = self.class_weights.to(logits.device)
            # Expand weights to match target spatial dims
            # weights: (C,) -> (1, C, 1, 1, 1)
            w = weights.view(1, -1, 1, 1, 1)
            # Weighted BCE: -[w*y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x))]
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            bce = (bce * w).mean()
            return bce
        else:
            return F.binary_cross_entropy_with_logits(logits, targets)


class BoundaryLoss(nn.Module):
    """
    Boundary Distance (BD) Loss — Kervadec et al., MIDL 2019.

    Reference:
        Kervadec, Bouchtiba, Desrosiers, et al.
        "Boundary loss for highly unbalanced segmentation"
        MIDL 2019. https://arxiv.org/abs/1812.07032

    HOW THE ORIGINAL BD LOSS WORKS (Kervadec 2019):
    ─────────────────────────────────────────────────
    Given a ground truth region G with boundary ∂G:

        1. Pre-compute φ_G(p) — the signed distance function:
           φ_G(p) < 0  ⇒  p is inside G (distance to ∂G, negative)
           φ_G(p) = 0  ⇒  p is ON ∂G
           φ_G(p) > 0  ⇒  p is outside G

        2. L_BD = ∫_Ω φ_G(p) · s_θ(p) dp

           where s_θ(p) is the softmax probability at pixel p.

        The intuition: s_θ(p) is multiplied by φ_G(p). If φ_G(p) is
        large positive (far outside G), the model is heavily penalized
        for predicting high probability there. If φ_G(p) is large
        negative (deep inside G), the model is penalized for predicting
        LOW probability there.

    OUR SIMPLIFIED IMPLEMENTATION (common in BraTS literature):
    ─────────────────────────────────────────────────────────────
        We use the distance transform to create a smooth "boundary
        band" weight map, then apply it to BCE:

        1. Extract GT surface via binary erosion:
           ∂G = mask  XOR  eroded(mask)

        2. Compute Euclidean distance d(p) from each foreground pixel
           to the nearest surface point using
           scipy.ndimage.distance_transform_edt.
           → surface: d ≈ 0
           → deep interior: d ≈ region_radius

        3. Weight map (exponential decay from boundary):
           w(p) = 1 + (W_max - 1) · exp(-α · d(p))

           → boundary (d=0):  w = W_max     (max penalty)
           → interior (d≫0):  w ≈ 1         (baseline penalty)
           → background:       w = 1         (unchanged)

        4. L_boundary = mean( w(p) · BCE(p) )

    WHY THIS IS BETTER THAN LAPLACIAN-BASED EDGE DETECTION:
    ─────────────────────────────────────────────────────
        Laplacian (old):  1-pixel sharp boundary line
        Distance (new):   Smooth 2-3 pixel "band", better gradients

        boundary pixel (d=0):     weight = 5.0  — max emphasis
        neighbor pixel (d=1):     weight ≈ 3.8  — still emphasized
        neighbor pixel (d=2):     weight ≈ 2.5  — moderate
        far interior (d≫5):       weight ≈ 1.0  — baseline BCE

        The smooth falloff means the model receives a "gradient signal"
        that gradually increases as predictions approach the boundary,
        instead of a sharp binary switch. This is the key insight of
        Kervadec 2019 — the distance function provides a spatially
        smooth supervisory signal.

    Args:
        max_weight:  W_max, boundary pixel multiplier (default 5.0).
                     Higher = stronger boundary emphasis.
        alpha:       Decay rate per voxel (default 1.0).
                     Larger = weight decays faster from boundary.
                     At d=1: w≈1+(W_max-1)·e^{-α} ≈ 1+4·0.37≈2.5
                     At d=3: w≈1+(W_max-1)·e^{-3α}≈1+4·0.05≈1.2
    """
    def __init__(self, max_weight: float = 5.0, alpha: float = 1.0):
        super(BoundaryLoss, self).__init__()
        self.max_weight = max_weight
        self.alpha = alpha

    def get_boundary_weights(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute per-pixel boundary weights using the distance transform.

        Steps:
          1. Erode the GT mask (6-connectivity) to get the inner region.
          2. boundary = mask XOR eroded → 1-pixel surface.
          3. Compute Euclidean distance from each pixel to the surface.
          4. w = 1 + (W_max - 1) * exp(-alpha * distance).

        Returns:
            weights: (B, C, D, H, W) float tensor, same device as targets.
                     Weights are computed on CPU per-channel and moved
                     back to GPU. Gradients are detached (weights are
                     treated as constants in the loss computation).
        """
        B, C, D, H, W = targets.shape
        weights = torch.ones_like(targets, device=targets.device)

        # 6-connectivity structure element for 3D erosion
        struct = ndimage.generate_binary_structure(3, 1)

        for b in range(B):
            for c in range(C):
                gt = targets[b, c].cpu().numpy().astype(bool)

                if gt.sum() == 0:
                    continue  # no GT → weights stay at 1.0

                # Step 1: Extract GT surface via binary erosion
                eroded = ndimage.binary_erosion(gt, structure=struct)
                surface = gt & (~eroded)

                if surface.sum() == 0:
                    continue  # no surface → weights stay at 1.0

                # Step 2: Distance transform of ~surface
                # ~surface: 1 everywhere, 0 at surface points
                # EDT computes distance from each pixel to nearest 0
                dist = ndimage.distance_transform_edt(~surface)

                # Step 3: Weight map with exponential decay
                # surface (d=0): exp(0)=1 → weight = max_weight
                # interior (d large): exp(-alpha*d)→0 → weight→1
                w = 1.0 + (self.max_weight - 1.0) * np.exp(-self.alpha * dist)
                w[~gt] = 1.0  # background stays at weight 1.0

                weights[b, c] = torch.from_numpy(w).float().to(targets.device)

        return weights.detach()  # no gradient through distance transform

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary-weighted BCE loss.

        L = (1/N) * Σ w(p) * BCE(logits(p), targets(p))

        where w(p) is computed from the GT surface via distance transform.
        """
        weights = self.get_boundary_weights(targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return (bce * weights).mean()


class DiceCEBoundaryLoss(nn.Module):
    """
    Combined Loss: Dice + Weighted CE + Boundary

    Loss = alpha * DiceLoss
         + beta  * CELoss (with class weights: ET > TC > WT)
         + gamma * BoundaryLoss

    Default weights tuned for BraTS:
        alpha=1.0 (dice), beta=0.5 (CE), gamma=lambda_b (boundary, tunable)
        Class weights: WT=1.0, TC=3.0, ET=5.0 (higher penalty for ET/TC)
    """
    def __init__(
        self,
        alpha: float = 1.0,         # Dice weight
        beta: float = 0.5,          # CE weight
        gamma: float = 0.3,         # Boundary weight (lambda_b)
        class_weights: list = None, # [WT, TC, ET] weights
        bd_max_weight: float = 5.0, # BoundaryLoss: max weight at surface
        bd_alpha: float = 1.0,      # BoundaryLoss: distance decay rate
    ):
        super(DiceCEBoundaryLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if class_weights is None:
            # Default: ET highest (5x), TC high (3x), WT baseline (1x)
            class_weights = [1.0, 3.0, 5.0]

        self.dice = DiceLoss()
        self.ce = CELoss(class_weights=torch.tensor(class_weights))
        self.boundary = BoundaryLoss(max_weight=bd_max_weight, alpha=bd_alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert logits.shape == targets.shape

        loss_dice = self.dice(logits, targets)
        loss_ce = self.ce(logits, targets)
        loss_boundary = self.boundary(logits, targets)

        total = self.alpha * loss_dice + self.beta * loss_ce + self.gamma * loss_boundary
        return total

    def log_components(self, logits, targets):
        """Return individual loss components for logging."""
        with torch.no_grad():
            d = self.dice(logits, targets).item()
            c = self.ce(logits, targets).item()
            b = self.boundary(logits, targets).item()
        return {'dice': d, 'ce': c, 'boundary': b}


# ================================================================
# Instance-Level (CC-Level) Dice Loss
# ================================================================

class CCLevelDiceLoss(nn.Module):
    """
    Instance-level Dice Loss — per-connected-component Dice.

    Reference:
        "Instance-level Dice Loss for Brain Tumor Segmentation"
        — Each ET connected component contributes equally to the loss,
          preventing large tumors from dominating small-lesion gradients.

    Motivation:
      Global Dice computes one scalar over the entire volume:
        L = 1 - 2·Σ(pred·GT) / (Σ(pred²) + Σ(GT²))
      This is dominated by large tumors (thousands of voxels → large
      gradient contribution). A 20-voxel ET lesion contributes ~0.4%
      of the gradient → functionally invisible.

    Solution:
      1. Extract all ET connected components from GT (3D 26-connectivity).
      2. For each component, compute Dice over the component's spatial
         extent (NOT the full volume — each lesion is evaluated in its
         own local context).
      3. Average the per-component Dice losses with EQUAL weight.
         → A 20-voxel lesion contributes as much as a 5000-voxel one.

    Args:
        min_component_size: ignore components smaller than this (noise).
        eps: numerical stability for Dice computation.
        n_classes: number of output classes (3 for BraTS: WT, TC, ET).
        et_channel: which channel index corresponds to ET (default 2).
    """

    def __init__(self, min_component_size=10, eps=1e-9,
                 n_classes=3, et_channel=2):
        super().__init__()
        self.min_size = min_component_size
        self.eps = eps
        self.n_classes = n_classes
        self.et_channel = et_channel

    def _extract_components(self, gt_et):
        """
        Extract ET connected components from GT.

        Args:
            gt_et: (D, H, W) numpy binary array, ET channel only.

        Returns:
            list of (D,H,W) numpy binary masks, one per component,
            or empty list if no components found.
        """
        from scipy.ndimage import label as connected_components

        labeled, n_comp = connected_components(gt_et)

        components = []
        for k in range(1, n_comp + 1):
            comp_mask = (labeled == k)
            if comp_mask.sum() < self.min_size:
                continue
            components.append(comp_mask.astype(bool))

        return components

    def _per_component_dice(self, pred_batch, gt_batch, components_batch):
        """
        Compute instance-level Dice for a batch of samples.

        Args:
            pred_batch: (B, D, H, W) probability map for ET channel (sigmoid output).
            gt_batch:   (B, D, H, W) binary GT for ET channel.
            components_batch: list of lists — for each sample in batch,
                              a list of component binary masks.

        Returns:
            cc_loss: scalar tensor — average (1 - Dice_k) across all
                     valid components in the batch.
            n_components: int — total number of components used.
        """
        total_loss = 0.0
        total_comp = 0

        for b in range(len(components_batch)):
            comps = components_batch[b]
            if len(comps) == 0:
                continue

            pred = pred_batch[b]  # (D, H, W)
            gt = gt_batch[b]      # (D, H, W)

            for comp_mask in comps:
                # Only evaluate Dice within this component's spatial extent.
                # This isolates each lesion — no cross-contamination from
                # other lesions or background.
                comp_t = torch.from_numpy(comp_mask).to(pred.device)

                # Crop to component bounding box for efficiency
                # (optional — keeps the math simple for now)
                p_comp = pred * comp_t.float()
                g_comp = gt  * comp_t.float()

                # Local Dice within the component's footprint
                intersection = 2.0 * (p_comp * g_comp).sum()
                union = (p_comp ** 2).sum() + (g_comp ** 2).sum()

                dice_k = (intersection + self.eps) / (union + self.eps)
                total_loss += (1.0 - dice_k)
                total_comp += 1

        if total_comp == 0:
            # No ET components found in this batch — return zero
            return torch.tensor(0.0, device=pred_batch.device,
                                requires_grad=True), 0

        return total_loss / total_comp, total_comp

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, D, H, W) raw logits.
            targets: (B, C, D, H, W) one-hot GT.

        Returns:
            cc_loss: scalar — average per-component Dice loss.
        """
        # Extract ET channel
        pred_et = torch.sigmoid(logits[:, self.et_channel])  # (B, D, H, W)
        gt_et_cpu = (targets[:, self.et_channel] > 0.5).cpu().numpy()

        # Extract components per sample (CPU, pre-computed once per forward)
        all_components = []
        for b in range(targets.shape[0]):
            comps = self._extract_components(gt_et_cpu[b])
            all_components.append(comps)

        cc_loss, n_comp = self._per_component_dice(pred_et, targets[:, self.et_channel], all_components)
        return cc_loss


# ================================================================
# BCEDiceLoss + CC-Level Dice (single-variable change from baseline)
# ================================================================

class BCEDiceCCLoss(nn.Module):
    """
    Baseline loss + Instance-Level Dice — single variable change.

    Formula:
        L = L_BCEDice + λ_cc · L_CCDice

    Where:
      - L_BCEDice = BCE(logits, GT) + Dice_global(logits, GT)
        **Identical** to the original baseline BCEDiceLoss.
      - L_CCDice = per-ET-connected-component Dice, averaged equally.
        A 20-voxel small ET lesion contributes the same weight as a
        5000-voxel large tumor in this term.

    Single-variable change:
        BCEDiceLoss  = BCE + Global Dice
        BCEDiceCCLoss = BCE + Global Dice + λ_cc · CC-Level Dice
                                                    └── only new term

        Model, data, optimizer, lr, scheduler — all unchanged.
        Delta = net contribution of instance-level Dice supervision.

    Reference:
        "Instance-level Dice Loss for Brain Tumor Segmentation"
        Each ET connected component contributes equally to the loss,
        preventing large tumors from dominating small-lesion gradients.

    Args:
        lambda_cc: weight for CC-level Dice term (default 1.0).
        cc_min_size: minimum ET component voxels (default 10, filter noise).
        eps: numerical stability.
    """

    def __init__(self, lambda_cc=1.0, cc_min_size=10, eps=1e-9):
        super().__init__()
        self.lambda_cc = lambda_cc

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_global = DiceLoss(eps=eps)
        self.dice_cc = CCLevelDiceLoss(
            min_component_size=cc_min_size, eps=eps,
            n_classes=3, et_channel=2,
        )

    def forward(self, logits, targets):
        assert logits.shape == targets.shape

        # Original baseline loss (BCE + Global Dice)
        loss_bce = self.bce(logits, targets)
        loss_global_dice = self.dice_global(logits, targets)

        # CC-level Dice: each ET lesion contributes equally
        loss_cc_dice = self.dice_cc(logits, targets)

        total = loss_bce + loss_global_dice + self.lambda_cc * loss_cc_dice
        return total

    def log_components(self, logits, targets):
        """Return individual loss components for logging."""
        with torch.no_grad():
            bce = self.bce(logits, targets).item()
            gd = self.dice_global(logits, targets).item()
            cd = self.dice_cc(logits, targets).item()
        return {'bce': bce, 'dice_global': gd, 'cc_dice': cd}


# ================================================================
# Pixel-wise Modulated (PM) Dice Loss — Hosseini, 2025
# ================================================================

class PMDiceLoss(nn.Module):
    """
    Pixel-wise Modulated Dice Loss.

    Reference:
        Hosseini, S.M. (2025). "Pixel-wise Modulated Dice Loss for
        Medical Image Segmentation." arXiv:2506.15744.

    Formula:
        L = 1 - (1/C) Σ_c [ 2 Σ_i m_i^c·y_i^c·p_i^c + ε ] / [ Σ_i m_i^c·((y_i^c)² + (p_i^c)²) + ε ]

        m_i^c = | y_i^c - p̂_i^c |^γ
        p̂ = sigmoid(logits).detach()  — stop-gradient through modulating term

    Intuition:
        m ≈ 0  →  easy pixel (pred ≈ GT), no contribution to loss
        m ≈ 1  →  hard pixel (pred far from GT), full contribution
        Larger γ → steeper focus on hardest pixels.

        γ = 0 → standard Dice (m = 1 for all pixels except trivial cases).
                Actually |y-p|^0 = 1 (for any non-zero argument), near-standard.

    Why this helps small lesions:
        Small ET lesions at boundaries are "hard" pixels — the model is
        uncertain about them. Background in deep brain interior is "easy"
        — model confidently predicts 0. PM Dice automatically shifts
        gradient budget toward small ET lesions and boundaries.

    Args:
        gamma: focusing parameter (default 2.0, paper tests γ∈{0.5,1,2,3}).
               Higher → more aggressive focus on hard pixels.
        eps: numerical stability.
    """

    def __init__(self, gamma=2.0, eps=1e-9):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, D, H, W) raw logits.
            targets: (B, C, D, H, W) binary GT.

        Returns:
            scalar PM Dice loss.
        """
        prob = torch.sigmoid(logits)

        # Modulating term: m = |y - p_detach|^gamma
        # Stop-gradient through p̂ (paper: "no gradient update through p̂")
        prob_detach = prob.detach()
        m = (targets - prob_detach).abs() ** self.gamma

        B, C = logits.shape[:2]

        total_loss = 0.0
        for c in range(C):
            y_c = targets[:, c]
            p_c = prob[:, c]
            m_c = m[:, c]

            num = 2.0 * (m_c * y_c * p_c).sum()
            denom = (m_c * (y_c ** 2 + p_c ** 2)).sum()

            dice_c = (num + self.eps) / (denom + self.eps)
            total_loss += (1.0 - dice_c)

        return total_loss / C


# ================================================================
# BCE + PM Dice (Step 4)
# ================================================================

# ================================================================
# BCE + CC-Level Dice (no global Dice) — new experiment
# ================================================================

class BCECCDiceLoss(nn.Module):
    """
    BCE + CC-Level Dice Loss — single-variable change from baseline.

    Formula:
        L = BCE + λ_cc · L_CCDice

    Where:
      - BCE = Binary Cross-Entropy per pixel (standard classification)
      - L_CCDice = per-ET-connected-component Dice, averaged equally.
        A 20-voxel small ET lesion contributes the same weight as a
        5000-voxel large tumor in this term.

    KEY DIFFERENCE from BCEDiceCCLoss:
        BCEDiceLoss   = BCE + Global Dice                        (baseline)
        BCEDiceCCLoss = BCE + Global Dice + λ_cc · CC Dice      (existing)
        BCECCDiceLoss = BCE + λ_cc · CC Dice   ← THIS           (NEW)

    Single-variable change from baseline:
        Replaces Global Dice with CC-Level Dice.
        Tests: can instance-level Dice replace global Dice entirely?

        Model, data, optimizer, lr, scheduler — all unchanged.
        Delta = CC-Level Dice (replacing Global Dice) vs pure BCE.

    Args:
        lambda_cc: weight for CC-level Dice term (default 1.0).
        cc_min_size: minimum ET component voxels (default 10, filter noise).
        eps: numerical stability.
    """

    def __init__(self, lambda_cc=1.0, cc_min_size=10, eps=1e-9):
        super().__init__()
        self.lambda_cc = lambda_cc

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_cc = CCLevelDiceLoss(
            min_component_size=cc_min_size, eps=eps,
            n_classes=3, et_channel=2,
        )

    def forward(self, logits, targets):
        assert logits.shape == targets.shape

        loss_bce = self.bce(logits, targets)
        loss_cc_dice = self.dice_cc(logits, targets)

        total = loss_bce + self.lambda_cc * loss_cc_dice
        return total

    def log_components(self, logits, targets):
        """Return individual loss components for logging."""
        with torch.no_grad():
            bce = self.bce(logits, targets).item()
            cd = self.dice_cc(logits, targets).item()
        return {'bce': bce, 'cc_dice': cd}


class BCEDicePMLoss(nn.Module):
    """
    BCEDiceLoss + Pixel-wise Modulated Dice — single variable from PM.

    Formula:
        L = BCE + Global Dice + λ_pm · PM Dice

    Single-variable change:
        BCEDiceLoss  = BCE + Global Dice
        BCEDicePMLoss = BCE + Global Dice + λ_pm · PM Dice
                                              └── new (Hosseini 2025)

    Args:
        lambda_pm: weight for PM Dice term (default 1.0).
        pm_gamma: focusing parameter for PM Dice (default 2.0).
        eps: numerical stability.
    """

    def __init__(self, lambda_pm=1.0, pm_gamma=2.0, eps=1e-9):
        super().__init__()
        self.lambda_pm = lambda_pm

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_global = DiceLoss(eps=eps)
        self.pm_dice = PMDiceLoss(gamma=pm_gamma, eps=eps)

    def forward(self, logits, targets):
        assert logits.shape == targets.shape

        loss_bce = self.bce(logits, targets)
        loss_global_dice = self.dice_global(logits, targets)
        loss_pm_dice = self.pm_dice(logits, targets)

        return loss_bce + loss_global_dice + self.lambda_pm * loss_pm_dice

    def log_components(self, logits, targets):
        with torch.no_grad():
            bce = self.bce(logits, targets).item()
            gd = self.dice_global(logits, targets).item()
            pm = self.pm_dice(logits, targets).item()
        return {'bce': bce, 'dice_global': gd, 'pm_dice': pm}


# ================================================================
# BCE + CC Dice + PM Dice (Step 5: A+B+C)
# ================================================================

class BCEDiceCCPMLoss(nn.Module):
    """
    All three Dice variants combined — ablation endpoint.

    Formula:
        L = BCE + Global Dice + λ_cc · CC Dice + λ_pm · PM Dice

    Components:
        Global Dice  — overall overlap (baseline)
        CC Dice      — per-ET-component equal weight (instance-level)
        PM Dice       — per-pixel difficulty modulation (Hosseini 2025)
        BCE           — per-pixel classification

    Each term addresses a different level:
        Global Dice  → volume-level    (all pixels equal)
        PM Dice      → pixel-level     (difficulty-weighted)
        CC Dice      → lesion-level    (equal per instance)

    Single-variable ablation:
        Step 1: CC-Dice      → BCEDiceCCLoss (BCE + Global + CC)
        Step 2: PM-Dice      → BCEDicePMLoss (BCE + Global + PM)
        Step 3: CC+PM        → BCEDiceCCPMLoss (BCE + Global + CC + PM)

        Compare each vs BCEDiceLoss to isolate contribution.

    Args:
        lambda_cc: weight for CC-level Dice (default 1.0).
        lambda_pm: weight for PM Dice (default 1.0).
        cc_min_size: minimum ET component voxels.
        pm_gamma: focusing parameter for PM Dice (default 2.0).
        eps: numerical stability.
    """

    def __init__(self, lambda_cc=1.0, lambda_pm=1.0,
                 cc_min_size=10, pm_gamma=2.0, eps=1e-9):
        super().__init__()
        self.lambda_cc = lambda_cc
        self.lambda_pm = lambda_pm

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_global = DiceLoss(eps=eps)
        self.dice_cc = CCLevelDiceLoss(
            min_component_size=cc_min_size, eps=eps,
            n_classes=3, et_channel=2,
        )
        self.pm_dice = PMDiceLoss(gamma=pm_gamma, eps=eps)

    def forward(self, logits, targets):
        assert logits.shape == targets.shape

        return (self.bce(logits, targets)
                + self.dice_global(logits, targets)
                + self.lambda_cc * self.dice_cc(logits, targets)
                + self.lambda_pm * self.pm_dice(logits, targets))

    def log_components(self, logits, targets):
        with torch.no_grad():
            bce = self.bce(logits, targets).item()
            gd = self.dice_global(logits, targets).item()
            cd = self.dice_cc(logits, targets).item()
            pm = self.pm_dice(logits, targets).item()
        return {'bce': bce, 'dice_global': gd, 'cc_dice': cd, 'pm_dice': pm}
