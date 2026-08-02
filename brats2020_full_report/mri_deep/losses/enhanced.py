"""
Enhanced Loss Functions for BraTS2020 Segmentation.
Extracted from: resunet_enhanced.py

New losses beyond the original BCEDiceLoss:
  - CELoss: Weighted Cross-Entropy with class weights (ET > TC > WT)
  - BoundaryLoss: Edge-aware loss using 3D Laplacian
  - DiceCEBoundaryLoss: Combined loss = alpha*Dice + beta*CE + gamma*Boundary
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
