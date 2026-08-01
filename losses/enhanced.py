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
    Boundary Loss using Distance Transform.

    What boundary losses do people commonly use?
    --------------------------------------------
    1. BD Loss (Kervadec et al., MIDL 2019):
       - Uses distance transform to weight CE loss
       - Boundary pixels get higher weight based on distance to contour
       - Simplest and most widely adopted for BraTS

    2. Surface Loss (Kervadec et al., MIDL 2019):
       - Integral approximation over boundary distance map
       - Mathematically elegant but complex implementation

    3. Hausdorff Distance Loss (Karimi et al., MICCAI 2019):
       - Directly optimizes Hausdorff distance
       - Computationally expensive, can be unstable

    THIS IMPLEMENTATION: Edge-aware Boundary Loss
    - Uses 3D Laplacian operator to detect edge regions
    - Computes BCE only on boundary pixels (weighted higher)
    - Simple, no pre-computation needed, differentiable

    Reference: Many BraTS papers use edge-aware weighting combined
    with Dice+CE for improved boundary delineation.
    """
    def __init__(self, edge_weight: float = 5.0):
        """
        Args:
            edge_weight: multiplier for boundary pixel loss.
                         Higher = more emphasis on boundaries.
        """
        super(BoundaryLoss, self).__init__()
        self.edge_weight = edge_weight

        # 3D Laplacian kernel for edge detection
        # This is a 3x3x3 kernel that highlights boundaries
        laplacian_kernel = torch.ones(1, 1, 3, 3, 3) * -1.0
        laplacian_kernel[0, 0, 1, 1, 1] = 26.0  # center = sum of neighbors
        self.register_buffer('laplacian_kernel', laplacian_kernel)

    def get_boundary_mask(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Extract boundary regions from ground truth masks using Laplacian.
        Boundary = pixels where Laplacian of mask != 0 (i.e., near edges).
        """
        B, C, D, H, W = targets.shape
        # Process each class channel
        boundaries = []
        for c in range(C):
            # (B, 1, D, H, W) -> apply Laplacian
            ch = targets[:, c:c+1, :, :, :]
            lap = F.conv3d(ch, self.laplacian_kernel, padding=1)
            # Boundary = where Laplacian != 0
            boundary = (lap.abs() > 1e-6).float()
            boundaries.append(boundary)
        boundary_mask = torch.cat(boundaries, dim=1)  # (B, C, D, H, W)
        return boundary_mask

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Get boundary mask from ground truth
        boundary_mask = self.get_boundary_mask(targets)

        # Compute BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Weight: edge_weight on boundaries, 1.0 elsewhere
        weight = 1.0 + boundary_mask * (self.edge_weight - 1.0)

        return (bce * weight).mean()


class DiceCEBoundaryLoss(nn.Module):
    """
    Combined Loss: Dice + Weighted CE + Boundary

    Loss = alpha * DiceLoss
         + beta  * CELoss (with class weights: ET > TC > WT)
         + gamma * BoundaryLoss

    Default weights tuned for BraTS:
        alpha=1.0 (dice), beta=0.5 (CE), gamma=0.3 (boundary)
        Class weights: WT=1.0, TC=2.0, ET=4.0
    """
    def __init__(
        self,
        alpha: float = 1.0,       # Dice weight
        beta: float = 0.5,        # CE weight
        gamma: float = 0.3,       # Boundary weight
        class_weights: list = None,  # [WT, TC, ET] weights
        edge_weight: float = 5.0,    # Boundary emphasis
    ):
        super(DiceCEBoundaryLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if class_weights is None:
            # Default: ET highest (4x), TC medium (2x), WT baseline (1x)
            class_weights = [1.0, 2.0, 4.0]

        self.dice = DiceLoss()
        self.ce = CELoss(class_weights=torch.tensor(class_weights))
        self.boundary = BoundaryLoss(edge_weight=edge_weight)

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
