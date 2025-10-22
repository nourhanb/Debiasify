## losses.py
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Losses:
    """
    Losses class for computing the hybrid loss in Debiasify.

    This class implements three loss components:
      1. Averaged Cross-Entropy Loss (L_ACE)
         - Trains both the shallow and deep classifiers to predict the target label.
      2. KL Divergence Loss (L_KL)
         - Aligns the predictions (logits) of the shallow classifier with those of the deep classifier.
      3. Maximum Mean Discrepancy (MMD) Loss (L_MMD)
         - Aligns the distributions of shallow and deep features using a Gaussian RBF kernel.
    
    The total hybrid loss is given by:
         L_total = L_ACE + alpha * L_KL + L_MMD

    Default Parameters:
         - alpha: 0.1 (weight for the KL divergence loss), as specified in config.yaml.
         - sigma: 1.0 (bandwidth for the Gaussian RBF kernel)
    """

    @staticmethod
    def classification_loss(
        logits_shallow: Tensor, 
        logits_deep: Tensor, 
        targets: Tensor
    ) -> Tensor:
        """
        Computes the averaged cross-entropy (classification) loss for both shallow and deep branches.

        Args:
            logits_shallow (Tensor): Logits from the shallow classifier with shape (batch_size, num_classes).
            logits_deep (Tensor): Logits from the deep classifier with shape (batch_size, num_classes).
            targets (Tensor): Ground truth labels with shape (batch_size).

        Returns:
            Tensor: Scalar tensor representing the averaged cross-entropy loss.
        """
        loss_ce_shallow: Tensor = F.cross_entropy(logits_shallow, targets)
        loss_ce_deep: Tensor = F.cross_entropy(logits_deep, targets)
        loss_ce: Tensor = 0.5 * (loss_ce_shallow + loss_ce_deep)
        return loss_ce

    @staticmethod
    def kl_divergence_loss(
        logits_shallow: Tensor, 
        logits_deep: Tensor
    ) -> Tensor:
        """
        Computes the KL divergence loss to align the shallow and deep classifier logits.

        Args:
            logits_shallow (Tensor): Logits from the shallow classifier (batch_size, num_classes).
            logits_deep (Tensor): Logits from the deep classifier (batch_size, num_classes).

        Returns:
            Tensor: Scalar tensor representing the KL divergence loss.
        """
        log_probs_shallow: Tensor = F.log_softmax(logits_shallow, dim=1)
        probs_deep: Tensor = F.softmax(logits_deep, dim=1)
        # Use reduction 'batchmean' for proper scaling across the batch.
        loss_kl: Tensor = F.kl_div(log_probs_shallow, probs_deep, reduction='batchmean')
        return loss_kl

    @staticmethod
    def gaussian_kernel(
        x: Tensor, 
        y: Tensor, 
        sigma: float = 1.0
    ) -> Tensor:
        """
        Computes the Gaussian RBF kernel matrix between two sets of features.

        Args:
            x (Tensor): Tensor of shape (n_samples_x, feature_dim).
            y (Tensor): Tensor of shape (n_samples_y, feature_dim).
            sigma (float, optional): Bandwidth parameter for the Gaussian kernel. Default is 1.0.

        Returns:
            Tensor: Kernel matrix of shape (n_samples_x, n_samples_y).
        """
        # Compute squared Euclidean norms for x and y.
        x_norm: Tensor = (x ** 2).sum(dim=1).view(-1, 1)  # Shape: (n_x, 1)
        y_norm: Tensor = (y ** 2).sum(dim=1).view(1, -1)   # Shape: (1, n_y)

        # Compute pairwise squared Euclidean distance: |x - y|^2 = ||x||^2 + ||y||^2 - 2 * x.y^T
        distances: Tensor = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        distances = torch.clamp(distances, min=0.0)

        # Gaussian RBF kernel computation.
        kernel: Tensor = torch.exp(-distances / (2.0 * sigma ** 2))
        return kernel

    @staticmethod
    def mmd_loss(
        features_shallow: Tensor, 
        features_deep: Tensor, 
        sigma: float = 1.0
    ) -> Tensor:
        """
        Computes the Maximum Mean Discrepancy (MMD) loss between shallow and deep feature distributions.

        Uses a Gaussian RBF kernel to measure the discrepancy:
          L_MMD = mean(kernel_xx) + mean(kernel_yy) - 2 * mean(kernel_xy)
        where kernel_xx, kernel_yy, and kernel_xy are the kernel matrices computed for the
        shallow features, deep features, and between shallow and deep features respectively.

        Args:
            features_shallow (Tensor): Shallow features with shape (batch_size, feature_dim).
            features_deep (Tensor): Deep features with shape (batch_size, feature_dim).
            sigma (float, optional): Bandwidth parameter for the Gaussian kernel. Default is 1.0.

        Returns:
            Tensor: Scalar tensor representing the MMD loss.
        """
        kernel_xx: Tensor = Losses.gaussian_kernel(features_shallow, features_shallow, sigma)
        kernel_yy: Tensor = Losses.gaussian_kernel(features_deep, features_deep, sigma)
        kernel_xy: Tensor = Losses.gaussian_kernel(features_shallow, features_deep, sigma)
        
        loss_mmd: Tensor = kernel_xx.mean() + kernel_yy.mean() - 2.0 * kernel_xy.mean()
        return loss_mmd

    @staticmethod
    def compute_total_loss(
        logits_shallow: Tensor, 
        logits_deep: Tensor, 
        features_shallow: Tensor, 
        features_deep: Tensor, 
        targets: Tensor, 
        alpha: float = 0.1, 
        sigma: float = 1.0
    ) -> Tensor:
        """
        Computes the total hybrid loss for Debiasify as:
        
            L_total = L_ACE + alpha * L_KL + L_MMD
        
        where:
            - L_ACE is the averaged cross-entropy loss for both shallow and deep classifiers.
            - L_KL is the KL divergence loss aligning shallow and deep logits.
            - L_MMD is the MMD loss aligning the shallow and deep feature distributions.

        Args:
            logits_shallow (Tensor): Logits from the shallow classifier (batch_size, num_classes).
            logits_deep (Tensor): Logits from the deep classifier (batch_size, num_classes).
            features_shallow (Tensor): Shallow feature representations (batch_size, feature_dim).
            features_deep (Tensor): Deep feature representations (batch_size, feature_dim).
            targets (Tensor): Ground truth labels (batch_size).
            alpha (float, optional): Weighting factor for the KL divergence loss. Default is 0.1.
            sigma (float, optional): Bandwidth parameter for the Gaussian kernel in MMD. Default is 1.0.

        Returns:
            Tensor: Scalar tensor representing the total loss.
        """
        loss_ce: Tensor = Losses.classification_loss(logits_shallow, logits_deep, targets)
        loss_kl: Tensor = Losses.kl_divergence_loss(logits_shallow, logits_deep)
        loss_mmd: Tensor = Losses.mmd_loss(features_shallow, features_deep, sigma)

        total_loss: Tensor = loss_ce + alpha * loss_kl + loss_mmd
        return total_loss
