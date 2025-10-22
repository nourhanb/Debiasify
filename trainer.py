## trainer.py
import os
from typing import Any, Dict, List

import torch
import torch.optim as optim
from torch import nn, Tensor
from tqdm import tqdm

from losses import Losses
from clustering import Clustering


class Trainer:
    """Trainer class for Debiasify.

    This class encapsulates the training loop including a warm-up phase,
    periodic clustering updates using shallow features, and loss computation
    that integrates classification loss, KL divergence loss, and MMD loss.
    """

    def __init__(
        self,
        model: nn.Module,
        data: Dict[str, Any],
        config: Dict[str, Any],
        dataset_name: str = "CelebA"
    ) -> None:
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): An instance of the Model class.
            data (Dict[str, Any]): Dictionary containing DataLoader objects. Expected to have a "train" key.
            config (Dict[str, Any]): Configuration parameters loaded from config.yaml.
            dataset_name (str, optional): Name of the dataset to determine clustering gamma. Default is "CelebA".
        """
        self.model: nn.Module = model
        self.data: Dict[str, Any] = data
        self.config: Dict[str, Any] = config

        # Device configuration.
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Training hyperparameters.
        training_config: Dict[str, Any] = self.config.get("training", {})
        self.learning_rate: float = float(training_config.get("learning_rate", 1e-4))
        self.batch_size: int = int(training_config.get("batch_size", 100))
        self.epochs: int = int(training_config.get("epochs", 50))
        self.weight_decay: float = float(training_config.get("weight_decay", 0.01))
        self.alpha: float = float(training_config.get("alpha", 0.1))

        # Warm-up phase duration. Default to 5 epochs if not specified.
        self.warmup_epochs: int = int(training_config.get("warmup_epochs", 5))

        # Determine clustering gamma from configuration.
        # For CelebA, the gamma is provided as a range string "0.003-0.01". Use the lower bound.
        clustering_config: Dict[str, Any] = self.config.get("clustering", {})
        gamma_value: Any = None
        if "gamma" in clustering_config:
            gamma_data: Any = clustering_config["gamma"]
            if isinstance(gamma_data, dict) and dataset_name in gamma_data:
                gamma_value = gamma_data[dataset_name]
            else:
                gamma_value = 0.01
        else:
            gamma_value = 0.01

        if isinstance(gamma_value, str):
            try:
                gamma_parts: List[str] = gamma_value.split("-")
                gamma_float: float = float(gamma_parts[0].strip())
                self.gamma: float = gamma_float
            except Exception:
                self.gamma = 0.01
        else:
            self.gamma = float(gamma_value)

        # Instantiate the Clustering object.
        # Here, we set use_pca to True and choose a default number of PCA components (e.g., 50).
        self.clustering: Clustering = Clustering(gamma=self.gamma, use_pca=True, pca_components=50)
        # Clustering update frequency after warm-up (update every epoch by default).
        self.clustering_update_frequency: int = int(training_config.get("clustering_update_frequency", 1))

        # Create the optimizer.
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Set up checkpointing directory.
        self.checkpoint_dir: str = self.config.get("checkpoint_dir", "checkpoints")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def update_clustering(self) -> None:
        """Update clustering assignments using shallow features from the training set."""
        self.model.eval()
        all_features: List[Tensor] = []
        all_labels: List[Tensor] = []
        train_loader = self.data.get("train", None)
        if train_loader is None:
            raise ValueError("Training DataLoader not found in provided data.")

        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Clustering Update", leave=False):
                # Expecting each batch as (inputs, labels).
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # Extract shallow features using the dedicated method.
                features = self.model.get_shallow_features(inputs)
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())
        # Concatenate features and labels from all batches.
        features_tensor: Tensor = torch.cat(all_features, dim=0)
        labels_tensor: Tensor = torch.cat(all_labels, dim=0)

        # Update clusters using the Clustering object.
        clustering_info = self.clustering.update_clusters(features_tensor, labels_tensor)
        # (Optional) The returned clustering_info can be used for further analysis.
        self.model.train()

    def train(self) -> None:
        """Runs the training loop for the model with Debiasify's self-distillation strategy."""
        train_loader = self.data.get("train", None)
        if train_loader is None:
            raise ValueError("Training DataLoader not found in provided data.")

        for epoch in range(self.epochs):
            epoch_loss_total: float = 0.0
            epoch_loss_ce: float = 0.0
            epoch_loss_kl: float = 0.0
            epoch_loss_mmd: float = 0.0
            num_batches: int = 0

            # If beyond warm-up, update clustering assignments periodically.
            if epoch >= self.warmup_epochs and ((epoch - self.warmup_epochs) % self.clustering_update_frequency == 0):
                print(f"Epoch {epoch + 1}: Updating clustering assignments.")
                self.update_clustering()

            self.model.train()
            progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{self.epochs}]", leave=True)

            for batch in progress_bar:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass through the model.
                outputs: Dict[str, Tensor] = self.model.forward(inputs)
                logits_shallow: Tensor = outputs.get("logits_shallow")
                logits_deep: Tensor = outputs.get("logits_deep")
                shallow_features: Tensor = outputs.get("shallow_features")
                deep_features: Tensor = outputs.get("deep_features")

                # Compute classification loss (averaged cross-entropy).
                loss_ce: Tensor = Losses.classification_loss(logits_shallow, logits_deep, targets)

                if epoch < self.warmup_epochs:
                    # Warm-up phase: use only classification loss.
                    total_loss: Tensor = loss_ce
                    loss_kl_value: float = 0.0
                    loss_mmd_value: float = 0.0
                else:
                    # Full training phase: compute additional loss components.
                    loss_kl: Tensor = Losses.kl_divergence_loss(logits_shallow, logits_deep)
                    loss_mmd: Tensor = Losses.mmd_loss(shallow_features, deep_features)
                    total_loss = loss_ce + self.alpha * loss_kl + loss_mmd
                    loss_kl_value = loss_kl.item()
                    loss_mmd_value = loss_mmd.item()

                total_loss.backward()
                self.optimizer.step()

                # Accumulate losses for reporting.
                batch_loss_ce: float = loss_ce.item()
                batch_loss_total: float = total_loss.item()
                epoch_loss_ce += batch_loss_ce
                epoch_loss_total += batch_loss_total
                epoch_loss_kl += loss_kl_value
                epoch_loss_mmd += loss_mmd_value
                num_batches += 1

                progress_bar.set_postfix({
                    "L_CE": f"{batch_loss_ce:.4f}",
                    "L_KL": f"{loss_kl_value:.4f}",
                    "L_MMD": f"{loss_mmd_value:.4f}",
                    "L_Total": f"{batch_loss_total:.4f}"
                })

            avg_loss_ce: float = epoch_loss_ce / num_batches
            avg_loss_kl: float = epoch_loss_kl / num_batches if num_batches > 0 else 0.0
            avg_loss_mmd: float = epoch_loss_mmd / num_batches if num_batches > 0 else 0.0
            avg_loss_total: float = epoch_loss_total / num_batches

            print(
                f"Epoch [{epoch + 1}/{self.epochs}] - "
                f"Avg L_CE: {avg_loss_ce:.4f}, Avg L_KL: {avg_loss_kl:.4f}, "
                f"Avg L_MMD: {avg_loss_mmd:.4f}, Avg L_Total: {avg_loss_total:.4f}"
            )

            # Save checkpoint at the end of the epoch.
            checkpoint_path: str = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            checkpoint: Dict[str, Any] = {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": avg_loss_total,
                "clustering_state": self.clustering.kmeans_models  # Save clustering state information.
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        print("Training complete.")
