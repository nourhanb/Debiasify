## evaluation.py
"""
Module for evaluating the Debiasify model on the evaluation dataset.
This module computes overall, average-group (unbiased), and worst-group accuracy
by grouping predictions based on (target, bias) combinations. It also provides
optional visualization functions for t-SNE feature embeddings and Grad-CAM heatmaps.
"""

from typing import Any, Dict, List, Tuple
import torch
from torch import nn, Tensor
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import cv2

class Evaluation:
    """
    Evaluation class for Debiasify.

    Attributes:
        model (nn.Module): The trained Debiasify model.
        data (Dict[str, Any]): Dictionary containing evaluation DataLoader(s) under key "eval".
        config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
        device (torch.device): Device on which evaluation is performed.
    """

    def __init__(self, model: nn.Module, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Initializes the Evaluation object.

        Args:
            model (nn.Module): An instance of the Debiasify Model class.
            data (Dict[str, Any]): Data dictionary containing evaluation DataLoader(s).
            config (Dict[str, Any]): Configuration parameters.
        """
        self.model: nn.Module = model
        self.data: Dict[str, Any] = data
        self.config: Dict[str, Any] = config
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluates the model on the evaluation dataset, computes prediction metrics, and measures
        subgroup performances based on (target, bias) combinations.

        The evaluation procedure:
          - Sets the model to evaluation mode.
          - Loops over the evaluation DataLoader in a no_grad context.
          - Uses the deep classifier logits to obtain predictions.
          - Accumulates true target labels, predicted labels, and bias attribute values.
          - Groups the samples by (target, bias), computes subgroup accuracies.
          - Returns overall accuracy, average-group accuracy, worst-group accuracy, and detailed group metrics.

        Returns:
            Dict[str, Any]: A dictionary with metrics, e.g.:
                {
                    "overall_accuracy": float,
                    "average_group_accuracy": float,
                    "worst_group_accuracy": float,
                    "group_metrics": { (target, bias): subgroup_accuracy, ... }
                }
        """
        self.model.eval()
        eval_loader = self.data.get("eval", None)
        if eval_loader is None:
            raise ValueError("Evaluation DataLoader not found in provided data.")

        all_predictions: List[int] = []
        all_targets: List[int] = []
        all_biases: List[int] = []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
                # Unpack batch: Expect either (inputs, targets, biases) or (inputs, targets).
                if isinstance(batch, (list, tuple)):
                    if len(batch) >= 3:
                        inputs, targets, biases = batch[0], batch[1], batch[2]
                    else:
                        inputs, targets = batch[0], batch[1]
                        biases = torch.zeros_like(targets)
                else:
                    raise ValueError("Batch data must be a tuple or list.")

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                biases = biases.to(self.device)

                # Forward pass through the model.
                outputs: Dict[str, Tensor] = self.model.forward(inputs)
                logits_deep: Tensor = outputs.get("logits_deep")
                predictions: Tensor = torch.argmax(logits_deep, dim=1)

                all_predictions.extend(predictions.cpu().tolist())
                all_targets.extend(targets.cpu().tolist())
                all_biases.extend(biases.cpu().tolist())

        # Compute overall accuracy.
        total_samples: int = len(all_targets)
        correct_overall: int = sum(1 for pred, true in zip(all_predictions, all_targets) if pred == true)
        overall_accuracy: float = correct_overall / total_samples if total_samples > 0 else 0.0

        # Compute subgroup accuracies based on (target, bias) groups.
        group_correct: Dict[Tuple[int, int], int] = {}
        group_counts: Dict[Tuple[int, int], int] = {}

        for pred, true, bias in zip(all_predictions, all_targets, all_biases):
            group_key: Tuple[int, int] = (true, bias)
            group_counts[group_key] = group_counts.get(group_key, 0) + 1
            if pred == true:
                group_correct[group_key] = group_correct.get(group_key, 0) + 1
            else:
                group_correct[group_key] = group_correct.get(group_key, 0)

        group_metrics: Dict[Tuple[int, int], float] = {}
        for key in group_counts:
            group_accuracy: float = group_correct[key] / group_counts[key]
            group_metrics[key] = group_accuracy

        if group_metrics:
            average_group_accuracy: float = sum(group_metrics.values()) / len(group_metrics)
            worst_group_accuracy: float = min(group_metrics.values())
        else:
            average_group_accuracy = overall_accuracy
            worst_group_accuracy = overall_accuracy

        results: Dict[str, Any] = {
            "overall_accuracy": overall_accuracy,
            "average_group_accuracy": average_group_accuracy,
            "worst_group_accuracy": worst_group_accuracy,
            "group_metrics": group_metrics
        }

        return results

    def visualize_tsne(self, feature_type: str = "deep", perplexity: float = 30.0, n_samples: int = 1000) -> None:
        """
        Visualizes the feature embeddings from the evaluation dataset using t-SNE.
        This can help assess whether debiased representations mix examples with different bias attributes.

        Args:
            feature_type (str, optional): Determines which features to extract; "deep" for deep features,
                                          "shallow" for shallow features. Defaults to "deep".
            perplexity (float, optional): t-SNE perplexity parameter. Defaults to 30.0.
            n_samples (int, optional): Maximum number of samples to visualize. Defaults to 1000.
        """
        self.model.eval()
        eval_loader = self.data.get("eval", None)
        if eval_loader is None:
            raise ValueError("Evaluation DataLoader not found in provided data.")

        features_list: List[np.ndarray] = []
        bias_list: List[int] = []
        target_list: List[int] = []
        samples_count: int = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Collecting features for t-SNE", leave=False):
                if isinstance(batch, (list, tuple)):
                    if len(batch) >= 3:
                        inputs, targets, biases = batch[0], batch[1], batch[2]
                    else:
                        inputs, targets = batch[0], batch[1]
                        biases = torch.zeros_like(targets)
                else:
                    raise ValueError("Batch data must be a tuple or list.")

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                biases = biases.to(self.device)

                if feature_type.lower() == "deep":
                    feats: Tensor = self.model.get_deep_features(inputs)
                else:
                    feats: Tensor = self.model.get_shallow_features(inputs)
                feats_np: np.ndarray = feats.cpu().detach().numpy()

                features_list.append(feats_np)
                target_list.extend(targets.cpu().tolist())
                bias_list.extend(biases.cpu().tolist())

                samples_count += feats_np.shape[0]
                if samples_count >= n_samples:
                    break

        features_array: np.ndarray = np.concatenate(features_list, axis=0)
        if features_array.shape[0] > n_samples:
            features_array = features_array[:n_samples]
            target_list = target_list[:n_samples]
            bias_list = bias_list[:n_samples]

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_results: np.ndarray = tsne.fit_transform(features_array)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=bias_list, cmap="viridis", alpha=0.7)
        plt.colorbar(scatter, label="Bias Attribute")
        plt.title(f"t-SNE Visualization of {feature_type.capitalize()} Features")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.tight_layout()
        plt.show()

    def generate_grad_cam(self, input_image: Tensor, target_class: int) -> None:
        """
        Generates and displays a Grad-CAM heatmap for a single input image to visualize the model's attention
        and interpret its predictions.

        The method registers forward and backward hooks on the last convolutional layer of the deep branch
        (i.e., the last module of layer4) to extract feature maps and gradients, computes the weighted activation
        map, and overlays it on the original image.

        Args:
            input_image (Tensor): A single preprocessed image tensor of shape (C, H, W). If not batched, a batch
                                  dimension is added.
            target_class (int): The target class index for which Grad-CAM is generated.
        """
        self.model.eval()
        # Add batch dimension if needed.
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
        input_image = input_image.to(self.device)

        # Lists to store activations and gradients.
        activations: List[Tensor] = []
        gradients: List[Tensor] = []

        def forward_hook(module: nn.Module, inp: Tuple[Tensor, ...], out: Tensor) -> None:
            activations.append(out.clone())

        def backward_hook(module: nn.Module, grad_in: Tuple[Tensor, ...], grad_out: Tuple[Tensor, ...]) -> None:
            gradients.append(grad_out[0].clone())

        # Select target layer: last convolutional layer from layer4.
        target_layer = self.model.layer4[-1]
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)

        # Forward pass.
        output = self.model.forward(input_image)
        logits_deep: Tensor = output.get("logits_deep")  # Shape: (1, num_classes)
        self.model.zero_grad()
        target_score: Tensor = logits_deep[0, target_class]
        target_score.backward()

        # Remove hooks.
        forward_handle.remove()
        backward_handle.remove()

        if not activations or not gradients:
            print("Failed to capture activations or gradients for Grad-CAM.")
            return

        activation_maps: Tensor = activations[0]  # Shape: (1, C, H, W)
        grads: Tensor = gradients[0]  # Shape: (1, C, H, W)

        # Compute channel weights by global-average pooling the gradients.
        weights: Tensor = torch.mean(grads, dim=(2, 3), keepdim=True)  # Shape: (1, C, 1, 1)
        grad_cam: Tensor = torch.sum(weights * activation_maps, dim=1, keepdim=True)  # Shape: (1, 1, H, W)
        grad_cam = torch.relu(grad_cam)
        grad_cam = grad_cam - grad_cam.min()
        if grad_cam.max() != 0:
            grad_cam = grad_cam / grad_cam.max()
        grad_cam_map: np.ndarray = grad_cam.cpu().detach().numpy()[0, 0]

        # Upsample the Grad-CAM map to the size of the input image.
        input_image_np: np.ndarray = input_image.cpu().detach().numpy()[0]
        # Unnormalize image using ImageNet means and stds.
        mean: np.ndarray = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std: np.ndarray = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        input_image_np = input_image_np * std + mean
        input_image_np = np.transpose(input_image_np, (1, 2, 0))
        input_image_np = np.clip(input_image_np, 0, 1)

        grad_cam_resized: np.ndarray = cv2.resize(grad_cam_map, (input_image_np.shape[1], input_image_np.shape[0]))
        heatmap: np.ndarray = cv2.applyColorMap(np.uint8(255 * grad_cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255.0
        overlay: np.ndarray = heatmap + input_image_np
        overlay = overlay / np.max(overlay)

        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.title(f"Grad-CAM for Target Class {target_class}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
