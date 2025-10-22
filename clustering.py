## clustering.py
"""
This module implements the Clustering class for generating pseudo bias groups from
shallow feature representations. It optionally applies PCA for dimensionality reduction
and then performs adaptive K-means clustering (per class) to determine groups whose 
average within-cluster variance is below a pre-defined threshold gamma.

The resulting clustering models (and PCA models if used) are stored internally and can be
used later to predict cluster assignments for new feature data.

Classes:
    Clustering
"""

from typing import Any, Dict, List, Optional
import numpy as np
import torch
from torch import Tensor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class Clustering:
    """
    Clustering class for generating pseudo bias groups from shallow features.

    Attributes:
        gamma (float): Threshold for average within-cluster variance.
        use_pca (bool): Whether to use PCA for dimensionality reduction.
        pca_components (Optional[int]): Number of PCA components to retain; if None,
            all components are preserved.
        kmeans_models (Dict[Any, Dict[str, Any]]): Dictionary mapping each class label
            to its corresponding clustering information which includes the fitted KMeans model,
            the PCA transformer (if used), and the cluster assignments.
    """

    def __init__(self, gamma: float, use_pca: bool = False, pca_components: Optional[int] = None) -> None:
        """
        Initializes the Clustering instance using the given gamma threshold,
        and optional PCA settings.

        Args:
            gamma (float): Variance threshold for adaptive K-means clustering.
            use_pca (bool, optional): Flag to enable PCA before clustering. Defaults to False.
            pca_components (Optional[int], optional): Number of PCA components to retain.
                If None, all components are used. Defaults to None.
        """
        self.gamma: float = gamma
        self.use_pca: bool = use_pca
        self.pca_components: Optional[int] = pca_components
        self.kmeans_models: Dict[Any, Dict[str, Any]] = {}

    def update_clusters(self, features: Tensor, labels: Tensor) -> Dict[Any, Dict[str, Any]]:
        """
        Updates the clustering assignments for each unique target class using adaptive K-means.

        For each unique class label, the method extracts the corresponding shallow feature vectors,
        optionally reduces dimensionality with PCA, and then searches for the smallest number of clusters
        (starting from 1) such that the average within-cluster variance (inertia / n_samples) is below
        the gamma threshold. The fitted KMeans model, PCA transformer (if used), and cluster assignments
        are stored and returned.

        Args:
            features (Tensor): Shallow feature representations with shape (n_samples, feature_dim).
            labels (Tensor): Target labels corresponding to each feature (shape: n_samples).

        Returns:
            Dict[Any, Dict[str, Any]]: A dictionary mapping each unique class label to its clustering info:
                {
                    class_label: {
                        "model": fitted KMeans model,
                        "pca": fitted PCA model (or None),
                        "assignments": NumPy array of cluster assignments
                    },
                    ...
                }
        """
        # Convert input tensors to numpy arrays.
        features_np: np.ndarray = features.cpu().detach().numpy()
        labels_np: np.ndarray = labels.cpu().detach().numpy()

        # Determine unique class labels.
        unique_labels: np.ndarray = np.unique(labels_np)
        cluster_info_dict: Dict[Any, Dict[str, Any]] = {}

        # Set maximum clusters to prevent endless loops.
        max_clusters: int = 20

        for class_label in unique_labels:
            # Extract features corresponding to the current class.
            indices: np.ndarray = np.where(labels_np == class_label)[0]
            class_features: np.ndarray = features_np[indices]

            # Optionally apply PCA to reduce dimensionality.
            pca_model: Optional[PCA] = None
            if self.use_pca:
                n_components: int = self.pca_components if self.pca_components is not None else class_features.shape[1]
                pca_model = PCA(n_components=n_components, random_state=42)
                class_features = pca_model.fit_transform(class_features)

            n_samples: int = class_features.shape[0]
            kmeans_model: Optional[KMeans] = None
            optimal_k: int = 1

            # Adaptive K-means: start with k=1 up to the smaller of max_clusters and n_samples.
            for k in range(1, min(max_clusters, n_samples) + 1):
                kmeans: KMeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(class_features)
                # Compute average within-cluster variance.
                avg_variance: float = kmeans.inertia_ / n_samples
                if avg_variance < self.gamma:
                    optimal_k = k
                    kmeans_model = kmeans
                    break

            # If no configuration meets the threshold, use the model with max_clusters (or n_samples).
            if kmeans_model is None:
                optimal_k = min(max_clusters, n_samples)
                kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                kmeans_model.fit(class_features)
                avg_variance = kmeans_model.inertia_ / n_samples

            # Store the clustering information for this class.
            assignments: np.ndarray = kmeans_model.labels_
            cluster_info: Dict[str, Any] = {
                "model": kmeans_model,
                "pca": pca_model,
                "assignments": assignments
            }
            cluster_info_dict[class_label] = cluster_info
            self.kmeans_models[class_label] = cluster_info

            # Debug output: print the chosen number of clusters and the average variance.
            print(f"Class {class_label}: optimal clusters = {optimal_k}, average variance = {avg_variance:.6f}")

        return cluster_info_dict

    def predict_cluster(self, features: Tensor) -> List[int]:
        """
        Predicts the cluster assignments for the given shallow features using a previously stored clustering model.
        
        This method assumes that the features belong to a single target class. If multiple clustering
        models exist in self.kmeans_models, the first one is used.

        Args:
            features (Tensor): Shallow feature representations with shape (n_samples, feature_dim).

        Returns:
            List[int]: A list of predicted cluster indices for the input features.
        """
        if not self.kmeans_models:
            raise ValueError("No clustering model is available. Please run update_clusters first.")

        # Use the first available clustering model.
        first_key: Any = next(iter(self.kmeans_models))
        cluster_info: Dict[str, Any] = self.kmeans_models[first_key]
        kmeans_model: KMeans = cluster_info["model"]
        pca_model: Optional[PCA] = cluster_info["pca"]

        # Convert features to a NumPy array.
        features_np: np.ndarray = features.cpu().detach().numpy()
        if pca_model is not None:
            features_np = pca_model.transform(features_np)

        # Predict cluster assignments.
        predictions: np.ndarray = kmeans_model.predict(features_np)
        return predictions.tolist() if predictions.ndim > 0 and predictions.size > 1 else [int(predictions[0])]


# Optional: Minimal test routine for debugging purposes.
if __name__ == "__main__":
    # Create dummy data: 100 samples with 128 features and binary labels (0 or 1).
    dummy_features: Tensor = torch.randn(100, 128)
    dummy_labels: Tensor = torch.randint(0, 2, (100,))

    # Instantiate the Clustering class with a gamma threshold.
    gamma_value: float = 0.01  # Example threshold; adjust as needed.
    clustering_instance: Clustering = Clustering(gamma=gamma_value, use_pca=True, pca_components=50)

    # Update clusters with the dummy data.
    clusters: Dict[Any, Dict[str, Any]] = clustering_instance.update_clusters(dummy_features, dummy_labels)
    print("Cluster information per class:")
    for key, value in clusters.items():
        print(f"Class {key}: {value['model'].n_clusters} clusters")

    # Predict clusters for a new batch (taking the first 5 samples).
    new_features: Tensor = dummy_features[:5]
    predicted_clusters: List[int] = clustering_instance.predict_cluster(new_features)
    print("Predicted cluster assignments for new features:", predicted_clusters)
