#!/usr/bin/env python3
"""
main.py

Entry point for the Debiasify project. This script loads configuration parameters,
instantiates the DatasetLoader, Model, Trainer, and Evaluation classes, and then
orchestrates the training and evaluation process for Debiasify.
"""

import os
import sys
import yaml
from typing import Any, Dict

# Import project modules
from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads the configuration parameters from a YAML file.

    Args:
        config_path (str, optional): Path to the configuration file. Defaults to "config.yaml".

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    with open(config_path, "r") as f:
        config_data: Dict[str, Any] = yaml.safe_load(f)
    return config_data


def main() -> None:
    """
    Main function that coordinates data loading, model creation, training, and evaluation.
    """
    # 1. Load Configuration Parameters
    config_path: str = "config.yaml"
    config: Dict[str, Any] = load_config(config_path)
    print(f"Configuration loaded successfully from '{config_path}'.")

    # 2. Instantiate the DatasetLoader and load data
    dataset_loader: DatasetLoader = DatasetLoader(config)
    all_data: Dict[str, Dict[str, Any]] = dataset_loader.load_data()
    
    # Select a dataset to train and evaluate; default is "CelebA"
    dataset_name: str = "CelebA"
    if dataset_name not in all_data:
        available_datasets = list(all_data.keys())
        raise ValueError(f"Dataset '{dataset_name}' not found in loaded data. Available datasets: {available_datasets}")
    selected_data: Dict[str, Any] = all_data[dataset_name]
    num_train: int = len(selected_data.get("train").dataset)
    num_eval: int = len(selected_data.get("eval").dataset)
    print(f"Using dataset '{dataset_name}' with {num_train} training samples and {num_eval} evaluation samples.")

    # 3. Instantiate the Model using the provided configuration.
    # Default number of classes is set to 2.
    model: Model = Model(config, num_classes=2)
    print("Model instantiated successfully.")

    # 4. Instantiate the Trainer with the model, selected data, config, and dataset name.
    trainer: Trainer = Trainer(model, selected_data, config, dataset_name)
    print("Starting training process...")
    trainer.train()
    print("Training completed successfully.")

    # 5. Instantiate the Evaluation module and perform evaluation.
    evaluator: Evaluation = Evaluation(model, selected_data, config)
    print("Evaluating the model on the evaluation dataset...")
    evaluation_results: Dict[str, Any] = evaluator.evaluate()

    # 6. Report Evaluation Results
    print("\n=== Evaluation Results ===")
    for metric_name, metric_value in evaluation_results.items():
        print(f"{metric_name}: {metric_value}")

    # Optional: Uncomment the following line to visualize t-SNE embeddings of the deep features.
    # evaluator.visualize_tsne(feature_type="deep", perplexity=30.0, n_samples=1000)


if __name__ == "__main__":
    main()
