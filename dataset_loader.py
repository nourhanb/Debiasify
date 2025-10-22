## dataset_loader.py
import os
from typing import Any, Dict, List
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class DatasetLoader:
    """Handles data loading, preprocessing, and augmentation for CelebA, Waterbirds, and Fitzpatrick datasets.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
        batch_size (int): Batch size to be used in DataLoader objects.
        num_workers (int): Number of worker threads for loading data.
        pin_memory (bool): Whether to pin memory in DataLoader for faster GPU transfers.
        data_config (Dict[str, Any]): Dataset-specific configuration.
        default_paths (Dict[str, str]): Default root directories for each dataset.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the DatasetLoader with the provided configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        self.config = config
        self.batch_size: int = int(config.get("training", {}).get("batch_size", 100))
        self.num_workers: int = int(config.get("num_workers", 4))  # Default number of workers if not specified
        self.pin_memory: bool = torch.cuda.is_available()
        self.data_config: Dict[str, Any] = config.get("data", {})

        # Default dataset root paths (can be overridden externally if needed)
        self.default_paths: Dict[str, str] = {
            "CelebA": "./data/CelebA",
            "Waterbirds": "./data/Waterbirds",
            "Fitzpatrick": "./data/Fitzpatrick"
        }

    def _build_transforms(self, phase: str, image_size: int, augmentations: List[str]) -> transforms.Compose:
        """Constructs a torchvision transformation pipeline based on the phase and configuration.

        Args:
            phase (str): 'train' for training transformations, 'eval' for evaluation transformations.
            image_size (int): Target image size (e.g., 224 or 256).
            augmentations (List[str]): List of augmentation techniques to apply.

        Returns:
            transforms.Compose: Composed transformation pipeline.
        """
        transform_list: List[Any] = []

        if phase == "train":
            # Use random resized crop if "random cropping" is specified; otherwise, a fixed resize.
            if any(a.lower() == "random cropping" for a in augmentations):
                transform_list.append(transforms.RandomResizedCrop(image_size))
            else:
                transform_list.append(transforms.Resize((image_size, image_size)))

            # Add random horizontal flipping if specified.
            if any(a.lower() == "horizontal flipping" for a in augmentations):
                transform_list.append(transforms.RandomHorizontalFlip())
        else:
            # For evaluation, resize first then center crop to ensure deterministic behavior.
            transform_list.append(transforms.Resize(image_size + 32))
            transform_list.append(transforms.CenterCrop(image_size))

        # Convert image to tensor.
        transform_list.append(transforms.ToTensor())

        # Normalize using ImageNet statistics if "normalization" is specified.
        if any(a.lower() == "normalization" for a in augmentations):
            normalize_mean = [0.485, 0.456, 0.406]
            normalize_std = [0.229, 0.224, 0.225]
            transform_list.append(transforms.Normalize(mean=normalize_mean, std=normalize_std))

        return transforms.Compose(transform_list)

    def load_data(self) -> Dict[str, Dict[str, DataLoader]]:
        """Loads datasets and returns a dictionary of DataLoader objects organized per dataset and split.

        Returns:
            Dict[str, Dict[str, DataLoader]]: Structured as:
                {
                    'CelebA': { 'train': train_loader, 'eval': eval_loader },
                    'Waterbirds': { 'train': train_loader, 'eval': eval_loader },
                    'Fitzpatrick': { 'train': train_loader, 'eval': eval_loader }
                }
        """
        data_loaders: Dict[str, Dict[str, DataLoader]] = {}

        # ---------------------- CelebA ----------------------
        celeba_config = self.data_config.get("CelebA", {})
        celeba_image_size: int = int(celeba_config.get("image_size", 224))
        celeba_augmentations: List[str] = celeba_config.get("augmentations", [])
        celeba_root: str = self.default_paths.get("CelebA", "./data/CelebA")

        celeba_train_transform = self._build_transforms("train", celeba_image_size, celeba_augmentations)
        celeba_eval_transform = self._build_transforms("eval", celeba_image_size, celeba_augmentations)

        try:
            celeba_train_dataset = datasets.CelebA(
                root=celeba_root,
                split="train",
                transform=celeba_train_transform,
                download=True
            )
            celeba_eval_dataset = datasets.CelebA(
                root=celeba_root,
                split="valid",
                transform=celeba_eval_transform,
                download=True
            )
        except Exception as e:
            raise RuntimeError(f"Error loading CelebA dataset: {e}")

        celeba_train_loader = DataLoader(
            celeba_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        celeba_eval_loader = DataLoader(
            celeba_eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        data_loaders["CelebA"] = {
            "train": celeba_train_loader,
            "eval": celeba_eval_loader
        }

        # ---------------------- Waterbirds ----------------------
        waterbirds_config = self.data_config.get("Waterbirds", {})
        waterbirds_image_size: int = int(waterbirds_config.get("image_size", 256))
        waterbirds_augmentations: List[str] = waterbirds_config.get("augmentations", [])
        waterbirds_root: str = self.default_paths.get("Waterbirds", "./data/Waterbirds")

        # Assume folder structure: <root>/train and <root>/val for official splits.
        waterbirds_train_dir = os.path.join(waterbirds_root, "train")
        waterbirds_eval_dir = os.path.join(waterbirds_root, "val")

        waterbirds_train_transform = self._build_transforms("train", waterbirds_image_size, waterbirds_augmentations)
        waterbirds_eval_transform = self._build_transforms("eval", waterbirds_image_size, waterbirds_augmentations)

        if not os.path.isdir(waterbirds_train_dir) or not os.path.isdir(waterbirds_eval_dir):
            raise RuntimeError(f"Waterbirds dataset directories not found at {waterbirds_train_dir} or {waterbirds_eval_dir}")

        waterbirds_train_dataset = datasets.ImageFolder(root=waterbirds_train_dir, transform=waterbirds_train_transform)
        waterbirds_eval_dataset = datasets.ImageFolder(root=waterbirds_eval_dir, transform=waterbirds_eval_transform)

        waterbirds_train_loader = DataLoader(
            waterbirds_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        waterbirds_eval_loader = DataLoader(
            waterbirds_eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        data_loaders["Waterbirds"] = {
            "train": waterbirds_train_loader,
            "eval": waterbirds_eval_loader
        }

        # ---------------------- Fitzpatrick ----------------------
        fitzpatrick_config = self.data_config.get("Fitzpatrick", {})
        fitzpatrick_image_size: int = int(fitzpatrick_config.get("image_size", 224))
        fitzpatrick_augmentations: List[str] = fitzpatrick_config.get("augmentations", [])
        fitzpatrick_root: str = self.default_paths.get("Fitzpatrick", "./data/Fitzpatrick")

        # Assume folder structure: <root>/train and <root>/val for official splits.
        fitzpatrick_train_dir = os.path.join(fitzpatrick_root, "train")
        fitzpatrick_eval_dir = os.path.join(fitzpatrick_root, "val")

        fitzpatrick_train_transform = self._build_transforms("train", fitzpatrick_image_size, fitzpatrick_augmentations)
        fitzpatrick_eval_transform = self._build_transforms("eval", fitzpatrick_image_size, fitzpatrick_augmentations)

        if not os.path.isdir(fitzpatrick_train_dir) or not os.path.isdir(fitzpatrick_eval_dir):
            raise RuntimeError(f"Fitzpatrick dataset directories not found at {fitzpatrick_train_dir} or {fitzpatrick_eval_dir}")

        fitzpatrick_train_dataset = datasets.ImageFolder(root=fitzpatrick_train_dir, transform=fitzpatrick_train_transform)
        fitzpatrick_eval_dataset = datasets.ImageFolder(root=fitzpatrick_eval_dir, transform=fitzpatrick_eval_transform)

        fitzpatrick_train_loader = DataLoader(
            fitzpatrick_train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        fitzpatrick_eval_loader = DataLoader(
            fitzpatrick_eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        data_loaders["Fitzpatrick"] = {
            "train": fitzpatrick_train_loader,
            "eval": fitzpatrick_eval_loader
        }

        return data_loaders


if __name__ == "__main__":
    import yaml

    # Load configuration from config.yaml
    config_path: str = "config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    # Initialize the DatasetLoader with the loaded configuration.
    dataset_loader = DatasetLoader(config_data)
    data: Dict[str, Dict[str, DataLoader]] = dataset_loader.load_data()
    
    # Verify the loaded datasets by printing the number of samples in each split.
    for dataset_name, splits in data.items():
        train_size = len(splits["train"].dataset)
        eval_size = len(splits["eval"].dataset)
        print(f"{dataset_name} - Train samples: {train_size}, Eval samples: {eval_size}")
