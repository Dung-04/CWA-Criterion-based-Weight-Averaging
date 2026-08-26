"""ImageFolder-style loader for the agricultural leaf-disease datasets.

Layout expected::

    <DATASET_PATH>/<class_name>/<image>.jpg

Each class is shuffled with the run seed and split TRAIN/VAL/TEST by ratio, so
every class keeps its proportion in all three splits.
"""
import os
import random
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset

from configs import Config
from .splits import PathSplit


class FolderImageDataset(Dataset):
    """Reads images from disk on access, tolerating corrupted files."""

    def __init__(self, split, transform=None):
        self.image_paths = split["image"]
        self.labels = split["label"]
        self.transform = transform
        self._corrupted_cache = set()  # Report each corrupted file only once

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, label

        except (OSError, IOError) as e:
            # Handle corrupted/broken images
            if idx not in self._corrupted_cache:
                self._corrupted_cache.add(idx)
                print(f"\n  Warning: Corrupted image at {img_path}: {str(e)[:50]}")

            # Return a black image with correct dimensions as fallback
            if self.transform:
                dummy_image = Image.new('RGB', (Config.IMAGE_SIZE, Config.IMAGE_SIZE), (0, 0, 0))
                dummy_image = self.transform(dummy_image)
                return dummy_image, label

            return torch.zeros(3, Config.IMAGE_SIZE, Config.IMAGE_SIZE), label


def load_splits(random_seed=42):
    """Split the dataset directory into train/val/test.

    IMPORTANT: uses a fixed random seed so every model sees identical
    train/val/test splits.

    Returns:
        train_data, train_labels, val_data, val_labels, test_data, test_labels,
        class_names
    """
    dataset_path = Config.DATASET_PATH
    train_ratio = Config.TRAIN_RATIO
    val_ratio = Config.VAL_RATIO

    # CRITICAL: Set random seed for reproducible train/val/test splits
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    class_dirs = sorted([d for d in os.listdir(dataset_path)
                        if os.path.isdir(os.path.join(dataset_path, d))])

    print(f"\nFound {len(class_dirs)} classes: {class_dirs}")

    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_dirs)}

    class_images = defaultdict(list)
    for class_name in class_dirs:
        class_path = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                class_images[class_name].append(os.path.join(class_path, img_name))

    print("\nDataset statistics:")
    total_images = 0
    for class_name in class_dirs:
        count = len(class_images[class_name])
        total_images += count
        print(f"  {class_name}: {count} images")
    print(f"  Total: {total_images} images")

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    for class_name in class_dirs:
        images = sorted(class_images[class_name])  # Deterministic base order
        random.shuffle(images)                     # Shuffle with seeded random

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        label = class_to_idx[class_name]

        for paths, labels, subset in (
            (train_paths, train_labels, images[:n_train]),
            (val_paths, val_labels, images[n_train:n_train + n_val]),
            (test_paths, test_labels, images[n_train + n_val:]),
        ):
            paths.extend(subset)
            labels.extend([label] * len(subset))

    print("\nData split:")
    print(f"  Train: {len(train_paths)} images")
    print(f"  Val: {len(val_paths)} images")
    print(f"  Test: {len(test_paths)} images")

    # VERIFICATION: Print first few samples for reproducibility check
    print(f"\n✓ Reproducibility check (random_seed={random_seed}):")
    print(f"  First train sample: {Path(train_paths[0]).name if train_paths else 'N/A'}")
    print(f"  First val sample: {Path(val_paths[0]).name if val_paths else 'N/A'}")
    print(f"  First test sample: {Path(test_paths[0]).name if test_paths else 'N/A'}")

    return (
        PathSplit(train_paths, train_labels),
        train_labels,
        PathSplit(val_paths, val_labels),
        val_labels,
        PathSplit(test_paths, test_labels),
        test_labels,
        class_dirs,
    )


def build_dataset(split, transform):
    return FolderImageDataset(split, transform=transform)


def verify(**_kwargs):
    """Report images that PIL cannot decode."""
    corrupted = []
    total = 0
    dataset_path = Config.DATASET_PATH

    class_dirs = sorted([d for d in os.listdir(dataset_path)
                        if os.path.isdir(os.path.join(dataset_path, d))])

    for class_name in class_dirs:
        class_path = os.path.join(dataset_path, class_name)
        for img_name in sorted(os.listdir(class_path)):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue
            img_path = os.path.join(class_path, img_name)
            total += 1
            try:
                with Image.open(img_path) as image:
                    image.convert('RGB')
            except Exception as exc:
                corrupted.append((img_path, str(exc)[:80]))

    return total, corrupted
