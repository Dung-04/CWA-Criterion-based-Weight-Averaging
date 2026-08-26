"""CIFAR-100 loader (``torchvision.datasets.CIFAR100``, read from disk).

torchvision ships 50,000 labelled training images and 10,000 labelled test
images. The official training split is stratified into 45,000 training and
5,000 validation samples; the official test split is left untouched as the
final test set. The dataset is read from disk only (``download=False``):
torchvision expects ``<DATA_ROOT>/cifar-100-python``.
"""
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR100

from configs import Config
from .cached import CachedImageDataset
from .splits import ArraySplit


def _official_splits():
    root = Config.DATA_ROOT
    train = CIFAR100(root=root, train=True, download=Config.DOWNLOAD_DATASET)
    test = CIFAR100(root=root, train=False, download=Config.DOWNLOAD_DATASET)
    return train, test


def load_splits(random_seed=42):
    """Load CIFAR-100 and reserve the official test split for testing."""
    val_ratio = Config.VALIDATION_RATIO
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be strictly between 0 and 1")

    print(f"\nLoading torchvision CIFAR-100 from: {Config.DATA_ROOT}")
    official_train, official_test = _official_splits()

    class_names = list(official_train.classes)
    official_train_labels = [int(label) for label in official_train.targets]
    test_labels = [int(label) for label in official_test.targets]

    # Fixed, stratified holdout so every seed/model sees the same 45k/5k split
    # for a given random_seed.
    train_idx, val_idx = train_test_split(
        np.arange(len(official_train_labels)),
        test_size=val_ratio,
        stratify=official_train_labels,
        random_state=random_seed,
        shuffle=True,
    )
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)

    train_data = ArraySplit(
        official_train.data[train_idx],
        [official_train_labels[idx] for idx in train_idx],
    )
    val_data = ArraySplit(
        official_train.data[val_idx],
        [official_train_labels[idx] for idx in val_idx],
    )
    test_data = ArraySplit(official_test.data, test_labels)

    print("\nDataset split (stratified):")
    print(f"  Train: {len(train_data):,} images")
    print(f"  Val:   {len(val_data):,} images (from official train)")
    print(f"  Test:  {len(test_data):,} images (official test, held out)")
    print(f"  Classes: {len(class_names)}")
    print(f"  Random seed: {random_seed}")

    return (
        train_data,
        list(train_data.labels),
        val_data,
        list(val_data.labels),
        test_data,
        test_labels,
        class_names,
    )


def build_dataset(split, transform):
    return CachedImageDataset(split, transform=transform)


def verify(**_kwargs):
    """Check that every on-disk CIFAR-100 image decodes."""
    official_train, official_test = _official_splits()
    corrupted = []
    total = 0

    for split_name, split in (
        (Config.OFFICIAL_TRAIN_SPLIT, official_train),
        (Config.OFFICIAL_TEST_SPLIT, official_test),
    ):
        label_counts = Counter(int(label) for label in split.targets)
        print(f"  {split_name}: {len(split):,} images, {len(label_counts)} classes")
        for idx in range(len(split)):
            total += 1
            try:
                image, _ = split[idx]
                image.convert("RGB")
            except Exception as exc:
                corrupted.append((f"{split_name}[{idx}]", str(exc)[:80]))

    return total, corrupted
