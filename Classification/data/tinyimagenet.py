"""Tiny ImageNet loader (Hugging Face ``datasets``).

The dataset contains 100,000 labelled training images and 10,000 labelled
validation images. The official training split is stratified into 90,000
training and 10,000 validation samples; the official ``valid`` split is left
untouched as the final test set.
"""
from collections import Counter

from datasets import load_dataset as load_hf_dataset  # Hugging Face `datasets`

from configs import Config
from .cached import CachedImageDataset


def _load_hf():
    dataset_name = Config.HF_DATASET_NAME
    print(f"\nLoading Hugging Face dataset: {dataset_name}")
    dataset = load_hf_dataset(dataset_name)

    required_splits = {Config.HF_TRAIN_SPLIT, Config.HF_TEST_SPLIT}
    missing_splits = required_splits.difference(dataset.keys())
    if missing_splits:
        raise ValueError(
            f"Dataset is missing required split(s): {sorted(missing_splits)}"
        )

    official_train = dataset[Config.HF_TRAIN_SPLIT]
    official_test = dataset[Config.HF_TEST_SPLIT]

    required_columns = {"image", "label"}
    for split_name, split_dataset in (
        (Config.HF_TRAIN_SPLIT, official_train),
        (Config.HF_TEST_SPLIT, official_test),
    ):
        missing_columns = required_columns.difference(split_dataset.column_names)
        if missing_columns:
            raise ValueError(
                f"Split '{split_name}' is missing column(s): {sorted(missing_columns)}"
            )

    return official_train, official_test


def load_splits(random_seed=42):
    """Load Tiny ImageNet and reserve the official valid split for testing."""
    val_ratio = Config.VALIDATION_RATIO
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be strictly between 0 and 1")

    official_train, official_test = _load_hf()

    split = official_train.train_test_split(
        test_size=val_ratio,
        stratify_by_column="label",
        seed=random_seed,
    )
    train_data = split["train"]
    val_data = split["test"]
    test_data = official_test

    class_names = list(official_train.features["label"].names)
    train_labels = [int(label) for label in train_data["label"]]
    val_labels = [int(label) for label in val_data["label"]]
    test_labels = [int(label) for label in test_data["label"]]

    print("\nDataset split (stratified):")
    print(f"  Train: {len(train_data):,} images")
    print(f"  Val:   {len(val_data):,} images (from official train)")
    print(f"  Test:  {len(test_data):,} images (official valid, held out)")
    print(f"  Classes: {len(class_names)}")
    print(f"  Random seed: {random_seed}")

    return (
        train_data,
        train_labels,
        val_data,
        val_labels,
        test_data,
        test_labels,
        class_names,
    )


def build_dataset(split, transform):
    return CachedImageDataset(split, transform=transform)


def verify(**_kwargs):
    """Check that every cached Tiny ImageNet image decodes."""
    official_train, official_test = _load_hf()
    corrupted = []
    total = 0

    for split_name, split in (
        (Config.HF_TRAIN_SPLIT, official_train),
        (Config.HF_TEST_SPLIT, official_test),
    ):
        label_counts = Counter(int(label) for label in split["label"])
        print(f"  {split_name}: {len(split):,} images, {len(label_counts)} classes")
        for idx in range(len(split)):
            total += 1
            try:
                split[idx]["image"].convert("RGB")
            except Exception as exc:
                corrupted.append((f"{split_name}[{idx}]", str(exc)[:80]))

    return total, corrupted
